"""Stage-major batch runs: every page of a set of chapters through one stage, then the next.

Not in Koharu. Koharu streams each page through every stage with every model resident
(``pipeline.py``). A batch instead runs one stage at a time over all of its pages and
unloads every model between stages, so each model has the accelerator to itself; that is
what lets an unattended batch use models too large to sit beside the others.

Stage semantics are unchanged: each page of a phase is an ordinary ``Pipeline.execute``
with ``Operation.only``, so prerequisites from earlier phases count as satisfied, each
result is its own undo step, and the stages' skip rules make a re-run continue where the
last one stopped without overwriting user edits. Unlike an interactive run, a failing
page does not fail the batch: it is recorded and left out of the later phases that need
the stage it failed (a page whose translation failed is still inpainted).
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from urakatsu.core.config import PipelineConfig
from urakatsu.core.progress import Progress, ProgressKind
from urakatsu.core.stage import PREREQUISITES, Operation, Stage
from urakatsu.core.stop import StopToken
from urakatsu.pipeline.pipeline import ErrorKind, Pipeline, PipelineError, Request
from urakatsu.pipeline.scope import PagesScope
from urakatsu.scene import Session

log = logging.getLogger(__name__)

# The order phases run in: inpainting only needs detection, so it goes last and the
# inpainting model never shares the accelerator with the translation model.
PHASE_ORDER = [Stage.DETECTION, Stage.OCR, Stage.TRANSLATION, Stage.INPAINTING]


@dataclass
class BatchChapter:
    id: str
    name: str
    pages: list[str]


@dataclass
class BatchStatus:
    """What the batch is doing now; read by whoever shows its progress."""

    phase: int = 0  # 1-based once started
    phases: int = 0
    stage: Optional[Stage] = None
    model: str = ""
    state: str = "starting"  # starting | loading | running | unloading | finished
    chapter: str = ""
    page: str = ""
    done: int = 0  # pages finished in this phase
    total: int = 0  # pages in this phase


@dataclass
class BatchResult:
    stopped: bool
    # page -> "<stage>: <error>" for every stage that failed on it
    failed: dict[str, list[str]] = field(default_factory=dict)
    # (stage, model) per phase that ran
    models: list[tuple[Stage, str]] = field(default_factory=list)
    elapsed_ms: float = 0.0


def phases_for(stages: list[Stage]) -> list[Stage]:
    wanted = set(stages)
    return [stage for stage in PHASE_ORDER if stage in wanted]


def needs(stage: Stage, other: Stage) -> bool:
    """Whether ``stage`` needs ``other`` to have run first, directly or through another stage."""
    return any(prerequisite is other or needs(prerequisite, other) for prerequisite in PREREQUISITES[stage])


def release_memory() -> None:
    """Hands what the unloaded models held back to the system before the next model loads."""
    gc.collect()
    from urakatsu.runtimes.torch_runtime import release_cache

    release_cache()


async def run_batch(
    pipeline: Pipeline,
    session: Session,
    chapters: list[BatchChapter],
    stages: list[Stage],
    stop: StopToken,
    config: Optional[PipelineConfig] = None,
    restore: Optional[Callable[[], PipelineConfig]] = None,
    status: Optional[BatchStatus] = None,
) -> BatchResult:
    """Runs ``stages`` over ``chapters`` one stage at a time.

    ``config`` selects the models for the batch (the pipeline's own when None); ``restore``
    returns the configuration the pipeline gets back afterwards, read when the batch ends so
    settings saved meanwhile apply.
    """
    started = time.perf_counter()
    status = status or BatchStatus()
    phases = phases_for(stages)
    status.phases = len(phases)
    result = BatchResult(stopped=False)
    chapter_of = {page: chapter for chapter in chapters for page in chapter.pages}
    failed_stages: dict[str, set[Stage]] = {}

    def fail(page: str, stage: Stage, error: BaseException) -> None:
        result.failed.setdefault(page, []).append(f"{stage.value}: {error}")
        failed_stages.setdefault(page, set()).add(stage)
    try:
        if config is not None:
            pipeline.configure(config)
        for index, stage in enumerate(phases, start=1):
            if stop.stopped:
                break
            status.state = "unloading"
            pipeline.unload_all()
            release_memory()
            pages = [
                page
                for chapter in chapters
                for page in chapter.pages
                if not any(needs(stage, failed) for failed in failed_stages.get(page, ()))
            ]
            model = pipeline.config.provider_for(stage)
            if stage is Stage.TRANSLATION and pipeline.config.translation.model.model:
                model = f"{model} · {pipeline.config.translation.model.model}"  # the provider alone says little
            status.phase, status.stage, status.model = index, stage, model
            status.done, status.total, status.page, status.chapter = 0, len(pages), "", ""
            result.models.append((stage, model))
            log.info("batch phase %d/%d: %s with %s on %d pages", index, len(phases), stage.value, model, len(pages))

            def sink(progress: Progress) -> None:
                if progress.kind is ProgressKind.LOADING:
                    status.state = "loading"
                elif progress.kind is ProgressKind.RUNNING:
                    status.state = "running"

            for position, page in enumerate(pages):
                if stop.stopped:
                    break
                status.page, status.chapter = page, chapter_of[page].name
                request = Request(operation=Operation.only(stage), scope=PagesScope((page,)), stop=stop, progress=sink)
                try:
                    await pipeline.execute(session, request)
                except PipelineError as error:
                    if error.kind is ErrorKind.MODEL_LOAD:
                        # The model will not load for any page: the phase cannot run.
                        log.warning("batch phase %s cannot load %s: %s", stage.value, model, error)
                        for rest in pages[position:]:
                            fail(rest, stage, error)
                        break
                    log.warning("batch %s failed on page %s: %s", stage.value, page, error)
                    fail(page, stage, error)
                except Exception as error:  # noqa: BLE001 - one page's crash does not end the night
                    log.exception("batch %s crashed on page %s", stage.value, page)
                    fail(page, stage, error)
                status.done = position + 1
        result.stopped = stop.stopped
    finally:
        status.state = "unloading"
        pipeline.unload_all()
        release_memory()
        if restore is not None:
            pipeline.configure(restore())
        status.state = "finished"
        result.elapsed_ms = (time.perf_counter() - started) * 1000
    return result


__all__ = ["BatchChapter", "BatchResult", "BatchStatus", "PHASE_ORDER", "needs", "phases_for", "release_memory", "run_batch"]
