"""Pipeline orchestration: port of pipeline.rs, execution.rs, stage_runner.rs.

- Pages enter in project order; a stage runs on one page; completed detection makes
  that page's later stages eligible immediately; at most one job per stage runs.
  a page finishes before the next one starts (``scheduler.py``).
- Every job reads the latest committed snapshot; each committed result is its own
  undo step; a patch whose observed state changed is rejected (rebase rule).
- Accelerator inference uses one admission lane, including remote translation. On
  out-of-memory: release the lane, unload the other models, wait, retry once. Any
  other failure unloads the failing stage and fails the run.
- Stopping is cooperative and discards an uncommitted result.
- Idle unload (an Urakatsu addition): once no run has used the models for
  ``PipelineConfig.idle_unload_minutes``, every loaded model is unloaded, so a model used
  once does not hold device memory other programs need. The next run loads it again.
  Code that uses a model outside ``execute`` holds ``in_use`` so it is not unloaded under it.
- Manual unload (an Urakatsu addition): the user can unload one stage's model or all of them
  (``unload``), outside a run.
- Every unload goes through ``_unload``, which also hands the memory back: dropping a model
  leaves PyTorch's share of it reserved, out of reach of native servers and other programs.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Optional

from urakatsu.core.config import PipelineConfig
from urakatsu.core.errors import ModelLoadError, ProcessingError, ProjectBusy, UrakatsuError, is_out_of_memory
from urakatsu.core.progress import Progress, ProgressKind, ProgressSink, StepPreview, StepSink, reporting_steps
from urakatsu.core.registry import Registry, registry as default_registry
from urakatsu.core.stage import Operation, Stage
from urakatsu.core.stop import StopToken, stopping
from urakatsu.pipeline.images import ImageCache
from urakatsu.pipeline.scheduler import Scheduler
from urakatsu.pipeline.scope import NormalizedScope, ProjectScope, Scope
from urakatsu.pipeline.stages import InpaintingMask, StageInput, StageProcessor, processors
from urakatsu.providers.base import Provider
from urakatsu.runtimes.manager import RuntimeManager
from urakatsu.runtimes.torch_runtime import release_memory
from urakatsu.scene import Patch, Session, Snapshot

log = logging.getLogger(__name__)

OOM_SETTLE_SECONDS = 1.0


class RunStatus(str, Enum):
    COMPLETED = "completed"
    STOPPED = "stopped"


class ErrorKind(str, Enum):
    INVALID_INPUT = "invalid_input"
    MODEL_LOAD = "model_load"
    PROCESSING = "processing"
    INVALID_OUTPUT = "invalid_output"
    COMMIT = "commit"


class PipelineError(UrakatsuError):
    def __init__(self, kind: ErrorKind, stage: Optional[Stage], message: str):
        super().__init__(message)
        self.kind = kind
        self.stage = stage


@dataclass
class Request:
    operation: Operation = field(default_factory=Operation.full)
    scope: Scope = field(default_factory=ProjectScope)
    stop: StopToken = field(default_factory=StopToken)
    progress: Optional[ProgressSink] = None
    inpainting_mask: Optional[InpaintingMask] = None
    # Redo: stages run on pages they already did (``StageInput.force``).
    force: bool = False


@dataclass
class Report:
    status: RunStatus
    base: int
    final_revision: int
    completed: int
    total: int
    elapsed_ms: float


class Pipeline:
    def __init__(
        self,
        config: PipelineConfig,
        runtimes: RuntimeManager | None = None,
        providers: Registry | None = None,
        stage_processors: dict[Stage, StageProcessor] | None = None,
    ):
        self.config = config
        self.runtimes = runtimes or RuntimeManager(config.device)
        self.registry = providers or default_registry
        self._providers: dict[Stage, Provider] = {}
        self._stage_processors = stage_processors
        self._accelerator = asyncio.Semaphore(1)
        self._execution = asyncio.Lock()
        self._last_used = time.monotonic()
        self._users = 0  # ``in_use`` blocks open now
        self._idle_watch: Optional[asyncio.Task] = None

    @property
    def _processors(self) -> dict[Stage, StageProcessor]:
        if self._stage_processors is None:
            self._stage_processors = processors()
        return self._stage_processors

    @_processors.setter
    def _processors(self, value: dict[Stage, StageProcessor]) -> None:
        self._stage_processors = value


    # -- configuration ---------------------------------------------------------------------------

    def configure(self, config: PipelineConfig) -> None:
        """Applies new settings; providers whose selection, runtime or settings changed are unloaded."""
        previous = self.config
        self.config = config
        if config.idle_unload_minutes != previous.idle_unload_minutes:
            self._watch_idle(restart=True)
        if config.device != previous.device:
            self.runtimes.select_device(config.device)
            self.unload_all()
            return
        stale = []
        for stage, provider in list(self._providers.items()):
            wanted = {config.provider_for(stage)} | ({config.fill_provider()} if stage is Stage.INPAINTING else set())
            changed = (
                provider.id not in wanted
                or config.runtime_for(provider.id) != previous.runtime_for(provider.id)
                or config.settings_for(stage, provider.id) != previous.settings_for(stage, provider.id)
            )
            if changed:
                stale.append(provider)
                del self._providers[stage]
        self._unload(*stale)

    def provider(self, stage: Stage, fill: bool = False) -> Provider:
        """The stage's provider. ``fill`` asks for the generative fill model, which takes the
        inpainting slot while it runs, so it is loaded only when used and never beside the
        inpainting model (the next inpainting run swaps them back)."""
        wanted = self.config.fill_provider() if fill else self.config.provider_for(stage)
        existing = self._providers.get(stage)
        if existing is not None and existing.id == wanted:
            return existing
        if existing is not None:
            self._unload(existing)
        self.registry.sync(self.config.providers)
        cls = self.registry.get(stage, wanted)
        provider = cls(self.config.settings_for(stage, wanted), self.runtimes, self.config.runtime_for(wanted))
        self._providers[stage] = provider
        return provider

    def unload_all(self, keep: Stage | None = None) -> list[Stage]:
        wanted = [(stage, provider) for stage, provider in self._providers.items() if stage is not keep]
        unloaded = self._unload(*[provider for _, provider in wanted])
        return [stage for stage, provider in wanted if provider in unloaded]

    def _unload(self, *providers: Provider) -> list[Provider]:
        """Unloads ``providers`` and hands back what they held; returns the ones that were loaded.
        One that fails stays loaded, for the next unload, and does not keep the others."""
        unloaded = []
        for provider in providers:
            try:
                if provider.unload():
                    unloaded.append(provider)
            except Exception:  # noqa: BLE001 - e.g. a native server that would not stop
                log.exception("could not unload %s", provider.id)
        if unloaded:
            release_memory()
        return unloaded

    def loaded(self) -> dict[Stage, str]:
        return {stage: p.id for stage, p in self.loaded_providers().items()}

    def loaded_providers(self) -> dict[Stage, Provider]:
        return {stage: p for stage, p in self._providers.items() if p.loaded}

    @property
    def busy(self) -> bool:
        """A run, or code outside one (restitching, an agent turn), is using the models."""
        return self._execution.locked() or self._users > 0

    async def unload(self, stage: Stage | None = None) -> list[Stage]:
        """Unloads one stage's model, or every model, when the user asks; returns the stages
        unloaded. Never under a run: it raises ``ProjectBusy`` while one uses the models."""
        if self.busy:
            raise ProjectBusy("a run is using the models; stop it or wait for it to finish, then unload them")
        async with self._execution:  # no run starts while the models unload
            if stage is None:
                unloaded = self.unload_all()
            else:
                provider = self._providers.get(stage)
                unloaded = [stage] if provider is not None and self._unload(provider) else []
        if unloaded:
            log.info("unloaded %s on request", ", ".join(s.value for s in unloaded))
        return unloaded

    # -- execution -------------------------------------------------------------------------------

    async def execute(self, session: Session, request: Request) -> Report:
        async with self._execution:
            try:
                return await _Execution(self, session, request).run()
            finally:
                self.touch()

    async def detect_layout(self, images: list, stop: StopToken | None = None) -> list:
        """The detection provider's ``LayoutDetections`` for each image, outside a stage: webtoon
        restitching (``pipeline/stitch.py``) asks where text is before it cuts. Inference takes
        the accelerator lane like a stage, so it interleaves with a running pipeline."""
        stop = stop or StopToken()
        provider = self.provider(Stage.DETECTION)
        results = []
        await self._accelerator.acquire()
        try:
            with self.in_use(), stopping(stop):
                await provider.load()
                for image in images:
                    if stop.stopped:
                        break
                    try:
                        results.append(await provider.detect(image))
                    except Exception as error:  # one retry: a remote worker's job can fail once
                        if stop.stopped or is_out_of_memory(error):
                            raise
                        log.warning("detection failed on a restitch tile, retrying: %s", error)
                        results.append(await provider.detect(image))
        finally:
            self._accelerator.release()
        return results

    # -- idle unload -----------------------------------------------------------------------------

    def touch(self) -> None:
        """Marks the models as used now, restarting the idle countdown. Code that runs a provider
        outside ``execute`` (the agent's local model) calls it too."""
        self._last_used = time.monotonic()
        self._watch_idle()

    @contextmanager
    def in_use(self) -> Iterator[None]:
        """Keeps idle unload away for the block: ``execute`` has the execution lock for that, code
        that uses a model outside it (restitching, an agent turn) has this."""
        self._users += 1
        self.touch()
        try:
            yield
        finally:
            self._users -= 1
            self.touch()

    def _watch_idle(self, restart: bool = False) -> None:
        """Starts the countdown task; ``restart`` replaces a running one (the timeout changed)."""
        if self._idle_watch is not None and not self._idle_watch.done():
            if not restart:
                return
            self._idle_watch.cancel()
        try:
            self._idle_watch = asyncio.get_running_loop().create_task(self._unload_when_idle())
        except RuntimeError:  # no event loop: nothing to schedule the countdown on
            self._idle_watch = None

    async def _unload_when_idle(self) -> None:
        while True:
            timeout = self.config.idle_unload_minutes * 60
            if timeout <= 0 or not self.loaded():
                return
            remaining = timeout if self._users else self._last_used + timeout - time.monotonic()
            if remaining > 0:
                await asyncio.sleep(remaining)
                continue
            async with self._execution:  # no run starts while the models unload
                if self._users or time.monotonic() - self._last_used < timeout:
                    continue
                unloaded = self.unload_all()
            if unloaded:
                log.info("unloaded %s after %g idle minutes", ", ".join(stage.value for stage in unloaded), self.config.idle_unload_minutes)
            return

    # -- stage runner ----------------------------------------------------------------------------

    async def _run_stage(self, stage: Stage, input: StageInput, stop: StopToken, progress: Optional[ProgressSink]):
        started = time.perf_counter()
        processor = self._processors[stage]
        fill = stage is Stage.INPAINTING and input.inpainting_mask is not None and input.inpainting_mask.generative
        try:
            provider = self.provider(stage, fill)
        except Exception as error:
            return _Completion(input.page, stage, "", 0.0, PipelineError(ErrorKind.MODEL_LOAD, stage, str(error)))
        model = provider.id
        if stop.stopped:
            return _Completion(input.page, stage, model, 0.0, _STOPPED)
        try:
            if processor.skip(input, provider):
                # Done before (a forced run keeps it only when the user edited it), or nothing to do.
                reason = ("kept" if input.force else "done") if processor.done(input) else None
                return _Completion(input.page, stage, model, 0.0, _SKIPPED, reason)
        except Exception as error:
            return _Completion(input.page, stage, model, 0.0, self._stage_error(stage, provider, ErrorKind.PROCESSING, error))

        await self._accelerator.acquire()
        released = False
        try:
            if stop.stopped:
                return _Completion(input.page, stage, model, 0.0, _STOPPED)
            outcome = await self._load_and_process(stage, processor, provider, input, stop, progress)
            if isinstance(outcome, _Attempt) and is_out_of_memory(outcome.error) and not stop.stopped:
                self._accelerator.release()
                released = True
                # The text alone: a log record that keeps the exception would keep what follows from working.
                log.warning("retrying %s on page %s after memory pressure: %s", stage.value, input.page, str(outcome.error))
                outcome = None  # its traceback's frames hold the failed attempt's tensors
                await self._accelerator.acquire()
                released = False
                self.unload_all(keep=stage)
                release_memory()  # always: the failed attempt's memory, even with nothing else to unload
                await asyncio.sleep(OOM_SETTLE_SECONDS)
                if stop.stopped:
                    return _Completion(input.page, stage, model, 0.0, _STOPPED)
                outcome = await self._load_and_process(stage, processor, provider, input, stop, progress)
            if isinstance(outcome, _Attempt):
                outcome = self._stage_error(stage, provider, outcome.kind, outcome.error)
            return _Completion(input.page, stage, model, (time.perf_counter() - started) * 1000, outcome)
        finally:
            if not released:
                self._accelerator.release()

    async def _load_and_process(self, stage, processor, provider, input, stop, progress):
        _emit(progress, Progress(kind=ProgressKind.LOADING, page=input.page, stage=stage, model=provider.id))
        try:
            with stopping(stop):
                await provider.load()
        except Exception as error:
            return _STOPPED if stop.stopped else _Attempt(ErrorKind.MODEL_LOAD, error)
        if stop.stopped:
            return _STOPPED
        _emit(progress, Progress(kind=ProgressKind.RUNNING, page=input.page, stage=stage, model=provider.id))
        try:
            with reporting_steps(_step_sink(progress, input.page, stage, provider.id)), stopping(stop):
                patch = await processor.process(input, provider)
        except Exception as error:
            return _STOPPED if stop.stopped else _Attempt(ErrorKind.PROCESSING, error)
        return _SKIPPED if patch.is_empty() else patch

    def _stage_error(self, stage: Stage, provider: Provider, kind: ErrorKind, error: BaseException) -> PipelineError:
        self._unload(provider)
        prefix = f"failed to load {provider.id}" if kind is ErrorKind.MODEL_LOAD else f"{provider.id} failed"
        log.debug("%s", prefix, exc_info=error)
        return PipelineError(kind, stage, f"{prefix}: {error}")


@dataclass
class _Attempt:
    kind: ErrorKind
    error: BaseException


@dataclass
class _Completion:
    page: str
    stage: Stage
    model: str
    elapsed_ms: float
    outcome: object
    # Why a skipped stage was skipped: "done" (its results are there), "kept" (a forced run left
    # what the user edited), or None (nothing to do).
    reason: Optional[str] = None


_SKIPPED = object()
_STOPPED = object()


PREVIEW_QUALITY = 80


def _step_sink(sink: Optional[ProgressSink], page: str, stage: Stage, model: str) -> Optional[StepSink]:
    """Turns a provider's sampling previews (reported from its worker thread) into STEP events on the loop."""
    if sink is None:
        return None
    loop = asyncio.get_running_loop()

    def report(step: StepPreview) -> None:
        jpeg = None
        if step.image is not None and step.box is not None:
            buffer = io.BytesIO()
            step.image.convert("RGB").save(buffer, format="JPEG", quality=PREVIEW_QUALITY)
            jpeg = buffer.getvalue()
        event = Progress(kind=ProgressKind.STEP, page=page, stage=stage, model=model, step=step.step, steps=step.total, preview=jpeg, preview_box=step.box if jpeg else None)
        loop.call_soon_threadsafe(_emit, sink, event)

    return report


def _emit(sink: Optional[ProgressSink], progress: Progress) -> None:
    if sink is not None:
        try:
            sink(progress)
        except Exception:  # a broken sink must not break the run
            log.exception("progress sink raised")


# The stages a user-painted mask applies to.
MASKED_STAGES = frozenset({Stage.INPAINTING, Stage.DECENSOR})


def _censorship_stages(stages: list[Stage], mode: str, operation: Operation) -> list[Stage]:
    """``stages`` without the censorship stages ``mode`` leaves out of this run."""
    dropped = set()
    named = operation.stage == Stage.DECENSOR or Stage.DECENSOR in operation.stages
    if mode != "remove" and not (mode == "detect" and named):
        dropped.add(Stage.DECENSOR)
    if mode == "off":
        dropped.add(Stage.CENSOR_DETECTION)
    kept = [stage for stage in stages if stage not in dropped]
    if not kept:
        raise ValueError("censorship is off (Settings, Pipeline, Censorship)")
    return kept


def _colorization_stages(stages: list[Stage], mode: str) -> list[Stage]:
    """``stages`` without colorization while ``mode`` is off."""
    if mode != "off":
        return stages
    kept = [stage for stage in stages if stage is not Stage.COLORIZATION]
    if not kept:
        raise ValueError("colorization is off (Settings, Pipeline, Colorization)")
    return kept


class _Execution:
    def __init__(self, pipeline: Pipeline, session: Session, request: Request):
        self.started = time.perf_counter()
        self.pipeline = pipeline
        self.session = session
        self.stop = request.stop
        self.progress = request.progress
        self.scene: Snapshot = session.snapshot()
        self.base = self.scene.revision
        try:
            stages = request.operation.resolve()
            if pipeline.config.sound_effects.mode != "redraw" and Stage.SOUND_EFFECTS in stages:
                # The sound effects stage only exists in a run while sound effects are redrawn.
                stages = [stage for stage in stages if stage is not Stage.SOUND_EFFECTS]
                if not stages:
                    raise ValueError("sound effects are not set to redraw (Settings, Pipeline, Sound effects)")
            elif pipeline.config.sound_effects.mode == "redraw" and Stage.INPAINTING in stages and request.inpainting_mask is None and Stage.SOUND_EFFECTS not in stages:
                # Inpainting leaves sound effects set to redraw in the art for the sound effects stage,
                # so a run that inpaints (Inpaint for a chapter or the project) redraws them too.
                stages = [*stages, Stage.SOUND_EFFECTS]
            if pipeline.config.watermarks.mode != "remove":
                # The watermarks stage only exists in a run while watermarks are removed.
                if Stage.WATERMARKS in stages:
                    stages = [stage for stage in stages if stage is not Stage.WATERMARKS]
                    if not stages:
                        raise ValueError("watermarks are not set to be removed (pipeline config, watermarks.mode)")
            elif Stage.DETECTION in stages and Stage.WATERMARKS not in stages:
                # Detection reads the page with its watermarks removed, so a run that detects removes them first.
                stages = [Stage.WATERMARKS, *stages]
            stages = _censorship_stages(stages, pipeline.config.censorship.mode, request.operation)
            stages = _colorization_stages(stages, pipeline.config.colorization.mode)
            self.scope = NormalizedScope(self.scene, request.scope, stages)
        except (ValueError, KeyError) as error:
            raise PipelineError(ErrorKind.INVALID_INPUT, None, str(error)) from error
        pages = list(self.scope.pages)
        mask = request.inpainting_mask
        if mask is not None and (mask.page not in pages or not MASKED_STAGES.intersection(stages)):
            raise PipelineError(
                ErrorKind.INVALID_INPUT, Stage.INPAINTING, "the inpainting mask page is outside the inpainting scope"
            )
        self.inpainting_mask = mask
        self.force = request.force
        self.scheduler = Scheduler(pages, stages)
        self.images: dict[str, ImageCache] = {}
        self.busy: set[Stage] = set()
        self.completed = 0
        self.failure: Optional[PipelineError] = None
        _emit(self.progress, Progress(kind=ProgressKind.STARTED, pages=pages, stages=stages, total=self.scheduler.total))

    async def run(self) -> Report:
        if self.stop.stopped:
            return self._report(RunStatus.STOPPED)
        running: set[asyncio.Task] = set()
        while True:
            while (job := self._take_ready_job()) is not None:
                stage, input = job
                running.add(asyncio.create_task(self.pipeline._run_stage(stage, input, self.stop, self.progress)))
            if not running:
                break
            done, running = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                completion: _Completion = task.result()
                self.busy.discard(completion.stage)
                if self.stop.stopped or self.failure is not None:
                    continue
                try:
                    self._apply(completion)
                except PipelineError as error:
                    self.failure = error
        return self._finalize()

    def _take_ready_job(self) -> Optional[tuple[Stage, StageInput]]:
        if self.stop.stopped or self.failure is not None:
            return None
        job = self.scheduler.start_next(self.busy)
        if job is None:
            return None
        page, stage = job
        self.busy.add(stage)
        images = self.images.setdefault(page, ImageCache())
        mask = self.inpainting_mask if (stage in MASKED_STAGES and self.inpainting_mask and self.inpainting_mask.page == page) else None
        sound_effects = self.pipeline.config.sound_effects.mode
        return stage, StageInput(self.scene, page, self.scope.entities, self.scope.region(page), images, mask, sound_effects, self.force)

    def _apply(self, completion: _Completion) -> None:
        outcome = completion.outcome
        if isinstance(outcome, PipelineError):
            _emit(self.progress, Progress(kind=ProgressKind.FAILED, page=completion.page, stage=completion.stage, model=completion.model, message=str(outcome)))
            raise outcome
        if outcome is _STOPPED:
            return
        if outcome is _SKIPPED:
            self._mark_complete(completion.page, completion.stage)
            _emit(self.progress, Progress(kind=ProgressKind.SKIPPED, page=completion.page, stage=completion.stage, model=completion.model, message=completion.reason or ""))
            return
        assert isinstance(outcome, Patch)
        if self.stop.stopped:
            return
        try:
            self.scene = self.session.commit(outcome)
        except Exception as error:
            raise PipelineError(ErrorKind.COMMIT, completion.stage, f"failed to commit {completion.stage.value} output for page {completion.page}: {error}") from error
        self._mark_complete(completion.page, completion.stage)
        _emit(
            self.progress,
            Progress(
                kind=ProgressKind.FINISHED,
                page=completion.page,
                stage=completion.stage,
                model=completion.model,
                elapsed_ms=completion.elapsed_ms,
                completed=self.completed,
                total=self.scheduler.total,
            ),
        )

    def _mark_complete(self, page: str, stage: Stage) -> None:
        if self.scheduler.complete(page, stage):
            self.images.pop(page, None)
        self.completed += 1

    def _finalize(self) -> Report:
        if self.failure is not None:
            raise self.failure
        if not self.stop.stopped and self.completed != self.scheduler.total:
            raise PipelineError(
                ErrorKind.INVALID_OUTPUT, None, f"pipeline scheduler stopped after {self.completed} of {self.scheduler.total} work items"
            )
        return self._report(RunStatus.STOPPED if self.stop.stopped else RunStatus.COMPLETED)

    def _report(self, status: RunStatus) -> Report:
        return Report(
            status=status,
            base=self.base,
            final_revision=self.scene.revision,
            completed=self.completed,
            total=self.scheduler.total,
            elapsed_ms=(time.perf_counter() - self.started) * 1000,
        )


__all__ = ["ErrorKind", "Pipeline", "PipelineError", "Report", "Request", "RunStatus", "ModelLoadError", "ProcessingError"]
