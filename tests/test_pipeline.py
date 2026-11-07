"""Orchestration (port of execution.rs / stage_runner.rs) with fake providers and stages."""

import asyncio

import pytest

import urakatsu.pipeline.pipeline as pipeline_module
from urakatsu.core import Operation, Progress, ProgressKind, Stage
from urakatsu.core.stop import interruptible
from urakatsu.pipeline import PagesScope, Pipeline, PipelineError, ProjectStore, Request, RunStatus
from urakatsu.pipeline.scheduler import Scheduler
from urakatsu.runtimes import RuntimeManager
from urakatsu.scene import RasterLayerKind, Translation

from fakes import FailingTranslator, FakeTranslator, fake_config, fake_processors, fake_registry, page_png


@pytest.fixture
def project(isolated_home):
    store = ProjectStore(isolated_home / "projects")
    project_id = store.create("test")
    pages = [store.import_page(project_id, f"p{i}.png", page_png()) for i in range(2)]
    return store, project_id, pages


def make_pipeline(config=None) -> Pipeline:
    return Pipeline(config or fake_config(), RuntimeManager("cpu"), fake_registry(), fake_processors())


def test_operation_resolution_matches_koharu():
    assert Operation.through(Stage.TRANSLATION).resolve() == [Stage.DETECTION, Stage.OCR, Stage.TRANSLATION]
    assert Operation.through(Stage.INPAINTING).resolve() == [Stage.DETECTION, Stage.INPAINTING]
    assert Operation.full().resolve() == Stage.all()
    with pytest.raises(ValueError):
        Operation.select([]).resolve()


def test_scheduler_runs_page_by_page():
    scheduler = Scheduler(["a", "b"], [Stage.DETECTION, Stage.OCR, Stage.INPAINTING])
    assert scheduler.start_next(set()) == ("a", Stage.DETECTION)
    assert scheduler.start_next({Stage.DETECTION}) is None
    scheduler.complete("a", Stage.DETECTION)
    # Both of a's later stages are eligible; b waits although its detection is free.
    assert scheduler.start_next(set()) == ("a", Stage.OCR)
    assert scheduler.start_next({Stage.OCR}) == ("a", Stage.INPAINTING)
    assert scheduler.start_next({Stage.OCR, Stage.INPAINTING}) is None
    scheduler.complete("a", Stage.OCR)
    assert scheduler.start_next({Stage.INPAINTING}) is None
    assert scheduler.complete("a", Stage.INPAINTING)
    assert scheduler.start_next(set()) == ("b", Stage.DETECTION)


def test_scheduler_moves_on_after_a_failed_page():
    scheduler = Scheduler(["a", "b"], [Stage.DETECTION, Stage.OCR])
    assert scheduler.start_next(set()) == ("a", Stage.DETECTION)
    scheduler.complete("a", Stage.DETECTION)
    assert scheduler.fail_page("a") == [Stage.OCR]
    assert scheduler.start_next(set()) == ("b", Stage.DETECTION)


def test_next_redoes_a_finished_page_after_the_running_stage():
    stages = [Stage.DETECTION, Stage.OCR]
    scheduler = Scheduler(["a", "b", "c"], stages)
    for stage in stages:
        assert scheduler.start_next(set()) == ("a", stage)
        scheduler.complete("a", stage)
    assert scheduler.start_next(set()) == ("b", Stage.DETECTION)
    # b's detection is running; a is put first again, forced (Redo).
    assert scheduler.prioritize("a", stages, force=True) == stages
    assert scheduler.total == 8
    assert scheduler.start_next({Stage.DETECTION}) is None  # a waits for the detection lane, b does not go on
    scheduler.complete("b", Stage.DETECTION)
    assert scheduler.start_next(set()) == ("a", Stage.DETECTION)
    assert scheduler.forced("a", Stage.DETECTION) and not scheduler.forced("b", Stage.OCR)
    scheduler.complete("a", Stage.DETECTION)
    assert scheduler.start_next(set()) == ("a", Stage.OCR)
    assert scheduler.complete("a", Stage.OCR)
    # Then the run goes on where it left off: b's OCR, then c.
    assert scheduler.start_next(set()) == ("b", Stage.OCR)
    assert scheduler.complete("b", Stage.OCR)
    assert scheduler.start_next(set()) == ("c", Stage.DETECTION)
    scheduler.complete("c", Stage.DETECTION)
    assert scheduler.start_next(set()) == ("c", Stage.OCR)
    scheduler.complete("c", Stage.OCR)
    assert scheduler.done() and scheduler.completed == scheduler.total == 8


def test_next_moves_a_waiting_page_up_in_the_order_asked():
    scheduler = Scheduler(["a", "b", "c", "d"], [Stage.DETECTION])
    assert scheduler.start_next(set()) == ("a", Stage.DETECTION)
    assert scheduler.prioritize("d", [Stage.DETECTION], force=False) == []  # already queued: nothing added
    scheduler.prioritize("c", [Stage.DETECTION], force=False)
    assert scheduler.total == 4 and not scheduler.forced("d", Stage.DETECTION)
    scheduler.complete("a", Stage.DETECTION)
    order = []
    while (job := scheduler.start_next(set())) is not None:
        order.append(job[0])
        scheduler.complete(*job)
    assert order == ["d", "c", "b"] and scheduler.done()


def test_next_adds_a_page_outside_the_run_and_repeats_a_running_stage():
    scheduler = Scheduler(["a"], [Stage.DETECTION, Stage.OCR])
    assert scheduler.start_next(set()) == ("a", Stage.DETECTION)
    assert scheduler.prioritize("z", [Stage.OCR, Stage.DETECTION], force=True) == [Stage.OCR, Stage.DETECTION]
    assert scheduler.prioritize("a", [Stage.DETECTION], force=True) == [Stage.DETECTION]  # running: once more after
    assert scheduler.total == 5
    scheduler.complete("a", Stage.DETECTION)
    assert scheduler.start_next(set()) == ("z", Stage.DETECTION)  # in stage order, though asked OCR first
    scheduler.complete("z", Stage.DETECTION)
    assert scheduler.start_next(set()) == ("z", Stage.OCR)
    assert scheduler.complete("z", Stage.OCR)
    assert scheduler.start_next(set()) == ("a", Stage.DETECTION)  # the repeat
    scheduler.complete("a", Stage.DETECTION)
    assert scheduler.start_next(set()) == ("a", Stage.OCR)
    assert scheduler.complete("a", Stage.OCR)
    assert scheduler.done() and scheduler.completed == scheduler.total == 5


@pytest.mark.asyncio
async def test_next_in_a_running_job_redoes_a_page_then_goes_on(project):
    store, project_id, pages = project
    first, second = pages
    pipeline = make_pipeline()
    request = Request(operation=Operation.through(Stage.INPAINTING))
    events: list[Progress] = []

    def sink(event: Progress):
        events.append(event)
        # As the second page starts, the user asks for the first again.
        if event.kind is ProgressKind.FINISHED and event.page == second and not any(e.kind is ProgressKind.REQUEUED for e in events):
            assert pipeline.prioritize(request.stop, first, None, force=True)

    request.progress = sink
    report = await pipeline.execute(store.session(project_id), request)
    assert report.status is RunStatus.COMPLETED
    assert report.completed == report.total == 6
    done = [(e.page, e.stage) for e in events if e.kind in (ProgressKind.FINISHED, ProgressKind.SKIPPED)]
    assert done == [
        (first, Stage.DETECTION), (first, Stage.INPAINTING),
        (second, Stage.DETECTION),
        (first, Stage.DETECTION), (first, Stage.INPAINTING),  # forced: done again, not skipped
        (second, Stage.INPAINTING),
    ]
    assert not any(e.kind is ProgressKind.SKIPPED for e in events if e.page == first)
    (requeued,) = [e for e in events if e.kind is ProgressKind.REQUEUED]
    assert requeued.page == first and requeued.total == 6
    # Once the run is over, Next is refused: the caller runs the page on its own.
    assert not pipeline.prioritize(request.stop, first)


@pytest.mark.asyncio
async def test_a_fill_while_a_job_runs_goes_first_in_it(project):
    import io

    from PIL import Image

    from urakatsu.pipeline.stages import InpaintingMask

    store, project_id, pages = project
    first, second = pages
    buffer = io.BytesIO()
    mask_image = Image.new("L", (400, 300), 0)
    mask_image.paste(255, (40, 40, 120, 120))
    mask_image.save(buffer, format="PNG")
    mask = InpaintingMask(page=first, png=buffer.getvalue())
    pipeline = make_pipeline()
    # Inpainting is not one of this run's stages: the fill runs it once all the same.
    request = Request(operation=Operation.through(Stage.TRANSLATION))
    events: list[Progress] = []

    def sink(event: Progress):
        events.append(event)
        if event.kind is ProgressKind.FINISHED and event.page == second and not any(e.kind is ProgressKind.REQUEUED for e in events):
            assert pipeline.prioritize(request.stop, first, [Stage.INPAINTING], mask=mask)

    request.progress = sink
    report = await pipeline.execute(store.session(project_id), request)
    assert report.status is RunStatus.COMPLETED
    assert report.completed == report.total == 7
    done = [(e.page, e.stage) for e in events if e.kind in (ProgressKind.FINISHED, ProgressKind.SKIPPED)]
    assert done == [
        (first, Stage.DETECTION), (first, Stage.OCR), (first, Stage.TRANSLATION),
        (second, Stage.DETECTION),
        (first, Stage.INPAINTING),  # the fill, then the run goes on
        (second, Stage.OCR), (second, Stage.TRANSLATION),
    ]


@pytest.mark.asyncio
async def test_full_run_commits_each_stage(project):
    store, project_id, pages = project
    session = store.session(project_id)
    base = session.snapshot().revision
    events: list[Progress] = []
    report = await make_pipeline().execute(session, Request(progress=events.append))
    assert report.status is RunStatus.COMPLETED
    assert report.completed == report.total == 8
    assert report.final_revision == base + 8  # one commit (undo step) per stage per page
    snapshot = session.snapshot()
    for page in pages:
        (layer,) = snapshot.text_layers(page)
        assert snapshot.component(snapshot.layer_content(layer), Translation).text.value == "EN:テスト"
        assert snapshot.raster_layers(page, RasterLayerKind.CLEANUP)
    assert sum(e.kind is ProgressKind.FINISHED for e in events) == 8

    # Detection and inpainting skip work that already exists.
    events.clear()
    report = await make_pipeline().execute(session, Request(operation=Operation.through(Stage.INPAINTING), progress=events.append))
    assert report.status is RunStatus.COMPLETED
    assert {e.kind for e in events if e.page} == {ProgressKind.SKIPPED} | {ProgressKind.LOADING} - {ProgressKind.LOADING}
    # ...and say so, so the UI can offer to redo them.
    assert {e.message for e in events if e.kind is ProgressKind.SKIPPED} == {"done"}

    # Redo: a forced run does them again.
    events.clear()
    report = await make_pipeline().execute(session, Request(operation=Operation.through(Stage.INPAINTING), progress=events.append, force=True))
    assert report.status is RunStatus.COMPLETED
    assert sum(e.kind is ProgressKind.FINISHED for e in events) == 4


@pytest.mark.asyncio
async def test_stop_discards_and_reports_stopped(project):
    store, project_id, pages = project
    request = Request()

    def stop_after_first(event: Progress):
        if event.kind is ProgressKind.FINISHED:
            request.stop.stop()

    request.progress = stop_after_first
    report = await make_pipeline().execute(store.session(project_id), request)
    assert report.status is RunStatus.STOPPED
    assert report.completed < report.total


@pytest.mark.asyncio
async def test_failure_unloads_and_fails_the_run(project):
    store, project_id, pages = project
    FailingTranslator.calls = 0
    FailingTranslator.oom_once = False
    with pytest.raises(PipelineError) as error:
        await make_pipeline(fake_config("failing-translator")).execute(store.session(project_id), Request(scope=PagesScope((pages[0],))))
    assert error.value.stage is Stage.TRANSLATION and "model exploded" in str(error.value)


@pytest.mark.asyncio
async def test_out_of_memory_retries_once(project, monkeypatch):
    store, project_id, pages = project
    monkeypatch.setattr(pipeline_module, "OOM_SETTLE_SECONDS", 0.0)
    FailingTranslator.calls = 0
    FailingTranslator.oom_once = True
    report = await make_pipeline(fake_config("failing-translator")).execute(store.session(project_id), Request(scope=PagesScope((pages[0],))))
    assert report.status is RunStatus.COMPLETED and FailingTranslator.calls == 2


def test_runtime_selection_per_provider():
    pipeline = make_pipeline(fake_config(runtimes={"fake-detector": "onnx"}))
    assert pipeline.provider(Stage.DETECTION).runtime_name == "onnx"
    pipeline.configure(fake_config(runtimes={"fake-detector": "torch"}))
    assert pipeline.provider(Stage.DETECTION).runtime_name == "torch"
    with pytest.raises(ValueError):
        pipeline.configure(fake_config(runtimes={"fake-detector": "llama"}))
        pipeline.provider(Stage.DETECTION)


def test_config_roundtrip(isolated_home):
    from urakatsu.core import load_config, save_config

    config = fake_config(runtimes={"paddleocr-vl-1.6": "llama"})
    config.providers["openai"] = {"api_key": "sk-test"}
    config.processor["flux2-klein"] = {"prompt": "saved prompt"}
    assert load_config(save_config(config)) == config


class HangingTranslator(FakeTranslator):
    """Waits on a request that never answers, as a slow cloud model does; only a stop ends it."""

    id = "hanging-translator"

    async def translate(self, request) -> list[str]:
        await interruptible(asyncio.sleep(60))
        return []


@pytest.mark.asyncio
async def test_stop_ends_a_stage_waiting_on_a_slow_request_and_frees_the_lane(project):
    store, project_id, pages = project
    registry = fake_registry()
    registry.add(HangingTranslator)
    pipeline = Pipeline(fake_config("hanging-translator"), RuntimeManager("cpu"), registry, fake_processors())
    request = Request(operation=Operation.through(Stage.TRANSLATION), scope=PagesScope((pages[0],)))
    events: list[Progress] = []

    def stop_when_translating(event: Progress):
        events.append(event)
        if event.kind is ProgressKind.RUNNING and event.stage is Stage.TRANSLATION:
            asyncio.get_running_loop().call_later(0.1, request.stop.stop)

    request.progress = stop_when_translating
    report = await asyncio.wait_for(pipeline.execute(store.session(project_id), request), timeout=5)
    assert report.status is RunStatus.STOPPED
    assert not any(e.kind is ProgressKind.FAILED for e in events)  # stopped, not failed

    # The lane is free: the next run starts and finishes.
    report = await asyncio.wait_for(make_pipeline().execute(store.session(project_id), Request(scope=PagesScope((pages[1],)))), timeout=5)
    assert report.status is RunStatus.COMPLETED
