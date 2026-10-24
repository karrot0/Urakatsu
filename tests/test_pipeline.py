"""Orchestration (port of execution.rs / stage_runner.rs) with fake providers and stages."""

import pytest

import urakatsu.pipeline.pipeline as pipeline_module
from urakatsu.core import Operation, Progress, ProgressKind, Stage
from urakatsu.pipeline import PagesScope, Pipeline, PipelineError, ProjectStore, Request, RunStatus
from urakatsu.pipeline.scheduler import Scheduler
from urakatsu.runtimes import RuntimeManager
from urakatsu.scene import RasterLayerKind, Translation

from fakes import FailingTranslator, fake_config, fake_processors, fake_registry, page_png


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


def test_scheduler_streams_pages_in_order():
    scheduler = Scheduler(["a", "b"], [Stage.DETECTION, Stage.OCR])
    assert scheduler.start_next(set()) == ("a", Stage.DETECTION)
    assert scheduler.start_next({Stage.DETECTION}) is None
    scheduler.complete("a", Stage.DETECTION)
    assert {scheduler.start_next(set()), scheduler.start_next({Stage.OCR})} == {("a", Stage.OCR), ("b", Stage.DETECTION)}


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
