"""Stage-major batches and scheduled automations, with fake providers and stages."""

import asyncio
from datetime import datetime

import httpx
import pytest

from urakatsu.core import Stage
from urakatsu.core.power import task_xml
from urakatsu.core.registry import registry
from urakatsu.core.stop import StopToken
from urakatsu.pipeline import Pipeline, ProjectStore
from urakatsu.pipeline.batch import BatchChapter, BatchStatus, needs, phases_for, run_batch
from urakatsu.runtimes import RuntimeManager
from urakatsu.scene import RasterLayerKind, Translation
from urakatsu.service import automations as automations_module
from urakatsu.service.app import build_app
from urakatsu.service.automations import AutomationBody, ModelProfile, Schedule, Target
from urakatsu.service.servicer import UrakatsuServicer
from urakatsu.service.workspace import Workspace

from fakes import FailingTranslator, FakeDetector, FakeInpainter, FakeOcr, FakeTranslator, fake_config, fake_processors, fake_registry, page_png

LOG: list[str] = []


class Logged:
    def _load(self):
        LOG.append(f"load {self.id}")

    def _unload(self):
        LOG.append(f"unload {self.id}")


class LoggedDetector(Logged, FakeDetector):
    id = "logged-detector"


class LoggedOcr(Logged, FakeOcr):
    id = "logged-ocr"


class LoggedTranslator(Logged, FakeTranslator):
    id = "logged-translator"


class BigTranslator(Logged, FakeTranslator):
    id = "big-translator"

    async def translate(self, request) -> list[str]:
        return [f"BIG:{segment}" for segment in request]


class LoggedInpainter(Logged, FakeInpainter):
    id = "logged-inpainter"


LOGGED = (LoggedDetector, LoggedOcr, LoggedTranslator, BigTranslator, LoggedInpainter)


def logged_config():
    config = fake_config("logged-translator")
    config.detection.model = "logged-detector"
    config.ocr.model = "logged-ocr"
    config.inpainting.model = "logged-inpainter"
    return config


def logged_pipeline() -> Pipeline:
    providers = fake_registry()
    for cls in LOGGED:
        providers.add(cls)
    return Pipeline(logged_config(), RuntimeManager("cpu"), providers, fake_processors())


@pytest.fixture
def project(isolated_home):
    store = ProjectStore(isolated_home / "projects")
    project_id = store.create("test")
    pages = [store.import_page(project_id, f"p{i}.png", page_png()) for i in range(3)]
    return store, project_id, pages


# ---- batch runner --------------------------------------------------------------------------------


def test_phases_run_in_dependency_order_with_inpainting_last():
    assert phases_for([Stage.INPAINTING, Stage.TRANSLATION, Stage.DETECTION, Stage.OCR]) == [Stage.DETECTION, Stage.OCR, Stage.TRANSLATION, Stage.INPAINTING]
    assert needs(Stage.TRANSLATION, Stage.DETECTION) and needs(Stage.TRANSLATION, Stage.OCR)
    assert not needs(Stage.INPAINTING, Stage.TRANSLATION) and not needs(Stage.DETECTION, Stage.DETECTION)


async def test_batch_runs_one_model_at_a_time_over_every_page(project):
    store, project_id, pages = project
    LOG.clear()
    pipeline = logged_pipeline()
    session = store.session(project_id)
    status = BatchStatus()
    result = await run_batch(pipeline, session, [BatchChapter("c", "Chapter 1", pages)], Stage.all(), StopToken(), status=status)

    assert not result.stopped and not result.failed
    # Each model loads once, does every page, and is unloaded before the next one loads.
    assert LOG == [
        "load logged-detector",
        "unload logged-detector",
        "load logged-ocr",
        "unload logged-ocr",
        "load logged-translator",
        "unload logged-translator",
        "load logged-inpainter",
        "unload logged-inpainter",
    ]
    assert pipeline.loaded() == {}
    assert status.state == "finished" and status.phases == 4 and status.done == status.total == 3
    snapshot = session.snapshot()
    for page in pages:
        (layer,) = snapshot.text_layers(page)
        assert snapshot.component(snapshot.layer_content(layer), Translation).text.value == "EN:テスト"
        assert snapshot.raster_layers(page, RasterLayerKind.CLEANUP)

    # A second batch finds nothing left for detection and inpainting, and keeps the results.
    LOG.clear()
    await run_batch(pipeline, session, [BatchChapter("c", "Chapter 1", pages)], [Stage.DETECTION, Stage.INPAINTING], StopToken())
    assert len(session.snapshot().text_layers(pages[0])) == 1


async def test_batch_profile_models_apply_for_the_batch_only(project):
    store, project_id, pages = project
    pipeline = logged_pipeline()
    base = pipeline.config
    profile = ModelProfile(name="Overnight", translation=base.translation.model.model_copy(update={"provider": "big-translator"}))
    session = store.session(project_id)
    result = await run_batch(pipeline, session, [BatchChapter("c", "C", pages)], Stage.all(), StopToken(), config=profile.apply(base), restore=lambda: base)
    assert any(stage is Stage.TRANSLATION and model.startswith("big-translator") for stage, model in result.models)
    snapshot = session.snapshot()
    (layer,) = snapshot.text_layers(pages[0])
    assert snapshot.component(snapshot.layer_content(layer), Translation).text.value == "BIG:テスト"
    assert pipeline.config is base and pipeline.config.provider_for(Stage.TRANSLATION) == "logged-translator"


async def test_a_failing_stage_skips_only_the_stages_that_need_it(project):
    store, project_id, pages = project
    FailingTranslator.calls = 0
    FailingTranslator.oom_once = False
    pipeline = Pipeline(fake_config("failing-translator"), RuntimeManager("cpu"), fake_registry(), fake_processors())
    session = store.session(project_id)
    result = await run_batch(pipeline, session, [BatchChapter("c", "C", pages)], Stage.all(), StopToken())
    assert set(result.failed) == set(pages)
    assert all(errors == [errors[0]] and errors[0].startswith("translation:") for errors in result.failed.values())
    # Inpainting does not need translation, so every page was still cleaned.
    assert all(session.snapshot().raster_layers(page, RasterLayerKind.CLEANUP) for page in pages)


async def test_a_stopped_batch_ends_and_unloads(project):
    store, project_id, pages = project
    LOG.clear()
    stop = StopToken()
    stop.stop()
    pipeline = logged_pipeline()
    result = await run_batch(pipeline, store.session(project_id), [BatchChapter("c", "C", pages)], Stage.all(), stop)
    assert result.stopped and not result.models and pipeline.loaded() == {}


# ---- schedules -----------------------------------------------------------------------------------


def at(text: str) -> float:
    return datetime.fromisoformat(text).timestamp()


def test_daily_schedule_finds_the_next_chosen_weekday():
    schedule = Schedule(kind="daily", time="1:30", days=[0, 2])  # Mondays and Wednesdays
    assert schedule.time == "01:30"
    # 2026-10-24 is a Saturday.
    assert schedule.next_after(at("2026-10-24T12:00")) == at("2026-10-26T01:30")
    assert schedule.next_after(at("2026-10-26T01:30")) == at("2026-10-28T01:30")
    assert Schedule(kind="daily", time="23:00").next_after(at("2026-10-24T22:00")) == at("2026-10-24T23:00")


def test_once_and_manual_schedules():
    once = Schedule(kind="once", time="02:00", date="2026-10-25")
    assert once.next_after(at("2026-10-24T20:00")) == at("2026-10-25T02:00")
    assert once.next_after(at("2026-10-25T03:00")) is None
    assert Schedule(kind="manual").next_after(0) is None
    with pytest.raises(ValueError):
        Schedule(time="25:00")


def test_wake_task_definition_wakes_the_computer():
    xml = task_xml(datetime(2026, 10, 25, 0, 58, 30, 123), "cmd.exe", '/c start "" "a&b"')
    assert "<WakeToRun>true</WakeToRun>" in xml
    assert "<StartBoundary>2026-10-25T00:58:30</StartBoundary>" in xml
    assert "a&amp;b" in xml


# ---- automations in the workspace ------------------------------------------------------------


@pytest.fixture
def workspace(isolated_home, monkeypatch):
    fakes = (FakeDetector, FakeOcr, FakeTranslator, FakeInpainter, *LOGGED)
    registry.discover()
    for cls in fakes:
        registry.add(cls)
    ws = Workspace(device="cpu")
    monkeypatch.setattr(ws.automations.wake, "schedule", lambda when: None)  # never touch the real Task Scheduler
    config = logged_config()
    ws.pipeline = Pipeline(config, RuntimeManager("cpu"), registry, fake_processors())
    ws.set_config(config)
    yield ws
    for cls in fakes:
        registry.remove(cls.stage, cls.id)


def chaptered(ws: Workspace) -> tuple[str, list[str]]:
    project_id = ws.create_project("Series").id
    for number in (1, 2, 3):
        chapter = ws.store.create_chapter(project_id, f"Chapter {number}")
        for page in range(2):
            ws.add_page(project_id, f"{number}-{page}.png", page_png(), None, chapter)
    view = ws.project_view(project_id)
    return project_id, [chapter.id for chapter in view.chapters if chapter.page_ids]


async def drain(ws: Workspace) -> None:
    while ws.automations._worker is not None and not ws.automations._worker.done():
        await asyncio.sleep(0.01)


async def test_pending_automation_processes_unfinished_chapters_in_order(workspace):
    ws = workspace
    project_id, chapters = chaptered(ws)
    profile = ws.automations.save_profile(ModelProfile(name="Overnight", translation=ws.config.translation.model.model_copy(update={"provider": "big-translator"})))
    automation = ws.automations.create(project_id, AutomationBody(name="Nightly", schedule=Schedule(kind="manual"), target=Target(kind="pending", limit=2), profile=profile.id))

    run = ws.automations.run_now(project_id, automation.id)
    assert ws.automations.run_now(project_id, automation.id) is run  # not queued twice
    await drain(ws)

    (record, *_) = ws.automations.runs(project_id)
    assert record.id == run.id and record.status == "completed" and record.profile == "Overnight"
    assert [c.id for c in record.chapters] == chapters[:2] and all(c.status == "done" for c in record.chapters)
    assert any(model.startswith("translation: big-translator") for model in record.models)
    assert ws.automations.list(project_id)[0].completed_chapters == chapters[:2]
    assert ws.pipeline.config.provider_for(Stage.TRANSLATION) == "logged-translator"  # settings are back
    assert ws.jobs[f"auto_{run.id}"].status == "completed"

    # The next run takes the chapter that is left, then there is nothing to do.
    ws.automations.run_now(project_id, automation.id)
    await drain(ws)
    assert [c.id for c in ws.automations.runs(project_id)[0].chapters] == chapters[2:]
    ws.automations.run_now(project_id, automation.id)
    await drain(ws)
    latest = ws.automations.runs(project_id)[0]
    assert latest.status == "completed" and latest.chapters == []


async def test_interactive_runs_wait_while_a_batch_holds_the_pipeline(workspace, monkeypatch):
    from urakatsu.core import Operation
    from urakatsu.core.errors import ProjectBusy

    ws = workspace
    project_id, chapters = chaptered(ws)
    automation = ws.automations.create(project_id, AutomationBody(schedule=Schedule(kind="manual"), target=Target(kind="chapters", chapter_ids=chapters[:1])))
    refused: list[bool] = []
    real = automations_module.run_batch

    async def observe(*args, **kwargs):
        try:
            await ws.start_run(project_id, Operation.full())
        except ProjectBusy:
            refused.append(True)
        return await real(*args, **kwargs)

    monkeypatch.setattr(automations_module, "run_batch", observe)
    ws.automations.run_now(project_id, automation.id)
    await drain(ws)
    assert refused == [True]
    assert ws.automations.runs(project_id)[0].status == "completed"


async def test_due_automations_are_queued_and_rescheduled(workspace, monkeypatch):
    ws = workspace
    project_id, _ = chaptered(ws)
    automation = ws.automations.create(project_id, AutomationBody(schedule=Schedule(kind="daily", time="01:00")))
    queued: list[str] = []
    monkeypatch.setattr(ws.automations, "_kick", lambda: queued.append("kick"))
    now = automation.next_run + 5
    ws.automations.tick(now)
    assert queued and ws.automations._queue[0].trigger == "schedule"
    assert automation.next_run == Schedule(kind="daily", time="01:00").next_after(now)

    once = ws.automations.create(project_id, AutomationBody(schedule=Schedule(kind="once", time="01:00", date="2099-01-01")))
    ws.automations.tick(once.next_run + 1)
    assert not once.enabled and once.next_run is None


def test_missed_runs_catch_up_only_when_recent(workspace, monkeypatch):
    ws = workspace
    project_id, _ = chaptered(ws)
    recent = ws.automations.create(project_id, AutomationBody(name="recent"))
    stale = ws.automations.create(project_id, AutomationBody(name="stale"))
    now = 2_000_000_000.0
    recent.next_run = now - 3600
    stale.next_run = now - 2 * automations_module.CATCH_UP_SECONDS
    ws.automations._save(project_id)

    fresh = Workspace(device="cpu")
    monkeypatch.setattr(fresh.automations.wake, "schedule", lambda when: None)
    monkeypatch.setattr(automations_module.time, "time", lambda: now)
    monkeypatch.setattr(automations_module.asyncio, "create_task", lambda coro: coro.close())
    fresh.automations.start()
    by_name = {a.name: a for a in fresh.automations.list(project_id)}
    assert by_name["recent"].next_run == now - 3600  # the next tick runs it
    assert by_name["stale"].next_run > now


def test_automation_validation(workspace):
    ws = workspace
    project_id, _ = chaptered(ws)
    with pytest.raises(ValueError):
        ws.automations.create(project_id, AutomationBody(stages=[]))
    with pytest.raises(ValueError):
        ws.automations.create(project_id, AutomationBody(target=Target(kind="chapters")))
    with pytest.raises(ValueError):
        ws.automations.create(project_id, AutomationBody(target=Target(download=True)))  # no source link
    with pytest.raises(KeyError):
        ws.automations.create(project_id, AutomationBody(profile="nope"))
    with pytest.raises(KeyError):
        ws.automations.save_profile(ModelProfile(name="bad", detection="no-such-model"))


async def test_json_routes(workspace):
    ws = workspace
    project_id, chapters = chaptered(ws)
    app = build_app(ws, UrakatsuServicer(ws), serve_ui=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test/api/v1") as client:
        profile = (await client.post("/automations/profiles", json={"name": "Best", "ocr": "logged-ocr"})).json()
        assert (await client.get("/automations/profiles")).json()["profiles"][0]["id"] == profile["id"]
        assert (await client.post("/automations/profiles", json={"name": "Bad", "ocr": "missing"})).status_code == 404

        body = {"name": "Night", "schedule": {"kind": "daily", "time": "02:15", "days": [5]}, "target": {"kind": "chapters", "chapter_ids": chapters}, "profile": profile["id"]}
        created = (await client.post(f"/projects/{project_id}/automations", json=body)).json()
        (automation,) = created["automations"]
        assert automation["schedule"]["time"] == "02:15" and automation["next_run"] is not None
        assert (await client.post(f"/projects/{project_id}/automations", json={"stages": []})).status_code == 400

        body["enabled"] = False
        updated = (await client.put(f"/projects/{project_id}/automations/{automation['id']}", json=body)).json()
        assert updated["automations"][0]["next_run"] is None

        ran = (await client.post(f"/projects/{project_id}/automations/{automation['id']}/run")).json()
        assert ran["runs"][0]["status"] in ("queued", "waiting", "running", "completed")
        await drain(ws)
        status = (await client.get("/automations/status")).json()
        assert status["active"] is None and "wake_supported" in status
        runs = (await client.get(f"/projects/{project_id}/automations")).json()["runs"]
        assert runs[0]["status"] == "completed" and len(runs[0]["chapters"]) == 3

        assert (await client.delete(f"/automations/profiles/{profile['id']}")).status_code == 204
        assert (await client.get(f"/projects/{project_id}/automations")).json()["automations"][0]["profile"] is None
        deleted = (await client.delete(f"/projects/{project_id}/automations/{automation['id']}")).json()
        assert deleted["automations"] == []
