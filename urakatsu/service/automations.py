"""Scheduled, unattended batches per project ("automations"); not a Koharu feature.

An automation names what to process (chosen chapters, or every chapter not yet processed,
optionally after downloading new ones from the project's source), which stages, which
model profile, and when (by hand, once, or daily at a time). When one is due it is queued;
one batch runs at a time, since there is one accelerator. A batch runs stage by stage
(``urakatsu.pipeline.batch``), so a profile can name models that only fit on the GPU alone.

- Automations and their run history live in ``projects/<id>/automations.json``; model
  profiles, shared by every project, in ``URAKATSU_HOME/automation-profiles.json``.
- A run that was due while Urakatsu was closed runs at startup if it is at most
  ``CATCH_UP_SECONDS`` late; older ones are skipped to the next time.
- While a batch runs the computer is kept awake, interactive runs wait their turn
  (``Workspace.start_run`` refuses them), and settings saved meanwhile apply afterwards.
- An automation with ``wake`` set has the computer woken for it (``core.power.WakeTask``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from datetime import date, datetime, timedelta
from datetime import time as clock
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from urakatsu.core import files
from urakatsu.core.config import ModelChoice, ModelSelection, PipelineConfig, home
from urakatsu.core.power import WakeTask, keep_awake
from urakatsu.core.registry import registry
from urakatsu.core.stage import Stage
from urakatsu.core.stop import StopToken
from urakatsu.pipeline.batch import BatchChapter, BatchStatus, phases_for, run_batch
from urakatsu.pipeline.projects import chapters_of

from .views import page_view

if TYPE_CHECKING:
    from .workspace import Job, Workspace

log = logging.getLogger(__name__)

TICK_SECONDS = 20.0
CATCH_UP_SECONDS = 6 * 3600.0
MAX_RUNS = 30
DOWNLOAD_POLL_SECONDS = 3.0
DOWNLOAD_TIMEOUT_SECONDS = 3 * 3600.0
FILE = "automations.json"


# ---- models ----------------------------------------------------------------------------------


class ModelProfile(BaseModel):
    """Models an automation runs instead of the current settings; an empty field keeps the setting."""

    id: str = ""
    name: str = "Profile"
    detection: Optional[str] = None
    ocr: Optional[str] = None
    inpainting: Optional[str] = None
    translation: Optional[ModelSelection] = None

    def apply(self, config: PipelineConfig) -> PipelineConfig:
        result = config.model_copy(deep=True)
        if self.detection:
            result.detection = ModelChoice(model=self.detection)
        if self.ocr:
            result.ocr = ModelChoice(model=self.ocr)
        if self.inpainting:
            result.inpainting = ModelChoice(model=self.inpainting)
        if self.translation and self.translation.provider:
            result.translation = result.translation.model_copy(update={"model": self.translation.model_copy()})
        return result

    def validate_providers(self) -> None:
        for stage, provider in ((Stage.DETECTION, self.detection), (Stage.OCR, self.ocr), (Stage.INPAINTING, self.inpainting)):
            if provider:
                registry.get(stage, provider)  # KeyError for an unknown model
        if self.translation and self.translation.provider:
            registry.get(Stage.TRANSLATION, self.translation.provider)


class Schedule(BaseModel):
    kind: Literal["manual", "once", "daily"] = "daily"
    time: str = "01:00"  # local HH:MM
    date: Optional[str] = None  # YYYY-MM-DD, for "once"
    days: list[int] = Field(default_factory=lambda: list(range(7)))  # weekdays for "daily", Monday = 0

    @field_validator("time")
    @classmethod
    def _time(cls, value: str) -> str:
        hours, minutes = value.split(":")
        clock(int(hours), int(minutes))
        return f"{int(hours):02d}:{int(minutes):02d}"

    @field_validator("days")
    @classmethod
    def _days(cls, value: list[int]) -> list[int]:
        return sorted({day for day in value if 0 <= day <= 6})

    def next_after(self, after: float) -> Optional[float]:
        """The first scheduled time later than ``after`` (epoch seconds), or None."""
        hours, minutes = (int(part) for part in self.time.split(":"))
        at = clock(hours, minutes)
        if self.kind == "once":
            if not self.date:
                return None
            when = datetime.combine(date.fromisoformat(self.date), at).timestamp()
            return when if when > after else None
        if self.kind == "daily" and self.days:
            start = datetime.fromtimestamp(after).date()
            for offset in range(8):
                day = start + timedelta(days=offset)
                if day.weekday() in self.days:
                    when = datetime.combine(day, at).timestamp()
                    if when > after:
                        return when
        return None


class Target(BaseModel):
    kind: Literal["chapters", "pending"] = "pending"
    chapter_ids: list[str] = Field(default_factory=list)  # "chapters"
    limit: Optional[int] = None  # "pending": at most this many chapters per run
    download: bool = False  # "pending": first download the source's chapters the project lacks


class Automation(BaseModel):
    id: str = ""
    name: str = "Overnight"
    enabled: bool = True
    schedule: Schedule = Field(default_factory=Schedule)
    target: Target = Field(default_factory=Target)
    stages: list[Stage] = Field(default_factory=Stage.all)
    profile: Optional[str] = None  # a ModelProfile id; None runs the current settings
    wake: bool = False
    next_run: Optional[float] = None
    created_at: float = 0.0
    # Chapters a run finished without failures; "pending" leaves them out afterwards.
    completed_chapters: list[str] = Field(default_factory=list)


class AutomationBody(BaseModel):
    """The fields a client sets; the rest are kept by the server."""

    name: str = "Overnight"
    enabled: bool = True
    schedule: Schedule = Field(default_factory=Schedule)
    target: Target = Field(default_factory=Target)
    stages: list[Stage] = Field(default_factory=Stage.all)
    profile: Optional[str] = None
    wake: bool = False


class ChapterResult(BaseModel):
    id: str
    name: str
    pages: int
    failed: int = 0
    errors: list[str] = Field(default_factory=list)
    status: Literal["done", "partial", "failed", "stopped"] = "done"


class RunProgress(BaseModel):
    phase: int = 0
    phases: int = 0
    stage: Optional[str] = None
    model: str = ""
    state: str = "starting"
    chapter: str = ""
    done: int = 0
    total: int = 0


RunState = Literal["queued", "waiting", "downloading", "running", "completed", "stopped", "failed"]


class RunRecord(BaseModel):
    id: str
    automation_id: str
    automation_name: str
    project_id: str
    project_name: str = ""
    trigger: Literal["schedule", "manual"] = "manual"
    status: RunState = "queued"
    queued_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    profile: Optional[str] = None  # the profile's name, as it was
    stages: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)  # "<stage>: <model>" per phase that ran
    downloaded: int = 0
    chapters: list[ChapterResult] = Field(default_factory=list)
    progress: Optional[RunProgress] = None
    error: Optional[str] = None


class ProjectAutomations(BaseModel):
    automations: list[Automation] = Field(default_factory=list)
    runs: list[RunRecord] = Field(default_factory=list)  # newest first


# ---- manager ---------------------------------------------------------------------------------


class Automations:
    def __init__(self, workspace: "Workspace"):
        self.workspace = workspace
        self.store = workspace.store
        self.wake = WakeTask()
        self._data: dict[str, ProjectAutomations] = {}
        self._queue: deque[RunRecord] = deque()
        self._active: Optional[RunRecord] = None
        self._stop: Optional[StopToken] = None
        self._status: Optional[BatchStatus] = None
        self._loop_task: Optional[asyncio.Task] = None
        self._worker: Optional[asyncio.Task] = None
        self._profiles_path = home() / "automation-profiles.json"

    @property
    def busy(self) -> bool:
        """Whether a batch holds the pipeline."""
        return self._active is not None and self._active.status == "running"

    # -- storage -----------------------------------------------------------------------------------

    def _path(self, project_id: str) -> Path:
        return self.store.root / project_id / FILE

    def _load(self, project_id: str) -> ProjectAutomations:
        if project_id not in self._data:
            self.store.session(project_id)  # KeyError for an unknown project
            path = self._path(project_id)
            try:
                self._data[project_id] = ProjectAutomations.model_validate_json(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                self._data[project_id] = ProjectAutomations()
            except ValueError as error:
                log.warning("ignoring unreadable %s: %s", path, error)
                self._data[project_id] = ProjectAutomations()
        return self._data[project_id]

    def _save(self, project_id: str) -> None:
        data = self._data.get(project_id)
        if data is None or not (self.store.root / project_id).exists():
            return
        files.write_text(self._path(project_id), data.model_dump_json(indent=1))

    def _load_all(self) -> None:
        for path in sorted(self.store.root.glob(f"*/{FILE}")):
            try:
                self._load(path.parent.name)
            except KeyError:
                continue

    def forget(self, project_id: str) -> None:
        """Drops a deleted project's automations and queued runs."""
        self._data.pop(project_id, None)
        self._queue = deque(run for run in self._queue if run.project_id != project_id)
        if self._active is not None and self._active.project_id == project_id and self._stop is not None:
            self._stop.stop()
        self._update_wake()

    # -- profiles ----------------------------------------------------------------------------------

    def profiles(self) -> list[ModelProfile]:
        try:
            raw = json.loads(self._profiles_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except ValueError as error:
            log.warning("ignoring unreadable %s: %s", self._profiles_path, error)
            return []
        return [ModelProfile.model_validate(item) for item in raw.get("profiles", [])]

    def _write_profiles(self, profiles: list[ModelProfile]) -> None:
        files.write_text(self._profiles_path, json.dumps({"profiles": [p.model_dump(mode="json") for p in profiles]}, indent=1))

    def profile(self, profile_id: str) -> ModelProfile:
        for profile in self.profiles():
            if profile.id == profile_id:
                return profile
        raise KeyError(f"model profile {profile_id} does not exist")

    def save_profile(self, profile: ModelProfile) -> ModelProfile:
        profile.validate_providers()
        profiles = self.profiles()
        if not profile.id:
            profile.id = uuid.uuid4().hex[:10]
            profiles.append(profile)
        else:
            index = next((i for i, p in enumerate(profiles) if p.id == profile.id), None)
            if index is None:
                raise KeyError(f"model profile {profile.id} does not exist")
            profiles[index] = profile
        self._write_profiles(profiles)
        return profile

    def delete_profile(self, profile_id: str) -> None:
        profiles = self.profiles()
        self._write_profiles([p for p in profiles if p.id != profile_id])
        for project_id, data in self._data.items():
            changed = False
            for automation in data.automations:
                if automation.profile == profile_id:
                    automation.profile = None
                    changed = True
            if changed:
                self._save(project_id)

    # -- automations -------------------------------------------------------------------------------

    def list(self, project_id: str) -> list[Automation]:
        return self._load(project_id).automations

    def _find(self, project_id: str, automation_id: str) -> Automation:
        for automation in self._load(project_id).automations:
            if automation.id == automation_id:
                return automation
        raise KeyError(f"automation {automation_id} does not exist")

    def _check(self, project_id: str, body: AutomationBody) -> None:
        if not body.stages:
            raise ValueError("choose at least one stage")
        if body.profile:
            self.profile(body.profile)
        if body.schedule.kind == "once":
            if not body.schedule.date:
                raise ValueError("a one-time schedule needs a date")
            date.fromisoformat(body.schedule.date)
        if body.schedule.kind == "daily" and not body.schedule.days:
            raise ValueError("choose at least one day")
        if body.target.kind == "chapters" and not body.target.chapter_ids:
            raise ValueError("choose at least one chapter")
        if body.target.limit is not None and body.target.limit < 1:
            raise ValueError("the chapter limit must be at least 1")
        if body.target.download and self.workspace.links.get(project_id) is None:
            raise ValueError("this project is not linked to a source, so there is nothing to download")

    def create(self, project_id: str, body: AutomationBody) -> Automation:
        self._check(project_id, body)
        automation = Automation(**body.model_dump(), id=uuid.uuid4().hex[:10], created_at=time.time())
        automation.next_run = automation.schedule.next_after(time.time()) if automation.enabled else None
        self._load(project_id).automations.append(automation)
        self._save(project_id)
        self._update_wake()
        return automation

    def update(self, project_id: str, automation_id: str, body: AutomationBody) -> Automation:
        self._check(project_id, body)
        automation = self._find(project_id, automation_id)
        for key in type(body).model_fields:
            setattr(automation, key, getattr(body, key))
        automation.next_run = automation.schedule.next_after(time.time()) if automation.enabled else None
        self._save(project_id)
        self._update_wake()
        return automation

    def delete(self, project_id: str, automation_id: str) -> None:
        data = self._load(project_id)
        self._find(project_id, automation_id)
        data.automations = [a for a in data.automations if a.id != automation_id]
        self._queue = deque(run for run in self._queue if run.automation_id != automation_id)
        self._save(project_id)
        self._update_wake()

    def reset(self, project_id: str, automation_id: str) -> Automation:
        """Forgets which chapters the automation finished, so "pending" considers them again."""
        automation = self._find(project_id, automation_id)
        automation.completed_chapters = []
        self._save(project_id)
        return automation

    # -- runs ----------------------------------------------------------------------------------------

    def runs(self, project_id: str) -> list[RunRecord]:
        """Queued and running runs of the project, then its history, newest first."""
        live = [self._live(run) for run in ([self._active] if self._active else []) + list(self._queue) if run.project_id == project_id]
        return live + self._load(project_id).runs

    def status(self) -> dict:
        return {
            "active": self._live(self._active).model_dump(mode="json") if self._active else None,
            "queued": [run.model_dump(mode="json") for run in self._queue],
            "wake_supported": self.wake.supported,
        }

    def _live(self, run: RunRecord) -> RunRecord:
        if run is self._active and self._status is not None:
            s = self._status
            run.progress = RunProgress(
                phase=s.phase, phases=s.phases, stage=s.stage.value if s.stage else None, model=s.model, state=s.state, chapter=s.chapter, done=s.done, total=s.total
            )
        return run

    def run_now(self, project_id: str, automation_id: str) -> RunRecord:
        automation = self._find(project_id, automation_id)
        return self._enqueue(project_id, automation, "manual")

    def _enqueue(self, project_id: str, automation: Automation, trigger: Literal["schedule", "manual"]) -> RunRecord:
        for run in ([self._active] if self._active else []) + list(self._queue):
            if run.automation_id == automation.id and run.status in ("queued", "waiting", "downloading", "running"):
                return run  # already on its way
        run = RunRecord(
            id=uuid.uuid4().hex[:12],
            automation_id=automation.id,
            automation_name=automation.name,
            project_id=project_id,
            project_name=self.store.name(project_id),
            trigger=trigger,
            queued_at=time.time(),
        )
        self._queue.append(run)
        self._kick()
        return run

    def stop(self, run_id: str) -> None:
        if self._active is not None and self._active.id == run_id:
            if self._stop is not None:
                self._stop.stop()
            return
        before = len(self._queue)
        self._queue = deque(run for run in self._queue if run.id != run_id)
        if len(self._queue) == before:
            raise KeyError(f"run {run_id} is not queued or running")

    # -- timer ---------------------------------------------------------------------------------------

    def start(self) -> None:
        """Starts the timer on the running event loop."""
        if self._loop_task is not None:
            return
        self._load_all()
        now = time.time()
        for project_id, data in self._data.items():
            changed = False
            for automation in data.automations:
                if not automation.enabled:
                    continue
                if automation.next_run is None or automation.next_run < now - CATCH_UP_SECONDS:
                    # Never scheduled, or missed by too much: the next time from now on.
                    upcoming = automation.schedule.next_after(now)
                    if upcoming != automation.next_run:
                        automation.next_run = upcoming
                        changed = True
            if changed:
                self._save(project_id)
        self._update_wake()
        self._loop_task = asyncio.create_task(self._timer())

    async def _timer(self) -> None:
        while True:
            try:
                self.tick(time.time())
            except Exception:  # noqa: BLE001 - the timer must keep going
                log.exception("automation timer failed")
            await asyncio.sleep(TICK_SECONDS)

    def tick(self, now: float) -> None:
        """Queues every automation that is due."""
        changed_any = False
        for project_id, data in list(self._data.items()):
            changed = False
            for automation in data.automations:
                if not automation.enabled or automation.next_run is None or automation.next_run > now:
                    continue
                self._enqueue(project_id, automation, "schedule")
                automation.next_run = automation.schedule.next_after(now)
                if automation.schedule.kind == "once":
                    automation.enabled = False
                changed = True
            if changed:
                self._save(project_id)
                changed_any = True
        if changed_any:
            self._update_wake()

    def _update_wake(self) -> None:
        upcoming = [
            automation.next_run
            for data in self._data.values()
            for automation in data.automations
            if automation.enabled and automation.wake and automation.next_run is not None
        ]
        self.wake.schedule(min(upcoming) if upcoming else None)

    def _kick(self) -> None:
        if self._worker is None or self._worker.done():
            try:
                self._worker = asyncio.get_running_loop().create_task(self._drain())
            except RuntimeError:  # no loop yet: start() drains
                pass

    async def _drain(self) -> None:
        while self._queue:
            run = self._queue.popleft()
            self._active = run
            try:
                await self._execute(run)
            except Exception as error:  # noqa: BLE001 - recorded on the run
                log.exception("automation run %s crashed", run.id)
                run.status = "failed"
                run.error = str(error) or error.__class__.__name__
            finally:
                run.finished_at = run.finished_at or time.time()
                run.progress = None
                self._status = None
                self._active = None
                self._stop = None
                self._record(run)

    def _record(self, run: RunRecord) -> None:
        try:
            data = self._load(run.project_id)
        except KeyError:
            return  # the project was deleted
        data.runs.insert(0, run)
        del data.runs[MAX_RUNS:]
        self._save(run.project_id)

    async def close(self) -> None:
        if self._loop_task is not None:
            self._loop_task.cancel()
        self._queue.clear()
        if self._stop is not None:
            self._stop.stop()
        if self._worker is not None and not self._worker.done():
            await asyncio.gather(self._worker, return_exceptions=True)
        # The wake task stays registered: it starts Urakatsu for the next scheduled run.

    # -- one run -----------------------------------------------------------------------------------

    async def _execute(self, run: RunRecord) -> None:
        from .workspace import Job

        ws = self.workspace
        automation = self._find(run.project_id, run.automation_id)
        stop = StopToken()
        self._stop = stop
        job = Job(id=f"auto_{run.id}", project_id=run.project_id, stop=stop)
        ws.jobs[job.id] = job
        run.stages = [stage.value for stage in phases_for(automation.stages)]
        profile = self.profile(automation.profile) if automation.profile else None
        run.profile = profile.name if profile else None
        try:
            with keep_awake():
                if automation.target.kind == "pending" and automation.target.download and not stop.stopped:
                    run.status = "downloading"
                    run.downloaded = await self._download(run.project_id, automation, stop)
                # Interactive runs finish first; from "running" on, the batch holds the pipeline.
                run.status = "waiting"
                while any(j.status == "running" for j in ws.jobs.values() if j is not job) and not stop.stopped:
                    await asyncio.sleep(1.0)
                chapters = self._chapters(run.project_id, automation)
                if stop.stopped:
                    run.status = "stopped"
                    return
                if not chapters:
                    run.status = "completed"
                    return
                run.status = "running"
                run.started_at = time.time()
                self._status = BatchStatus()
                config = profile.apply(ws.config) if profile else None
                result = await run_batch(
                    ws.pipeline,
                    ws.session(run.project_id),
                    chapters,
                    automation.stages,
                    stop,
                    config=config,
                    restore=lambda: ws.config,
                    status=self._status,
                )
                run.models = [f"{stage.value}: {model}" for stage, model in result.models]
                run.chapters = [self._result(chapter, result.failed, result.stopped) for chapter in chapters]
                run.status = "stopped" if result.stopped else "completed"
                known = set(automation.completed_chapters)
                finished = [chapter.id for chapter in run.chapters if chapter.status == "done" and chapter.id not in known]
                if finished and automation.target.kind == "pending":
                    automation.completed_chapters = [*automation.completed_chapters, *finished]
                    self._save(run.project_id)
        except Exception as error:
            run.status = "failed"
            run.error = str(error) or error.__class__.__name__
            log.exception("automation %s on project %s failed", automation.name, run.project_id)
        finally:
            run.finished_at = time.time()
            job.status = "failed" if run.status == "failed" else ("stopped" if run.status == "stopped" else "completed")
            job.error = run.error
            job.finish()
            try:
                self.store.touch(run.project_id)
            except Exception:  # noqa: BLE001 - the project may be gone
                pass

    @staticmethod
    def _result(chapter: BatchChapter, failed: dict[str, list[str]], stopped: bool) -> ChapterResult:
        broken = [page for page in chapter.pages if page in failed]
        errors = sorted({error for page in broken for error in failed[page]})
        if broken:
            status = "failed" if len(broken) == len(chapter.pages) else "partial"
        else:
            status = "stopped" if stopped else "done"
        return ChapterResult(id=chapter.id, name=chapter.name, pages=len(chapter.pages), failed=len(broken), errors=errors[:5], status=status)

    def _chapters(self, project_id: str, automation: Automation) -> list[BatchChapter]:
        """The chapters a run processes, in project order."""
        snapshot = self.workspace.session(project_id).snapshot()
        chapters = [chapter for chapter in chapters_of(snapshot) if chapter.pages]
        target = automation.target
        if target.kind == "chapters":
            wanted = set(target.chapter_ids)
            chosen = [chapter for chapter in chapters if chapter.id in wanted]
        else:
            stages = {stage.value for stage in automation.stages}
            done = set(automation.completed_chapters)

            def pending(chapter) -> bool:
                for page in chapter.pages:
                    view = page_view(snapshot, page)
                    if any(status.stage in stages and not status.done for status in view.stages):
                        return True
                return False

            chosen = [chapter for chapter in chapters if chapter.id not in done and pending(chapter)]
            if target.limit:
                chosen = chosen[: target.limit]
        return [BatchChapter(id=chapter.id, name=chapter.name, pages=list(chapter.pages)) for chapter in chosen]

    async def _download(self, project_id: str, automation: Automation, stop: StopToken) -> int:
        """Downloads the source's chapters the project lacks (at most the target's limit); returns how many arrived."""
        links = self.workspace.links
        if links.get(project_id) is None:
            return 0
        try:
            view = await asyncio.to_thread(links.refresh, project_id)
        except Exception as error:  # noqa: BLE001 - work on what is already downloaded
            log.warning("could not refresh the source of project %s: %s", project_id, error)
            view = await asyncio.to_thread(links.view, project_id)
        if view is None:
            return 0
        have = {chapter.number for chapter in view.chapters if chapter.chapter_id and chapter.number is not None}
        missing = [(index, chapter) for index, chapter in enumerate(view.chapters) if not chapter.chapter_id]
        # Reading order: by chapter number; sources usually list the newest first.
        missing.sort(key=lambda item: (item[1].number is None, item[1].number or 0.0, -item[0]))
        picked: list[str] = []
        for _, chapter in missing:
            if chapter.number is not None:
                if chapter.number in have:
                    continue  # another scan group's copy of a chapter the project has
                have.add(chapter.number)
            picked.append(chapter.id)
        if automation.target.limit:
            picked = picked[: automation.target.limit]
        if not picked:
            return 0
        await asyncio.to_thread(links.download, project_id, picked)
        deadline = time.monotonic() + DOWNLOAD_TIMEOUT_SECONDS
        while not stop.stopped and time.monotonic() < deadline:
            await asyncio.sleep(DOWNLOAD_POLL_SECONDS)
            view = await asyncio.to_thread(links.view, project_id)
            if view is None:
                return 0
            states = {chapter.id: chapter for chapter in view.chapters if chapter.id in picked}
            waiting = [c for c in states.values() if not c.chapter_id and c.download is not None and c.download.state != "failed"]
            if not waiting:
                return sum(1 for c in states.values() if c.chapter_id)
        if stop.stopped:
            for chapter_id in picked:
                links.cancel(project_id, chapter_id)
        view = await asyncio.to_thread(links.view, project_id)
        return sum(1 for c in (view.chapters if view else []) if c.id in picked and c.chapter_id)


__all__ = [
    "Automation",
    "AutomationBody",
    "Automations",
    "ChapterResult",
    "ModelProfile",
    "RunRecord",
    "Schedule",
    "Target",
]
