"""Project links: one project follows one series on one source.

The link lives next to the project's scene, in ``<project>/source/``:

    link.json     the system, source, series and its last fetched chapter list
    cover         the series cover, which becomes the project's cover

Downloading a chapter fetches its pages through the source system and adds them to the
project as one chapter, in one undoable commit. The new chapter records which source
chapter it came from (``Chapter.remote``, see :func:`remote_key`), which is how the sidebar tells downloaded chapters from
the ones still on the source; reordering or renaming it keeps that id. A downloaded chapter
is never fetched again, so user edits are never overwritten.

Downloads run one chapter at a time per project, up to PAGE_WORKERS pages at a time.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

from urakatsu.pipeline.projects import ProjectStore, chapters_of, write_chapters
from urakatsu.scene import Chapter, Chapters

from .base import Image, RemoteChapter, Series, SourceChallenge, SourceSystemError
from .registry import SourceSystems

log = logging.getLogger(__name__)

PAGE_WORKERS = 8  # measured: 8 fetch a chapter ~35% faster than 4; 16 is no faster (sources rate-limit)
UNTITLED = ("", "Untitled")


class SourceLink(BaseModel):
    system: str
    source: str
    source_name: str
    source_language: str = ""
    series: Series
    chapters: list[RemoteChapter]
    linked_at: float
    refreshed_at: float
    cover_type: Optional[str] = None


class DownloadState(BaseModel):
    # "stitching": a webtoon project joins the pages and cuts them again (pipeline/stitch.py).
    state: Literal["queued", "downloading", "importing", "stitching", "failed"]
    done: int = 0
    total: int = 0
    error: Optional[str] = None
    challenge: Optional[str] = None  # a page to clear in the system's browser before retrying


class LinkedChapter(RemoteChapter):
    chapter_id: Optional[str] = None  # the project's chapter, once downloaded
    download: Optional[DownloadState] = None


class LinkView(BaseModel):
    system: str
    system_name: str
    source: str
    source_name: str
    source_language: str
    series: Series
    refreshed_at: float
    chapters: list[LinkedChapter]


def remote_key(link: SourceLink, chapter_id: str) -> str:
    """What ``Chapter.remote`` stores: chapter ids are only unique within one series on one source."""
    return f"{link.system}:{link.source}:{link.series.id}:{chapter_id}"


def placement(order: dict[str, int], remote: str):
    """Where a downloaded chapter goes: before the first downloaded chapter that follows it on the source."""

    def place(chapters: list[Chapter]) -> int:
        target = order[remote]
        after: Optional[int] = None
        for index, chapter in enumerate(chapters):
            position = order.get(chapter.remote or "")
            if position is None:
                continue
            if position > target:
                return index
            after = index
        return len(chapters) if after is None else after + 1

    return place


class SourceLinks:
    def __init__(self, store: ProjectStore, systems: SourceSystems):
        self.store = store
        self.systems = systems
        self._downloads: dict[str, dict[str, DownloadState]] = {}
        self._queues: dict[str, deque[str]] = {}
        self._workers: dict[str, threading.Thread] = {}
        self._cancelled: set[tuple[str, str]] = set()
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(PAGE_WORKERS, thread_name_prefix="source-page")
        self._closed = False

    # -- storage -----------------------------------------------------------------------------------

    def _dir(self, project_id: str) -> Path:
        self.store.session(project_id)  # raises KeyError for an unknown project
        return self.store.root / project_id / "source"

    def get(self, project_id: str) -> Optional[SourceLink]:
        path = self._dir(project_id) / "link.json"
        try:
            return SourceLink.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _save(self, project_id: str, link: SourceLink, cover: Optional[Image] = None) -> None:
        folder = self._dir(project_id)
        folder.mkdir(parents=True, exist_ok=True)
        if cover is not None:
            (folder / "cover").write_bytes(cover.data)
            link.cover_type = cover.media_type
        tmp = folder / "link.tmp"
        tmp.write_text(link.model_dump_json(), encoding="utf-8")
        tmp.replace(folder / "link.json")

    def _cover(self, link: SourceLink) -> Optional[Image]:
        if not link.series.cover:
            return None
        try:
            return self.systems.get(link.system).cover(link.source, link.series.cover)
        except Exception as error:  # noqa: BLE001 - a missing cover must not fail the link
            log.warning("could not fetch the cover of %s: %s", link.series.title, error)
            return None

    # -- linking -----------------------------------------------------------------------------------

    def link(self, project_id: str, system_id: str, source_id: str, series_id: str) -> LinkView:
        system = self.systems.get(system_id)
        source = next((s for s in system.sources() if s.id == source_id), None)
        if source is None:
            raise KeyError(f"{system.name} has no installed source {source_id}")
        series = system.series(source_id, series_id)
        now = time.time()
        link = SourceLink(
            system=system_id,
            source=source_id,
            source_name=source.name,
            source_language=source.language,
            series=series,
            chapters=system.chapters(source_id, series_id),
            linked_at=now,
            refreshed_at=now,
        )
        self.cancel_all(project_id)
        folder = self._dir(project_id)
        (folder / "cover").unlink(missing_ok=True)
        self._save(project_id, link, self._cover(link))
        if self.store.name(project_id).strip() in UNTITLED:
            self.store.rename(project_id, series.title)
        self._drop_placeholder(project_id)
        return self.view(project_id)  # type: ignore[return-value]

    def _drop_placeholder(self, project_id: str) -> None:
        """A new project's implicit empty "Chapter 1" gives way to the series' own chapters."""
        session = self.store.session(project_id)
        snapshot = session.snapshot()
        if snapshot.pages() or snapshot.project_component(Chapters) is not None:
            return
        edit = snapshot.edit()
        write_chapters(edit, [])
        session.commit(edit.finish())

    def unlink(self, project_id: str) -> None:
        self.cancel_all(project_id)
        shutil.rmtree(self._dir(project_id), ignore_errors=True)

    def refresh(self, project_id: str) -> LinkView:
        link = self._require(project_id)
        system = self.systems.get(link.system)
        link.series = system.series(link.source, link.series.id)
        link.chapters = system.chapters(link.source, link.series.id)
        link.refreshed_at = time.time()
        self._save(project_id, link, self._cover(link))
        return self.view(project_id)  # type: ignore[return-value]

    def _require(self, project_id: str) -> SourceLink:
        link = self.get(project_id)
        if link is None:
            raise KeyError(f"project {project_id} is not linked to a source")
        return link

    def cover(self, project_id: str) -> Optional[Image]:
        link = self.get(project_id)
        path = self._dir(project_id) / "cover"
        if link is None or not path.exists():
            return None
        return Image(path.read_bytes(), link.cover_type or "image/jpeg")

    def view(self, project_id: str) -> Optional[LinkView]:
        link = self.get(project_id)
        if link is None:
            return None
        local = {chapter.remote: chapter.id for chapter in chapters_of(self.store.session(project_id).snapshot()) if chapter.remote}
        local = {chapter.id: local[remote_key(link, chapter.id)] for chapter in link.chapters if remote_key(link, chapter.id) in local}
        with self._lock:
            states = {key: state.model_copy() for key, state in self._downloads.get(project_id, {}).items()}
        system = self.systems.get(link.system)
        return LinkView(
            system=link.system,
            system_name=system.name,
            source=link.source,
            source_name=link.source_name,
            source_language=link.source_language,
            series=link.series,
            refreshed_at=link.refreshed_at,
            chapters=[
                LinkedChapter(**chapter.model_dump(), chapter_id=local.get(chapter.id), download=None if chapter.id in local else states.get(chapter.id))
                for chapter in link.chapters
            ],
        )

    # -- downloads ---------------------------------------------------------------------------------

    def download(self, project_id: str, chapter_ids: list[str]) -> LinkView:
        link = self._require(project_id)
        known = {chapter.id for chapter in link.chapters}
        local = {chapter.remote for chapter in chapters_of(self.store.session(project_id).snapshot()) if chapter.remote}
        with self._lock:
            states = self._downloads.setdefault(project_id, {})
            queue = self._queues.setdefault(project_id, deque())
            for chapter_id in chapter_ids:
                if chapter_id not in known:
                    raise KeyError(f"{link.series.title} has no chapter {chapter_id}")
                if remote_key(link, chapter_id) in local or (chapter_id in states and states[chapter_id].state != "failed"):
                    continue
                self._cancelled.discard((project_id, chapter_id))
                states[chapter_id] = DownloadState(state="queued")
                queue.append(chapter_id)
            worker = self._workers.get(project_id)
            if queue and (worker is None or not worker.is_alive()):
                worker = threading.Thread(target=self._work, args=(project_id,), name=f"source-download-{project_id}", daemon=True)
                self._workers[project_id] = worker
                worker.start()
        return self.view(project_id)  # type: ignore[return-value]

    def cancel(self, project_id: str, chapter_id: str) -> Optional[LinkView]:
        with self._lock:
            self._cancelled.add((project_id, chapter_id))
            queue = self._queues.get(project_id)
            if queue and chapter_id in queue:
                queue.remove(chapter_id)
            self._downloads.get(project_id, {}).pop(chapter_id, None)
        return self.view(project_id)

    def cancel_all(self, project_id: str) -> None:
        with self._lock:
            for chapter_id in list(self._downloads.get(project_id, {})):
                self._cancelled.add((project_id, chapter_id))
            self._queues.pop(project_id, None)
            self._downloads.pop(project_id, None)

    def _work(self, project_id: str) -> None:
        while not self._closed:
            with self._lock:
                queue = self._queues.get(project_id)
                if not queue:
                    self._workers.pop(project_id, None)
                    return
                chapter_id = queue.popleft()
                state = self._downloads.get(project_id, {}).get(chapter_id)
                if state is None:
                    continue
                state.state = "downloading"
            try:
                self._fetch(project_id, chapter_id, state)
                with self._lock:
                    self._downloads.get(project_id, {}).pop(chapter_id, None)
            except _Cancelled:
                continue
            except SourceChallenge as challenge:
                with self._lock:
                    state.state = "failed"
                    state.error = str(challenge)
                    state.challenge = challenge.url
            except Exception as error:  # noqa: BLE001 - reported on the chapter, the queue goes on
                log.warning("downloading chapter %s of project %s failed: %s", chapter_id, project_id, error)
                with self._lock:
                    state.state = "failed"
                    state.error = str(error) or error.__class__.__name__

    def _fetch(self, project_id: str, chapter_id: str, state: DownloadState) -> None:
        link = self._require(project_id)
        chapter = next(c for c in link.chapters if c.id == chapter_id)
        system = self.systems.get(link.system)
        handles = system.pages(link.source, link.series.id, chapter_id)
        if not handles:
            raise ValueError("the source returned no pages for this chapter")
        state.total = len(handles)

        def fetch(numbered: tuple[int, str]) -> bytes:
            number, handle = numbered
            if (project_id, chapter_id) in self._cancelled or self._closed:
                raise _Cancelled()
            try:
                data = system.page(link.source, link.series.id, chapter_id, handle).data
            except (_Cancelled, SourceChallenge):
                raise
            except Exception as error:
                raise SourceSystemError(f"page {number} of {len(handles)} did not download: {error}") from error
            with self._lock:
                state.done += 1
            return data

        images = list(self._pool.map(fetch, enumerate(handles, start=1)))
        if (project_id, chapter_id) in self._cancelled:
            raise _Cancelled()
        state.state = "stitching" if self.store.format(project_id).kind == "webtoon" else "importing"
        width = len(str(len(images)))
        order = {remote_key(link, c.id): index for index, c in enumerate(link.chapters)}
        name = chapter.name
        if chapter.scanlator and sum(c.name == chapter.name for c in link.chapters) > 1:
            name = f"{chapter.name} ({chapter.scanlator})"  # sources list one chapter per scan group
        self.store.import_chapter(
            project_id,
            name,
            [(f"{number:0{width}d}", data) for number, data in enumerate(images, start=1)],
            place=placement(order, remote_key(link, chapter_id)),
            remote=remote_key(link, chapter_id),
        )

    def close(self) -> None:
        self._closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)


class _Cancelled(Exception):
    pass
