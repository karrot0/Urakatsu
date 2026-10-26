"""Projects on Komga: a project is one Komga series, and each of its chapters one book.

The link lives next to the project's scene, in ``<project>/komga.json``: the library, the
series' folder (relative to the library's), the series' id once Komga lists it, the book
written for each chapter and how the last export went.

- **Link** a series Komga already has: its folder comes from Komga's path for it.
- **Add** the project as a new series: a folder named after it in the chosen library. Komga
  lists a series once its folder holds a book, so it appears with the first export. Urakatsu
  then fills in the series' metadata (from the linked source: summary, status, genres, link)
  and cover, once; what is edited in Komga afterwards stays.

Exporting a chapter renders its pages, packs them as a CBZ with ``ComicInfo.xml``
(``urakatsu.sources.komga.archive``) and writes it to the series' folder, replacing the chapter's
earlier book; then Komga is asked to scan the library. Files Urakatsu did not write are
never replaced: a linked series keeps its own books. Deleting a chapter leaves its book.

One export runs at a time per project. The UI starts exports in the background
(:meth:`KomgaLinks.start_export`) and reads their progress from the view; automations call
:meth:`KomgaLinks.export` on their own worker.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field

from urakatsu.core import files
from urakatsu.core.stop import StopToken
from urakatsu.pipeline.projects import ProjectStore, chapters_of
from urakatsu.scene import Chapter
from urakatsu.sources.links import SourceLink, SourceLinks, remote_key

from .archive import BookInfo, pack_cbz
from .client import KomgaClient, KomgaError
from .connection import KomgaConnection, relative_folder

log = logging.getLogger(__name__)

FILE = "komga.json"
SCAN_WAIT_SECONDS = 60.0  # how long an export waits for Komga to list a new series
SCAN_POLL_SECONDS = 1.0

# Source statuses (Suwayomi's, Mana's) as Komga's series statuses.
STATUS = {
    "ongoing": "ONGOING",
    "publishing": "ONGOING",
    "completed": "ENDED",
    "publishing_finished": "ENDED",
    "finished": "ENDED",
    "ended": "ENDED",
    "cancelled": "ABANDONED",
    "canceled": "ABANDONED",
    "abandoned": "ABANDONED",
    "on_hiatus": "HIATUS",
    "hiatus": "HIATUS",
}

RenderChapter = Callable[[str, str], tuple[Chapter, list[tuple[str, bytes]]]]


class Book(BaseModel):
    file: str  # in the series' folder
    exported_at: float


class ExportReport(BaseModel):
    started_at: float
    finished_at: Optional[float] = None
    books: int = 0  # chapters written
    error: Optional[str] = None
    stopped: bool = False  # stopped before every chapter was written (and before the scan)
    pending: bool = False  # Komga had not listed the new series yet; the next export links it


class ExportState(BaseModel):
    """An export on its way."""

    state: Literal["queued", "rendering", "scanning"] = "queued"
    done: int = 0
    total: int = 0
    chapter: str = ""


class KomgaLink(BaseModel):
    library_id: str
    library_name: str = ""
    folder: str  # the series' folder, relative to the library's, with "/" between parts
    title: str
    series_id: Optional[str] = None  # once Komga lists the series
    created: bool = False  # Urakatsu made the series: it sends the metadata and cover
    described: bool = False  # ...and has sent them
    linked_at: float
    books: dict[str, Book] = Field(default_factory=dict)  # project chapter id -> its book
    last_export: Optional[ExportReport] = None


class KomgaLinkView(BaseModel):
    library_id: str
    library_name: str
    folder: str
    title: str
    series_id: Optional[str]
    series_url: Optional[str]  # the series in Komga's web reader
    created: bool
    exported: dict[str, float]  # project chapter id -> when its book was written
    last_export: Optional[ExportReport]
    export: Optional[ExportState]


def series_folder(library_folder: Path, folder: str) -> Path:
    """The series' folder on this computer; never outside the library's."""
    parts = PurePosixPath(folder).parts
    if not parts or any(part in ("..", ".") for part in parts) or PurePosixPath(folder).is_absolute():
        raise ValueError(f"{folder!r} is not a folder inside the library")
    return library_folder.joinpath(*parts)


class KomgaLinks:
    def __init__(
        self,
        store: ProjectStore,
        sources: SourceLinks,
        connection: KomgaConnection,
        render: RenderChapter,
        language: Callable[[], Optional[str]] = lambda: None,
    ):
        self.store = store
        self.sources = sources
        self.connection = connection
        self.render = render
        self.language = language  # the translation's language, for the books' metadata
        self._exports: dict[str, ExportState] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()

    # -- storage -----------------------------------------------------------------------------------

    def _path(self, project_id: str) -> Path:
        self.store.session(project_id)  # raises KeyError for an unknown project
        return self.store.root / project_id / FILE

    def get(self, project_id: str) -> Optional[KomgaLink]:
        try:
            return KomgaLink.model_validate_json(self._path(project_id).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except ValueError as error:
            log.warning("ignoring the unreadable Komga link of project %s: %s", project_id, error)
            return None

    def _save(self, project_id: str, link: KomgaLink) -> None:
        if (self.store.root / project_id).exists():  # not for a project deleted meanwhile
            files.write_text(self._path(project_id), link.model_dump_json(indent=1))

    def _require(self, project_id: str) -> KomgaLink:
        link = self.get(project_id)
        if link is None:
            raise KeyError(f"project {project_id} is not on Komga")
        return link

    def view(self, project_id: str) -> Optional[KomgaLinkView]:
        link = self.get(project_id)
        if link is None:
            return None
        url = self.connection.settings().url
        with self._lock:
            live = self._exports.get(project_id)
        return KomgaLinkView(
            library_id=link.library_id,
            library_name=link.library_name,
            folder=link.folder,
            title=link.title,
            series_id=link.series_id,
            series_url=f"{url}/series/{link.series_id}" if url and link.series_id else None,
            created=link.created,
            exported={chapter_id: book.exported_at for chapter_id, book in link.books.items()},
            last_export=link.last_export,
            export=live.model_copy() if live else None,
        )

    # -- linking -----------------------------------------------------------------------------------

    def link(self, project_id: str, series_id: str) -> KomgaLinkView:
        """Links the project to a series Komga already has; its books are written to that series' folder."""
        self._idle(project_id)
        with self.connection.client() as client:
            series = client.series(series_id)
            library = client.library(series.library_id)
            folder = relative_folder(library.root, series.url)
            self.connection.folder(library.id, client)  # the library must be writable from here
        self._save(project_id, KomgaLink(library_id=library.id, library_name=library.name, folder=folder, title=series.title, series_id=series.id, linked_at=time.time()))
        return self.view(project_id)  # type: ignore[return-value]

    def add(self, project_id: str, library_id: str, title: Optional[str] = None) -> KomgaLinkView:
        """Adds the project to a library as a new series, in a folder named after it.

        The title defaults to the linked series' title, else the project's name."""
        self._idle(project_id)
        source = self.sources.get(project_id)
        title = (title or "").strip() or (source.series.title if source else "") or self.store.name(project_id)
        folder = files.safe_stem(title, extension=False)
        if not folder:
            raise ValueError("choose a title for the series")
        with self.connection.client() as client:
            library = client.library(library_id)
            path = series_folder(self.connection.folder(library_id, client), folder)
            if (path.exists() and any(path.iterdir())) or self._claimed(library_id, folder, project_id):
                raise ValueError(f"{library.name} already has a folder named {folder}; link that series instead, or choose another title")
            path.mkdir(parents=True, exist_ok=True)
        self._save(project_id, KomgaLink(library_id=library.id, library_name=library.name, folder=folder, title=title, created=True, linked_at=time.time()))
        return self.view(project_id)  # type: ignore[return-value]

    def _claimed(self, library_id: str, folder: str, project_id: str) -> bool:
        """Whether another project was added as a series in that folder (empty until its first export)."""
        for path in self.store.root.glob(f"*/{FILE}"):
            if path.parent.name == project_id:
                continue
            try:
                other = KomgaLink.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if other.library_id == library_id and other.folder.lower() == folder.lower():
                return True
        return False

    def unlink(self, project_id: str) -> None:
        """Forgets the series; its books stay on Komga."""
        self._idle(project_id)
        self._path(project_id).unlink(missing_ok=True)

    def _idle(self, project_id: str) -> None:
        with self._lock:
            if project_id in self._exports:
                raise ValueError("wait for the export to Komga to finish")

    # -- exporting ---------------------------------------------------------------------------------

    def _project_lock(self, project_id: str) -> threading.Lock:
        with self._lock:
            return self._locks.setdefault(project_id, threading.Lock())

    def start_export(self, project_id: str, chapter_ids: Optional[list[str]] = None) -> KomgaLinkView:
        """Exports in the background; the view shows its progress, then its report."""
        self._require(project_id)
        with self._lock:
            if project_id in self._exports:
                raise ValueError("an export of this project to Komga is already running")
            self._exports[project_id] = ExportState()

        def run() -> None:
            try:
                self.export(project_id, chapter_ids)
            except Exception:  # noqa: BLE001 - recorded on the link where it can be
                log.exception("exporting project %s to Komga failed", project_id)
            finally:
                with self._lock:
                    self._exports.pop(project_id, None)

        threading.Thread(target=run, name=f"komga-export-{project_id}", daemon=True).start()
        return self.view(project_id)  # type: ignore[return-value]

    def _progress(self, project_id: str, **changes) -> None:
        with self._lock:
            state = self._exports.get(project_id)
            if state is not None:
                for key, value in changes.items():
                    setattr(state, key, value)

    def export(self, project_id: str, chapter_ids: Optional[list[str]] = None, stop: Optional[StopToken] = None) -> ExportReport:
        """Writes the chapters (all with pages, by default) as books of the project's series, then
        has Komga scan them. Failures end the export and are reported, not raised; so is a
        project that is not on Komga. ``stop`` ends it between chapters (the books written stay;
        Komga finds them at its next scan)."""
        with self._lock:
            owned = project_id not in self._exports  # not started by start_export: shown all the same
            if owned:
                self._exports[project_id] = ExportState()
        try:
            with self._project_lock(project_id):
                return self._export(project_id, chapter_ids, stop)
        finally:
            if owned:
                with self._lock:
                    self._exports.pop(project_id, None)

    def _export(self, project_id: str, chapter_ids: Optional[list[str]], stop: Optional[StopToken]) -> ExportReport:
        report = ExportReport(started_at=time.time())
        link = self.get(project_id)
        if link is None:  # removed from Komga since the automation was set up
            report.finished_at, report.error = report.started_at, "the project is not on Komga any more"
            return report
        wanted = set(chapter_ids) if chapter_ids is not None else None
        chapters = [c for c in chapters_of(self.store.session(project_id).snapshot()) if c.pages and (wanted is None or c.id in wanted)]
        self._progress(project_id, state="rendering", total=len(chapters))
        try:
            with self.connection.client() as client:
                folder = series_folder(self.connection.folder(link.library_id, client), link.folder)
                folder.mkdir(parents=True, exist_ok=True)
                source = self.sources.get(project_id)
                numbers = _chapter_numbers(source, project_id, self.store)
                webtoon = self.store.format(project_id).kind == "webtoon"
                for chapter in chapters:
                    if stop is not None and stop.stopped:
                        report.stopped = True
                        break
                    self._progress(project_id, chapter=chapter.name)
                    _, pages = self.render(project_id, chapter.id)
                    book = self._book_info(link, chapter, numbers.get(chapter.id), source, webtoon)
                    name = self._file_name(link, chapter, folder)
                    files.write_bytes(folder / name, pack_cbz(pages, book))
                    link.books[chapter.id] = Book(file=name, exported_at=time.time())
                    report.books += 1
                    self._save(project_id, link)
                    self._progress(project_id, done=report.books)
                if not report.stopped:
                    self._progress(project_id, state="scanning", chapter="")
                    client.scan(link.library_id)
                    if link.series_id is None and report.books:
                        link.series_id = self._find_series(client, link, stop)
                        report.pending = link.series_id is None
                        report.stopped = bool(stop and stop.stopped)
                    if link.series_id and link.created and not link.described:
                        self._describe(client, link, project_id, source)
        except Exception as error:  # noqa: BLE001 - every failure (rendering too) is reported on the export
            report.error = str(error) or error.__class__.__name__
            log.warning("exporting project %s to Komga stopped: %s", project_id, report.error)
        report.finished_at = time.time()
        link.last_export = report
        self._save(project_id, link)
        return report

    @staticmethod
    def _file_name(link: KomgaLink, chapter: Chapter, folder: Path) -> str:
        """The chapter's book: the file it was written to before, else one named after it that
        neither another chapter's book nor a file Urakatsu did not write has."""
        if chapter.id in link.books:
            return link.books[chapter.id].file
        ours = {book.file.lower() for book in link.books.values()}
        stem = files.safe_stem(chapter.name, extension=False) or "Chapter"
        for attempt in range(1, 1000):
            name = f"{stem}.cbz" if attempt == 1 else f"{stem} ({attempt}).cbz"
            if name.lower() not in ours and not (folder / name).exists():
                return name
        raise ValueError(f"no free file name for {chapter.name} in {folder}")

    def _book_info(self, link: KomgaLink, chapter: Chapter, number: Optional[float], source: Optional[SourceLink], webtoon: bool = False) -> BookInfo:
        series = source.series if source else None
        return BookInfo(
            series=link.title,
            title=chapter.name,
            number=number,
            writers=list(series.authors) if series else [],
            genres=list(series.tags) if series else [],
            web=series.url if series else None,
            language=self.language(),
            right_to_left=bool(source and source.source_language.lower().startswith("ja")),
            webtoon=webtoon,
        )

    def _find_series(self, client: KomgaClient, link: KomgaLink, stop: Optional[StopToken] = None) -> Optional[str]:
        """Waits for Komga's scan to list the series in the link's folder; its id, or None (also
        when stopped)."""
        root = client.library(link.library_id).root
        names = list(dict.fromkeys([link.title, PurePosixPath(link.folder).name]))
        deadline = time.monotonic() + SCAN_WAIT_SECONDS
        while True:
            last = time.monotonic() >= deadline
            candidates = client.all_series(link.library_id) if last else (s for name in names for s in client.search(link.library_id, name))
            for series in candidates:
                try:
                    if series.url and relative_folder(root, series.url) == link.folder:
                        return series.id
                except ValueError:
                    continue
            if last or (stop is not None and stop.stopped):
                return None
            time.sleep(SCAN_POLL_SECONDS)

    def _describe(self, client: KomgaClient, link: KomgaLink, project_id: str, source: Optional[SourceLink]) -> None:
        """Gives a series Urakatsu created its title, the source's metadata and the cover. Locked,
        so later scans keep them; a failure here leaves the books exported."""
        metadata: dict = {"title": link.title, "titleLock": True}
        language = self.language()
        if language:
            metadata |= {"language": language, "languageLock": True}
        if source is not None:
            series = source.series
            if series.summary:
                metadata |= {"summary": series.summary, "summaryLock": True}
            status = STATUS.get((series.status or "").strip().lower().replace(" ", "_"))
            if status:
                metadata |= {"status": status, "statusLock": True}
            if series.tags:
                metadata |= {"genres": sorted(set(series.tags)), "genresLock": True}
            if series.url:
                metadata |= {"links": [{"label": source.source_name or "Source", "url": series.url}], "linksLock": True}
            if source.source_language.lower().startswith("ja"):
                metadata |= {"readingDirection": "RIGHT_TO_LEFT", "readingDirectionLock": True}
        if self.store.format(project_id).kind == "webtoon":
            metadata |= {"readingDirection": "WEBTOON", "readingDirectionLock": True}
        try:
            client.update_series(link.series_id, metadata)  # type: ignore[arg-type]
            cover = self.sources.cover(project_id)
            if cover is not None:
                client.upload_series_cover(link.series_id, cover.data, cover.media_type)  # type: ignore[arg-type]
            link.described = True
        except KomgaError as error:
            log.warning("could not set the metadata of Komga series %s: %s", link.series_id, error)


def _chapter_numbers(source: Optional[SourceLink], project_id: str, store: ProjectStore) -> dict[str, float]:
    """Each chapter's number: the source's for a downloaded chapter, else its place in the project."""
    by_remote = {remote_key(source, chapter.id): chapter.number for chapter in source.chapters if chapter.number is not None} if source else {}
    numbers: dict[str, float] = {}
    for index, chapter in enumerate(chapters_of(store.session(project_id).snapshot()), start=1):
        number = by_remote.get(chapter.remote or "")
        numbers[chapter.id] = number if number is not None else float(index)
    return numbers


__all__ = ["Book", "ExportReport", "ExportState", "KomgaLink", "KomgaLinkView", "KomgaLinks", "series_folder"]
