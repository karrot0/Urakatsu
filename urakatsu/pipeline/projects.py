"""Project store: one scene session per project directory under ``URAKATSU_HOME/projects``.

A page entity sized to the image with the
decoded image attached under the ``source`` asset role.

Projects are organized as Project -> Chapters -> Pages. Chapters live in the
project's scene state (the ``Chapters`` project component), so arranging them is
an ordinary undoable commit, and the scene page order is always the chapters'
pages flattened in order. A project that never stored chapters has one implicit
"Chapter 1" holding every page.

A project's format (``project.json``) says whether it holds manga pages or webtoon strips
(webtoon, manhwa, manhua). Webtoon chapters are restitched (``pipeline/stitch.py``): imported
chapters on arrival, others on request.
"""

from __future__ import annotations

import atexit
import io
import json
import logging
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from urakatsu.core import files
from urakatsu.core.config import home
from urakatsu.core.errors import ProjectBusy
from urakatsu.core.results import LayoutDetections
from urakatsu.pipeline.stitch import Detector, ProjectFormat, restitch
from urakatsu.scene import At, AssetInput, AssetMetadata, Chapter, Chapters, Edit, Geometry, Page, Region, RemovePolicy, Session, Snapshot
from urakatsu.scene.components import RegionKinds

log = logging.getLogger(__name__)

DEFAULT_CHAPTER_ID = "default"


@dataclass
class ProjectSummary:
    id: str
    name: str
    pages: int
    created_at: float
    updated_at: float
    chapters: int = 1


def chapters_of(snapshot: Snapshot) -> list[Chapter]:
    """The project's chapters reconciled with its pages.

    Stored page ids that no longer exist are dropped; pages no chapter lists (imported
    before chapters existed) are appended to the last chapter.
    """
    stored = snapshot.project_component(Chapters)
    pages = snapshot.pages()
    present = set(pages)
    if stored is None:
        chapters = [Chapter(id=DEFAULT_CHAPTER_ID, name="Chapter 1")]
    else:
        chapters = [Chapter(id=ch.id, name=ch.name, remote=ch.remote, pages=[p for p in ch.pages if p in present]) for ch in stored.chapters]
    listed = {page for chapter in chapters for page in chapter.pages}
    orphans = [page for page in pages if page not in listed]
    if orphans:
        if not chapters:
            chapters.append(Chapter(id=DEFAULT_CHAPTER_ID, name="Chapter 1"))
        chapters[-1].pages.extend(orphans)
    return chapters


def write_chapters(edit: Edit, chapters: list[Chapter]) -> None:
    """Stores ``chapters`` and makes the scene page order their pages, flattened."""
    edit.set_page_order([page for chapter in chapters for page in chapter.pages])
    edit.set_project(Chapters(chapters=chapters))


def _find(chapters: list[Chapter], chapter_id: str) -> Chapter:
    for chapter in chapters:
        if chapter.id == chapter_id:
            return chapter
    raise KeyError(f"chapter {chapter_id} does not exist")


# Every edit marks its project updated; the time reaches project.json at most this often (seconds).
TOUCH_WRITE_INTERVAL = 5.0


class ProjectStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else home() / "projects"
        self.root.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self.lossless_pages = False  # StorageConfig.lossless_pages, set by the workspace
        # Runs the detection model while restitching webtoon chapters; set by the workspace.
        self.detector: Optional[Detector] = None
        # Update times not yet written, and the timers that will write them.
        self._touched: dict[str, float] = {}
        self._touch_timers: dict[str, threading.Timer] = {}
        self._touch_lock = threading.Lock()
        atexit.register(self.flush)

    def _meta_path(self, project_id: str) -> Path:
        return self.root / project_id / "project.json"

    def _read_meta(self, project_id: str) -> dict:
        return json.loads(self._meta_path(project_id).read_text(encoding="utf-8"))

    def _write_meta(self, project_id: str, meta: dict) -> None:
        files.write_text(self._meta_path(project_id), json.dumps(meta))

    def create(self, name: str) -> str:
        project_id = uuid.uuid4().hex[:12]
        (self.root / project_id).mkdir(parents=True)
        now = time.time()
        self._write_meta(project_id, {"name": name or "Untitled", "created_at": now, "updated_at": now})
        return project_id

    def session(self, project_id: str) -> Session:
        with self._lock:
            if project_id not in self._sessions:
                if not self._meta_path(project_id).exists():
                    raise KeyError(f"project {project_id} does not exist")
                self._sessions[project_id] = Session(self.root / project_id / "scene")
            return self._sessions[project_id]

    def name(self, project_id: str) -> str:
        return self._read_meta(project_id)["name"]

    def rename(self, project_id: str, name: str) -> None:
        meta = self._read_meta(project_id)
        meta.update(name=name, updated_at=time.time())
        self._write_meta(project_id, meta)

    def format(self, project_id: str) -> ProjectFormat:
        return ProjectFormat.model_validate(self._read_meta(project_id).get("format") or {})

    def set_format(self, project_id: str, value: ProjectFormat) -> None:
        meta = self._read_meta(project_id)
        meta.update(format=value.model_dump(mode="json"), updated_at=time.time())
        self._write_meta(project_id, meta)

    def touch(self, project_id: str) -> None:
        """Marks the project updated. Every edit (each brush stroke) calls this, so the time is kept
        in memory and written at most every TOUCH_WRITE_INTERVAL: on a USB hard disk one write of
        project.json costs ~45 ms."""
        with self._touch_lock:
            self._touched[project_id] = time.time()
            if project_id in self._touch_timers:
                return
            timer = threading.Timer(TOUCH_WRITE_INTERVAL, self._write_touch, (project_id,))
            timer.daemon = True
            self._touch_timers[project_id] = timer
            timer.start()

    def _write_touch(self, project_id: str) -> None:
        with self._touch_lock:
            self._touch_timers.pop(project_id, None)
            when = self._touched.pop(project_id, None)
        if when is None:
            return
        try:
            meta = self._read_meta(project_id)
            meta["updated_at"] = max(meta.get("updated_at", 0.0), when)
            self._write_meta(project_id, meta)
        except FileNotFoundError:
            pass  # deleted meanwhile

    def flush(self) -> None:
        """Writes pending update times now (at exit)."""
        with self._touch_lock:
            pending = list(self._touch_timers.items())
        for project_id, timer in pending:
            timer.cancel()
            self._write_touch(project_id)

    def _updated_at(self, project_id: str, meta: dict) -> float:
        with self._touch_lock:
            return max(meta["updated_at"], self._touched.get(project_id, 0.0))

    def delete(self, project_id: str) -> None:
        with self._lock:
            self._sessions.pop(project_id, None)
        shutil.rmtree(self.root / project_id, ignore_errors=True)

    def list(self) -> list[ProjectSummary]:
        summaries = []
        for meta_path in self.root.glob("*/project.json"):
            project_id = meta_path.parent.name
            try:
                meta = self._read_meta(project_id)
                snapshot = self.session(project_id).snapshot()
            except Exception:
                continue
            summaries.append(
                ProjectSummary(
                    project_id,
                    meta["name"],
                    len(snapshot.pages()),
                    meta["created_at"],
                    self._updated_at(project_id, meta),
                    len(chapters_of(snapshot)),
                )
            )
        return sorted(summaries, key=lambda s: s.updated_at, reverse=True)

    # -- chapters ------------------------------------------------------------------------------------

    def create_chapter(self, project_id: str, name: str, index: Optional[int] = None) -> str:
        session = self.session(project_id)
        snapshot = session.snapshot()
        chapters = chapters_of(snapshot)
        chapter = Chapter(id=uuid.uuid4().hex[:12], name=name.strip() or f"Chapter {len(chapters) + 1}")
        position = len(chapters) if index is None else max(0, min(index, len(chapters)))
        chapters.insert(position, chapter)
        edit = snapshot.edit()
        write_chapters(edit, chapters)
        session.commit(edit.finish())
        self.touch(project_id)
        return chapter.id

    def rename_chapter(self, project_id: str, chapter_id: str, name: str) -> None:
        session = self.session(project_id)
        snapshot = session.snapshot()
        chapters = chapters_of(snapshot)
        _find(chapters, chapter_id).name = name.strip() or "Untitled chapter"
        edit = snapshot.edit()
        write_chapters(edit, chapters)
        session.commit(edit.finish())
        self.touch(project_id)

    def delete_chapter(self, project_id: str, chapter_id: str) -> None:
        """Removes the chapter and every page in it, in one undoable commit."""
        session = self.session(project_id)
        snapshot = session.snapshot()
        chapters = chapters_of(snapshot)
        doomed = _find(chapters, chapter_id)
        edit = snapshot.edit()
        for page in doomed.pages:
            edit.remove_entity(page, RemovePolicy.CASCADE)
        write_chapters(edit, [chapter for chapter in chapters if chapter.id != chapter_id])
        session.commit(edit.finish())
        self.touch(project_id)

    def arrange(self, project_id: str, layout: list[tuple[str, list[str]]]) -> None:
        """Reorders chapters and moves pages within and between them.

        ``layout`` lists every chapter id with its pages, in the new order. It must
        name exactly the current chapters and pages; names are kept.
        """
        session = self.session(project_id)
        snapshot = session.snapshot()
        chapters = chapters_of(snapshot)
        names = {chapter.id: chapter.name for chapter in chapters}
        remotes = {chapter.id: chapter.remote for chapter in chapters}
        if sorted(cid for cid, _ in layout) != sorted(names):
            raise ValueError("arrangement must list every chapter exactly once")
        pages = [page for _, chapter_pages in layout for page in chapter_pages]
        if len(pages) != len(set(pages)) or set(pages) != set(snapshot.pages()):
            raise ValueError("arrangement must list every page exactly once")
        edit = snapshot.edit()
        write_chapters(edit, [Chapter(id=cid, name=names[cid], remote=remotes[cid], pages=list(chapter_pages)) for cid, chapter_pages in layout])
        session.commit(edit.finish())
        self.touch(project_id)

    def restitch_chapter(self, project_id: str, chapter_id: str) -> None:
        """Joins the chapter's source images and cuts them again where no text or art is cut,
        replacing its pages in one undoable commit. Page work (text, cleanup) is discarded:
        it belonged to the old cuts. Text the pages already had detected is never cut."""
        session = self.session(project_id)
        snapshot = session.snapshot()
        chapter = _find(chapters_of(snapshot), chapter_id)
        if not chapter.pages:
            raise ValueError(f"{chapter.name} has no pages to restitch")
        images = [_source_image(snapshot, page) for page in chapter.pages]
        avoid = [_text_bands(snapshot, page) for page in chapter.pages]
        lossless = all(snapshot.asset(page, "source").media_type == "image/png" for page in chapter.pages)
        slices = restitch(images, self.format(project_id).stitch, avoid, self.detector)
        encoded = _encode_slices(slices, lossless, self.lossless_pages)

        snapshot = session.snapshot()
        chapters = chapters_of(snapshot)
        target = _find(chapters, chapter_id)
        if target.pages != chapter.pages:
            raise ProjectBusy(f"{chapter.name} changed while it was restitched; try again")
        edit = snapshot.edit()
        for page in target.pages:
            edit.remove_entity(page, RemovePolicy.CASCADE)
        width = len(str(len(encoded)))
        target.pages = [
            _add_page(edit, f"{number:0{width}d}", data, media_type, w, h) for number, (data, media_type, w, h) in enumerate(encoded, start=1)
        ]
        write_chapters(edit, chapters)
        session.commit(edit.finish())
        self.touch(project_id)

    def remove_page(self, project_id: str, page_id: str) -> None:
        session = self.session(project_id)
        snapshot = session.snapshot()
        snapshot.page(page_id)
        chapters = chapters_of(snapshot)
        for chapter in chapters:
            if page_id in chapter.pages:
                chapter.pages.remove(page_id)
        edit = snapshot.edit()
        edit.remove_entity(page_id, RemovePolicy.CASCADE)
        write_chapters(edit, chapters)
        session.commit(edit.finish())
        self.touch(project_id)

    # -- pages ---------------------------------------------------------------------------------------

    def import_page(
        self,
        project_id: str,
        label: str,
        data: bytes,
        index: Optional[int] = None,
        chapter_id: Optional[str] = None,
    ) -> str:
        """Adds a page to ``chapter_id`` (default: the last chapter) at ``index`` within it (default: its end)."""
        data, media_type, width, height = _normalize(data, self.lossless_pages)
        session = self.session(project_id)
        snapshot = session.snapshot()
        chapters = chapters_of(snapshot)
        if not chapters:
            chapters.append(Chapter(id=uuid.uuid4().hex[:12], name="Chapter 1"))
        target = _find(chapters, chapter_id) if chapter_id else chapters[-1]
        edit = snapshot.edit()
        page = _add_page(edit, label, data, media_type, width, height)
        position = len(target.pages) if index is None else max(0, min(index, len(target.pages)))
        target.pages.insert(position, page)
        write_chapters(edit, chapters)
        session.commit(edit.finish())
        self.touch(project_id)
        return page

    def import_chapter(
        self,
        project_id: str,
        name: str,
        pages: list[tuple[str, bytes]],
        place: Optional[Callable[[list[Chapter]], int]] = None,
        remote: Optional[str] = None,
    ) -> str:
        """Adds a chapter of ``pages`` (label, image bytes) in one undoable commit.

        ``place`` picks the chapter position from the current chapters (default: the end).
        A project that never stored chapters loses its empty implicit "Chapter 1".
        In a webtoon project the pages are restitched first.
        """
        format = self.format(project_id)
        if format.kind == "webtoon" and pages:
            images = [Image.open(io.BytesIO(data)) for _, data in pages]
            lossless = all(image.format == "PNG" for image in images)
            slices = restitch([_upright(image) for image in images], format.stitch, detect=_lenient(self.detector))
            width = len(str(len(slices)))
            pages = [(f"{number:0{width}d}", data) for number, (data, *_) in enumerate(_encode_slices(slices, lossless), start=1)]
        # Pages that need converting do so in parallel: Pillow releases the GIL while encoding.
        with ThreadPoolExecutor(min(8, max(1, len(pages)))) as pool:
            decoded = [(label, *found) for (label, _), found in zip(pages, pool.map(partial(_normalize, lossless=self.lossless_pages), [data for _, data in pages]), strict=True)]
        session = self.session(project_id)
        snapshot = session.snapshot()
        chapters = chapters_of(snapshot) if snapshot.project_component(Chapters) is not None or snapshot.pages() else []
        chapter = Chapter(id=uuid.uuid4().hex[:12], name=name.strip() or f"Chapter {len(chapters) + 1}", remote=remote)
        edit = snapshot.edit()
        chapter.pages = [_add_page(edit, label, data, media_type, width, height) for label, data, media_type, width, height in decoded]
        position = len(chapters) if place is None else max(0, min(place(chapters), len(chapters)))
        chapters.insert(position, chapter)
        write_chapters(edit, chapters)
        session.commit(edit.finish())
        self.touch(project_id)
        return chapter.id


# Formats stored as they arrive: every browser shows them and Pillow reads them.
KEPT_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
EXIF_ORIENTATION = 0x0112
# Lossless WebP at low effort: on real manga pages 15% smaller than PNG in ~0.2 s a page; the
# highest effort saves 1% more and takes ~17 s. WebP cannot hold a side over 16383 pixels.
LOSSLESS_WEBP = {"format": "WEBP", "lossless": True, "method": 3, "quality": 0}
WEBP_MAX_SIDE = 16383


def _normalize(data: bytes, lossless: bool = False) -> tuple[bytes, str, int, int]:
    """A page image as stored: its own bytes when they can be kept, else PNG; with its media type and size.

    Bytes are kept for JPEG, PNG and WebP in RGB, RGBA or grayscale without an EXIF rotation
    (browsers apply the rotation, the stages would not). Anything else is converted to PNG.
    With ``lossless``, a page that would be stored as PNG becomes lossless WebP when that is
    smaller (the same pixels); JPEG and WebP pages are left alone.
    """
    image = Image.open(io.BytesIO(data))  # reads the header only
    media_type = KEPT_FORMATS.get(image.format or "")
    if media_type and image.mode in ("RGB", "RGBA", "L") and image.getexif().get(EXIF_ORIENTATION, 1) == 1:
        if not (lossless and media_type == "image/png"):
            return data, media_type, image.width, image.height
        stored = (data, media_type)
        image.load()
    else:
        image.load()
        if image.mode not in ("RGB", "RGBA", "L"):
            image = image.convert("RGB")
        stored = (_encode(image, format="PNG"), "image/png")
    if lossless and max(image.width, image.height) <= WEBP_MAX_SIDE:
        webp = _encode(image, **LOSSLESS_WEBP)
        if len(webp) < len(stored[0]):
            stored = (webp, "image/webp")
    return stored[0], stored[1], image.width, image.height


# Restitched pages from lossy sources: one more lossy encode at this quality is invisible.
SLICE_WEBP = {"format": "WEBP", "quality": 95, "method": 4}


def _lenient(detect: Optional[Detector]) -> Optional[Detector]:
    """``detect`` for chapters restitched on import: when the detection model cannot run, the
    chapter still arrives, cut by pixel comparison alone (Restitch can try again later)."""
    if detect is None:
        return None

    def run(tiles: list[Image.Image]):
        try:
            return detect(tiles)
        except Exception as error:  # noqa: BLE001 - a download must not fail on detection
            log.warning("restitching an imported chapter without text detection: %s", error)
            return [LayoutDetections(tile.width, tile.height) for tile in tiles]

    return run


def _upright(image: Image.Image) -> Image.Image:
    """``image`` decoded with its EXIF rotation applied, as the pages store it."""
    from PIL import ImageOps

    return ImageOps.exif_transpose(image) or image


def _source_image(snapshot: Snapshot, page: str) -> Image.Image:
    asset = snapshot.asset(page, "source")
    if asset is None:
        raise ValueError(f"page {page} has no source image")
    image = Image.open(io.BytesIO(snapshot.blobs.read(asset.blob)))
    image.load()
    return image


def _text_bands(snapshot: Snapshot, page: str) -> list[tuple[float, float]]:
    """Rows of ``page`` covered by text or bubbles it already has (detected or drawn)."""
    bands = []
    for entity in snapshot.descendants(page):
        region = snapshot.component(entity, Region)
        geometry = snapshot.component(entity, Geometry)
        if region is None or geometry is None or not geometry.points or region.kind not in (RegionKinds.TEXT, RegionKinds.BUBBLE):
            continue
        ys = [point.y for point in geometry.points]
        bands.append((min(ys), max(ys)))
    return bands


def _encode_slices(slices: list[Image.Image], lossless: bool, lossless_pages: bool = False) -> list[tuple[bytes, str, int, int]]:
    """Restitched pages as stored: PNG (through :func:`_normalize`, so ``lossless_pages`` applies)
    when every source was lossless, else WebP at SLICE_WEBP; encoded in parallel."""

    def encode(image: Image.Image) -> tuple[bytes, str, int, int]:
        if lossless or max(image.size) > WEBP_MAX_SIDE:
            return _normalize(_encode(image, format="PNG", compress_level=3), lossless_pages)
        return _encode(image, **SLICE_WEBP), "image/webp", image.width, image.height

    with ThreadPoolExecutor(min(8, max(1, len(slices)))) as pool:
        return list(pool.map(encode, slices))


def _encode(image: Image.Image, **options) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, **options)
    return buffer.getvalue()


def _add_page(edit: Edit, label: str, data: bytes, media_type: str, width: int, height: int) -> str:
    page = edit.add_page(Page(label=label, width=float(width), height=float(height)), At.END)
    edit.set_asset(page, "source", AssetInput(data=data, media_type=media_type, metadata=AssetMetadata(width=width, height=height)))
    return page
