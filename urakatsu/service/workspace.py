"""The workspace: one process-wide owner of configuration, runtimes, projects, renders, and jobs.

Both transports (gRPC and JSON HTTP) call into this object; it keeps no
transport-specific state so both can run in one service.
"""

from __future__ import annotations

import asyncio
import io
import logging
import threading
import time
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, Optional

from PIL import Image
from pydantic import BaseModel

from urakatsu import __version__
from urakatsu.core.config import PipelineConfig, load_config, save_config
from urakatsu.core.errors import ProjectBusy
from urakatsu.core.progress import Progress, ProgressKind
from urakatsu.core.registry import registry
from urakatsu.core.stage import Operation, Stage
from urakatsu.core.stop import StopToken
from urakatsu.pipeline import (
    EntitiesScope,
    PagesScope,
    Pipeline,
    PipelineError,
    ProjectScope,
    ProjectStore,
    ProjectSummary,
    RegionScope,
    Request,
)
from urakatsu.pipeline.projects import chapters_of
from urakatsu.pipeline.scope import Bounds
from urakatsu.pipeline.stages import InpaintingMask
from urakatsu.runtimes.manager import RuntimeManager, RuntimeReport
from urakatsu.scene import components as c
from urakatsu.scene import At, RemovePolicy, Session
from urakatsu.scene.scene import Snapshot
from urakatsu.sources import SourceSystems
from urakatsu.sources.links import SourceLinks

from .automations import Automations
from .views import PageView, page_view, parse_rgba

log = logging.getLogger(__name__)

RENDER_CACHE = 16


class ChapterView(BaseModel):
    id: str
    name: str
    page_ids: list[str]


class ProjectView(BaseModel):
    id: str
    name: str
    created_at: float
    updated_at: float
    revision: int
    pages: list[PageView]
    can_undo: bool
    can_redo: bool
    chapters: list[ChapterView] = []


class JobInfo(BaseModel):
    id: str
    project_id: str
    status: str
    completed: int
    total: int
    started_at: float
    error: Optional[str] = None


@dataclass
class Job:
    id: str
    project_id: str
    stop: StopToken
    started_at: float = field(default_factory=time.time)
    status: str = "running"
    completed: int = 0
    total: int = 0
    error: Optional[str] = None
    events: list[Progress] = field(default_factory=list)
    queues: list[asyncio.Queue] = field(default_factory=list)
    task: Optional[asyncio.Task] = None

    def publish(self, progress: Progress) -> None:
        self.events.append(progress)
        if progress.total:
            self.total = progress.total
        if progress.completed:
            self.completed = progress.completed
        for queue in list(self.queues):
            queue.put_nowait(progress)

    def finish(self) -> None:
        for queue in list(self.queues):
            queue.put_nowait(None)


class Workspace:
    def __init__(self, device: str | None = None):
        self.config: PipelineConfig = load_config()
        if device:
            self.config.device = device
        self.runtimes = RuntimeManager(self.config.device)
        self.store = ProjectStore()
        self.store.lossless_pages = self.config.storage.lossless_pages
        self.pipeline = Pipeline(self.config, self.runtimes)
        self.systems = SourceSystems()
        self.links = SourceLinks(self.store, self.systems)
        self.automations = Automations(self)
        self.jobs: dict[str, Job] = {}
        self._renders: OrderedDict[tuple[str, str, int], bytes] = OrderedDict()
        self._thumbnails: dict[str, bytes] = {}
        self._render_lock = threading.Lock()
        # Assets are served from worker threads; the renderer draws one page at a time.
        self._draw_lock = threading.Lock()
        self._renderer = None
        registry.discover()

    # -- capabilities / configuration ------------------------------------------------------------

    def report(self) -> RuntimeReport:
        return self.runtimes.describe()

    def providers(self) -> list:
        return registry.info()

    def loaded(self) -> dict[Stage, str]:
        return self.pipeline.loaded()

    def languages(self) -> list[tuple[str, str]]:
        try:
            from urakatsu.providers.translation import languages

            return [tuple(item) for item in languages()]  # type: ignore[misc]
        except Exception:
            return []

    def translation_models(self) -> list[dict[str, Any]]:
        try:
            from urakatsu.providers.translation import models
        except Exception:
            return []
        result = []
        for model in models(self.config):
            result.append(model.model_dump(mode="json") if hasattr(model, "model_dump") else dict(vars(model)))
        return result

    def memory(self) -> dict[str, Any]:
        """GPU and system memory with Urakatsu's share of each; the UI polls this."""
        from urakatsu.runtimes.memory import report

        return report(self.runtimes.device).model_dump(mode="json")

    def capabilities(self) -> dict[str, Any]:
        report = self.report()
        return {
            "version": __version__,
            "device": report.device.model_dump(mode="json"),
            "devices": [d.model_dump(mode="json") for d in report.hardware.candidates()],
            "platform": report.hardware.platform,
            "runtimes": report.runtimes,
            "providers": [p.model_dump(mode="json") for p in self.providers()],
            "stages": [s.value for s in Stage.all()],
            "store_root": report.store_root,
            "store_bytes": report.store_bytes,
            "loaded": {stage.value: provider for stage, provider in self.loaded().items()},
            "languages": [{"tag": tag, "name": name} for tag, name in self.languages()],
        }

    def get_config(self) -> PipelineConfig:
        return self.config

    def set_config(self, config: PipelineConfig) -> PipelineConfig:
        for stage in Stage.all():
            cls = registry.get(stage, config.provider_for(stage))
            runtime = config.runtime_for(cls.id)
            if runtime is not None and runtime not in cls.runtimes_declared():
                raise ValueError(f"{cls.id} cannot run on {runtime!r}; supported: {list(cls.runtimes_declared())}")
        for provider_id, runtime in config.runtimes.items():
            if runtime not in RuntimeManager.RUNTIMES:
                raise ValueError(f"unknown runtime {runtime!r} for {provider_id}")
        if not self.automations.busy:  # a running batch applies the saved settings when it ends
            self.pipeline.configure(config)
        if config.typesetting != self.config.typesetting:
            self._renderer = None
            with self._render_lock:
                self._renders.clear()
        self.config = config
        self.store.lossless_pages = config.storage.lossless_pages
        save_config(config)
        return config

    # -- projects ------------------------------------------------------------------------------------

    def list_projects(self) -> list[ProjectSummary]:
        return self.store.list()

    def session(self, project_id: str) -> Session:
        return self.store.session(project_id)

    def project_view(self, project_id: str) -> ProjectView:
        session = self.session(project_id)
        snapshot = session.snapshot()
        meta = self.store._read_meta(project_id)
        return ProjectView(
            id=project_id,
            name=meta["name"],
            created_at=meta["created_at"],
            updated_at=self.store._updated_at(project_id, meta),
            revision=snapshot.revision,
            pages=[page_view(snapshot, page) for page in snapshot.pages()],
            can_undo=session.can_undo,
            can_redo=session.can_redo,
            chapters=[ChapterView(id=ch.id, name=ch.name, page_ids=ch.pages) for ch in chapters_of(snapshot)],
        )

    def create_project(self, name: str) -> ProjectView:
        return self.project_view(self.store.create(name))

    def rename_project(self, project_id: str, name: str) -> ProjectView:
        self.store.rename(project_id, name)
        return self.project_view(project_id)

    def delete_project(self, project_id: str) -> None:
        for job in self.jobs.values():
            if job.project_id == project_id and job.status == "running":
                job.stop.stop()
        self.links.cancel_all(project_id)
        self.automations.forget(project_id)
        self.store.delete(project_id)
        with self._render_lock:
            for key in [k for k in self._renders if k[0] == project_id]:
                del self._renders[key]

    def undo(self, project_id: str) -> ProjectView:
        self.session(project_id).undo()
        return self.project_view(project_id)

    def redo(self, project_id: str) -> ProjectView:
        self.session(project_id).redo()
        return self.project_view(project_id)

    # -- chapters ------------------------------------------------------------------------------------

    def _idle(self, project_id: str) -> None:
        for job in self.jobs.values():
            if job.project_id == project_id and job.status == "running":
                raise ProjectBusy("stop the running job before restructuring the project")

    def create_chapter(self, project_id: str, name: str, index: int | None = None) -> ProjectView:
        self.store.create_chapter(project_id, name, index)
        return self.project_view(project_id)

    def rename_chapter(self, project_id: str, chapter_id: str, name: str) -> ProjectView:
        self.store.rename_chapter(project_id, chapter_id, name)
        return self.project_view(project_id)

    def export_chapter(self, project_id: str, chapter_id: str, format: str) -> tuple[bytes, str]:
        """A chapter's pages as rendered (translation typeset), in chapter order, packed as a
        comic book archive (``cbz``) or a plain ``zip``; returns the archive and its file name.

        CBZ export: ``0001_<page>.png`` entries, stored uncompressed
        since PNG is already compressed.
        """
        if format not in ("cbz", "zip"):
            raise ValueError("format must be cbz or zip")
        snapshot = self.session(project_id).snapshot()
        chapter = next((ch for ch in chapters_of(snapshot) if ch.id == chapter_id), None)
        if chapter is None:
            raise KeyError(f"chapter {chapter_id} does not exist")
        if not chapter.pages:
            raise ValueError(f"{chapter.name} has no pages to export")
        width = max(4, len(str(len(chapter.pages))))
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
            for index, page_id in enumerate(chapter.pages, start=1):
                label = _file_stem(snapshot.page(page_id).label or "") or "page"
                archive.writestr(f"{index:0{width}d}_{label}.png", self.render(project_id, page_id, snapshot))
        name = _file_stem(f"{self.store.name(project_id)} - {chapter.name}") or "chapter"
        return buffer.getvalue(), f"{name}.{format}"

    def delete_chapter(self, project_id: str, chapter_id: str) -> ProjectView:
        self._idle(project_id)
        self.store.delete_chapter(project_id, chapter_id)
        return self.project_view(project_id)

    def arrange(self, project_id: str, layout: list[tuple[str, list[str]]]) -> ProjectView:
        self._idle(project_id)
        self.store.arrange(project_id, layout)
        return self.project_view(project_id)

    # -- pages ---------------------------------------------------------------------------------------

    def add_page(
        self, project_id: str, name: str, data: bytes, index: int | None = None, chapter_id: str | None = None
    ) -> PageView:
        page = self.store.import_page(project_id, name, data, index, chapter_id)
        return self.page(project_id, page)

    def remove_page(self, project_id: str, page_id: str) -> ProjectView:
        self._idle(project_id)
        self.store.remove_page(project_id, page_id)
        return self.project_view(project_id)

    def page(self, project_id: str, page_id: str) -> PageView:
        snapshot = self.session(project_id).snapshot()
        snapshot.page(page_id)
        return page_view(snapshot, page_id)

    # -- assets / rendering ----------------------------------------------------------------------

    def renderer(self):
        if self._renderer is None:
            from urakatsu.renderer import Renderer

            self._renderer = Renderer(self.config.typesetting)
        return self._renderer

    def render(self, project_id: str, page_id: str, snapshot: Optional[Snapshot] = None) -> bytes:
        snapshot = snapshot or self.session(project_id).snapshot()
        key = (project_id, page_id, snapshot.revision)
        with self._render_lock:
            cached = self._renders.get(key)
            if cached is not None:
                self._renders.move_to_end(key)
                return cached
        with self._draw_lock:
            image = self.renderer().render(snapshot, page_id).composite()
        data = _png(image)
        with self._render_lock:
            self._renders[key] = data
            while len(self._renders) > RENDER_CACHE:
                self._renders.popitem(last=False)
        return data

    def asset(self, project_id: str, page_id: str, kind: str) -> Optional[tuple[bytes, str, int, int]]:
        snapshot = self.session(project_id).snapshot()
        meta = snapshot.page(page_id)
        size = (int(meta.width), int(meta.height))
        if kind == "render":
            return self.render(project_id, page_id, snapshot), "image/png", *size
        if kind == "thumbnail":
            source = snapshot.asset(page_id, "source")
            if source is None:
                return None
            if source.blob not in self._thumbnails:
                image = Image.open(io.BytesIO(snapshot.blobs.read(source.blob))).convert("RGB")
                image.thumbnail((256, 256))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=80)
                self._thumbnails[source.blob] = buffer.getvalue()
            return self._thumbnails[source.blob], "image/jpeg", *size
        if kind.startswith("layer:"):
            layer = kind.removeprefix("layer:")
            if layer not in snapshot.raster_layers(page_id):
                return None
            asset = snapshot.asset(layer, "source")
            return (snapshot.blobs.read(asset.blob), asset.media_type, *size) if asset else None
        if kind == "cleanup":
            for layer in snapshot.raster_layers(page_id, c.RasterLayerKind.CLEANUP):
                asset = snapshot.asset(layer, "source")
                if asset is not None:
                    return snapshot.blobs.read(asset.blob), asset.media_type, *size
            return None
        asset = snapshot.asset(page_id, kind)
        if asset is None:
            return None
        return snapshot.blobs.read(asset.blob), asset.media_type, *size

    # -- text blocks (user edits, committed with user authorship) ------------------------------------

    def update_text_block(self, project_id: str, page_id: str, layer_id: str, patch: dict[str, Any]) -> PageView:
        session = self.session(project_id)
        snapshot = session.snapshot()
        if layer_id not in snapshot.text_layers(page_id):
            raise KeyError(f"page {page_id} has no text layer {layer_id}")
        content = snapshot.layer_content(layer_id)
        edit = snapshot.edit()
        if "source_text" in patch:
            current = snapshot.component(content, c.SourceText)
            edit.set(content, c.SourceText(text=c.Authored.user(patch["source_text"] or ""), language=current.language if current else None))
        if "translation" in patch:
            current = snapshot.component(content, c.Translation)
            language = current.language if current else self.config.translation.target_language
            text = patch["translation"] or ""
            # Rich text comes with the text it formats; a translation without styles is plain.
            styles = [
                c.TextStyle(**{**style, "color": parse_rgba(style.get("color"))})
                for style in patch.get("translation_styles") or []
            ]
            styles = c.normalize_text_styles(styles, len(text))
            edit.set(content, c.Translation(text=c.Authored.user(text), language=language, styles=styles))
        if "typography" in patch:
            edit.set(layer_id, _typography(patch["typography"], snapshot.component(layer_id, c.Typography)))
        if "frame" in patch:
            points = [c.Point(x=float(p["x"]), y=float(p["y"])) for p in patch["frame"]]
            if len(points) < 3:
                raise ValueError("a frame needs at least three points")
            edit.set(layer_id, c.Geometry(points=points))
            layout = snapshot.component(layer_id, c.TextLayout)
            if layout is not None:
                edit.set(layer_id, layout.model_copy(update={"origin": c.user(), "angle_degrees": None}))
        if "visible" in patch:
            visibility = snapshot.component(layer_id, c.Visibility) or c.Visibility()
            visibility.visible = bool(patch["visible"])
            edit.set(layer_id, visibility)
        session.commit(edit.finish())
        self.store.touch(project_id)
        return self.page(project_id, page_id)

    def create_text_block(self, project_id: str, page_id: str, bounds: dict[str, float]) -> PageView:
        """A manual paragraph layer with its own frame (no detected region)."""
        session = self.session(project_id)
        snapshot = session.snapshot()
        snapshot.page(page_id)
        edit = snapshot.edit()
        content = edit.add_text_content(page_id, At.END)
        layer = edit.add_text_layer(page_id, At.END, content, c.TextLayout(kind=c.TextLayoutKind.PARAGRAPH))
        edit.set(layer, c.Geometry.rectangle(bounds["x"], bounds["y"], bounds["width"], bounds["height"]))
        session.commit(edit.finish())
        self.store.touch(project_id)
        return self.page(project_id, page_id)

    def delete_text_block(self, project_id: str, page_id: str, layer_id: str) -> PageView:
        session = self.session(project_id)
        snapshot = session.snapshot()
        if layer_id not in snapshot.text_layers(page_id):
            raise KeyError(f"page {page_id} has no text layer {layer_id}")
        content = snapshot.layer_content(layer_id)
        edit = snapshot.edit()
        edit.remove_entity(layer_id, RemovePolicy.CASCADE)
        still_presented = [r for r in snapshot.relations_to(content, c.Relations.PRESENTS) if r.source != layer_id]
        if not still_presented:
            edit.remove_entity(content, RemovePolicy.CASCADE)
        session.commit(edit.finish())
        self.store.touch(project_id)
        return self.page(project_id, page_id)

    # -- layers (any layer: text or raster) ------------------------------------------------------------

    def _layer_siblings(self, snapshot: Snapshot, page_id: str, layer_id: str) -> list[str]:
        if layer_id in snapshot.text_layers(page_id):
            return snapshot.text_layers(page_id)
        if layer_id in snapshot.raster_layers(page_id):
            return snapshot.raster_layers(page_id)
        raise KeyError(f"page {page_id} has no layer {layer_id}")

    def set_layer_visibility(
        self, project_id: str, page_id: str, layer_ids: list[str], visible: Optional[bool], opacity: Optional[float]
    ) -> PageView:
        session = self.session(project_id)
        snapshot = session.snapshot()
        edit = snapshot.edit()
        for layer_id in layer_ids:
            self._layer_siblings(snapshot, page_id, layer_id)
            visibility = snapshot.component(layer_id, c.Visibility) or c.Visibility()
            if visible is not None:
                visibility.visible = visible
            if opacity is not None:
                visibility.opacity = min(1.0, max(0.0, float(opacity)))
            edit.set(layer_id, visibility)
        session.commit(edit.finish())
        self.store.touch(project_id)
        return self.page(project_id, page_id)

    def move_layer(self, project_id: str, page_id: str, layer_id: str, delta: int) -> PageView:
        """Moves a layer ``delta`` places among its siblings (positive is later, which draws on top)."""
        session = self.session(project_id)
        snapshot = session.snapshot()
        siblings = self._layer_siblings(snapshot, page_id, layer_id)
        index = siblings.index(layer_id)
        target = max(0, min(len(siblings) - 1, index + delta))
        if target == index:
            return self.page(project_id, page_id)
        anchor = siblings[target]
        edit = snapshot.edit()
        edit.move_entity(layer_id, snapshot.parent(layer_id), At.after(anchor) if target > index else At.before(anchor))
        session.commit(edit.finish())
        self.store.touch(project_id)
        return self.page(project_id, page_id)

    def set_raster_layer(
        self, project_id: str, page_id: str, kind: str, layer_id: Optional[str], png: bytes, at: Optional[tuple[int, int]] = None
    ) -> PageView:
        """Replaces a raster layer's pixels (brush and eraser strokes); creates the paint layer on first use.

        With ``at``, ``png`` is a patch: it replaces the layer's pixels in its rectangle at that offset.
        """
        layer_kind = c.RasterLayerKind(kind)
        image = Image.open(io.BytesIO(png))
        image.load()  # decodes it all, so a damaged upload fails here
        session = self.session(project_id)
        snapshot = session.snapshot()
        meta = snapshot.page(page_id)
        size = (int(meta.width), int(meta.height))
        if at is None and image.size != size:
            raise ValueError(f"layer is {image.width}x{image.height}, the page is {size[0]}x{size[1]}")
        if at is not None and not (0 <= at[0] and 0 <= at[1] and at[0] + image.width <= size[0] and at[1] + image.height <= size[1]):
            raise ValueError(f"a {image.width}x{image.height} patch at {at} is outside the {size[0]}x{size[1]} page")
        edit = snapshot.edit()
        rasters = snapshot.raster_layers(page_id)
        if layer_id:
            if layer_id not in rasters:
                raise KeyError(f"page {page_id} has no raster layer {layer_id}")
            entity = layer_id
        else:
            existing = snapshot.raster_layers(page_id, layer_kind)
            if existing:
                entity = existing[-1]
            else:
                entity = edit.add_entity(page_id, At.after(rasters[-1]) if rasters else At.START)
                edit.set(entity, c.RasterLayer(origin=c.user(), name=layer_kind.value.capitalize(), kind=layer_kind))
        if at is not None:
            current = snapshot.asset(entity, "source") if entity in rasters else None
            layer = Image.open(io.BytesIO(snapshot.blobs.read(current.blob))).convert("RGBA") if current else Image.new("RGBA", size)
            layer.paste(image.convert("RGBA"), at)
            data = _png(layer)
        elif image.format == "PNG" and image.mode == "RGBA":
            data = png  # already what is stored: re-encoding would cost ~40 ms for the same pixels
        else:
            data = _png(image.convert("RGBA"))
        edit.set_asset(
            entity,
            "source",
            c.AssetInput(
                data=data,
                media_type="image/png",
                metadata=c.AssetMetadata(width=size[0], height=size[1], attributes={}),
            ),
        )
        session.commit(edit.finish())
        self.store.touch(project_id)
        return self.page(project_id, page_id)

    def page_art(self, project_id: str, page_id: str, base: str = "source") -> Image.Image:
        """The page's pixels without text, as RGB: the original scan (``source``), or what the
        editor shows under the text layers (``current``: the scan with cleanup, paint and edits)."""
        snapshot = self.session(project_id).snapshot()
        if base == "source":
            asset = snapshot.asset(page_id, "source")
            if asset is None:
                raise KeyError(f"page {page_id} has no source image")
            return Image.open(io.BytesIO(snapshot.blobs.read(asset.blob))).convert("RGB")
        if base != "current":
            raise ValueError(f"unknown base {base!r}; use source or current")
        with self._draw_lock:
            frame = self.renderer().render(snapshot, page_id)
        art = replace(frame, layers=[layer for layer in frame.layers if not layer.is_text()]).composite()
        flat = Image.new("RGB", art.size, (255, 255, 255))
        flat.paste(art, mask=art.getchannel("A"))
        return flat

    def add_raster_layer(
        self,
        project_id: str,
        page_id: str,
        name: str,
        image: Image.Image,
        kind: str = "edit",
        hide_layers: tuple[str, ...] = (),
        hide_all_text: bool = False,
    ) -> tuple[PageView, str, list[str]]:
        """Adds a raster layer above the others and hides text layers in the same commit, so one
        undo removes both: ``hide_layers``, or every visible one with ``hide_all_text``. Which
        exist is read now, not when the caller looked (an image edit takes minutes, and a
        re-run in between replaces the page's text layers); ids that are gone are skipped.
        Returns the page, the new layer's id and the layers hidden."""
        session = self.session(project_id)
        snapshot = session.snapshot()
        meta = snapshot.page(page_id)
        if image.size != (int(meta.width), int(meta.height)):
            raise ValueError(f"layer is {image.width}x{image.height}, the page is {int(meta.width)}x{int(meta.height)}")
        texts = snapshot.text_layers(page_id)
        buffer = io.BytesIO()
        image.convert("RGBA").save(buffer, format="PNG")
        edit = snapshot.edit()
        rasters = snapshot.raster_layers(page_id)
        entity = edit.add_entity(page_id, At.after(rasters[-1]) if rasters else At.START)
        edit.set(entity, c.RasterLayer(origin=c.user(), name=name, kind=c.RasterLayerKind(kind)))
        edit.set_asset(
            entity,
            "source",
            c.AssetInput(
                data=buffer.getvalue(),
                media_type="image/png",
                metadata=c.AssetMetadata(width=image.width, height=image.height, attributes={}),
            ),
        )
        hidden = []
        for layer_id in texts if hide_all_text else [t for t in hide_layers if t in texts]:
            visibility = snapshot.component(layer_id, c.Visibility) or c.Visibility()
            if visibility.visible:
                visibility.visible = False
                edit.set(layer_id, visibility)
                hidden.append(layer_id)
        session.commit(edit.finish())
        self.store.touch(project_id)
        return self.page(project_id, page_id), entity, hidden

    def add_empty_layer(self, project_id: str, page_id: str, name: Optional[str] = None, above: Optional[str] = None) -> PageView:
        """Adds an empty paint layer (Photoshop's New Layer) directly above raster layer ``above``, or
        above every raster layer. It has no pixels until its first stroke; the renderer skips it till then."""
        session = self.session(project_id)
        snapshot = session.snapshot()
        snapshot.page(page_id)
        rasters = snapshot.raster_layers(page_id)
        if above is not None and above not in rasters:
            raise KeyError(f"page {page_id} has no raster layer {above}")
        if not name:
            taken = {snapshot.component(layer, c.RasterLayer).name for layer in rasters}
            name = next(f"Layer {n}" for n in range(1, len(rasters) + 2) if f"Layer {n}" not in taken)
        edit = snapshot.edit()
        anchor = above or (rasters[-1] if rasters else None)
        entity = edit.add_entity(page_id, At.after(anchor) if anchor else At.START)
        edit.set(entity, c.RasterLayer(origin=c.user(), name=name, kind=c.RasterLayerKind.PAINT))
        session.commit(edit.finish())
        self.store.touch(project_id)
        return self.page(project_id, page_id)

    def delete_layer(self, project_id: str, page_id: str, layer_id: str) -> PageView:
        snapshot = self.session(project_id).snapshot()
        if layer_id in snapshot.text_layers(page_id):
            return self.delete_text_block(project_id, page_id, layer_id)
        if layer_id not in snapshot.raster_layers(page_id):
            raise KeyError(f"page {page_id} has no layer {layer_id}")
        session = self.session(project_id)
        edit = snapshot.edit()
        edit.remove_entity(layer_id, RemovePolicy.CASCADE)
        session.commit(edit.finish())
        self.store.touch(project_id)
        return self.page(project_id, page_id)

    # -- jobs ----------------------------------------------------------------------------------------

    async def start_run(
        self,
        project_id: str,
        operation: Operation,
        page_ids: list[str] | None = None,
        region: Optional[Bounds] = None,
        entities: list[str] | None = None,
        inpainting_mask: Optional[bytes] = None,
        fill_prompt: Optional[str] = None,
    ) -> Job:
        """Starts a pipeline run. ``fill_prompt`` (with ``inpainting_mask``) makes it a generative fill."""
        session = self.session(project_id)
        if self.automations.busy:
            raise ProjectBusy("an automation is running; stop it to run the pipeline by hand")
        if entities:
            scope = EntitiesScope(tuple(entities))
        elif region is not None:
            if not page_ids or len(page_ids) != 1:
                raise ValueError("region scope requires exactly one page")
            scope = RegionScope(page_ids[0], region)
        elif page_ids:
            scope = PagesScope(tuple(page_ids))
        else:
            scope = ProjectScope()
        mask = None
        if fill_prompt is not None and not inpainting_mask:
            raise ValueError("generative fill requires a selection mask")
        if inpainting_mask:
            if not page_ids or len(page_ids) != 1:
                raise ValueError("an inpainting mask requires exactly one page")
            mask = InpaintingMask(page=page_ids[0], png=inpainting_mask, prompt=fill_prompt.strip() if fill_prompt else None, generative=fill_prompt is not None)
        job = Job(id=f"job_{int(time.time() * 1000):x}{len(self.jobs)}", project_id=project_id, stop=StopToken())
        request = Request(operation=operation, scope=scope, stop=job.stop, progress=job.publish, inpainting_mask=mask)
        self.jobs[job.id] = job

        async def runner() -> None:
            try:
                report = await self.pipeline.execute(session, request)
                job.completed, job.total = report.completed, report.total
                job.status = report.status.value
                job.publish(
                    Progress(
                        kind=ProgressKind.STOPPED if job.status == "stopped" else ProgressKind.COMPLETED,
                        completed=report.completed,
                        total=report.total,
                        elapsed_ms=report.elapsed_ms,
                    )
                )
            except PipelineError as error:
                job.status = "failed"
                job.error = str(error)
                job.publish(Progress(kind=ProgressKind.FAILED, stage=error.stage, message=str(error)))
            except Exception as error:
                log.exception("run %s crashed", job.id)
                job.status = "failed"
                job.error = str(error)
                job.publish(Progress(kind=ProgressKind.FAILED, message=str(error)))
            finally:
                self.store.touch(project_id)
                job.finish()

        job.task = asyncio.create_task(runner())
        return job

    async def events(self, job_id: str) -> AsyncIterator[Progress]:
        job = self.jobs[job_id]
        queue: asyncio.Queue = asyncio.Queue()
        for event in job.events:
            queue.put_nowait(event)
        if job.status != "running":
            queue.put_nowait(None)
        else:
            job.queues.append(queue)
        try:
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event
        finally:
            if queue in job.queues:
                job.queues.remove(queue)

    def stop(self, job_id: str) -> None:
        self.jobs[job_id].stop.stop()

    def job_info(self) -> list[JobInfo]:
        return [
            JobInfo(id=j.id, project_id=j.project_id, status=j.status, completed=j.completed, total=j.total, started_at=j.started_at, error=j.error)
            for j in sorted(self.jobs.values(), key=lambda j: j.started_at, reverse=True)
        ]

    # -- fonts ----------------------------------------------------------------------------------------

    def fonts(self) -> list[tuple[str, list[str]]]:
        families = self.renderer().available_fonts()
        result = []
        for family in families:
            name = getattr(family, "name", None) or family.get("name")  # type: ignore[union-attr]
            faces = getattr(family, "faces", None) or family.get("faces", [])  # type: ignore[union-attr]
            result.append((name, [getattr(f, "post_script_name", str(f)) for f in faces]))
        return result

    def font_face(self, family: str, weight: int = 400, italic: bool = False) -> Optional[tuple[bytes, str]]:
        """The font file the renderer draws ``family`` with at that weight and style, with its media type, for
        the browser: the on-canvas text editor shows text in the layer's font, which may come from the bundled
        catalog rather than the system. None when the family is not installed, or is a face in a collection
        (browsers load whole files only)."""
        from urakatsu.renderer.types import FontStyle

        for font in self.renderer().fonts.resolve(family, weight, FontStyle.ITALIC if italic else None, [], "A"):
            if font.family_name.lower() != family.lower():
                continue
            data = font.data()
            if font.index != 0 or data[:4] == b"ttcf":
                return None
            return data, "font/otf" if data[:4] == b"OTTO" else "font/ttf"
        return None

    async def close(self) -> None:
        await self.automations.close()
        for job in self.jobs.values():
            job.stop.stop()
        tasks = [j.task for j in self.jobs.values() if j.task and not j.task.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.links.close()
        self.systems.close()
        self.pipeline.unload_all()


def _file_stem(label: str) -> str:
    """A page or archive name safe on every file system: no extension, no reserved characters
    (the export rule)."""
    name = label.strip().rstrip(". ")
    stem, dot, extension = name.rpartition(".")
    if dot and stem and 1 <= len(extension) <= 5 and extension.isalnum():
        name = stem
    return "".join("_" if ch in '<>:"/\\|?*' or ord(ch) < 32 else ch for ch in name).strip()


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=3)
    return buffer.getvalue()


def _typography(data: dict[str, Any], current: Optional[c.Typography]) -> c.Typography:
    """Merges ``data`` (the properties the user set) into the layer's style, which becomes user-authored.

    Differs from Koharu, whose ``set_typography`` stores the inspector's whole style as user-authored:
    a writing mode the detector guessed (vertical for vertical Japanese) is only a hint the renderer
    ignores for a horizontal-script translation, but once user-authored it is obeyed, so changing
    just the color turned an English free-text block into vertical letter columns. A generated
    writing mode is therefore dropped (back to automatic) unless the user sets one.
    """
    base = current.model_dump() if current else {}
    base.pop("origin", None)
    if current is not None and not isinstance(current.origin, c.UserOrigin) and "writing_mode" not in data:
        base["writing_mode"] = None
    value = dict(base)
    for key in ("preferred_font", "font_weight", "size", "stroke_width"):
        if key in data:
            value[key] = data[key]
    if "auto_fit" in data:
        value["auto_fit"] = bool(data["auto_fit"])
    if "font_style" in data:
        value["font_style"] = c.FontStyle(data["font_style"]) if data["font_style"] else None
    if "alignment" in data:
        value["alignment"] = c.TextAlignment(data["alignment"]) if data["alignment"] else None
    if "writing_mode" in data:
        value["writing_mode"] = c.WritingMode(data["writing_mode"]) if data["writing_mode"] else None
    for key in ("color", "stroke_color"):
        if key in data:
            value[key] = parse_rgba(data[key])
    for key in ("fill_gradient", "stroke_gradient"):
        if key in data:
            gradient = data[key]
            stops = [c.GradientStop(offset=float(stop["offset"]), color=parse_rgba(stop["color"])) for stop in (gradient or {}).get("stops", []) if parse_rgba(stop.get("color"))]
            # Fewer than two stops is no gradient.
            value[key] = c.Gradient(kind=c.GradientKind(gradient.get("kind") or "linear"), angle=float(gradient.get("angle", 90.0)), stops=stops) if len(stops) >= 2 else None
    if not value.get("auto_fit", True) and value.get("size") is None:
        raise ValueError("a fixed-size typography needs a size")
    return c.Typography(**value)
