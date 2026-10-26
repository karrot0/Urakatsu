"""Watermarks stage: removes repeated watermarks before the page is read. Not in Koharu.

With ``WatermarksConfig.mode`` set to ``remove`` this stage runs first on each page. Its first page
in a chapter analyses the whole chapter: the watermark detector (the stage's provider) proposes
boxes on every page, and ``pipeline/watermarks.py`` learns each mark that repeats across pages and
unblends it, which keeps the art under the mark instead of repainting it. The analysis is kept for
the chapter's other pages, so each page then commits only its own result.

The result is a generated raster layer per page, "Watermarks", at the bottom of the layer stack
(right over the source): opaque where a mark was removed, transparent elsewhere. Like the Cleanup
layer it is not redone while it exists, a forced run (Redo) replaces it unless the user edited
it, and hiding or deleting it brings the marks back.

The later stages read the page through ``page_image``: the source with a visible Watermarks layer
drawn over it, so detection does not box a scanlator URL as text, OCR does not read a platform
logo into a speech balloon, and inpainting does not paint a mark back into its cleanup.
"""

from __future__ import annotations

import asyncio
import io
import logging
from collections import OrderedDict
from typing import Optional

import numpy as np
from PIL import Image

from urakatsu.pipeline.watermarks import Candidate, ChapterResult, remove_watermarks
from urakatsu.providers.base import Provider
from urakatsu.scene import At, Patch, Snapshot
from urakatsu.scene import components as c

from .base import StageInput, StageProcessor, finish, generation

log = logging.getLogger(__name__)

PRODUCER = "dev.urakatsu.pipeline.watermarks"
LAYER_NAME = "Watermarks"
SOURCE = "source"
# Chapter analyses kept, so a run across a chapter analyses it once.
CACHED_CHAPTERS = 2


def watermark_layer(scene: Snapshot, page: str) -> Optional[str]:
    """The page's Watermarks layer, if it has one with pixels."""
    for entity in scene.raster_layers(page, c.RasterLayerKind.WATERMARK):
        if scene.asset(entity, SOURCE) is not None:
            return entity
    return None


def page_image(input: StageInput) -> Optional[Image.Image]:
    """The page as the stages read it: the source, with a visible Watermarks layer drawn over it."""
    source = input.images.get(input.scene, input.page, SOURCE)
    layer = watermark_layer(input.scene, input.page)
    if source is None or layer is None:
        return source
    visibility = input.scene.component(layer, c.Visibility)
    if visibility is not None and not visibility.visible:
        return source
    overlay = input.images.get(input.scene, layer, SOURCE)
    if overlay is None or overlay.size != source.size:
        return source
    key = f"{input.scene.asset(input.page, SOURCE).blob}+{input.scene.asset(layer, SOURCE).blob}"
    return input.images.derived(key, lambda: Image.alpha_composite(source.convert("RGBA"), overlay.convert("RGBA")).convert("RGB"))


def chapter_pages(scene: Snapshot, page: str) -> list[str]:
    """The pages of the chapter holding ``page``, in order (every page when there are no chapters)."""
    pages = scene.pages()
    chapters = scene.project_component(c.Chapters)
    if chapters is not None:
        present = set(pages)
        for chapter in chapters.chapters:
            if page in chapter.pages:
                return [p for p in chapter.pages if p in present]
    return pages


def _torch_device(provider: Provider) -> str:
    """The torch device on the provider's GPU (the mattes are fitted with torch), else the CPU."""
    try:
        runtimes = provider.runtimes
        if runtimes is None:
            return "cpu"
        return str(runtimes.runtime("torch").torch_device)
    except Exception:  # no torch runtime: fit on the CPU
        return "cpu"


class WatermarksStage(StageProcessor):
    def __init__(self) -> None:
        self._analyses: "OrderedDict[tuple, ChapterResult]" = OrderedDict()

    def skip(self, input: StageInput, provider: Provider) -> bool:
        layer = watermark_layer(input.scene, input.page)
        # A forced run (Redo) removes the marks again, but never over a layer the user edited.
        return layer is not None and (not getattr(input, "force", False) or input.scene.user_authored(layer))

    def done(self, input: StageInput) -> bool:
        return watermark_layer(input.scene, input.page) is not None

    async def process(self, input: StageInput, provider: Provider) -> Patch:
        pages = chapter_pages(input.scene, input.page)
        sources = [(page, input.scene.asset(page, SOURCE)) for page in pages]
        sources = [(page, asset.blob) for page, asset in sources if asset is not None]
        key = (tuple(sources), provider.id, provider.settings.model_dump_json())
        result = self._analyses.get(key)
        if result is None:
            result = await self._analyse(input, provider, [page for page, _ in sources])
            self._analyses[key] = result
            while len(self._analyses) > CACHED_CHAPTERS:
                self._analyses.popitem(last=False)
        else:
            self._analyses.move_to_end(key)
        page_result = result.pages.get(input.page)
        if page_result is None:
            return finish(input.scene.edit())  # no repeated mark on this page
        return self._commit(input, provider, page_result.image, page_result.footprint)

    async def _analyse(self, input: StageInput, provider: Provider, pages: list[str]) -> ChapterResult:
        images: dict[str, np.ndarray] = {}
        candidates: list[Candidate] = []
        for page in pages:
            image = input.images.get(input.scene, page, SOURCE)
            if image is None:
                continue
            images[page] = np.asarray(image.convert("RGB"))
            for box in await provider.detect_watermarks(image):
                candidates.append(Candidate(page, box.box, box.score))
        device = _torch_device(provider)
        result = await asyncio.to_thread(remove_watermarks, images, candidates, device)
        log.info("watermarks: %d repeated in a chapter of %d pages (%d candidates); %d pages changed", len(result.watermarks), len(images), len(candidates), len(result.pages))
        return result

    def _commit(self, input: StageInput, provider: Provider, image: np.ndarray, footprint: np.ndarray) -> Patch:
        page = input.page
        gen = generation(PRODUCER, provider.id)
        edit = input.scene.edit_as(gen)
        edit.observe_assets(page)
        entity = watermark_layer(input.scene, page)
        if entity is not None:
            edit.observe(entity, c.RasterLayer)
            edit.observe_assets(entity)
        else:
            entity = edit.add_entity(page, At.START)
            edit.set(entity, c.RasterLayer(origin=c.generated(gen), name=LAYER_NAME, kind=c.RasterLayerKind.WATERMARK))
        overlay = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
        overlay[footprint, :3] = image[footprint]
        overlay[footprint, 3] = 255
        buffer = io.BytesIO()
        Image.fromarray(overlay, "RGBA").save(buffer, format="PNG")
        edit.set_asset(
            entity,
            SOURCE,
            c.AssetInput(
                data=buffer.getvalue(),
                media_type="image/png",
                metadata=c.AssetMetadata(width=overlay.shape[1], height=overlay.shape[0], attributes={}),
            ),
        )
        return finish(edit)
