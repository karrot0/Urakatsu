"""Colorization stage. Not in Koharu.

The provider paints the page as the reader would see it, cleaned and decensored, so it never
colors lettering that is about to go. The result is a raster layer of its own, "Color", over the
Decensor and Cleanup layers, so it can be hidden, deleted or redone without touching them. Like
those it is not redone while it exists, a forced run (Redo) replaces it unless the user edited it,
and a regional run replaces only what is inside the region.

A diffusion model sees an empty bubble as a mistake and letters it, in text of its own that no
prompt keeps out, so lettering is the stage's to protect, whatever the provider. The layer is
transparent over detected bubbles: they stay as cleaned, and a cleanup edit made there shows
through. Where text outside a bubble was erased, what the model drew on the paper is painted over
from the color around it. Elsewhere the layer is opaque, so a cleanup edit made after it shows
only once the page is redone.

A page that already has color is left alone, unless a forced run asks for it.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw

from urakatsu.providers.base import Provider
from urakatsu.scene import At, Patch, Snapshot
from urakatsu.scene import components as c

from .base import StageInput, StageProcessor, finish, generation
from .censorship import SOURCE, _asset, _png, _visible_over, cleanup_layer, decensor_layer
from .inpainting import TEXT_FLAT_FILL_PAD, _clear_outside
from .watermarks import page_image, watermark_layer

PRODUCER = "dev.urakatsu.pipeline.colorization"
LAYER_NAME = "Color"
# A page is compared at this size: JPEG color noise averages out, and it costs next to nothing.
SAMPLE_SIZE = 128
# Channel spread above which a pixel has color: above tinted paper and scanner casts.
COLOR_SPREAD = 48
# Share of pixels with color above which the page is a color page.
COLOR_SHARE = 0.05
# Page gray level from which a pixel in erased text counts as bare paper, with no art to keep.
PAPER_LEVEL = 200
PAINT_OVER_RADIUS = 5


def is_colored(image: Image.Image) -> bool:
    """Whether ``image`` already has color."""
    small = np.asarray(image.convert("RGB").resize((SAMPLE_SIZE, SAMPLE_SIZE), Image.Resampling.BILINEAR), dtype=np.int16)
    spread = small.max(axis=2) - small.min(axis=2)
    return float((spread > COLOR_SPREAD).mean()) > COLOR_SHARE


def color_layer(scene: Snapshot, page: str) -> Optional[str]:
    """The page's Color layer, if it has one."""
    layers = scene.raster_layers(page, c.RasterLayerKind.COLOR)
    return layers[0] if layers else None


def cleaned_page(input: StageInput, source: Image.Image) -> Image.Image:
    """``source`` with the visible Cleanup and Decensor layers drawn over it."""
    image = _visible_over(input, source, cleanup_layer(input.scene, input.page))
    return _visible_over(input, image, decensor_layer(input.scene, input.page))


def lettering_masks(scene: Snapshot, page: str, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """The page's bubbles, and the boxes of text in no bubble, as HxW masks of ``size``."""
    bubbles = Image.new("L", size, 0)
    free_text = Image.new("L", size, 0)
    for entity in scene.descendants(page):
        region = scene.component(entity, c.Region)
        geometry = scene.component(entity, c.Geometry)
        if region is None or geometry is None or len(geometry.points) < 3:
            continue
        if region.kind == c.RegionKinds.BUBBLE:
            ImageDraw.Draw(bubbles).polygon([(point.x, point.y) for point in geometry.points], fill=255)
        elif region.kind == c.RegionKinds.TEXT and not scene.relations_from(entity, c.Relations.INSIDE):
            left, top, right, bottom = geometry.bounds()
            box = (left - TEXT_FLAT_FILL_PAD, top - TEXT_FLAT_FILL_PAD, right + TEXT_FLAT_FILL_PAD, bottom + TEXT_FLAT_FILL_PAD)
            ImageDraw.Draw(free_text).rectangle(box, fill=255)
    return np.asarray(bubbles), np.asarray(free_text)


def paint_over_paper(image: Image.Image, color: Image.Image, free_text: np.ndarray) -> Image.Image:
    """``color`` with what the model drew over bare paper in ``free_text`` painted over from the
    color around it. Art in the box is not paper, so it stays."""
    paper = (free_text > 0) & (np.asarray(image.convert("L")) >= PAPER_LEVEL)
    if not paper.any():
        return color
    return Image.fromarray(cv2.inpaint(np.asarray(color.convert("RGB")), paper.astype(np.uint8), PAINT_OVER_RADIUS, cv2.INPAINT_TELEA), "RGB")


class ColorizationStage(StageProcessor):
    def skip(self, input: StageInput, provider: Provider) -> bool:
        layer = color_layer(input.scene, input.page)
        # A forced run (Redo) repaints the page, but never over a layer the user edited.
        if layer is not None:
            return not input.force or input.scene.user_authored(layer)
        source = page_image(input)
        return source is None or (not input.force and is_colored(source))

    def done(self, input: StageInput) -> bool:
        return color_layer(input.scene, input.page) is not None

    async def process(self, input: StageInput, provider: Provider) -> Patch:
        page = input.page
        source = page_image(input)
        if source is None:
            raise ValueError(f"page {page} has no source image")
        image = cleaned_page(input, source)
        result = await provider.colorize(image)
        if result.size != image.size:
            raise ValueError(f"colorized image dimensions do not match page {page}")
        bubbles, free_text = lettering_masks(input.scene, page, image.size)
        result = paint_over_paper(image, result, free_text)
        entity = color_layer(input.scene, page)
        previous = input.images.get(input.scene, entity, SOURCE) if entity is not None and input.region is not None else None
        return self._commit(input, provider, entity, result, bubbles, previous)

    def _commit(self, input: StageInput, provider: Provider, entity: Optional[str], result: Image.Image, bubbles: np.ndarray, previous: Optional[Image.Image]) -> Patch:
        page = input.page
        gen = generation(PRODUCER, provider.id)
        edit = input.scene.edit_as(gen)
        edit.observe_assets(page)

        overlay = np.array(result.convert("RGBA"))
        overlay[bubbles > 0, 3] = 0
        if input.region is not None:
            # A regional run replaces the layer inside the region and keeps the rest.
            inside = np.ones(overlay.shape[:2], dtype=np.uint8)
            _clear_outside(inside, inside, input.region)
            kept = np.array(previous.convert("RGBA")) if previous is not None else np.zeros_like(overlay)
            overlay = np.where(inside[..., None] > 0, overlay, kept)

        if entity is not None:
            edit.observe(entity, c.RasterLayer)
            edit.observe_assets(entity)
        else:
            scene = input.scene
            below = decensor_layer(scene, page) or cleanup_layer(scene, page) or watermark_layer(scene, page)
            entity = edit.add_entity(page, At.after(below) if below is not None else At.START)
            edit.set(entity, c.RasterLayer(origin=c.generated(gen), name=LAYER_NAME, kind=c.RasterLayerKind.COLOR))
        edit.set_asset(entity, SOURCE, _asset(_png(Image.fromarray(overlay, "RGBA")), overlay.shape[1], overlay.shape[0]))
        return finish(edit)
