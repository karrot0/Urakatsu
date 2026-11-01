"""Censor detection and decensor stages. Not in Koharu.

Censor detection maps the bars on each page into the ``censor-mask`` asset, to review on the
canvas before anything is repainted. Decensor reconstructs under that mask, or under a mask the
user painted, and can run without detection. Ported from the Koharu fork at
https://github.com/spgsroot/koharu (``stages/censor_detection.rs``, ``stages/decensor.rs``).

Unlike the fork, the result is a raster layer of its own, "Decensor", right over the Cleanup
layer, and not part of it: inpainting does not redo a page with a Cleanup layer, and this way
either stage can be redone without the other. Like the Cleanup layer it is not redone while it
exists, a forced run (Redo) replaces it unless the user edited it, and hiding or deleting it
brings the bars back. Its alpha is the mask, so a soft mask edge blends into the art.

Each group of nearby bars goes to the model as one tight crop, since it edits what it is shown
from an instruction, and a whole page would reach it at a fraction of its size.
"""

from __future__ import annotations

import asyncio
import io
from typing import Callable, Optional

import numpy as np
from PIL import Image
from scipy import ndimage

from urakatsu.providers.base import Provider
from urakatsu.scene import At, Patch, Snapshot
from urakatsu.scene import components as c

from .base import StageInput, StageProcessor, finish, generation
from .inpainting import EDIT_CONTEXT, EDIT_CONTEXT_MIN, _clear_outside, fill_region
from .watermarks import page_image, watermark_layer

DETECTION_PRODUCER = "dev.urakatsu.pipeline.censor-detection"
DECENSOR_PRODUCER = "dev.urakatsu.pipeline.decensor"
LAYER_NAME = "Decensor"
SOURCE = "source"
CENSOR_MASK = "censor-mask"
# Bars closer than this go to the model in one crop.
GROUP_DISTANCE = 32
# Median brightness above which a group of bars is paper-white rather than ink.
WHITE_LEVEL = 128

Reconstruct = Callable[[Image.Image, np.ndarray, bool], Image.Image]


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _asset(data: bytes, width: int, height: int) -> c.AssetInput:
    return c.AssetInput(data=data, media_type="image/png", metadata=c.AssetMetadata(width=width, height=height, attributes={}))


class CensorDetectionStage(StageProcessor):
    def skip(self, input: StageInput, provider: Provider) -> bool:
        return self.done(input) and not input.force

    def done(self, input: StageInput) -> bool:
        return input.scene.asset(input.page, CENSOR_MASK) is not None

    async def process(self, input: StageInput, provider: Provider) -> Patch:
        image = page_image(input)
        if image is None:
            raise ValueError(f"page {input.page} has no source image")
        mask = np.array(await provider.detect_censorship(image), dtype=np.uint8)
        if mask.shape != (image.height, image.width):
            raise ValueError(f"censor mask dimensions do not match page {input.page}")
        if input.region is not None:
            # A regional run replaces the mask inside the region and keeps the rest.
            inside = np.ones_like(mask)
            _clear_outside(inside, inside, input.region)
            previous = input.images.get(input.scene, input.page, CENSOR_MASK)
            kept = np.asarray(previous.convert("L")) if previous is not None else np.zeros_like(mask)
            mask = np.where(inside > 0, mask, kept)
        edit = input.scene.edit_as(generation(DETECTION_PRODUCER, provider.id))
        edit.observe_assets(input.page)
        edit.set_asset(input.page, CENSOR_MASK, _asset(_png(Image.fromarray(mask, "L")), mask.shape[1], mask.shape[0]))
        return finish(edit)


def decensor_layer(scene: Snapshot, page: str) -> Optional[str]:
    """The page's Decensor layer, if it has one."""
    layers = scene.raster_layers(page, c.RasterLayerKind.DECENSOR)
    return layers[0] if layers else None


def cleanup_layer(scene: Snapshot, page: str) -> Optional[str]:
    layers = scene.raster_layers(page, c.RasterLayerKind.CLEANUP)
    return layers[0] if layers else None


def _visible_over(input: StageInput, base: Image.Image, layer: Optional[str]) -> Image.Image:
    """``base`` with ``layer`` drawn over it, while the layer is visible."""
    if layer is None:
        return base
    visibility = input.scene.component(layer, c.Visibility)
    overlay = input.images.get(input.scene, layer, SOURCE)
    if overlay is None or overlay.size != base.size or (visibility is not None and not visibility.visible):
        return base
    return Image.alpha_composite(base.convert("RGBA"), overlay.convert("RGBA")).convert("RGB")


def bar_groups(mask: np.ndarray) -> list[np.ndarray]:
    """``mask`` split into groups of nearby bars, each a mask of its own, top to bottom."""
    labels, count = ndimage.label(ndimage.binary_dilation(mask > 0, iterations=GROUP_DISTANCE // 2))
    if count == 0:
        return []
    rows = ndimage.find_objects(labels)
    return [np.where(labels == index + 1, mask, 0) for index in sorted(range(count), key=lambda i: rows[i][0].start)]


def is_white(image: np.ndarray, group: np.ndarray) -> bool:
    """Whether the bars are paper-white, from the pixels well inside the group's mask."""
    return float(np.median(image[group >= 127].mean(axis=1))) >= WHITE_LEVEL


def reconstruct_page(image: Image.Image, mask: np.ndarray, reconstruct: Reconstruct) -> Image.Image:
    """``image`` with each group of bars under ``mask`` reconstructed in a crop of its own."""
    output = image.convert("RGB")
    for group in bar_groups(mask):
        if not (group >= 127).any():
            continue
        white = is_white(np.asarray(output), group)
        whole = np.where(group > 0, 255, 0).astype(np.uint8)  # the soft edge is the layer's alpha
        output = fill_region(output, whole, lambda crop, crop_mask: reconstruct(crop, crop_mask, white), EDIT_CONTEXT, EDIT_CONTEXT_MIN)
    return output


class DecensorStage(StageProcessor):
    def skip(self, input: StageInput, provider: Provider) -> bool:
        if input.inpainting_mask is not None:
            return False
        layer = decensor_layer(input.scene, input.page)
        # A forced run (Redo) repaints the page, but never over a layer the user edited.
        if layer is not None and (not input.force or input.scene.user_authored(layer)):
            return True
        mask = input.images.get(input.scene, input.page, CENSOR_MASK)
        return mask is None or not np.asarray(mask.convert("L")).any()

    def done(self, input: StageInput) -> bool:
        return input.inpainting_mask is None and decensor_layer(input.scene, input.page) is not None

    def _mask(self, input: StageInput, size: tuple[int, int]) -> np.ndarray:
        if input.inpainting_mask is not None:
            mask = np.array(Image.open(io.BytesIO(input.inpainting_mask.png)).convert("L"))
        else:
            image = input.images.get(input.scene, input.page, CENSOR_MASK)
            mask = np.array(image.convert("L")) if image is not None else np.zeros(size[::-1], dtype=np.uint8)
        if mask.shape != size[::-1]:
            raise ValueError(f"censor mask dimensions do not match page {input.page}")
        if input.region is not None:
            _clear_outside(mask, mask, input.region)
        return mask

    async def process(self, input: StageInput, provider: Provider) -> Patch:
        page = input.page
        source = page_image(input)
        if source is None:
            raise ValueError(f"page {page} has no source image")
        entity = decensor_layer(input.scene, page)
        manual = input.inpainting_mask is not None
        # A later pass builds on what is kept of the layer; a whole-page run replaces it.
        keep = entity is not None and (manual or input.region is not None)
        image = _visible_over(input, source, cleanup_layer(input.scene, page))
        if keep:
            image = _visible_over(input, image, entity)
        mask = self._mask(input, source.size)
        if not mask.any():
            return finish(input.scene.edit())
        result = await asyncio.to_thread(reconstruct_page, image, mask, provider._reconstruct)
        previous = input.images.get(input.scene, entity, SOURCE) if keep else None
        return self._commit(input, provider, entity, np.asarray(result), mask, previous)

    def _commit(self, input: StageInput, provider: Provider, entity: Optional[str], result: np.ndarray, mask: np.ndarray, previous: Optional[Image.Image]) -> Patch:
        page = input.page
        manual = input.inpainting_mask is not None
        gen = generation(DECENSOR_PRODUCER, provider.id)
        edit = input.scene.edit() if manual else input.scene.edit_as(gen)
        edit.observe_assets(page)

        overlay = np.array(previous.convert("RGBA")) if previous is not None else np.zeros((*mask.shape, 4), dtype=np.uint8)
        painted = mask > 0
        overlay[painted, :3] = result[painted]
        overlay[painted, 3] = np.maximum(overlay[painted, 3], mask[painted])

        if entity is not None:
            edit.observe(entity, c.RasterLayer)
            edit.observe_assets(entity)
            layer = input.scene.component(entity, c.RasterLayer)
            if manual and layer is not None and not c.is_user(layer.origin):
                layer.origin = c.user()
                edit.set(entity, layer)
        else:
            below = cleanup_layer(input.scene, page) or watermark_layer(input.scene, page)
            entity = edit.add_entity(page, At.after(below) if below is not None else At.START)
            origin = c.user() if manual else c.generated(gen)
            edit.set(entity, c.RasterLayer(origin=origin, name=LAYER_NAME, kind=c.RasterLayerKind.DECENSOR))
        edit.set_asset(entity, SOURCE, _asset(_png(Image.fromarray(overlay, "RGBA")), overlay.shape[1], overlay.shape[0]))
        return finish(edit)
