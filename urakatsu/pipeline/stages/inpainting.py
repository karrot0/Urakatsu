"""Inpainting stage: port of stages/inpainting.rs.

The stage builds the removal mask (detection mask assets, or the user-painted
mask), fills uniform bubble backgrounds without inference, runs the provider on
bounded tiles (``inpaint_tiled``), and commits the result as the page's Cleanup
raster layer. The provider is any ``InpaintingProvider``: LaMa and AOT run on the
torch runtime, FLUX.2 Klein and RORem on the diffusion runtime; all four go through
the same tiling here.

Generative fill is a user mask with ``generative`` set, run by the fill
model. A model that follows prompts gets the whole selection as one crop with its
surroundings (``fill_region``), so a sound effect is painted as one piece rather than in
tiles; LaMa and AOT go through the tiling like the remove tool. Either way the result is
committed to the Cleanup layer as a user edit.
"""

from __future__ import annotations

import asyncio
import functools
import io
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

import numpy as np
from PIL import Image
from scipy import ndimage

from urakatsu.core.progress import StepPreview, reporting_steps, step_sink
from urakatsu.pipeline.images import ImageCache
from urakatsu.pipeline.scope import Bounds
from urakatsu.providers.base import Provider
from urakatsu.scene import (
    AssetInput,
    AssetMetadata,
    At,
    EntityOrigin,
    Geometry,
    Patch,
    RasterLayer,
    RasterLayerKind,
    Region,
    RegionKinds,
    Snapshot,
    user,
)
from urakatsu.scene.components import is_user

from .base import StageInput, StageProcessor, finish, generation

PRODUCER = "dev.koharu.pipeline.inpainting"
SOURCE = "source"
TEXT_MASK = "text-mask"
COO_MASK = "coo-mask"

TILE_SIZE = 512
TILE_CONTEXT = 128
UNIFORM_BACKGROUND_MIN_PIXELS = 16
FLAT_FILL_EDGE_MARGIN = 3.0
# Context around a generative fill selection, as a share of its longer side (at least TILE_CONTEXT).
FILL_CONTEXT = 0.5

Inference = Callable[[Image.Image, np.ndarray], Image.Image]


@dataclass
class FlatFillRegion:
    bounds: tuple[int, int, int, int]  # left, top, right, bottom (exclusive)
    polygon: np.ndarray  # (N, 2) float32


@dataclass
class _Tile:
    components: list[int]
    core: tuple[int, int, int, int]
    crop: tuple[int, int, int, int]


@dataclass
class _Prepared:
    original: Image.Image
    image: Image.Image
    cleanup_entity: Optional[str]
    cleanup: Optional[np.ndarray]  # RGBA overlay of the committed Cleanup layer
    mask: np.ndarray
    text_mask: np.ndarray
    flat_fill_regions: list[FlatFillRegion] = field(default_factory=list)


class InpaintingStage(StageProcessor):
    def skip(self, input: StageInput, provider: Provider) -> bool:
        if input.inpainting_mask is not None:
            return False
        for entity in input.scene.children(input.page):
            layer = input.scene.component(entity, RasterLayer)
            if layer is not None and layer.kind is RasterLayerKind.CLEANUP and input.scene.asset(entity, SOURCE) is not None:
                return True
        return False

    async def process(self, input: StageInput, provider: Provider) -> Patch:
        prepared = prepare(input)
        if not prepared.mask.any():
            return finish(input.scene.edit())
        mask = input.inpainting_mask
        if mask is not None and mask.generative:
            fill = functools.partial(provider._fill, prompt=mask.prompt or None)
            if getattr(provider, "prompted", False):
                image = await asyncio.to_thread(fill_region, prepared.image, prepared.mask, fill)
            else:
                image = await asyncio.to_thread(inpaint_tiled, prepared.image, prepared.mask, prepared.text_mask, [], fill)
            return commit(input, provider, prepared, image)
        image = await asyncio.to_thread(
            inpaint_tiled,
            prepared.image,
            prepared.mask,
            prepared.text_mask,
            prepared.flat_fill_regions,
            provider._inpaint,
        )
        return commit(input, provider, prepared, image)


def prepare(input: StageInput) -> _Prepared:
    page = input.page
    original = input.images.get(input.scene, page, SOURCE)
    if original is None:
        raise ValueError(f"page {page} has no source image")
    cleanup_entity = next(
        (
            entity
            for entity in input.scene.children(page)
            if (layer := input.scene.component(entity, RasterLayer)) is not None and layer.kind is RasterLayerKind.CLEANUP
        ),
        None,
    )
    cleanup: Optional[np.ndarray] = None
    if cleanup_entity is not None:
        cleanup_image = input.images.get(input.scene, cleanup_entity, SOURCE)
        if cleanup_image is not None:
            cleanup = np.asarray(cleanup_image.convert("RGBA"))
    if cleanup is not None and cleanup.shape[:2] != (original.height, original.width):
        raise ValueError(f"cleanup layer dimensions do not match page {page}")

    if input.inpainting_mask is not None and cleanup is not None:
        composite = Image.alpha_composite(original.convert("RGBA"), Image.fromarray(cleanup, "RGBA"))
        source = composite
    else:
        source = original

    width, height = source.width, source.height
    mask = np.zeros((height, width), dtype=np.uint8)
    text_mask = np.zeros((height, width), dtype=np.uint8)
    if input.inpainting_mask is not None:
        layer = np.asarray(Image.open(io.BytesIO(input.inpainting_mask.png)).convert("L"))
        if layer.shape != mask.shape:
            raise ValueError(f"inpainting mask dimensions do not match page {page}")
        mask = layer.copy()
    else:
        for role in (TEXT_MASK, COO_MASK):
            image = input.images.get(input.scene, page, role)
            if image is None:
                continue
            layer = np.asarray(image.convert("L"))
            if layer.shape != mask.shape:
                raise ValueError(f"{role} dimensions do not match page {page}")
            mask = np.maximum(mask, layer)
            if role == TEXT_MASK:
                text_mask = layer.copy()

    if input.region is not None:
        _clear_outside(mask, text_mask, input.region)

    return _Prepared(
        original=original,
        image=source,
        cleanup_entity=cleanup_entity,
        cleanup=cleanup,
        mask=mask,
        text_mask=text_mask,
        flat_fill_regions=flat_fill_regions(input.scene, page, width, height),
    )


def _clear_outside(mask: np.ndarray, text_mask: np.ndarray, bounds: Bounds) -> None:
    height, width = mask.shape
    xs = np.arange(width, dtype=np.float64)[None, :]
    ys = np.arange(height, dtype=np.float64)[:, None]
    outside = (
        (xs + 1.0 <= bounds.x)
        | (ys + 1.0 <= bounds.y)
        | (xs >= bounds.x + bounds.width)
        | (ys >= bounds.y + bounds.height)
    )
    mask[outside] = 0
    text_mask[outside] = 0


def flat_fill_regions(scene: Snapshot, page: str, width: int, height: int) -> list[FlatFillRegion]:
    regions: list[FlatFillRegion] = []
    for entity in scene.descendants(page):
        region = scene.component(entity, Region)
        if region is None or region.kind != RegionKinds.BUBBLE:
            continue
        geometry = scene.component(entity, Geometry)
        if geometry is None:
            continue
        polygon = np.array([(point.x, point.y) for point in geometry.points], dtype=np.float32).reshape(-1, 2)
        if len(polygon) < 3:
            continue
        left, top = polygon.min(axis=0)
        right, bottom = polygon.max(axis=0)
        bounds = (
            int(np.clip(np.floor(left), 0, width)),
            int(np.clip(np.floor(top), 0, height)),
            int(np.clip(np.ceil(right), 0, width)),
            int(np.clip(np.ceil(bottom), 0, height)),
        )
        if bounds[2] > bounds[0] and bounds[3] > bounds[1]:
            regions.append(FlatFillRegion(bounds=bounds, polygon=polygon))
    return regions


def commit(input: StageInput, provider: Provider, prepared: _Prepared, image: Image.Image) -> Patch:
    page = input.page
    manual = input.inpainting_mask is not None
    edit = input.scene.edit() if manual else input.scene.edit_as(generation(PRODUCER, provider.id))
    edit.observe_assets(page)
    if prepared.cleanup_entity is not None:
        edit.observe(prepared.cleanup_entity, RasterLayer)
        edit.observe_assets(prepared.cleanup_entity)

    original = prepared.original
    mask = prepared.mask
    generated = np.asarray(image.convert("RGB"))
    if generated.shape[:2] != (original.height, original.width) or mask.shape != (original.height, original.width):
        raise ValueError(f"inpainted image dimensions do not match page {page}")

    if manual and prepared.cleanup is not None:
        overlay = prepared.cleanup.copy()
    else:
        overlay = np.zeros((original.height, original.width, 4), dtype=np.uint8)
    keep = mask >= 127
    # Keep the complete cleanup mask opaque: transparent texels next to edited pixels
    # let linear canvas sampling reveal the artwork below.
    overlay[keep, :3] = generated[keep]
    overlay[keep, 3] = 255

    entity = prepared.cleanup_entity
    if entity is not None:
        if manual:
            layer = input.scene.component(entity, RasterLayer)
            if layer is None:
                raise ValueError("cleanup entity has no raster layer component")
            entity_origin = input.scene.component(entity, EntityOrigin)
            if not is_user(layer.origin) or (entity_origin is not None and not is_user(entity_origin.origin)):
                layer.origin = user()
                edit.set(entity, layer)
    else:
        entity = edit.add_entity(page, At.START)
        edit.set(entity, RasterLayer(origin=user(), name="Cleanup", kind=RasterLayerKind.CLEANUP))

    buffer = io.BytesIO()
    Image.fromarray(overlay, "RGBA").save(buffer, format="PNG")
    edit.set_asset(
        entity,
        SOURCE,
        AssetInput(
            data=buffer.getvalue(),
            media_type="image/png",
            metadata=AssetMetadata(width=overlay.shape[1], height=overlay.shape[0], attributes={}),
        ),
    )
    return finish(edit)


# ---- inpaint_tiled ---------------------------------------------------------------------------

# BallonsTranslator avoids model inference for a text block when the non-text pixels inside its
# balloon are nearly uniform, and otherwise sends an enlarged block crop to the inpainter:
# https://github.com/dmMaze/BallonsTranslator/blob/4bcc635c19f6c63a902872cf77b3d554e14ed1b7/ballontranslator/modules/inpaint/base.py#L168-L200
# The detected bubble polygons serve as those blocks. Uniform bubbles are filled first;
# only the remaining mask is split into bounded model crops.
def inpaint_tiled(
    image: Image.Image,
    mask: np.ndarray,
    text_mask: np.ndarray,
    flat_regions: list[FlatFillRegion],
    inference: Inference,
) -> Image.Image:
    if not (image.size == (mask.shape[1], mask.shape[0]) and mask.shape == text_mask.shape):
        raise ValueError("image and mask dimensions differ")
    output = np.asarray(image.convert("RGB")).copy()
    pending = mask.copy()
    fill_uniform_regions(output, pending, text_mask, flat_regions)
    labels, tiles = inpaint_tiles(pending)
    for tile in tiles:
        left, top, right, bottom = tile.crop
        crop_width, crop_height = right - left, bottom - top
        crop_image = Image.fromarray(output[top:bottom, left:right].copy(), "RGB")
        crop_mask = crop_tile_mask(pending, tile)

        with placed_previews(crop_image, crop_mask, (left, top)):
            generated = inference(crop_image, crop_mask)
        if generated.size != (crop_width, crop_height):
            generated = generated.resize((crop_width, crop_height), Image.Resampling.LANCZOS)
        composite_generated(output, labels, tile, np.asarray(generated.convert("RGB")))
    return Image.fromarray(output, "RGB")


def fill_region(image: Image.Image, mask: np.ndarray, fill: Inference) -> Image.Image:
    """Generative fill: the masked area in one crop, with context around it on every side."""
    if image.size != (mask.shape[1], mask.shape[0]):
        raise ValueError("image and mask dimensions differ")
    output = np.asarray(image.convert("RGB")).copy()
    ys, xs = np.nonzero(mask >= 127)
    if xs.size == 0:
        return Image.fromarray(output, "RGB")
    height, width = mask.shape
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    margin = max(TILE_CONTEXT, round(max(right - left, bottom - top) * FILL_CONTEXT))
    left, top = max(left - margin, 0), max(top - margin, 0)
    right, bottom = min(right + margin, width), min(bottom + margin, height)
    crop_mask = np.where(mask[top:bottom, left:right] >= 127, 255, 0).astype(np.uint8)
    crop = Image.fromarray(output[top:bottom, left:right].copy(), "RGB")
    with placed_previews(crop, crop_mask, (left, top)):
        generated = fill(crop, crop_mask)
    if generated.size != (right - left, bottom - top):
        generated = generated.resize((right - left, bottom - top), Image.Resampling.LANCZOS)
    keep = crop_mask >= 127
    output[top:bottom, left:right][keep] = np.asarray(generated.convert("RGB"))[keep]
    return Image.fromarray(output, "RGB")


# Longest side of a step preview: the browser stretches it over the crop, so full size buys nothing.
PREVIEW_SIDE = 768


@contextmanager
def placed_previews(crop: Image.Image, mask: np.ndarray, origin: tuple[int, int]) -> Iterator[None]:
    """Reports each sampling preview as it would land on the page: inside ``mask``, at the crop."""
    outer = step_sink()
    if outer is None:
        yield
        return
    scale = min(1.0, PREVIEW_SIDE / max(crop.width, crop.height))
    size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
    base = crop.convert("RGB").resize(size, Image.Resampling.BILINEAR)
    keep = Image.fromarray(mask.astype(np.uint8), "L").resize(size, Image.Resampling.BILINEAR)
    box = (origin[0], origin[1], crop.width, crop.height)

    def place(preview: StepPreview) -> None:
        if preview.image is None:
            outer(preview)
            return
        frame = preview.image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
        outer(preview.placed(Image.composite(frame, base, keep), box))

    with reporting_steps(place):
        yield


def fill_uniform_regions(output: np.ndarray, pending: np.ndarray, text_mask: np.ndarray, regions: list[FlatFillRegion]) -> None:
    for region in regions:
        left, top, right, bottom = region.bounds
        pending_view = pending[top:bottom, left:right]
        text_view = text_mask[top:bottom, left:right]
        ys, xs = np.mgrid[top:bottom, left:right]
        inside = point_in_polygon(xs.astype(np.float32) + np.float32(0.5), ys.astype(np.float32) + np.float32(0.5), region.polygon)
        targets = inside & (pending_view >= 127) & (text_view >= 127)
        if not targets.any():
            continue
        color = uniform_region_color(output, pending, region)
        if color is None:
            continue
        output[top:bottom, left:right][targets] = color
        pending_view[targets] = 0


def inpaint_tiles(mask: np.ndarray) -> tuple[np.ndarray, list[_Tile]]:
    labels, components = mask_components(mask)
    height, width = mask.shape
    bounded: list[_Tile] = []
    split: list[_Tile] = []

    for label, bounds in components:
        left, top, right, bottom = bounds
        if right - left <= TILE_SIZE and bottom - top <= TILE_SIZE:
            best: Optional[tuple[tuple[int, int], int, tuple[int, int, int, int]]] = None
            for index, tile in enumerate(bounded):
                union = union_bounds(tile.core, bounds)
                if union[2] - union[0] <= TILE_SIZE and union[3] - union[1] <= TILE_SIZE:
                    key = (bounds_area(union) - bounds_area(tile.core), index)
                    if best is None or key < best[0]:
                        best = (key, index, union)
            if best is not None:
                _, index, union = best
                tile = bounded[index]
                tile.components.append(label)
                tile.core = union
                tile.crop = expand_bounds(union, width, height)
            else:
                bounded.append(_Tile([label], bounds, expand_bounds(bounds, width, height)))
            continue

        core_top = top
        while core_top < bottom:
            core_bottom = min(core_top + TILE_SIZE, bottom)
            core_left = left
            while core_left < right:
                core_right = min(core_left + TILE_SIZE, right)
                core = (core_left, core_top, core_right, core_bottom)
                owned = component_bounds_in(labels, label, core)
                if owned is not None:
                    split.append(_Tile([label], core, expand_bounds(owned, width, height)))
                core_left = core_right
            core_top = core_bottom

    bounded.extend(split)
    bounded.sort(key=lambda tile: (tile.core[1], tile.core[0], tile.core[3], tile.core[2]))
    return labels, bounded


def mask_components(mask: np.ndarray) -> tuple[np.ndarray, list[tuple[int, tuple[int, int, int, int]]]]:
    binary = (mask >= 127).astype(np.uint8)
    labels, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    components = []
    for label, slices in enumerate(ndimage.find_objects(labels, max_label=count), start=1):
        if slices is None:
            continue
        ys, xs = slices
        components.append((label, (xs.start, ys.start, xs.stop, ys.stop)))
    return labels, components


def union_bounds(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (min(left[0], right[0]), min(left[1], right[1]), max(left[2], right[2]), max(left[3], right[3]))


def bounds_area(bounds: tuple[int, int, int, int]) -> int:
    return (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])


def expand_bounds(bounds: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = bounds
    return (
        max(left - TILE_CONTEXT, 0),
        max(top - TILE_CONTEXT, 0),
        min(right + TILE_CONTEXT, width),
        min(bottom + TILE_CONTEXT, height),
    )


def component_bounds_in(labels: np.ndarray, label: int, region: tuple[int, int, int, int]) -> Optional[tuple[int, int, int, int]]:
    left, top, right, bottom = region
    ys, xs = np.nonzero(labels[top:bottom, left:right] == label)
    if xs.size == 0:
        return None
    return (left + int(xs.min()), top + int(ys.min()), left + int(xs.max()) + 1, top + int(ys.max()) + 1)


def crop_tile_mask(mask: np.ndarray, tile: _Tile) -> np.ndarray:
    left, top, right, bottom = tile.crop
    return np.where(mask[top:bottom, left:right] >= 127, 255, 0).astype(np.uint8)


def uniform_region_color(image: np.ndarray, mask: np.ndarray, region: FlatFillRegion) -> Optional[np.ndarray]:
    left, top, right, bottom = region.bounds
    ys, xs = np.mgrid[top:bottom, left:right]
    px = xs.astype(np.float32) + np.float32(0.5)
    py = ys.astype(np.float32) + np.float32(0.5)
    inside = point_in_polygon(px, py, region.polygon)
    edge_distance = polygon_edge_distance_squared(px, py, region.polygon)
    valid = (
        (mask[top:bottom, left:right] < 127)
        & inside
        & (edge_distance >= np.float32(FLAT_FILL_EDGE_MARGIN * FLAT_FILL_EDGE_MARGIN))
    )
    values = image[top:bottom, left:right][valid]
    if values.shape[0] < UNIFORM_BACKGROUND_MIN_PIXELS:
        return None

    medians = [_median(values[:, channel]) for channel in range(3)]
    deviations = [
        float(np.sqrt(np.mean((values[:, channel].astype(np.float64) - medians[channel]) ** 2))) for channel in range(3)
    ]
    mean_deviation = sum(deviations) / len(deviations)
    channel_spread = float(np.sqrt(sum((d - mean_deviation) ** 2 for d in deviations) / len(deviations)))
    threshold = 7.0 if channel_spread > 1.0 else 10.0
    if max(deviations) < threshold:
        return np.array(medians, dtype=np.uint8)
    return None


def _median(values: np.ndarray) -> int:
    ordered = np.sort(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (int(ordered[middle - 1]) + int(ordered[middle])) // 2
    return int(ordered[middle])


def point_in_polygon(px: np.ndarray, py: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    inside = np.zeros(px.shape, dtype=bool)
    previous = polygon[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        for current in polygon:
            crosses = (current[1] > py) != (previous[1] > py)
            intersection_x = (previous[0] - current[0]) * (py - current[1]) / (previous[1] - current[1]) + current[0]
            inside ^= crosses & (px < intersection_x)
            previous = current
    return inside


def polygon_edge_distance_squared(px: np.ndarray, py: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    minimum = np.full(px.shape, np.inf, dtype=np.float32)
    start = polygon[-1]
    for end in polygon:
        segment_x = end[0] - start[0]
        segment_y = end[1] - start[1]
        length_squared = segment_x * segment_x + segment_y * segment_y
        if length_squared > 0:
            projection = ((px - start[0]) * segment_x + (py - start[1]) * segment_y) / length_squared
        else:
            projection = np.zeros(px.shape, dtype=np.float32)
        projection = np.clip(projection, 0.0, 1.0)
        closest_x = start[0] + segment_x * projection
        closest_y = start[1] + segment_y * projection
        distance = (px - closest_x) ** 2 + (py - closest_y) ** 2
        minimum = np.minimum(minimum, distance)
        start = end
    return minimum


def composite_generated(output: np.ndarray, labels: np.ndarray, tile: _Tile, generated: np.ndarray) -> None:
    left, top, _, _ = tile.crop
    core_left, core_top, core_right, core_bottom = tile.core
    member = np.isin(labels[core_top:core_bottom, core_left:core_right], tile.components)
    source = generated[core_top - top : core_bottom - top, core_left - left : core_right - left]
    output[core_top:core_bottom, core_left:core_right][member] = source[member]
