"""Inpainting stage: port of stages/inpainting.rs.

The stage builds the removal mask (detection mask assets, or the user-painted
mask), fills uniform bubble backgrounds without inference, runs the provider on
bounded tiles (``inpaint_tiled``), and commits the result as the page's Cleanup
raster layer. The provider is any ``InpaintingProvider``: LaMa and AOT run on the
torch runtime, FLUX.2 Klein and RORem on the diffusion runtime; all four go through
the same tiling here.

Urakatsu also flat-fills text that is in no bubble when the background around it is uniform
(captions: white lettering on a black box, as webtoons and manhua use). Koharu sends those to
the model, and a generative model with a crop of almost nothing but lettering on one colour
paints a scene into it.

Urakatsu also flat-fills a bubble that is one colour but for a little else in it (white blocks
over a black bubble's rim, a stroke of art across it), which Koharu's strict check sends to the
model (``dominant_region_color``).

Urakatsu also fills text on a smooth gradient without the model, where the flat fill finds no one
colour (a chapter title on a banner that fades from white to black). The same model paints a pale
blob over such a gradient. ``fill_smooth_background`` carries the pixels around the text across it
when they show nothing but the gradient; anything else there, art or screentone, goes to the model.

Generative fill is a user mask with ``generative`` set, run by the fill
model. A model that follows prompts gets the whole selection as one crop with its
surroundings (``fill_region``), so a sound effect is painted as one piece rather than in
tiles; a model that edits the crop it is given (``crop_edits``) gets only a rim of
surroundings, since its instruction applies to everything in view. LaMa and AOT go through
the tiling like the remove tool. Either way the result is
committed to the Cleanup layer as a user edit.

Difference from Koharu: inside a generative fill's selection the model sees the original page,
not the Cleanup layer over it (``prepare``); outside it, the Cleanup layer stays as context. A
retry then starts from the page as drawn instead of from the last attempt (half-drawn lettering).
Measured on one sound effect with FLUX.2 Klein (8 runs each), this made no clear difference: the
word came out whole in about one run in four from either input. A fill cannot build on the one
before it; the remove tool still does.
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
from scipy import ndimage, sparse
from scipy.sparse import linalg as sparse_linalg

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
    Relations,
    Snapshot,
    user,
)
from urakatsu.scene.components import is_user

from .base import StageInput, StageProcessor, finish, generation
from .watermarks import page_image, watermark_layer

PRODUCER = "dev.koharu.pipeline.inpainting"
SOURCE = "source"
TEXT_MASK = "text-mask"
COO_MASK = "coo-mask"

TILE_SIZE = 512
TILE_CONTEXT = 128
UNIFORM_BACKGROUND_MIN_PIXELS = 16
FLAT_FILL_EDGE_MARGIN = 3.0
# Pixels around a free text region's box sampled for its background, beyond the edge margin.
TEXT_FLAT_FILL_PAD = 8
# fill_smooth_background: pixels around the text that must show nothing but the background.
SMOOTH_RIM = 3
# Below these, neighbouring rim pixels differ by a gradient's slope and JPEG noise; a line or
# screentone exceeds the first, a faint tone the second.
SMOOTH_MAX_STEP = 32
SMOOTH_MAX_MEAN_STEP = 4.0
# Share of the text's outline that must have known background beside it.
SMOOTH_MIN_ANCHORED = 0.5
# Mask beyond the text's box: at most this share of what is filled, or it is not a stray spill. It
# ran onto art the fill must not erase, so the rim keeps this far from it.
SMOOTH_MAX_OVERFLOW = 0.1
SMOOTH_OVERFLOW_CLEARANCE = 8
# dominant_region_color: a bubble is one colour when this share of it (and DOMINANT_RIM_SHARE of
# what borders its text) lies within DOMINANT_TOLERANCE of the median on every channel.
DOMINANT_TOLERANCE = 24
DOMINANT_SHARE = 0.9
DOMINANT_RIM_SHARE = 0.95
# Context around a generative fill selection, as a share of its longer side (at least TILE_CONTEXT).
FILL_CONTEXT = 0.5
# The same for a model that edits the whole crop it sees from an instruction (Qwen Image, FLUX.2
# Klein): with a page's worth of context, "remove the text" goes to a speech balloon beside the
# selected sound effect, outside the mask, and the selection comes back unchanged.
EDIT_CONTEXT = 0.1
EDIT_CONTEXT_MIN = 48

Inference = Callable[[Image.Image, np.ndarray], Image.Image]

SQUARE = np.ones((3, 3), dtype=bool)
PixelBounds = tuple[int, int, int, int]  # left, top, right, bottom (exclusive)


@dataclass
class FlatFillRegion:
    bounds: PixelBounds
    polygon: np.ndarray  # (N, 2) float32
    texts: list[PixelBounds] = field(default_factory=list)  # the text in it, widened by TEXT_FLAT_FILL_PAD


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
        cleanup = self._cleanup(input)
        # A forced run (Redo) paints the page again, but never over a cleanup the user painted.
        return cleanup is not None and (not input.force or input.scene.user_authored(cleanup))

    def done(self, input: StageInput) -> bool:
        return input.inpainting_mask is None and self._cleanup(input) is not None

    @staticmethod
    def _cleanup(input: StageInput) -> Optional[str]:
        for entity in input.scene.children(input.page):
            layer = input.scene.component(entity, RasterLayer)
            if layer is not None and layer.kind is RasterLayerKind.CLEANUP and input.scene.asset(entity, SOURCE) is not None:
                return entity
        return None

    async def process(self, input: StageInput, provider: Provider) -> Patch:
        prepared = prepare(input)
        if not prepared.mask.any():
            return finish(input.scene.edit())
        mask = input.inpainting_mask
        if mask is not None and mask.generative:
            fill = functools.partial(provider._fill, prompt=mask.prompt or None)
            if getattr(provider, "prompted", False):
                context = (EDIT_CONTEXT, EDIT_CONTEXT_MIN) if getattr(provider, "crop_edits", False) else (FILL_CONTEXT, TILE_CONTEXT)
                image = await asyncio.to_thread(fill_region, prepared.image, prepared.mask, fill, *context)
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
    original = page_image(input)
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

    if input.inpainting_mask is not None and input.inpainting_mask.generative and cleanup is not None:
        # Under the selection the fill model sees the page as drawn, not an earlier attempt.
        selected = mask >= 127
        pixels = np.asarray(source.convert("RGBA")).copy()
        pixels[selected] = np.asarray(original.convert("RGBA"))[selected]
        source = Image.fromarray(pixels, "RGBA")

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


def _pixel_bounds(polygon: np.ndarray, width: int, height: int) -> PixelBounds:
    left, top = polygon.min(axis=0)
    right, bottom = polygon.max(axis=0)
    return (
        int(np.clip(np.floor(left), 0, width)),
        int(np.clip(np.floor(top), 0, height)),
        int(np.clip(np.ceil(right), 0, width)),
        int(np.clip(np.ceil(bottom), 0, height)),
    )


def flat_fill_regions(scene: Snapshot, page: str, width: int, height: int) -> list[FlatFillRegion]:
    """Bubbles, then text in no bubble (its box widened by TEXT_FLAT_FILL_PAD): each is filled
    without the model when the unmasked pixels in it share one colour (``fill_uniform_regions``)
    or show a smooth gradient (``fill_smooth_background``)."""
    bubbles: list[tuple[str, np.ndarray]] = []
    free_text: list[np.ndarray] = []
    texts: dict[str, list[PixelBounds]] = {}
    for entity in scene.descendants(page):
        region = scene.component(entity, Region)
        if region is None or region.kind not in (RegionKinds.BUBBLE, RegionKinds.TEXT):
            continue
        geometry = scene.component(entity, Geometry)
        if geometry is None:
            continue
        polygon = np.array([(point.x, point.y) for point in geometry.points], dtype=np.float32).reshape(-1, 2)
        if len(polygon) < 3:
            continue
        if region.kind == RegionKinds.TEXT:
            (left, top), (right, bottom) = polygon.min(axis=0) - TEXT_FLAT_FILL_PAD, polygon.max(axis=0) + TEXT_FLAT_FILL_PAD
            box = np.array([(left, top), (right, top), (right, bottom), (left, bottom)], dtype=np.float32)
            containers = scene.relations_from(entity, Relations.INSIDE)
            if not containers:
                free_text.append(box)
            for relation in containers:
                texts.setdefault(relation.target, []).append(_pixel_bounds(box, width, height))
            continue
        bubbles.append((entity, polygon))
    regions: list[FlatFillRegion] = []
    for entity, polygon in bubbles:
        bounds = _pixel_bounds(polygon, width, height)
        if bounds[2] > bounds[0] and bounds[3] > bounds[1]:
            regions.append(FlatFillRegion(bounds=bounds, polygon=polygon, texts=texts.get(entity, [])))
    for polygon in free_text:
        bounds = _pixel_bounds(polygon, width, height)
        if bounds[2] > bounds[0] and bounds[3] > bounds[1]:
            regions.append(FlatFillRegion(bounds=bounds, polygon=polygon, texts=[bounds]))
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
        # Over the Watermarks layer (not in Koharu), which sits right over the source.
        below = watermark_layer(input.scene, page)
        entity = edit.add_entity(page, At.after(below) if below is not None else At.START)
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


def fill_region(image: Image.Image, mask: np.ndarray, fill: Inference, context: float = FILL_CONTEXT, minimum: int = TILE_CONTEXT) -> Image.Image:
    """Generative fill: the masked area in one crop, with ``context`` of its longer side (at least
    ``minimum`` pixels) around it on every side."""
    if image.size != (mask.shape[1], mask.shape[0]):
        raise ValueError("image and mask dimensions differ")
    output = np.asarray(image.convert("RGB")).copy()
    ys, xs = np.nonzero(mask >= 127)
    if xs.size == 0:
        return Image.fromarray(output, "RGB")
    height, width = mask.shape
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    margin = max(minimum, round(max(right - left, bottom - top) * context))
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
    text_area = np.zeros(pending.shape, dtype=bool)
    for region in regions:
        for left, top, right, bottom in region.texts:
            text_area[top:bottom, left:right] = True
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
            before = np.count_nonzero(pending_view >= 127)
            fill_smooth_background(output, pending, targets, region, text_area)
            if np.count_nonzero(pending_view >= 127) != before:
                continue  # carried across a gradient; what it left past the text is the model's
            color = dominant_region_color(output, pending, region, targets)
        if color is None:
            continue
        output[top:bottom, left:right][targets] = color
        pending_view[targets] = 0


def fill_smooth_background(output: np.ndarray, pending: np.ndarray, targets: np.ndarray, region: FlatFillRegion, text_area: np.ndarray) -> None:
    """Text on a smooth gradient: the pixels around it are carried across it (``harmonic_fill``).

    ``targets`` (the region's window) is what the flat fill would have painted. Only what lies in
    the region's text boxes is filled; the page stays as it is where the same patch of mask spills
    past them onto art. The rest of the mask, and a region with anything but a gradient around its
    text, is left to the model.
    """
    left, top, right, bottom = region.bounds
    boxes = np.zeros(pending.shape, dtype=bool)
    for box_left, box_top, box_right, box_bottom in region.texts:
        boxes[box_top:box_bottom, box_left:box_right] = True
    targets = targets & boxes[top:bottom, left:right]
    if not targets.any():
        return
    mask = pending >= 127
    labels, _ = ndimage.label(mask, structure=SQUARE)
    filled = np.zeros(pending.shape, dtype=bool)
    filled[top:bottom, left:right] = targets
    overflow = np.isin(labels, np.unique(labels[filled])) & ~filled & ~text_area
    if np.count_nonzero(overflow) > np.count_nonzero(targets) * SMOOTH_MAX_OVERFLOW:
        return

    view = output[top:bottom, left:right]
    rim = smooth_rim(view, mask[top:bottom, left:right], targets, overflow[top:bottom, left:right], region)
    if rim is None:
        return
    solved, values = harmonic_fill(view, targets, rim)
    view[solved] = values
    pending[top:bottom, left:right][solved] = 0
    pending[overflow] = 0


def smooth_rim(image: np.ndarray, mask: np.ndarray, targets: np.ndarray, overflow: np.ndarray, region: FlatFillRegion) -> Optional[np.ndarray]:
    """The unmasked pixels around ``targets``, or None unless they show only a smooth background."""
    left, top, right, bottom = region.bounds
    ys, xs = np.mgrid[top:bottom, left:right]
    px = xs.astype(np.float32) + np.float32(0.5)
    py = ys.astype(np.float32) + np.float32(0.5)
    rim = (
        ndimage.binary_dilation(targets, SQUARE, iterations=SMOOTH_RIM)
        & ~mask
        & point_in_polygon(px, py, region.polygon)
        & (polygon_edge_distance_squared(px, py, region.polygon) >= np.float32(FLAT_FILL_EDGE_MARGIN * FLAT_FILL_EDGE_MARGIN))
    )
    if overflow.any():
        rim &= ~ndimage.binary_dilation(overflow, SQUARE, iterations=SMOOTH_OVERFLOW_CLEARANCE)
    outline = targets & ndimage.binary_dilation(~targets)
    if np.count_nonzero(outline & ndimage.binary_dilation(rim)) < np.count_nonzero(outline) * SMOOTH_MIN_ANCHORED:
        return None
    values = image.astype(np.int16)
    steps = np.concatenate(
        [
            np.abs(values[:, 1:] - values[:, :-1])[rim[:, 1:] & rim[:, :-1]],
            np.abs(values[1:] - values[:-1])[rim[1:] & rim[:-1]],
        ]
    )
    if steps.size < UNIFORM_BACKGROUND_MIN_PIXELS or steps.max() >= SMOOTH_MAX_STEP or steps.mean() >= SMOOTH_MAX_MEAN_STEP:
        return None
    return rim


def harmonic_fill(image: np.ndarray, targets: np.ndarray, known: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solves Laplace's equation over ``targets`` with the pixels of ``known`` held fixed; a target
    pixel with neither beside it has no neighbour there. Returns the pixels solved, which are the
    targets connected to ``known``, and their colours."""
    labels, _ = ndimage.label(targets)
    anchored = np.unique(labels[targets & ndimage.binary_dilation(known)])
    solved = np.isin(labels, anchored[anchored > 0])
    ys, xs = np.nonzero(solved)
    index = np.full(targets.shape, -1, dtype=np.int64)
    index[ys, xs] = np.arange(len(ys))
    height, width = targets.shape
    degree = np.zeros(len(ys))
    sides = np.zeros((len(ys), 3))
    rows, columns = [], []
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        beside = (ys + dy >= 0) & (ys + dy < height) & (xs + dx >= 0) & (xs + dx < width)
        here, near_y, near_x = np.nonzero(beside)[0], ys[beside] + dy, xs[beside] + dx
        open_side = index[near_y, near_x] >= 0
        rows.append(here[open_side])
        columns.append(index[near_y, near_x][open_side])
        fixed = known[near_y, near_x]
        np.add.at(sides, here[fixed], image[near_y[fixed], near_x[fixed]].astype(np.float64))
        np.add.at(degree, here[open_side | fixed], 1.0)
    row, column = np.concatenate(rows), np.concatenate(columns)
    laplacian = sparse.coo_matrix((-np.ones(len(row)), (row, column)), shape=(len(ys), len(ys))) + sparse.diags(degree)
    values = sparse_linalg.splu(laplacian.tocsc()).solve(sides)
    return solved, np.clip(np.rint(values), 0, 255).astype(np.uint8)


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
                    # The core is the component's part of the cell, not the whole cell: the crop is
                    # built around that part, and a cell the component only crosses in a thin band
                    # would otherwise be taller or wider than its crop when composited.
                    split.append(_Tile([label], owned, expand_bounds(owned, width, height)))
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


def dominant_region_color(image: np.ndarray, mask: np.ndarray, region: FlatFillRegion, targets: np.ndarray) -> Optional[np.ndarray]:
    """One colour that nearly all of a bubble shows, and all but a little of what borders the text.

    ``uniform_region_color`` rejects a bubble when anything else is in it: a black caption bubble
    with white blocks drawn over its rim, or a stroke of art crossing it. The model then gets a
    crop of lettering on one colour and paints a scene into it. Here the odd pixels are set aside:
    at least DOMINANT_SHARE of the bubble, and DOMINANT_RIM_SHARE of the pixels around ``targets``
    (the bubble's window), must lie within DOMINANT_TOLERANCE of the median, and those must be as
    even as the strict check asks.
    """
    left, top, right, bottom = region.bounds
    ys, xs = np.mgrid[top:bottom, left:right]
    px = xs.astype(np.float32) + np.float32(0.5)
    py = ys.astype(np.float32) + np.float32(0.5)
    valid = (
        (mask[top:bottom, left:right] < 127)
        & point_in_polygon(px, py, region.polygon)
        & (polygon_edge_distance_squared(px, py, region.polygon) >= np.float32(FLAT_FILL_EDGE_MARGIN * FLAT_FILL_EDGE_MARGIN))
    )
    view = image[top:bottom, left:right].astype(np.int16)
    values = view[valid]
    if values.shape[0] < UNIFORM_BACKGROUND_MIN_PIXELS:
        return None
    median = np.array([_median(values[:, channel]) for channel in range(3)], dtype=np.int16)
    inliers = (np.abs(values - median) <= DOMINANT_TOLERANCE).all(axis=1)
    if inliers.mean() < DOMINANT_SHARE:
        return None
    rim = ndimage.binary_dilation(targets, SQUARE, iterations=SMOOTH_RIM) & valid
    rim_values = view[rim]
    if rim_values.shape[0] < UNIFORM_BACKGROUND_MIN_PIXELS:
        return None
    if (np.abs(rim_values - median) <= DOMINANT_TOLERANCE).all(axis=1).mean() < DOMINANT_RIM_SHARE:
        return None
    deviations = np.sqrt(np.mean((values[inliers] - median).astype(np.float64) ** 2, axis=0))
    if deviations.max() >= 10.0:
        return None
    return median.astype(np.uint8)


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
