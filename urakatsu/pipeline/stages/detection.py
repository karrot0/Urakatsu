"""Detection stage: port of ``stages/detection.rs``.

Each function mirrors the Rust function named in its comment. The imageproc primitives the Rust
uses (distance transform, morphology, Suzuki-Abe contour tracing, Ramer-Douglas-Peucker
simplification, polygon area and length) are reproduced with the same integer and f64/f32
semantics, including tie-breaking of ``min_by_key`` (first) and ``max_by_key`` (last). OpenCV is
used only for exact connected-component labelling and the exact Euclidean distance transform.

Difference from Koharu: ``infer_outline`` only finds a stroke whose colour differs from the page
behind the text, so the common manga case of dark text with a white stroke on white paper, where
the stroke shows only where the text crosses art or screentone, came back as unstroked text.
``infer_background_halo`` (no Rust counterpart) runs when no outline is found and recognises that
stroke by its rim: the art around the glyphs stops at one constant distance from them.

Difference from Koharu: ``measured_stroke_width`` takes the median rim distance in Rust. When the
stroke colour joins larger areas of the same colour in the window (white clothes, a white text
box), their far edges outnumber the true rim and pull the median to two or three times the real
width. The width is taken at the peak of the rim distances instead, and a rim with no clear peak
(a box behind the text rather than a stroke traced around it) measures no stroke. ``infer_outline``
also skips a fill that lies mostly outside the text's mask: when a neighbouring text reaches into
the sampling window, the counters of its glyphs otherwise read as a fill "outlined" by the glyphs.

Difference from Koharu: without an outline, Rust picks the ink as the cluster most unlike the
window border's colour, counting a cluster's pixels outside the mask when it has none inside. Text
on a coloured band that the window cuts across then took the colour of the page beside the band,
or of a neighbouring text. ``infer_text_paint`` judges ink against what lines the inside of the
mask's edge when the border colour barely reaches into the mask (``local_background_index``), and
passes over clusters with no pixels in the mask while another cluster has some.
"""

from __future__ import annotations

import io
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from urakatsu.core.errors import ProcessingError
from urakatsu.core.results import LayoutDetection, LayoutDetections, LayoutMask
from urakatsu.pipeline.scope import Bounds, geometry_extents
from urakatsu.scene import At, AssetInput, AssetMetadata, Edit, Patch, RemovePolicy, Snapshot
from urakatsu.scene import components as c

from .base import StageInput, StageProcessor, finish, generation
from .sound_effects import LABEL as SOUND_EFFECT_LABEL, group_sound_effects, write_sound_effect_mask, write_sound_effects
from .watermarks import page_image

MODEL_ID = "mayocream/koharu-layout-rfdetr-seg-2xl-1152"
MODEL_NAME = "koharu-layout-rfdetr-seg-2xl"
PRODUCER = "dev.koharu.pipeline.detection"
ANGLE_SNAP_DEGREES = 3.0
ANGLE_SEARCH_HALF_STEPS = 90
ANGLE_SEARCH_STEP_DEGREES = 0.5
TYPOGRAPHY_SAMPLE_MARGIN = np.float32(2.0)
TYPOGRAPHY_BACKGROUND_MARGIN = 2  # ceil(TYPOGRAPHY_SAMPLE_MARGIN)
COLOR_SNAP_DARK_LUMINANCE = 64 * 256
COLOR_SNAP_LIGHT_LUMINANCE = 191 * 256
COLOR_CLUSTER_MIN_DISTANCE_SQUARED = 32 * 32
COLOR_CLUSTER_COUNT = 4
MIN_EXTREME_COLOR_PIXELS = 4
MIN_MEASURED_STROKE_WIDTH = 2
# measured_stroke_width: share of the nearby rim (out to twice the peak distance) that must lie
# within one pixel of the peak for the band to count as a stroke.
OUTLINE_RIM_CONCENTRATION = 0.35
# local_background_index: below this share of the mask, the window border's colour is not what the
# text sits on.
LOCAL_BACKGROUND_MIN_SHARE = 0.3
# infer_background_halo: rings around the glyphs are searched out to this distance; inside the
# stroke almost no ring pixel contrasts with the halo colour, and at its rim a clear share does.
HALO_SEARCH_DISTANCE = 24
HALO_QUIET_FRACTION = 0.06
HALO_RIM_FRACTION = 0.15
# How much that share must rise across the rim (two rings inside it to one ring past it): art that
# merely gets denser with distance, as other lettering around white text on a black panel does,
# drifts past HALO_RIM_FRACTION without a rim.
HALO_RIM_STEP = 0.15
HALO_MIN_RIM_PIXELS = 16
HALO_RIM_CONCENTRATION = 0.5
# A wider "stroke" than this many glyph depths (distance from glyph centre line to its edge) is
# plain paper between unstroked text and nearby art.
HALO_MAX_WIDTH_PER_GLYPH_DEPTH = 5
DIALOGUE_MASK_CONTAINMENT_THRESHOLD = np.float32(0.9)
NON_MAXIMUM_SUPPRESSION_THRESHOLD = np.float32(0.5)
F32_ZERO = np.float32(0.0)
F32_ONE = np.float32(1.0)
DBL_EPSILON = 2.220446049250313e-16
UNKNOWN_REGION_KIND = "dev.koharu.region.unknown"


class DetectionStage(StageProcessor):
    """``detection::Processor``: skip rule, provider call, patch construction."""

    model = MODEL_NAME

    def skip(self, input: StageInput, provider=None) -> bool:
        # Processor::skip: a page that already has a text region is not re-detected, unless the run
        # is forced (Redo, an Urakatsu addition).
        return not input.force and self.done(input)

    def done(self, input: StageInput) -> bool:
        # Urakatsu counts only regions that still have their text: once the user deletes every text
        # block, the page is detected again (a page that keeps some keeps its deletions).
        for entity in input.scene.descendants(input.page):
            if (
                input.contains_entity(entity)
                and _region_kind_of(input.scene, entity) == c.RegionKinds.TEXT
                and input.scene.relations_to(entity, c.Relations.RECOGNIZED_FROM)
            ):
                return True
        return False

    async def process(self, input: StageInput, provider) -> Patch:
        page = input.page
        image = page_image(input)
        if image is None:
            raise ProcessingError(f"page {page} has no source image")
        output: LayoutDetections = await provider.detect(image)
        provider_id = getattr(provider, "id", MODEL_NAME)
        model = MODEL_ID if provider_id == MODEL_NAME else provider_id
        return build_patch(input, image, output, generation(PRODUCER, model))


# ---- scene helpers -----------------------------------------------------------------------------


def _region_kind_of(snapshot: Snapshot, entity: str) -> Optional[str]:
    region = snapshot.component(entity, c.Region)
    return region.kind if region is not None else None


def build_patch(input: StageInput, image: Image.Image, output: LayoutDetections, gen: c.Generation) -> Patch:
    """``build_patch``: observe the page, replace previous detection regions, write the page."""
    page = input.page
    edit = input.scene.edit_as(gen)
    edit.observe_subtree(page)
    kept = remove_previous_regions(input, edit, gen)
    write_page(input, edit, page, image, output, gen, kept)
    return finish(edit)


def _owned(scene: Snapshot, entity: str, gen: c.Generation) -> bool:
    origin = scene.component(entity, c.EntityOrigin)
    return origin is not None and isinstance(origin.origin, c.GeneratedOrigin) and origin.origin.generation.producer == gen.producer


def remove_previous_regions(input: StageInput, edit: Edit, gen: c.Generation) -> list[tuple[float, float, float, float]]:
    """``remove_previous_regions``: drop regions this producer generated earlier on the page.

    Urakatsu: the text blocks detection made go with them (a forced run detects a page that has
    text), except a block the user edited (its text, translation, style or frame): it stays with
    its region, whose extents are returned so a new detection over it is not written twice.
    """
    scene = input.scene
    kept_regions: set[str] = set()
    removed_contents: set[str] = set()
    for layer in scene.text_layers(input.page):
        presents = scene.relation_from(layer, c.Relations.PRESENTS)
        content = presents.target if presents is not None else None
        if not input.contains_entity(layer) or content is None or content in removed_contents or not _owned(scene, content, gen):
            continue
        layers = [relation.source for relation in scene.relations_to(content, c.Relations.PRESENTS)]
        if scene.user_authored(content) or any(scene.user_authored(each) for each in layers):
            kept_regions.update(relation.target for relation in scene.relations_from(content, c.Relations.RECOGNIZED_FROM))
            continue
        for each in layers:
            edit.remove_entity(each, RemovePolicy.CASCADE)
        edit.remove_entity(content, RemovePolicy.CASCADE)
        removed_contents.add(content)
    remove = []
    for entity in scene.descendants(input.page):
        if input.contains_entity(entity) and entity not in kept_regions and _owned(scene, entity, gen) and scene.component(entity, c.Region) is not None:
            remove.append(entity)
    for entity in remove:
        edit.remove_entity(entity, RemovePolicy.CASCADE)
    kept = []
    for entity in kept_regions:
        geometry = scene.component(entity, c.Geometry)
        extents = geometry_extents(geometry) if geometry is not None else None
        if extents is not None:
            kept.append(extents)
    return kept


# ---- write_page ----------------------------------------------------------------------------------


@dataclass
class _Bubble:
    entity: str
    mask: LayoutMask
    area: int


@dataclass
class _Text:
    entity: str
    mask: LayoutMask
    bounds: tuple
    content: str
    layer: str


@dataclass
class _PageRegions:
    bubbles: list
    texts: list


def write_page(
    input: StageInput,
    edit: Edit,
    page: str,
    image: Image.Image,
    output: LayoutDetections,
    gen: c.Generation,
    kept: list[tuple[float, float, float, float]] = (),
) -> None:
    """``write_page``: region scope filter, NMS, layout order, regions, dialogue links, masks.

    ``kept`` are text regions the user's edited blocks keep (``remove_previous_regions``): text
    found again over one is not written a second time."""
    detections = list(output.detections)
    if input.region is not None:
        detections = [d for d in detections if _intersects(d.bbox, input.region)]
    if kept:
        detections = [d for d in detections if d.label != "text" or not any(_same_text(d.bbox, box) for box in kept)]
    non_maximum_suppression(detections, NON_MAXIMUM_SUPPRESSION_THRESHOLD)
    suppress_nested_text(detections)
    sort_by_layout(detections)

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if input.sound_effects != "off":
        # Not in Koharu: sound effects become text blocks (pipeline/stages/sound_effects.py).
        groups = group_sound_effects(detections)
        regions = write_regions(input.scene, edit, page, rgb, [d for d in detections if d.label != SOUND_EFFECT_LABEL], gen)
        write_sound_effects(edit, page, rgb, groups, input.sound_effects, gen)
    else:
        regions = write_regions(input.scene, edit, page, rgb, detections, gen)
    link_dialogue_regions(edit, regions, gen)
    write_masks(input, edit, page, detections, (output.image_width, output.image_height))
    if input.sound_effects != "off":
        write_sound_effect_mask(input, edit, page, detections, (output.image_width, output.image_height))


# A new text detection is the same text as a kept region when this much of it lies inside it.
KEPT_OVERLAP = 0.5


def _same_text(bbox, box) -> bool:
    """Whether a new detection is text a kept block already holds: it lies mostly in the kept
    box, or the kept box (a line the user edited) lies in it (its paragraph)."""
    return containment(box, bbox) >= np.float32(KEPT_OVERLAP) or containment(bbox, box) >= NESTED_TEXT_CONTAINMENT


def _intersects(bbox, region: Bounds) -> bool:
    """``intersects``: f32 comparison of a detection box against the region."""
    left, top, right, bottom = (np.float32(v) for v in bbox)
    return bool(
        left < np.float32(region.x + region.width)
        and right > np.float32(region.x)
        and top < np.float32(region.y + region.height)
        and bottom > np.float32(region.y)
    )


def write_regions(
    snapshot: Snapshot,
    edit: Edit,
    page: str,
    image: np.ndarray,
    detections: list,
    gen: c.Generation,
) -> _PageRegions:
    """``write_regions``: typography inference per text box, then one region entity per detection."""
    regions = _PageRegions(bubbles=[], texts=[])
    inferred = [infer_typography(image, d) if d.label == "text" else None for d in detections]
    text_group = snapshot.text_group(page)
    managed_text_group = None
    if text_group is not None:
        origin = snapshot.component(text_group, c.EntityOrigin)
        if origin is not None and not isinstance(origin.origin, c.UserOrigin):
            managed_text_group = text_group
    for index, (detection, typography) in enumerate(zip(detections, inferred)):
        try:
            output = write_region(edit, page, detection, typography, gen)
        except Exception as error:
            raise ProcessingError(f"failed to write {detection.label} detection {index}: {error}") from error
        if output[0] == "bubble":
            regions.bubbles.append(output[1])
        elif output[0] == "text":
            text = output[1]
            if managed_text_group is not None:
                edit.move_entity(text.layer, managed_text_group, At.END)
            regions.texts.append(text)
    return regions


def region_kind(label: str) -> str:
    """``region_kind``."""
    return {
        "text": c.RegionKinds.TEXT,
        "bubble": c.RegionKinds.BUBBLE,
        "panel": c.RegionKinds.PANEL,
    }.get(label, UNKNOWN_REGION_KIND)


def write_region(
    edit: Edit,
    page: str,
    detection: LayoutDetection,
    typography: Optional["_Inferred"],
    gen: c.Generation,
):
    """``write_region``: geometry, region metadata, detection analysis; text adds content, layer and paint."""
    entity = edit.add_entity(page, At.END)
    kind = region_kind(detection.label)
    if detection.label == "bubble":
        geometry = mask_geometry(detection.mask) or rectangle_geometry(detection.bbox)
    elif detection.label == "text":
        geometry = (
            rotated_rectangle_geometry(detection.bbox, typography.angle_degrees)
            if typography is not None
            else rectangle_geometry(detection.bbox)
        )
    else:
        geometry = rectangle_geometry(detection.bbox)
    edit.set(entity, geometry)
    edit.set(entity, c.Region(origin=c.generated(gen), kind=kind, label=detection.label))
    edit.set(
        entity,
        c.DetectionAnalysis(
            origin=c.generated(gen),
            labels=[c.DetectionLabel(kind=kind, confidence=detection.score)],
        ),
    )
    if detection.label != "text":
        if detection.label == "bubble":
            return "bubble", _Bubble(entity=entity, mask=detection.mask, area=detection.area)
        return "other", None

    content = edit.add_text_content(page, At.END)
    layer = edit.add_text_layer(
        page,
        At.END,
        content,
        c.TextLayout(origin=c.generated(gen), kind=c.TextLayoutKind.PARAGRAPH, angle_degrees=None),
    )
    write_text_role(edit, content, "dev.koharu.text.free-text", gen)
    edit.set(
        layer,
        c.Typography(
            origin=c.generated(gen),
            preferred_font=None,
            font_weight=None,
            font_style=None,
            size=None,
            auto_fit=True,
            color=(*typography.color, 255) if typography is not None else None,
            stroke_color=(*typography.stroke_color, 255) if typography is not None and typography.stroke_color is not None else None,
            stroke_width=typography.stroke_width if typography is not None else None,
            alignment=None,
            writing_mode=typography.writing_mode if typography is not None else None,
            extensions={},
        ),
    )
    edit.relate(c.Relations.RECOGNIZED_FROM, content, entity)
    return "text", _Text(entity=entity, mask=detection.mask, bounds=detection.bbox, content=content, layer=layer)


def link_dialogue_regions(edit: Edit, regions: _PageRegions, gen: c.Generation) -> None:
    """``link_dialogue_regions``: text inside a bubble flows in it; other text fits its own box."""
    for text in regions.texts:
        bubble = containing_bubble(regions.bubbles, text)
        if bubble is not None:
            edit.relate(c.Relations.INSIDE, text.entity, bubble.entity)
            edit.relate(c.Relations.FLOWS_IN, text.layer, bubble.entity)
            write_text_role(edit, text.content, "dev.koharu.text.dialogue", gen)
        else:
            edit.relate(c.Relations.FITS_TO, text.layer, text.entity)


def containing_bubble(bubbles: list, text: _Text) -> Optional[_Bubble]:
    """``containing_bubble``: smallest bubble whose ink contains the text's ink."""
    candidates = [
        b for b in bubbles if mask_containment(b.mask, text.mask, text.bounds) >= DIALOGUE_MASK_CONTAINMENT_THRESHOLD
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda b: b.area)  # min_by_key: first minimum


def write_text_role(edit: Edit, entity: str, role: str, gen: c.Generation) -> None:
    """``write_text_role``."""
    edit.set(entity, c.TextRole(origin=c.generated(gen), role=role))


# ---- typography (infer_typography and colour clustering) ----------------------------------------


@dataclass
class _Inferred:
    color: tuple
    stroke_color: Optional[tuple]
    stroke_width: Optional[float]
    angle_degrees: float
    writing_mode: c.WritingMode


@dataclass
class _Pixels:
    """All sampled window pixels in row-major order, with local grid coordinates (window offset by 1)."""

    lx: np.ndarray
    ly: np.ndarray
    color: np.ndarray  # (n, 3) int64
    inside: np.ndarray  # (n,) bool


def _mask_pixels(mask: LayoutMask) -> Optional[np.ndarray]:
    """``valid_mask``: the pixel plane must match the declared extent."""
    pixels = np.asarray(mask.pixels)
    if pixels.shape != (mask.height, mask.width):
        return None
    return pixels


def _plane(mask: LayoutMask, left: int, top: int, right: int, bottom: int) -> np.ndarray:
    """``contains`` over the window [left,right) x [top,bottom)."""
    out = np.zeros((bottom - top, right - left), dtype=bool)
    pixels = _mask_pixels(mask)
    if pixels is None:
        return out
    x0, x1 = max(left, mask.x), min(right, mask.x + mask.width)
    y0, y1 = max(top, mask.y), min(bottom, mask.y + mask.height)
    if x0 < x1 and y0 < y1:
        out[y0 - top : y1 - top, x0 - left : x1 - left] = pixels[y0 - mask.y : y1 - mask.y, x0 - mask.x : x1 - mask.x] != 0
    return out


def _mask_window(bbox_margin, width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    """``mask_window``: floor/ceil the expanded box, clamped to the image."""
    if width == 0 or height == 0:
        return None
    left_value, top_value, right_value, bottom_value = bbox_margin
    left = int(np.clip(np.floor(left_value), F32_ZERO, np.float32(width)))
    top = int(np.clip(np.floor(top_value), F32_ZERO, np.float32(height)))
    right = int(np.clip(np.ceil(right_value), F32_ZERO, np.float32(width)))
    bottom = int(np.clip(np.ceil(bottom_value), F32_ZERO, np.float32(height)))
    if right > left and bottom > top:
        return left, top, right, bottom
    return None


def infer_typography(image: np.ndarray, detection: LayoutDetection) -> Optional[_Inferred]:
    """``infer_typography``: sample the text window, measure angle, infer paint."""
    mask = detection.mask
    height, width = image.shape[:2]
    bl, bt, br, bb = (np.float32(v) for v in detection.bbox)
    m = TYPOGRAPHY_SAMPLE_MARGIN
    window = _mask_window((bl - m, bt - m, br + m, bb + m), width, height)
    if window is None:
        return None
    left, top, right, bottom = window
    local_width = right - left + 2
    local_height = bottom - top + 2
    window_height, window_width = bottom - top, right - left
    colors = image[top:bottom, left:right].astype(np.int64)  # (h, w, 3)
    inside = _plane(mask, left, top, right, bottom)
    if not inside.any():
        return None  # points.is_empty()

    yy = np.arange(window_height)[:, None]
    xx = np.arange(window_width)[None, :]
    margin = TYPOGRAPHY_BACKGROUND_MARGIN
    border = (xx < margin) | (xx + margin >= window_width) | (yy < margin) | (yy + margin >= window_height)
    flat_colors = colors.reshape(-1, 3)
    background = median_color(colors[border]) if border.any() else median_color(flat_colors)

    rows, cols = np.nonzero(inside)  # row-major, as the Rust loops
    points_x = (cols + left).astype(np.float64) + 0.5
    points_y = (rows + top).astype(np.float64) + 0.5
    angle_degrees, vertical = mask_angle(points_x, points_y, detection.bbox)

    lx = np.broadcast_to(xx + 1, (window_height, window_width)).reshape(-1).astype(np.int64)
    ly = np.broadcast_to(yy + 1, (window_height, window_width)).reshape(-1).astype(np.int64)
    pixels = _Pixels(lx=lx, ly=ly, color=flat_colors, inside=inside.reshape(-1))
    color, stroke_color, stroke_width = infer_text_paint(pixels, background, local_width, local_height)
    return _Inferred(
        color=color,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        angle_degrees=angle_degrees,
        writing_mode=c.WritingMode.VERTICAL if vertical else c.WritingMode.HORIZONTAL,
    )


def mask_angle(points_x: np.ndarray, points_y: np.ndarray, bbox) -> tuple[float, bool]:
    """``mask_angle``: search line angles on a half-degree grid; pick the sharper of horizontal or vertical."""
    bounds = (
        float(points_x.min()),
        float(points_y.min()),
        float(points_x.max()),
        float(points_y.max()),
    )
    best_horizontal = (-math.inf, 0.0)
    best_vertical = (-math.inf, 0.0)
    for step in range(-ANGLE_SEARCH_HALF_STEPS, ANGLE_SEARCH_HALF_STEPS + 1):
        angle_degrees = float(step) * ANGLE_SEARCH_STEP_DEGREES
        radians = math.radians(angle_degrees)
        sin, cos = math.sin(radians), math.cos(radians)
        horizontal_score, vertical_score = projection_scores(points_x, points_y, bounds, sin, cos)
        if horizontal_score > best_horizontal[0]:
            best_horizontal = (horizontal_score, angle_degrees)
        if vertical_score > best_vertical[0]:
            best_vertical = (vertical_score, angle_degrees)
    maximum_score = max(best_horizontal[0], best_vertical[0])
    scores_are_close = abs(best_horizontal[0] - best_vertical[0]) <= maximum_score * 0.02
    left, top, right, bottom = (np.float32(v) for v in bbox)
    if scores_are_close:
        is_vertical = bool((bottom - top) > (right - left))
    else:
        is_vertical = best_vertical[0] > best_horizontal[0]
    angle = np.float32(best_vertical[1] if is_vertical else best_horizontal[1])
    if abs(float(angle)) < ANGLE_SNAP_DEGREES:
        angle = F32_ZERO
    return float(angle), is_vertical


def projection_scores(points_x: np.ndarray, points_y: np.ndarray, bounds, sin: float, cos: float) -> tuple[float, float]:
    """``projection_scores``: sum of squared projection-profile bins, normalised by point count."""
    horizontal_origin, horizontal_length = projection_extent(bounds, -sin, cos)
    vertical_origin, vertical_length = projection_extent(bounds, cos, sin)
    count = len(points_x)

    horizontal_projection = points_y * cos - points_x * sin - horizontal_origin
    horizontal_index = np.floor(horizontal_projection)
    horizontal_fraction = horizontal_projection - horizontal_index
    horizontal_index = np.maximum(horizontal_index, 0).astype(np.int64)  # Rust `as usize` saturates
    horizontal = _split_bins(horizontal_index, horizontal_fraction, horizontal_length)

    vertical_projection = points_x * cos + points_y * sin - vertical_origin
    vertical_index = np.floor(vertical_projection)
    vertical_fraction = vertical_projection - vertical_index
    vertical_index = np.maximum(vertical_index, 0).astype(np.int64)
    vertical = _split_bins(vertical_index, vertical_fraction, vertical_length)

    return _sum_squares(horizontal) / count, _sum_squares(vertical) / count


def _split_bins(index: np.ndarray, fraction: np.ndarray, length: int) -> np.ndarray:
    """Linear splatting of each point into two adjacent bins, accumulated in point order."""
    interleaved_index = np.empty(2 * len(index), dtype=np.int64)
    interleaved_index[0::2] = index
    interleaved_index[1::2] = index + 1
    interleaved_weight = np.empty(2 * len(index), dtype=np.float64)
    interleaved_weight[0::2] = 1.0 - fraction
    interleaved_weight[1::2] = fraction
    return np.bincount(interleaved_index, weights=interleaved_weight, minlength=length)[:length].astype(np.float64)


def _sum_squares(values: np.ndarray) -> float:
    total = 0.0
    for value in values.tolist():  # sequential fold, as Rust's iterator sum
        total += value * value
    return total


def projection_extent(bounds, axis_x: float, axis_y: float) -> tuple[float, int]:
    """``projection_extent``."""
    left, top, right, bottom = bounds
    minimum_x = left if axis_x >= 0.0 else right
    minimum_y = top if axis_y >= 0.0 else bottom
    maximum_x = right if axis_x >= 0.0 else left
    maximum_y = bottom if axis_y >= 0.0 else top
    minimum = minimum_x * axis_x + minimum_y * axis_y
    maximum = maximum_x * axis_x + maximum_y * axis_y
    origin = math.floor(minimum)
    length = int(max(math.ceil(maximum) - origin, 0.0)) + 2
    return float(origin), length


def _histogram_median_values(values: np.ndarray) -> int:
    """``histogram_median``: mean of the two middle order statistics, integer division."""
    count = len(values)
    if count == 0:
        return 0
    ordered = np.sort(values)
    lower = int(ordered[(count - 1) // 2])
    upper = int(ordered[count // 2])
    return (lower + upper) // 2


def median_color(colors: np.ndarray) -> tuple[int, int, int]:
    """``median_color`` / ``median_pixel_color``: per-channel median over a set of colours."""
    colors = np.asarray(colors).reshape(-1, 3)
    if len(colors) == 0:
        return (0, 0, 0)
    return tuple(_histogram_median_values(colors[:, channel]) for channel in range(3))


def color_distance_squared(left, right) -> int:
    """``color_distance_squared``."""
    return int(sum((int(a) - int(b)) ** 2 for a, b in zip(left, right)))


def color_luminance(color) -> int:
    """``color_luminance``."""
    return int(color[0]) * 54 + int(color[1]) * 183 + int(color[2]) * 19


def normalize_text_color(color) -> tuple[int, int, int]:
    """``normalize_text_color``: snap only at luminance extremes."""
    luminance = color_luminance(color)
    if luminance <= COLOR_SNAP_DARK_LUMINANCE:
        return (0, 0, 0)
    if luminance >= COLOR_SNAP_LIGHT_LUMINANCE:
        return (255, 255, 255)
    return (int(color[0]), int(color[1]), int(color[2]))


def color_lies_between(candidate, start, end) -> bool:
    """``color_lies_between``: candidate projects inside the segment and is close to it."""
    start = [float(v) for v in start]
    end = [float(v) for v in end]
    candidate = [float(v) for v in candidate]
    direction = [end[i] - start[i] for i in range(3)]
    length_squared = sum(d * d for d in direction)
    if length_squared <= DBL_EPSILON:
        return False
    projection = sum((candidate[i] - start[i]) * direction[i] for i in range(3)) / length_squared
    if not (0.0 <= projection <= 1.0):
        return False
    distance_squared = 0.0
    for i in range(3):
        difference = candidate[i] - (start[i] + direction[i] * projection)
        distance_squared += difference * difference
    return distance_squared <= 24.0**2 * 3.0


def _palette(color: np.ndarray):
    """``color_palette`` as used by ``color_clusters``: 16^3 bins, non-empty bins in index order."""
    index = ((color[:, 0] >> 4) << 8) | ((color[:, 1] >> 4) << 4) | (color[:, 2] >> 4)
    counts = np.bincount(index, minlength=16 * 16 * 16).astype(np.int64)
    sums = np.stack(
        [np.bincount(index, weights=color[:, channel].astype(np.float64), minlength=16 * 16 * 16) for channel in range(3)],
        axis=1,
    ).astype(np.int64)
    keep = counts > 0
    counts = counts[keep]
    sums = sums[keep]
    # ColorBin::color: (sum + count / 2) / count per channel.
    colors = (sums + (counts // 2)[:, None]) // counts[:, None]
    return colors, counts, sums


def _extreme_palette_color(colors: np.ndarray, counts: np.ndarray, darkest: bool) -> tuple[int, int, int]:
    """``extreme_palette_color``: min_by_key (first) for darkest, max_by_key (last) for lightest."""
    luminance = colors @ np.array([54, 183, 19], dtype=np.int64)

    def select(significant_only: bool):
        candidates = (counts >= MIN_EXTREME_COLOR_PIXELS) if significant_only else np.ones(len(counts), dtype=bool)
        if not candidates.any():
            return None
        if darkest:
            index = int(np.argmin(np.where(candidates, luminance, np.iinfo(np.int64).max)))
        else:
            values = np.where(candidates, luminance, -1)
            index = len(values) - 1 - int(np.argmax(values[::-1]))
        return colors[index]

    chosen = select(True)
    if chosen is None:
        chosen = select(False)
    if chosen is None:
        return (0, 0, 0)
    return (int(chosen[0]), int(chosen[1]), int(chosen[2]))


def _distant_palette_color(colors: np.ndarray, counts: np.ndarray, centers) -> tuple[int, int, int]:
    """``distant_palette_color``: max_by_key (last) of distance-to-nearest-centre weighted by count."""
    if len(colors) == 0:
        return (0, 0, 0)
    centers = np.asarray(centers, dtype=np.int64)
    distances = ((colors[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).min(axis=1)
    scores = distances * np.minimum(counts, 64)
    index = len(scores) - 1 - int(np.argmax(scores[::-1]))
    return (int(colors[index][0]), int(colors[index][1]), int(colors[index][2]))


def _distances_to_centers(colors: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return ((colors[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)


def color_clusters(pixels: _Pixels, background) -> tuple[np.ndarray, np.ndarray]:
    """``color_clusters``: four seeded k-means centres over palette bins; returns centres and per-pixel cluster."""
    palette_colors, palette_counts, palette_sums = _palette(pixels.color)
    darkest = _extreme_palette_color(palette_colors, palette_counts, darkest=True)
    lightest = _extreme_palette_color(palette_colors, palette_counts, darkest=False)
    distant = _distant_palette_color(palette_colors, palette_counts, [background, darkest, lightest])
    centers = np.array([background, darkest, lightest, distant], dtype=np.int64)
    for _ in range(4):
        assignment = _distances_to_centers(palette_colors, centers).argmin(axis=1)
        accumulated_count = np.bincount(assignment, weights=palette_counts.astype(np.float64), minlength=COLOR_CLUSTER_COUNT)
        accumulated_sums = [
            np.bincount(assignment, weights=palette_sums[:, channel].astype(np.float64), minlength=COLOR_CLUSTER_COUNT)
            for channel in range(3)
        ]
        for index in range(COLOR_CLUSTER_COUNT):
            count = int(accumulated_count[index])
            if count != 0:
                centers[index] = [(int(accumulated_sums[channel][index]) + count // 2) // count for channel in range(3)]
    pixel_assignment = _distances_to_centers(pixels.color, centers).argmin(axis=1)
    return centers, pixel_assignment


@dataclass
class _Outline:
    fill: tuple
    stroke: tuple
    width: float
    score: int


def infer_text_paint(pixels: _Pixels, background, width: int, height: int):
    """``infer_text_paint``: a measured outline first, else the ink cluster on a single-background text."""
    centers, assignment = color_clusters(pixels, background)
    outline = infer_outline(pixels, centers, assignment, background, width, height)
    if outline is not None:
        return outline.fill, outline.stroke, outline.width

    nonempty = [index for index in range(COLOR_CLUSTER_COUNT) if np.any(assignment == index)]
    if len(nonempty) == 1:
        return normalize_text_color(centers[nonempty[0]]), None, None

    background_index = min(nonempty, key=lambda index: color_distance_squared(centers[index], background))
    background_index = local_background_index(pixels, assignment, background_index, width, height)
    # Not in Rust: a cluster with no pixels in the mask is a neighbour's ink, unless no cluster has any.
    in_mask = {index: int(np.count_nonzero(pixels.inside[assignment == index])) >= 8 for index in nonempty}
    candidates = [index for index in nonempty if index != background_index]
    if any(in_mask[index] for index in candidates):
        candidates = [index for index in candidates if in_mask[index]]
    fill_index = None
    fill_score = -1
    for index in candidates:
        score = cluster_ink_score(pixels, assignment, index, centers[index], centers[background_index])
        if fill_index is None or score >= fill_score:  # max_by_key: last maximum
            fill_index, fill_score = index, score
    if fill_index is None:
        return normalize_text_color(centers[background_index]), None, None
    color = representative_ink_color(pixels, assignment, fill_index)
    halo = infer_background_halo(pixels, assignment, fill_index, color, background, width, height)
    if halo is not None:
        return normalize_text_color(color), halo[0], halo[1]
    return normalize_text_color(color), None, None


def local_background_index(pixels: _Pixels, assignment: np.ndarray, background_index: int, width: int, height: int) -> int:
    """The cluster the text sits on (no Rust counterpart).

    Rust judges ink against the window border's colour. When that colour barely reaches into the
    mask (text on a band or box that the axis-aligned window cuts across), the text sits on whatever
    lines the inside of the mask's edge instead; the edge rather than the whole mask, since bold
    lettering can fill most of a tight mask.
    """
    inside_counts = np.bincount(assignment[pixels.inside], minlength=COLOR_CLUSTER_COUNT)
    total = int(inside_counts.sum())
    if total == 0 or inside_counts[background_index] >= total * LOCAL_BACKGROUND_MIN_SHARE:
        return background_index
    inside_grid = np.zeros((height, width), dtype=bool)
    inside_grid[pixels.ly[pixels.inside], pixels.lx[pixels.inside]] = True
    edge = (inside_grid & _dilate3(~inside_grid))[pixels.ly, pixels.lx]
    edge_counts = np.bincount(assignment[edge], minlength=COLOR_CLUSTER_COUNT)
    if int(edge_counts.sum()) == 0:
        return background_index
    local = int(np.argmax(edge_counts))
    # A background runs on past the mask; ink lining a glyph-tight mask stops at it.
    outside = int(np.count_nonzero((assignment == local) & ~pixels.inside))
    if outside < inside_counts[local] * LOCAL_BACKGROUND_MIN_SHARE:
        return background_index
    return local


def cluster_ink_score(pixels: _Pixels, assignment: np.ndarray, index: int, center, background) -> int:
    """``cluster_ink_score``."""
    members = assignment == index
    inside = int(np.count_nonzero(pixels.inside[members]))
    evidence = inside + int(np.count_nonzero(members)) if inside >= 8 else int(np.count_nonzero(members))
    return color_distance_squared(center, background) * min(evidence, 4096)


def representative_ink_color(pixels: _Pixels, assignment: np.ndarray, index: int) -> tuple[int, int, int]:
    """``representative_ink_color``: median of inside-mask pixels when there are at least eight."""
    members = assignment == index
    inside = members & pixels.inside
    if int(np.count_nonzero(inside)) >= 8:
        return median_color(pixels.color[inside])
    return median_color(pixels.color[members])


def _reachable_without_cluster(assignments: np.ndarray, barrier: int) -> np.ndarray:
    """``reachable_without_cluster``: 4-connected pixels not in ``barrier`` that touch the window border."""
    passable = (assignments != barrier).astype(np.uint8)
    _, labels = cv2.connectedComponents(passable, connectivity=4)
    border = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    touching = np.unique(border[border > 0])
    return passable.astype(bool) & np.isin(labels, touching)


def _dilate3(grid: np.ndarray) -> np.ndarray:
    """3x3 box dilation of a boolean grid (zero outside)."""
    out = grid.copy()
    out[1:, :] |= grid[:-1, :]
    out[:-1, :] |= grid[1:, :]
    widened = out.copy()
    widened[:, 1:] |= out[:, :-1]
    widened[:, :-1] |= out[:, 1:]
    return widened


def _enclosing_cluster_selection(assignments: np.ndarray, cluster: int, fill_grid: np.ndarray) -> np.ndarray:
    """``enclosing_cluster_pixels``: cluster pixels 8-connected to the cluster's ring around the fill."""
    cluster_grid = assignments == cluster
    seeds = cluster_grid & _dilate3(fill_grid)
    _, labels = cv2.connectedComponents(cluster_grid.astype(np.uint8), connectivity=8)
    hit = np.unique(labels[seeds])
    hit = hit[hit > 0]
    return cluster_grid & np.isin(labels, hit)


def measured_stroke_width(
    pixels: _Pixels,
    stroke_index: np.ndarray,
    fill_index: np.ndarray,
    outside: np.ndarray,
    width: int,
    height: int,
) -> Optional[float]:
    """``measured_stroke_width``: the peak distance from fill to the outer rim of the stroke band.

    Rust takes the median; see the module docstring for why the peak is used here.
    """
    if len(stroke_index) < 8 or len(fill_index) < 8:
        return None
    fill_grid = np.zeros((height, width), dtype=bool)
    fill_grid[pixels.ly[fill_index], pixels.lx[fill_index]] = True
    distances = _distance_from_sites(fill_grid)
    touches_outside = _dilate3(outside)
    sy, sx = pixels.ly[stroke_index], pixels.lx[stroke_index]
    keep = touches_outside[sy, sx] & (distances[sy, sx] != 0)
    values = distances[sy, sx][keep].astype(np.int64)
    if len(values) < 8:
        return None
    counts = np.bincount(values, minlength=2)
    windows = counts + np.append(counts[1:], 0) + np.insert(counts[:-1], 0, 0)
    counts[0] = 0
    peak = int(np.argmax(counts))
    if windows[peak] < counts[1 : 2 * peak + 3].sum() * OUTLINE_RIM_CONCENTRATION:
        return None
    return float(peak) if peak >= MIN_MEASURED_STROKE_WIDTH else None


def infer_outline(
    pixels: _Pixels,
    centers: np.ndarray,
    assignment: np.ndarray,
    background,
    width: int,
    height: int,
) -> Optional[_Outline]:
    """``infer_outline``: a fill enclosed by a stroke band with contrasting colour and measurable width."""
    assignments = np.full((height, width), -1, dtype=np.int64)
    assignments[pixels.ly, pixels.lx] = assignment
    nonempty = [index for index in range(COLOR_CLUSTER_COUNT) if np.any(assignment == index)]
    background_index = min(nonempty, key=lambda index: color_distance_squared(centers[index], background))
    members = [np.nonzero(assignment == index)[0] for index in range(COLOR_CLUSTER_COUNT)]

    best: Optional[_Outline] = None
    for stroke_index in range(COLOR_CLUSTER_COUNT):
        stroke_members = members[stroke_index]
        if len(stroke_members) < 8 or stroke_index == background_index:
            continue
        outside = _reachable_without_cluster(assignments, stroke_index)
        for fill_index in range(COLOR_CLUSTER_COUNT):
            fill_members = members[fill_index]
            if fill_index == stroke_index or len(fill_members) == 0:
                continue
            enclosed = fill_members[~outside[pixels.ly[fill_members], pixels.lx[fill_members]]]
            if len(enclosed) < 8:
                continue
            # Not in Rust: a fill outside this text's mask is a neighbour's glyph counters, and the
            # glyph around them is not this text's stroke.
            inside_fill = int(np.count_nonzero(pixels.inside[enclosed]))
            if inside_fill * 2 < len(enclosed):
                continue
            fill_grid = np.zeros((height, width), dtype=bool)
            fill_grid[pixels.ly[enclosed], pixels.lx[enclosed]] = True
            selection = _enclosing_cluster_selection(assignments, stroke_index, fill_grid)
            stroke = stroke_members[selection[pixels.ly[stroke_members], pixels.lx[stroke_members]]]
            if len(stroke) < 8:
                continue

            fill_color = median_color(pixels.color[enclosed])
            stroke_color = median_color(pixels.color[stroke])
            contrast = color_distance_squared(fill_color, stroke_color)
            if (
                contrast < COLOR_CLUSTER_MIN_DISTANCE_SQUARED
                or color_distance_squared(fill_color, background) < COLOR_CLUSTER_MIN_DISTANCE_SQUARED
                or color_distance_squared(stroke_color, background) < COLOR_CLUSTER_MIN_DISTANCE_SQUARED
                or color_lies_between(stroke_color, fill_color, background)
                or color_lies_between(fill_color, stroke_color, background)
            ):
                continue
            normalized_fill = normalize_text_color(fill_color)
            normalized_stroke = normalize_text_color(stroke_color)
            if normalized_fill == normalized_stroke:
                continue

            stroke_width = measured_stroke_width(pixels, stroke, enclosed, outside, width, height)
            if stroke_width is None:
                continue
            evidence = len(enclosed) + inside_fill
            score = contrast * min(evidence, 4096)
            if best is None or score > best.score:
                best = _Outline(fill=normalized_fill, stroke=normalized_stroke, width=stroke_width, score=score)
    return best


def infer_background_halo(
    pixels: _Pixels,
    assignment: np.ndarray,
    fill_index: int,
    fill_color,
    background,
    width: int,
    height: int,
) -> Optional[tuple[tuple[int, int, int], float]]:
    """A stroke in the page's own colour, seen only where the text crosses art (no Rust counterpart).

    Glyphs are the fill cluster's components inside the mask that do not touch the window border, so
    art in the fill colour is not mistaken for text. Rings at each distance from the glyphs then show
    the stroke: the first rings past the anti-aliasing hold almost nothing but the halo colour, and
    the ring just past the stroke's rim suddenly holds art. The art's edge must sit at that one
    distance, as a stroke traced around the glyphs makes it, rather than spread like a panel or
    bubble line that merely passes nearby, and the band must stay in proportion to the glyph lines.
    """
    valid = np.zeros((height, width), dtype=bool)
    valid[pixels.ly, pixels.lx] = True
    members = assignment == fill_index
    fill_grid = np.zeros((height, width), dtype=bool)
    fill_grid[pixels.ly[members], pixels.lx[members]] = True
    inside_grid = np.zeros((height, width), dtype=bool)
    inside_grid[pixels.ly[pixels.inside], pixels.lx[pixels.inside]] = True
    _, labels = cv2.connectedComponents(fill_grid.astype(np.uint8), connectivity=8)
    border = np.concatenate([labels[1], labels[-2], labels[:, 1], labels[:, -2]])
    glyph_labels = np.setdiff1d(np.unique(labels[fill_grid & inside_grid]), np.append(border, 0))
    glyph = np.isin(labels, glyph_labels)
    if int(np.count_nonzero(glyph)) < 8:
        return None

    distance = _distance_from_sites(glyph).astype(np.int64)
    colors = np.zeros((height, width, 3), dtype=np.int64)
    colors[pixels.ly, pixels.lx] = pixels.color
    ring = valid & (distance >= 2) & (distance <= 4)
    if int(np.count_nonzero(ring)) < 8:
        return None
    halo = median_color(colors[ring])
    # A halo unlike the background is infer_outline's to judge, and it already declined.
    if (
        color_distance_squared(halo, background) >= COLOR_CLUSTER_MIN_DISTANCE_SQUARED
        or color_distance_squared(halo, fill_color) < COLOR_CLUSTER_MIN_DISTANCE_SQUARED
    ):
        return None
    near = valid & (((colors - np.array(halo, dtype=np.int64)) ** 2).sum(axis=2) < COLOR_CLUSTER_MIN_DISTANCE_SQUARED)
    contrast = valid & ~near & ~glyph
    edge = contrast & _dilate3(near)

    searched = valid & (distance <= HALO_SEARCH_DISTANCE)
    length = HALO_SEARCH_DISTANCE + 1
    totals = np.bincount(distance[searched], minlength=length)
    contrasts = np.bincount(distance[searched & contrast], minlength=length)
    edges = np.bincount(distance[searched & edge], minlength=length)
    fractions = contrasts / np.maximum(totals, 1)
    quiet = next((d for d in range(2, length) if totals[d] > 0 and fractions[d] <= HALO_QUIET_FRACTION), None)
    if quiet is None:
        return None
    rim = next((d for d in range(quiet + 1, length) if fractions[d] >= HALO_RIM_FRACTION), None)
    if rim is None or rim - 1 < MIN_MEASURED_STROKE_WIDTH or contrasts[rim] < HALO_MIN_RIM_PIXELS:
        return None
    if fractions[min(rim + 1, length - 1)] - fractions[max(rim - 2, quiet)] < HALO_RIM_STEP:
        return None
    depths = np.sort(_distance_from_sites(~glyph)[glyph])
    if rim - 1 > HALO_MAX_WIDTH_PER_GLYPH_DEPTH * int(depths[(len(depths) - 1) * 99 // 100]):
        return None
    nearby_edges = int(edges[rim : rim + 6].sum())
    if nearby_edges == 0 or int(edges[rim : rim + 2].sum()) < nearby_edges * HALO_RIM_CONCENTRATION:
        return None
    stroke_color = normalize_text_color(halo)
    if stroke_color == normalize_text_color(fill_color):
        return None
    return stroke_color, float(rim - 1)


# ---- geometry (rectangle, rotated rectangle, mask outline) ---------------------------------------


def rectangle_geometry(bbox) -> c.Geometry:
    """``rectangle_geometry``: box with a minimum extent of one pixel, computed in f32 like Rust."""
    left, top, right, bottom = (np.float32(v) for v in bbox)
    width = float(max(right - left, F32_ONE))
    height = float(max(bottom - top, F32_ONE))
    return c.Geometry.rectangle(float(left), float(top), width, height)


def rotated_rectangle_geometry(bbox, angle_degrees: float) -> c.Geometry:
    """``rotated_rectangle_geometry``: box rotated about its centre by the inferred angle."""
    left, top, right, bottom = (np.float32(v) for v in bbox)
    width = float(max(right - left, F32_ONE))
    height = float(max(bottom - top, F32_ONE))
    center_x = float(left + right) * 0.5
    center_y = float(top + bottom) * 0.5
    radians = math.radians(float(np.float32(angle_degrees)))
    sin, cos = math.sin(radians), math.cos(radians)
    corners = [(-width * 0.5, -height * 0.5), (width * 0.5, -height * 0.5), (width * 0.5, height * 0.5), (-width * 0.5, height * 0.5)]
    return c.Geometry(
        origin=c.user(),
        points=[
            c.Point(x=center_x + x * cos - y * sin, y=center_y + x * sin + y * cos)
            for x, y in corners
        ],
    )


def mask_geometry(mask: LayoutMask) -> Optional[c.Geometry]:
    """``mask_geometry``: outer contour of the instance mask, simplified by Ramer-Douglas-Peucker."""
    pixels = _mask_pixels(mask)
    if pixels is None:
        return None
    padded = np.zeros((mask.height + 2, mask.width + 2), dtype=bool)
    padded[1 : mask.height + 1, 1 : mask.width + 1] = pixels != 0
    contours = trace_contours(padded)
    best = None
    best_area = -1.0
    for points, is_outer in contours:
        if not is_outer:
            continue
        area = contour_area(points)
        if best is None or area >= best_area:  # max_by: last maximum
            best, best_area = points, area
    if best is None or len(best) < 3:
        return None
    epsilon = max(arc_length(best, closed=True) * 0.001, DBL_EPSILON)
    simplified = approximate_polygon_dp(best, epsilon, closed=True)
    points = [
        c.Point(x=float(px - 1) + float(mask.x), y=float(py - 1) + float(mask.y)) for px, py in simplified
    ]
    if len(points) < 3:
        return None
    return c.Geometry(origin=c.user(), points=points)


_CONTOUR_DIFFS = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]


def trace_contours(binary: np.ndarray) -> list[tuple[list[tuple[int, int]], bool]]:
    """Suzuki-Abe border following, mirroring ``imageproc::contours::find_contours_with_threshold``.

    Returns ``(points, is_outer)`` for every border in discovery order. The parent bookkeeping of
    the Rust code only affects contour parents, which the stage does not use, so it is omitted.
    """
    height, width = binary.shape
    values = binary.astype(np.int64).reshape(-1).tolist()

    def at(x: int, y: int) -> int:
        return x + width * y

    def nonzero(x: int, y: int) -> bool:
        return 0 <= x < width and 0 <= y < height and values[at(x, y)] != 0

    # Only pixels next to a zero can start a border. Zeros never become non-zero during tracing and
    # non-zero pixels never become zero, so this candidate set is fixed in advance.
    padded_values = binary.astype(bool)
    left_zero = np.zeros_like(padded_values)
    left_zero[:, 1:] = ~padded_values[:, :-1]
    right_zero = np.zeros_like(padded_values)
    right_zero[:, :-1] = ~padded_values[:, 1:]
    candidate_rows, candidate_cols = np.nonzero(padded_values & (left_zero | right_zero))

    diffs = deque(_CONTOUR_DIFFS)
    border_number = 1
    contours: list[tuple[list[tuple[int, int]], bool]] = []

    def rotate_to_value(value: tuple[int, int]) -> None:
        position = diffs.index(value)
        diffs.rotate(-position)

    for y, x in zip(candidate_rows.tolist(), candidate_cols.tolist()):
        current = values[at(x, y)]
        if current == 0:
            continue
        if current == 1 and x > 0 and values[at(x - 1, y)] == 0:
            adjacent = (x - 1, y)
            is_outer = True
        elif current > 0 and x + 1 < width and values[at(x + 1, y)] == 0:
            adjacent = (x + 1, y)
            is_outer = False
        else:
            continue

        border_number += 1
        contour_points: list[tuple[int, int]] = []
        curr = (x, y)
        rotate_to_value((adjacent[0] - curr[0], adjacent[1] - curr[1]))
        first = None
        for dx, dy in diffs:
            if nonzero(curr[0] + dx, curr[1] + dy):
                first = (curr[0] + dx, curr[1] + dy)
                break
        if first is not None:
            pos1 = first
            pos2 = pos1
            pos3 = curr
            while True:
                contour_points.append(pos3)
                rotate_to_value((pos2[0] - pos3[0], pos2[1] - pos3[1]))
                pos4 = None
                for dx, dy in reversed(diffs):  # counter-clockwise
                    if nonzero(pos3[0] + dx, pos3[1] + dy):
                        pos4 = (pos3[0] + dx, pos3[1] + dy)
                        break
                if pos4 is None:
                    raise ProcessingError("contour tracing lost its neighbourhood")
                is_right_edge = False
                for dx, dy in reversed(diffs):
                    if (dx, dy) == (pos4[0] - pos3[0], pos4[1] - pos3[1]):
                        break
                    if (dx, dy) == (1, 0):
                        is_right_edge = True
                        break
                if pos3[0] + 1 == width or is_right_edge:
                    values[at(pos3[0], pos3[1])] = -border_number
                elif values[at(pos3[0], pos3[1])] == 1:
                    values[at(pos3[0], pos3[1])] = border_number
                if pos4 == curr and pos3 == pos1:
                    break
                pos2 = pos3
                pos3 = pos4
        else:
            contour_points.append((x, y))
            values[at(x, y)] = -border_number
        contours.append((contour_points, is_outer))
    return contours


def contour_area(points: list[tuple[int, int]]) -> float:
    """``contour_area``: absolute shoelace area."""
    if len(points) < 3:
        return 0.0
    previous = points[-1]
    accumulated = 0.0
    for point in points:
        accumulated += float(previous[0]) * float(point[1]) - float(previous[1]) * float(point[0])
        previous = point
    return abs(0.5 * accumulated)


def arc_length(points: list[tuple[int, int]], closed: bool) -> float:
    """``arc_length``."""
    total = 0.0
    for left, right in zip(points, points[1:]):
        total += _distance(left, right)
    if len(points) > 2 and closed:
        total += _distance(points[0], points[-1])
    return total


def _distance(left, right) -> float:
    dx = float(left[0]) - float(right[0])
    dy = float(left[1]) - float(right[1])
    return math.sqrt(dx * dx + dy * dy)


def approximate_polygon_dp(points: list[tuple[int, int]], epsilon: float, closed: bool) -> list[tuple[int, int]]:
    """``approximate_polygon_dp``: Ramer-Douglas-Peucker, recursing on the farthest point."""
    if epsilon <= 0.0:
        raise ValueError("epsilon must be greater than 0.0")
    result = _dp_open(points, epsilon)
    if closed:
        result.pop()
    return result


def _dp_open(curve: list[tuple[int, int]], epsilon: float) -> list[tuple[int, int]]:
    end = len(curve) - 1
    p, q = (float(curve[0][0]), float(curve[0][1])), (float(curve[end][0]), float(curve[end][1]))
    a = p[1] - q[1]
    b = q[0] - p[0]
    cc = p[0] * q[1] - q[0] * p[1]
    norm = math.sqrt(a * a + b * b)
    dmax = 0.0
    index = 0
    for i in range(1, len(curve)):
        if norm == 0.0:
            distance = math.nan
        else:
            distance = abs(a * float(curve[i][0]) + b * float(curve[i][1]) + cc) / norm
        if distance > dmax:
            index = i
            dmax = distance
    if dmax > epsilon:
        partial1 = _dp_open(curve[0 : index + 1], epsilon)
        partial2 = _dp_open(curve[index : end + 1], epsilon)
        partial1.pop()
        partial1.extend(partial2)
        return partial1
    return [curve[0], curve[end]]


# ---- morphology and distance transform (imageproc semantics) -----------------------------------


def _distance_from_sites(sites: np.ndarray) -> np.ndarray:
    """``distance_transform`` (L2, foreground = ``sites``): ceil of the Euclidean distance, saturating at 255.

    No site means every distance is infinite and saturates to 255, as in imageproc.
    """
    if not sites.any():
        return np.full(sites.shape, 255, dtype=np.uint8)
    if sites.all():
        return np.zeros(sites.shape, dtype=np.uint8)
    # cv2 measures from non-zero pixels to the nearest zero pixel, so zeros carry the sites.
    source = np.where(sites, 0, 1).astype(np.uint8)
    distance = cv2.distanceTransform(source, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.minimum(np.ceil(distance), 255).astype(np.uint8)


def dilate_l2(image: np.ndarray, k: int) -> np.ndarray:
    """``morphology::dilate`` with ``Norm::L2``: 255 where the nearest foreground is within ``k``."""
    return np.where(_distance_from_sites(image != 0) <= k, 255, 0).astype(np.uint8)


def erode_l2(image: np.ndarray, k: int) -> np.ndarray:
    """``morphology::erode`` with ``Norm::L2``: 0 where the nearest background is within ``k``."""
    return np.where(_distance_from_sites(image == 0) <= k, 0, 255).astype(np.uint8)


def close_l2(image: np.ndarray, k: int) -> np.ndarray:
    """``morphology::close``: dilate then erode."""
    return erode_l2(dilate_l2(image, k), k)


# ---- masks ----------------------------------------------------------------------------------------


def _valid_mask(mask: LayoutMask) -> bool:
    return _mask_pixels(mask) is not None


def _stamp_mask(target: np.ndarray, mask: LayoutMask, origin_x: int, origin_y: int) -> None:
    """``stamp_mask``: copy non-zero mask pixels into a target placed at (origin_x, origin_y)."""
    pixels = _mask_pixels(mask)
    if pixels is None:
        return
    target_height, target_width = target.shape
    for local_y in range(mask.height):
        target_y = mask.y + local_y - origin_y
        if target_y < 0 or target_y >= target_height:
            continue
        local_start = max(0, origin_x - mask.x)
        local_end = min(mask.width, target_width + origin_x - mask.x)
        if local_start >= local_end:
            continue
        segment = pixels[local_y, local_start:local_end] != 0
        target_start = mask.x + local_start - origin_x
        row = target[target_y, target_start : target_start + len(segment)]
        row[segment] = 255


def mask_for(detections: list, label: str, size: tuple[int, int]) -> np.ndarray:
    """``mask_for``: union of one label's instance masks on the page."""
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    for detection in detections:
        if detection.label == label:
            _stamp_mask(mask, detection.mask, 0, 0)
    return mask


def _mask_bounds(mask: LayoutMask, size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    return (min(mask.x, width), min(mask.y, height), min(mask.x + mask.width, width), min(mask.y + mask.height, height))


def _expand_bounds(bounds, radius: int, size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    left, top, right, bottom = bounds
    return (max(left - radius, 0), max(top - radius, 0), min(right + radius, width), min(bottom + radius, height))


def _union_bounds(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _rectangles_touch(a, b) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def closed_mask_for(detections: list, label: str, size: tuple[int, int], radius: int) -> np.ndarray:
    """``closed_mask_for``: per-group local dilate-dilate-erode, equal to the full-page operation."""
    width, height = size
    masks = [
        d.mask
        for d in detections
        if d.label == label and _valid_mask(d.mask) and d.mask.width > 0 and d.mask.height > 0
    ]
    output = np.zeros((height, width), dtype=np.uint8)
    if not masks:
        return output

    interaction_radius = radius * 2
    parents = list(range(len(masks)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    expanded = [_expand_bounds(_mask_bounds(m, size), interaction_radius, size) for m in masks]
    for left in range(len(masks)):
        for right in range(left + 1, len(masks)):
            if _rectangles_touch(expanded[left], expanded[right]):
                root_left, root_right = find(left), find(right)
                if root_left != root_right:
                    parents[root_right] = root_left

    groups: dict[int, list[int]] = {}
    for index in range(len(masks)):
        groups.setdefault(find(index), []).append(index)

    for group in groups.values():
        bounds = _mask_bounds(masks[group[0]], size)
        for index in group[1:]:
            bounds = _union_bounds(bounds, _mask_bounds(masks[index], size))
        left, top, right, bottom = _expand_bounds(bounds, radius * 3, size)
        if left >= right or top >= bottom:
            continue
        local = np.zeros((bottom - top, right - left), dtype=np.uint8)
        for index in group:
            _stamp_mask(local, masks[index], left, top)
        local = close_l2(dilate_l2(local, radius), radius)
        region = output[top:bottom, left:right]
        region[local != 0] = 255
    return output


def _radius_for(size: tuple[int, int]) -> int:
    """``write_mask``: closing radius scales with the page's longest side."""
    width, height = size
    scaled = np.float32(max(width, height)) / np.float32(1024.0) * np.float32(6.0)
    return int(min(max(math.floor(float(scaled) + 0.5), 1), 255))


def write_masks(input: StageInput, edit: Edit, page: str, detections: list, size: tuple[int, int]) -> None:
    """``write_masks``: text mask asset, then drop the legacy bubble mask."""
    write_mask(input, edit, page, detections, size)
    edit.remove_asset(page, "bubble-mask")


def write_mask(input: StageInput, edit: Edit, page: str, detections: list, size: tuple[int, int]) -> None:
    """``write_mask``: closed text mask, preserved outside the scope region, stored as a PNG asset."""
    width, height = size
    if width > 0 and height > 0:
        mask = closed_mask_for(detections, "text", size, _radius_for(size))
    else:
        mask = mask_for(detections, "text", size)
    if input.region is not None:
        preserve_mask_outside_region(input, page, "text-mask", input.region, mask)
    buffer = io.BytesIO()
    Image.fromarray(mask).save(buffer, format="PNG", optimize=False)
    edit.set_asset(
        page,
        "text-mask",
        AssetInput(
            data=buffer.getvalue(),
            media_type="image/png",
            metadata=AssetMetadata(width=width, height=height, attributes={}),
        ),
    )


def preserve_mask_outside_region(input: StageInput, page: str, role: str, bounds: Bounds, mask: np.ndarray) -> None:
    """``preserve_mask_outside_region``: keep the previous mask wherever the region is not being re-detected."""
    previous_image = input.images.get(input.scene, page, role)
    previous = np.asarray(previous_image.convert("L"), dtype=np.uint8) if previous_image is not None else None
    if previous is not None and previous.shape != mask.shape:
        raise ProcessingError(f"existing {role} dimensions do not match page {page}")
    height, width = mask.shape
    xs = np.arange(width, dtype=np.float64)[None, :]
    ys = np.arange(height, dtype=np.float64)[:, None]
    outside = (
        (xs + 1.0 <= bounds.x)
        | (ys + 1.0 <= bounds.y)
        | (xs >= bounds.x + bounds.width)
        | (ys >= bounds.y + bounds.height)
    )
    replacement = previous if previous is not None else np.zeros_like(mask)
    mask[outside] = replacement[outside]


# ---- non-maximum suppression and layout order ---------------------------------------------------


def _area32(bbox) -> np.float32:
    left, top, right, bottom = (np.float32(v) for v in bbox)
    return max(right - left, F32_ZERO) * max(bottom - top, F32_ZERO)


def _intersection_area32(left_box, right_box) -> np.float32:
    l0, t0, r0, b0 = (np.float32(v) for v in left_box)
    l1, t1, r1, b1 = (np.float32(v) for v in right_box)
    return max(min(r0, r1) - max(l0, l1), F32_ZERO) * max(min(b0, b1) - max(t0, t1), F32_ZERO)


def intersection_over_union(left_box, right_box) -> np.float32:
    """``intersection_over_union`` in f32."""
    intersection = _intersection_area32(left_box, right_box)
    union = _area32(left_box) + _area32(right_box) - intersection
    if union <= F32_ZERO:
        return F32_ZERO
    return intersection / union


def containment(container, value) -> np.float32:
    """``containment``: fraction of ``value`` covered by ``container``."""
    value_area = _area32(value)
    if value_area <= F32_ZERO:
        return F32_ZERO
    return _intersection_area32(container, value) / value_area


def detection_order_key(detection) -> tuple:
    """``detection_order``: top edge ascending, then right-to-left, then label."""
    return (float(np.float32(detection.bbox[1])), -float(np.float32(detection.bbox[0])), detection.label)


def non_maximum_suppression(detections: list, threshold: np.float32) -> None:
    """``non_maximum_suppression``: per-label greedy suppression by IoU, highest score first."""
    detections.sort(key=lambda d: (-d.score, *detection_order_key(d)))
    kept: list = []
    for candidate in detections:
        suppressed = any(
            existing.label == candidate.label and intersection_over_union(existing.bbox, candidate.bbox) >= threshold
            for existing in kept
        )
        if not suppressed:
            kept.append(candidate)
    detections[:] = kept


# A text box this much inside a larger one is part of it (a line of its paragraph; a line's box
# can stick out of a paragraph box the detector drew narrower than the text).
NESTED_TEXT_CONTAINMENT = np.float32(0.75)


def suppress_nested_text(detections: list) -> None:
    """Not in Koharu: the layout model can find a paragraph and also its lines, and IoU
    suppression keeps all of them (a line is a small part of its paragraph), so the text would be
    read, translated and typeset twice. Lines inside a paragraph box are dropped. When the boxes
    inside a large one sit in different bubbles, the large box is the mistake and is dropped."""
    texts = [d for d in detections if d.label == "text"]
    bubbles = [d for d in detections if d.label == "bubble"]

    def home(text) -> Optional[int]:
        held = [i for i, bubble in enumerate(bubbles) if containment(bubble.bbox, text.bbox) >= np.float32(0.5)]
        return min(held, key=lambda i: _area32(bubbles[i].bbox)) if held else None

    dropped: set[int] = set()
    for outer in sorted(texts, key=lambda d: -_area32(d.bbox)):
        if id(outer) in dropped:
            continue
        inner = [
            text
            for text in texts
            if text is not outer and id(text) not in dropped and _area32(text.bbox) < _area32(outer.bbox) and containment(outer.bbox, text.bbox) >= NESTED_TEXT_CONTAINMENT
        ]
        if not inner:
            continue
        if len({home(text) for text in inner}) > 1:
            dropped.add(id(outer))
        else:
            dropped.update(id(text) for text in inner)
    detections[:] = [d for d in detections if id(d) not in dropped]


def sort_by_layout(detections: list) -> None:
    """``sort_by_layout``: reorder by the panel/bubble/text tree."""
    order = layout_order(detections)
    values = list(detections)
    detections[:] = [values[index] for index in order]


def _indices_with_label(detections: list, label: str) -> list[int]:
    return [index for index, d in enumerate(detections) if d.label == label]


def _best_container(detections: list, value: int, candidates: list[int]) -> Optional[int]:
    """``best_container``: smallest candidate covering at least half of the value's area."""
    best = None
    best_key = None
    for candidate in candidates:
        if containment(detections[candidate].bbox, detections[value].bbox) < np.float32(0.5):
            continue
        key = (float(_area32(detections[candidate].bbox)), *detection_order_key(detections[candidate]), candidate)
        if best_key is None or key < best_key:
            best, best_key = candidate, key
    return best


def layout_order(detections: list) -> list[int]:
    """``layout_order``: panels contain bubbles, bubbles contain text; depth-first in reading order."""
    panels = _indices_with_label(detections, "panel")
    bubbles = _indices_with_label(detections, "bubble")
    texts = _indices_with_label(detections, "text")

    panel_for_bubble: dict[int, Optional[int]] = {}
    for bubble in bubbles:
        panel_for_bubble[bubble] = _best_container(detections, bubble, panels)
    bubble_for_text: dict[int, Optional[int]] = {}
    panel_for_text: dict[int, Optional[int]] = {}
    for text in texts:
        bubble = _best_container(detections, text, bubbles)
        bubble_for_text[text] = bubble
        panel_for_text[text] = (
            panel_for_bubble[bubble] if bubble is not None and panel_for_bubble.get(bubble) is not None
            else _best_container(detections, text, panels)
        )

    parent: dict[int, Optional[int]] = {}
    for bubble in bubbles:
        parent[bubble] = panel_for_bubble[bubble]
    for text in texts:
        parent[text] = bubble_for_text[text] if bubble_for_text[text] is not None else panel_for_text[text]

    roots: list[int] = []
    children: dict[int, list[int]] = {index: [] for index in range(len(detections))}
    for index in range(len(detections)):
        owner = parent.get(index)
        if owner is not None:
            children[owner].append(index)
        else:
            roots.append(index)

    def sort_spatial(indices: list[int]) -> list[int]:
        return sorted(indices, key=lambda i: (*detection_order_key(detections[i]), -detections[i].score, i))

    roots = sort_spatial(roots)
    for siblings in children.values():
        siblings[:] = sort_spatial(siblings)

    order: list[int] = []
    stack = list(reversed(roots))
    while stack:
        index = stack.pop()
        order.append(index)
        stack.extend(reversed(children[index]))
    return order


# ---- containment of ink masks (dialogue membership) ---------------------------------------------


def mask_containment(container: LayoutMask, value: LayoutMask, bounds) -> np.float32:
    """``mask_containment``: fraction of the value's ink that lies inside the container's ink, within bounds."""
    if not _valid_mask(container) or not _valid_mask(value):
        return F32_ZERO
    if not all(math.isfinite(float(v)) for v in bounds):
        return F32_ZERO
    value_right = value.x + value.width
    value_bottom = value.y + value.height
    bl, bt, br, bb = (np.float32(v) for v in bounds)
    left = int(min(max(np.floor(bl), np.float32(value.x)), np.float32(value_right)))
    top = int(min(max(np.floor(bt), np.float32(value.y)), np.float32(value_bottom)))
    right = int(min(max(np.ceil(br), np.float32(value.x)), np.float32(value_right)))
    bottom = int(min(max(np.ceil(bb), np.float32(value.y)), np.float32(value_bottom)))
    if left >= right or top >= bottom:
        return F32_ZERO
    value_plane = _plane(value, left, top, right, bottom)
    container_plane = _plane(container, left, top, right, bottom)
    value_area = int(np.count_nonzero(value_plane))
    if value_area == 0:
        return F32_ZERO
    intersection = int(np.count_nonzero(value_plane & container_plane))
    return np.float32(intersection) / np.float32(value_area)
