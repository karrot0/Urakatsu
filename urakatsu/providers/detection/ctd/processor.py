"""Comic text detector pre- and post-processing: port of Koharu's ``comic_text_detector/processor.rs``.

Upstream: BallonsTranslator ``ctd/inference.py``, ``db_utils.py``, ``ctd/textmask.py``,
``utils/textblock.py`` and ``utils/imgproc_utils.py`` (commit 4bcc635c, cited by Koharu). Function
names follow the Rust ones (``extract_line_polygons``, ``minimum_area_rect``, ``refine_mask``, ...).

Inputs and outputs are numpy unless noted:

- Images are ``(H, W, 3)`` uint8 RGB arrays (a PIL image is accepted wherever an image is taken).
- Probability maps are ``(map_height, map_width)`` float arrays in [0, 1], row-major.
- Masks are ``(H, W)`` uint8 with 0 for background and 255 for text (intermediate masks may be
  non-binary, exactly where Rust keeps them ``u8``).
- Quads are ``(4, 2)`` arrays of ``(x, y)`` corners in destination pixels.

The public surface that a DBNet ResNet34 detector reuses (numpy only, no torch):
``extract_line_polygons``, ``group_text_lines``, ``refine_mask``, with the Koharu thresholds as
keyword defaults. Torch is imported only inside ``preprocess``, ``postprocess``,
``rearranged_inference`` and ``inference``.

Deviations from the Rust source, each deliberate:

- Geometry (``minimum_area_rect``, ``polygon_score``, the merge rules) runs in float64 where Rust
  uses ``f32``. Results differ only where a comparison lands within float rounding of a threshold.
- Contour tracing is a line-by-line port of imageproc's Suzuki-Abe ``find_contours_with_threshold``
  (outer and hole borders, same scan order). Only the foreground pixels are visited, which gives the
  same borders in the same order because foreground pixels never change to background.
- ``fast_image_resize`` is replaced by equivalent numpy/torch code: ``Interpolation(Bilinear)`` as
  ``torch`` bilinear without antialiasing (sample at ``(i + 0.5) * scale - 0.5``), ``Convolution(Box)``
  as exact area-weighted box averaging. Results can differ by one 8-bit level at rounding boundaries.
- Sorts: Rust ``sort_unstable`` is replaced by stable sorts. The two agree on short slices, which the
  quad and candidate sorts are; for longer lists they can order exact ties differently.
- Connected components use ``scipy.ndimage.label`` (same 4- or 8-connected partition, labelled in
  raster order of first pixel, as the Rust flood fill visits them).
- Luma uses imageproc's integer formula ``(2126 R + 7152 G + 722 B) // 10000`` from the ``image`` crate.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import ndimage

# Koharu's constants (processor.rs).
DBNET_BINARY_THRESHOLD = 0.3
LINE_SCORE_THRESHOLD = 0.6
MAX_LINE_CANDIDATES = 1000
LINE_UNCLIP_RATIO = 1.5
MASK_SCORE_THRESHOLD = 0.1
INPUT_SIZE = 1280
REFINE_WINDOW_RATIO = 2.5
FLOAT_EPSILON = float(np.finfo(np.float32).eps)

# Border directions in imageproc order: w, nw, n, ne, e, se, s, sw.
_DIFFS = ((-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1))
_OUTER = "outer"
_HOLE = "hole"

Point = tuple[float, float]
Quad = tuple[Point, Point, Point, Point]


@dataclass(frozen=True)
class LinePolygon:
    """One DBNet line candidate: ``quad`` is ``(4, 2)`` in destination pixels, ``score`` the mean shrink-map value inside it."""

    quad: np.ndarray
    score: float


@dataclass
class TextBlock:
    """``TextBlock``: ``xyxy`` ints, ``lines`` ``(4, 2)`` int32 quads, plus ``line_scores`` (Python-side, one per line)."""

    xyxy: tuple[int, int, int, int]
    lines: list[np.ndarray]
    language: str = "unknown"
    vertical: bool = False
    angle: int = 0
    detected_font_size: float = 0.0
    line_scores: list[float] = field(default_factory=list)


# ---- rounding and image helpers ---------------------------------------------------------------------


def _round_half_away(value):
    """Rust ``f64::round`` (ties away from zero), for scalars and arrays."""
    return np.sign(value) * np.floor(np.abs(value) + 0.5)


def _as_rgb(image) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        return np.ascontiguousarray(image[..., :3], dtype=np.uint8)
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _resize_bilinear(array: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """``Resizer`` with ``Interpolation(Bilinear)`` on u8 data: sample-based bilinear, no antialias."""
    if array.shape[0] == out_h and array.shape[1] == out_w:
        return array.copy()
    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(np.array(array, copy=True)).float()
    if array.ndim == 2:
        tensor = tensor[None, None]
    else:
        tensor = tensor.permute(2, 0, 1)[None]
    out = F.interpolate(tensor, size=(out_h, out_w), mode="bilinear", align_corners=False, antialias=False)
    out = out[0, 0] if array.ndim == 2 else out[0].permute(1, 2, 0)
    return out.round().clamp(0, 255).to(torch.uint8).numpy()


def _box_weights(in_size: int, out_size: int) -> np.ndarray:
    """Area-weighted box filter as a ``(out, in)`` matrix (``ResizeAlg::Convolution(Box)``)."""
    scale = in_size / out_size
    weights = np.zeros((out_size, in_size), dtype=np.float64)
    for index in range(out_size):
        start, end = index * scale, (index + 1) * scale
        for source in range(int(math.floor(start)), min(int(math.ceil(end)), in_size)):
            overlap = min(end, source + 1) - max(start, source)
            if overlap > 0:
                weights[index, source] = overlap / scale
    return weights


def _resize_box(array: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Box-filter resize of an ``(H, W, 3)`` uint8 image, rounded to uint8."""
    in_h, in_w = array.shape[:2]
    rows = _box_weights(in_h, out_h)
    cols = _box_weights(in_w, out_w)
    data = array.astype(np.float64)
    # Separable filter as two matrix products: rows first, then columns.
    out = (rows @ data.reshape(in_h, in_w * 3)).reshape(out_h, in_w, 3)
    out = (cols @ out.transpose(1, 0, 2).reshape(in_w, out_h * 3)).reshape(out_w, out_h, 3)
    return np.clip(np.round(out.transpose(1, 0, 2)), 0, 255).astype(np.uint8)


def _to_tensor(image: np.ndarray, device):
    """``image_to_tensor``: ``(1, 3, H, W)`` float in [0, 1] on ``device``."""
    import torch

    tensor = torch.from_numpy(np.array(image, copy=True)).to(device)
    return tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0


def _tensor_to_u8(tensor) -> np.ndarray:
    """``tensor_to_u8_vec``: clamp to [0, 1], scale, truncate to uint8 in the tensor's own dtype."""
    import torch

    return (tensor.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()


# ---- pre- and post-processing (torch) ---------------------------------------------------------------


def preprocess(image, device):
    """Letterbox to 1280 px (bilinear, top-left, bottom/right padding) and convert to a float tensor.

    Returns ``(pixel_values, (original_width, original_height, resized_width, resized_height))``.
    """
    rgb = _as_rgb(image)
    original_height, original_width = rgb.shape[:2]
    if original_width == 0 or original_height == 0:
        raise ValueError("empty image")
    scale = min(INPUT_SIZE / original_width, INPUT_SIZE / original_height)
    resized_width = max(int(round(original_width * scale)), 1)
    resized_height = max(int(round(original_height * scale)), 1)
    resized = _resize_bilinear(rgb, resized_height, resized_width)
    letterboxed = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    letterboxed[:resized_height, :resized_width] = resized
    return _to_tensor(letterboxed, device), (original_width, original_height, resized_width, resized_height)


def postprocess(maps, dimensions, source) -> tuple[np.ndarray, list[TextBlock]]:
    """``postprocess``: ``maps`` is the ``(1, 3, H, W)`` tensor ``cat(mask, shrink, thresh)``.

    Returns ``(mask (H, W) uint8 after dilation, text blocks)`` for the source image.
    """
    original_width, original_height, resized_width, resized_height = dimensions
    cropped = maps[:, :, :resized_height, :resized_width]
    raw_mask = _resize_bilinear(_tensor_to_u8(cropped[0, 0]), original_height, original_width)
    shrink = cropped[0, 1].float().cpu().numpy()
    lines = extract_line_polygons(shrink, resized_width, resized_height, original_width, original_height)
    return finalize_detection(source, raw_mask, lines)


def rearranged_inference(source, device, forward: Callable) -> Optional[tuple[np.ndarray, list[TextBlock]]]:
    """``rearranged_inference``: split very tall pages into 1280 px strips, run them in batches of four.

    Returns ``None`` when the page is not tall enough to need it (downscale ratio <= 2.5 or aspect <= 3).
    ``forward`` maps a ``(N, 3, 1280, 1280)`` tensor to ``(mask, line_maps)``.
    """
    import torch

    source = _as_rgb(source)
    source_height, source_width = source.shape[:2]
    if source_width == 0 or source_height == 0:
        raise ValueError("empty image")
    transposed = source_height < source_width
    width, height = (source_height, source_width) if transposed else (source_width, source_height)
    aspect_ratio = height / width
    downscale_ratio = height / INPUT_SIZE
    if downscale_ratio <= 2.5 or aspect_ratio <= 3.0:
        return None

    oriented = np.transpose(source, (1, 0, 2)) if transposed else source
    strips_per_composite = max((2 * INPUT_SIZE) // width, 2)
    patch_height = width * strips_per_composite
    patch_count = -(-height // patch_height)
    patch_step = (height - patch_height) // (patch_count - 1) if patch_count > 1 else 0
    patch_starts = [index * patch_step for index in range(patch_count)]
    composite_count = -(-patch_count // strips_per_composite)
    pad_count = composite_count * strips_per_composite - patch_count
    square_size = max(patch_height, INPUT_SIZE)

    composite_maps: list[np.ndarray] = []
    map_size = 0
    for first_composite in range(0, composite_count, 4):
        batch_len = min(4, composite_count - first_composite)
        tensors = []
        padding = []
        for offset in range(batch_len):
            composite_index = first_composite + offset
            composite = np.zeros((square_size, square_size, 3), dtype=np.uint8)
            for strip_index in range(strips_per_composite):
                patch_index = composite_index * strips_per_composite + strip_index
                if patch_index >= patch_count:
                    break
                top = patch_starts[patch_index]
                if transposed:
                    # Rust: composite(col = y, row = strip * width + x) = oriented(x, top + y).
                    block = oriented[top : top + patch_height, :width]
                    composite[strip_index * width : (strip_index + 1) * width, :patch_height] = block.transpose(1, 0, 2)
                else:
                    composite[:patch_height, strip_index * width : (strip_index + 1) * width] = oriented[top : top + patch_height, :width]
            resized = composite if square_size == INPUT_SIZE else _resize_box(composite, INPUT_SIZE, INPUT_SIZE)
            tensors.append(_to_tensor(resized, device))
            padding.append(max(INPUT_SIZE - patch_height, 0))
        mask, line_maps = forward(torch.cat(tensors, 0))
        maps = torch.cat([mask, line_maps], 1)
        for local_index, pad in enumerate(padding):
            single = maps[local_index : local_index + 1]
            if pad > 0:
                output_pad = single.shape[3] * pad // INPUT_SIZE
                keep = single.shape[3] - output_pad
                single = single[:, :, :keep, :keep]
            map_size = single.shape[3]
            composite_maps.append(single[0].float().cpu().numpy())

    output_step = patch_step * map_size // patch_height
    strip_width = map_size // strips_per_composite
    output_height = strip_width * height // width
    restored = np.zeros((3, output_height, strip_width), dtype=np.float32)
    patch_total = len(composite_maps) * strips_per_composite - pad_count
    for composite_index, composite_map in enumerate(composite_maps):
        for strip_index in range(strips_per_composite):
            patch_index = composite_index * strips_per_composite + strip_index
            if patch_index >= patch_total:
                break
            relative_top = patch_starts[patch_index] / height
            top = int(round(relative_top * output_height))
            bottom = min(top + map_size, output_height)
            left = strip_index * strip_width
            rows = bottom - top
            if transposed:
                block = composite_map[:, left : left + strip_width, 0:rows].transpose(0, 2, 1)
            else:
                block = composite_map[:, 0:rows, left : left + strip_width]
            restored[:, top:bottom, :] += block
            if patch_index > 0:
                overlap = map_size - output_step
                restored[:, top : min(top + overlap, output_height), :] /= 2.0

    if transposed:
        maps_out = restored.transpose(0, 2, 1)
        map_width, map_height = output_height, strip_width
    else:
        maps_out = restored
        map_width, map_height = strip_width, output_height
    raw_mask_map = (np.clip(maps_out[0], 0.0, 1.0) * 255.0).astype(np.uint8)
    mask = _resize_bilinear(raw_mask_map, source_height, source_width)
    lines = extract_line_polygons(maps_out[1], map_width, map_height, source_width, source_height)
    return finalize_detection(source, mask, lines)


def inference(image, device, forward: Callable) -> tuple[np.ndarray, list[TextBlock]]:
    """``ComicTextDetector::inference``: rearranged inference when the page needs it, else one pass."""
    import torch

    rgb = _as_rgb(image)
    rearranged = rearranged_inference(rgb, device, forward)
    if rearranged is not None:
        return rearranged
    pixel_values, dimensions = preprocess(rgb, device)
    mask, line_maps = forward(pixel_values)
    maps = torch.cat([mask, line_maps], 1)
    return postprocess(maps, dimensions, rgb)


def finalize_detection(source, raw_mask: np.ndarray, lines: list[LinePolygon]) -> tuple[np.ndarray, list[TextBlock]]:
    """``finalize_detection``: group lines, refine the mask around them, then dilate by an ellipse of radius 2."""
    blocks = group_text_lines(lines, raw_mask)
    refined = refine_mask(source, raw_mask, blocks)
    return dilate_binary(refined, 2, "ellipse"), blocks


# ---- DBNet decoding (numpy) -------------------------------------------------------------------------


@dataclass(frozen=True)
class RotatedRect:
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    cos: float
    sin: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    def corners(self, expand: float) -> np.ndarray:
        """``RotatedRect::corners``: the four corners, expanded by ``expand`` and rotated back to image space."""
        points = (
            (self.min_x - expand, self.min_y - expand),
            (self.max_x + expand, self.min_y - expand),
            (self.max_x + expand, self.max_y + expand),
            (self.min_x - expand, self.max_y + expand),
        )
        return np.array(
            [[x * self.cos - y * self.sin, x * self.sin + y * self.cos] for x, y in points],
            dtype=np.float64,
        )


def extract_line_polygons(
    prob_map,
    map_width: int,
    map_height: int,
    dest_width: int,
    dest_height: int,
    *,
    binary_threshold: float = DBNET_BINARY_THRESHOLD,
    score_threshold: float = LINE_SCORE_THRESHOLD,
    max_candidates: int = MAX_LINE_CANDIDATES,
    unclip_ratio: float = LINE_UNCLIP_RATIO,
) -> list[LinePolygon]:
    """``extract_line_polygons``: shrink map to text-line quads in destination coordinates.

    ``prob_map`` is ``(map_height, map_width)`` (flattened input is reshaped). Each contour of the
    thresholded map (outer and hole borders) becomes one candidate, filtered by minimum side length
    (2 px) and mean probability (``> score_threshold``), then unclipped and scaled to
    ``(dest_width, dest_height)``.
    """
    prob = np.asarray(prob_map, dtype=np.float32).reshape(map_height, map_width)
    bitmap = np.where(prob > np.float32(binary_threshold), 255, 0).astype(np.uint8)
    lines: list[LinePolygon] = []
    for contour in find_contours_with_threshold(bitmap, 0)[:max_candidates]:
        points = contour.points
        if len(points) < 3:
            continue
        rect = minimum_area_rect(points)
        if rect is None or min(rect.width, rect.height) < 2.0:
            continue
        score = polygon_score(prob, map_width, map_height, points)
        if score <= score_threshold:
            continue
        perimeter = 2.0 * (rect.width + rect.height)
        expand = rect.width * rect.height * unclip_ratio / perimeter if perimeter > FLOAT_EPSILON else 0.0
        quad = rect.corners(expand)
        quad[:, 0] = np.clip(_round_half_away(quad[:, 0] / map_width * dest_width), 0.0, dest_width)
        quad[:, 1] = np.clip(_round_half_away(quad[:, 1] / map_height * dest_height), 0.0, dest_height)
        lines.append(LinePolygon(quad=quad, score=float(score)))
    return lines


@dataclass(frozen=True)
class Contour:
    points: list[tuple[int, int]]
    border_type: str
    parent: Optional[int]


def find_contours_with_threshold(bitmap: np.ndarray, threshold: int = 0) -> list[Contour]:
    """imageproc ``find_contours_with_threshold``: Suzuki-Abe border following, outer and hole borders.

    Pixels strictly greater than ``threshold`` are foreground. Borders are returned in discovery
    order, which is the raster order of their first pixel, as in imageproc.
    """
    height, width = bitmap.shape
    values = (bitmap > threshold).astype(np.int64).ravel().tolist()
    foreground = np.flatnonzero(bitmap > threshold).tolist()
    diffs: deque = deque(_DIFFS)

    def at(x: int, y: int) -> int:
        return x + width * y

    def nonzero(x: int, y: int) -> bool:
        return 0 <= x < width and 0 <= y < height and values[x + width * y] != 0

    contours: list[Contour] = []
    curr_border_num = 1
    current_row = -1
    parent_border_num = 1
    for index in foreground:
        y, x = divmod(index, width)
        if y != current_row:
            current_row = y
            parent_border_num = 1
        if values[index] == 0:
            continue
        adjacency = None
        border_type = None
        if values[index] == 1 and x > 0 and values[at(x - 1, y)] == 0:
            adjacency, border_type = (x - 1, y), _OUTER
        elif values[index] > 0 and x + 1 < width and values[at(x + 1, y)] == 0:
            if values[index] > 1:
                parent_border_num = values[index]
            adjacency, border_type = (x + 1, y), _HOLE
        if adjacency is not None:
            curr_border_num += 1
            if parent_border_num > 1:
                parent_index = parent_border_num - 2
                parent_contour = contours[parent_index]
                if (border_type == _OUTER) != (parent_contour.border_type == _OUTER):
                    parent = parent_index
                else:
                    parent = parent_contour.parent
            else:
                parent = None

            contour_points: list[tuple[int, int]] = []
            curr = (x, y)
            _rotate_to_value(diffs, (adjacency[0] - x, adjacency[1] - y))
            pos1 = None
            for dx, dy in diffs:
                if nonzero(x + dx, y + dy):
                    pos1 = (x + dx, y + dy)
                    break
            if pos1 is not None:
                pos2 = pos1
                pos3 = curr
                while True:
                    contour_points.append(pos3)
                    _rotate_to_value(diffs, (pos2[0] - pos3[0], pos2[1] - pos3[1]))
                    pos4 = None
                    for dx, dy in reversed(diffs):
                        if nonzero(pos3[0] + dx, pos3[1] + dy):
                            pos4 = (pos3[0] + dx, pos3[1] + dy)
                            break
                    is_right_edge = False
                    for diff in reversed(diffs):
                        if diff == (pos4[0] - pos3[0], pos4[1] - pos3[1]):
                            break
                        if diff == (1, 0):
                            is_right_edge = True
                            break
                    if pos3[0] + 1 == width or is_right_edge:
                        values[at(*pos3)] = -curr_border_num
                    elif values[at(*pos3)] == 1:
                        values[at(*pos3)] = curr_border_num
                    if pos4 == curr and pos3 == pos1:
                        break
                    pos2 = pos3
                    pos3 = pos4
            else:
                contour_points.append((x, y))
                values[at(x, y)] = -curr_border_num
            contours.append(Contour(points=contour_points, border_type=border_type, parent=parent))
        if values[index] != 1:
            parent_border_num = abs(values[index])
    return contours


def _rotate_to_value(values: deque, value) -> None:
    """``rotate_to_value``: rotate the deque so that ``value`` is first."""
    values.rotate(-values.index(value))


def minimum_area_rect(points) -> Optional[RotatedRect]:
    """``minimum_area_rect``: smallest-area rotated rectangle around the convex hull, aligned to one hull edge."""
    hull = convex_hull(points)
    if len(hull) < 3:
        return None
    best: Optional[RotatedRect] = None
    best_area = math.inf
    for index in range(len(hull)):
        ax, ay = hull[index]
        bx, by = hull[(index + 1) % len(hull)]
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length <= FLOAT_EPSILON:
            continue
        cos, sin = dx / length, dy / length
        min_x = min_y = math.inf
        max_x = max_y = -math.inf
        for px, py in hull:
            x = px * cos + py * sin
            y = -px * sin + py * cos
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        area = (max_x - min_x) * (max_y - min_y)
        if area < best_area:
            best_area = area
            best = RotatedRect(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y, cos=cos, sin=sin)
    return best


def convex_hull(points) -> list[tuple[float, float]]:
    """``convex_hull``: monotone chain over the sorted, de-duplicated points (counter-clockwise)."""
    unique = sorted({(float(p[0]), float(p[1])) for p in points})
    if len(unique) <= 2:
        return unique
    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    lower.pop()
    upper.pop()
    return lower + upper


def _cross(origin, a, b) -> float:
    return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])


def polygon_score(prob_map, map_width: int, map_height: int, polygon) -> float:
    """``polygon_score``: mean of the map over the polygon's integer pixels (its bounding box, clamped to the map)."""
    prob = np.asarray(prob_map, dtype=np.float32).reshape(map_height, map_width)
    xs = [int(p[0]) for p in polygon]
    ys = [int(p[1]) for p in polygon]
    min_x = max(min(xs), 0)
    min_y = max(min(ys), 0)
    max_x = min(max(max(xs), 0), map_width - 1)
    max_y = min(max(max(ys), 0), map_height - 1)
    if max_x < min_x or max_y < min_y:
        return 0.0
    grid_x, grid_y = np.meshgrid(np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1))
    inside = _points_in_polygon(grid_x.astype(np.float64), grid_y.astype(np.float64), polygon)
    values = prob[min_y : max_y + 1, min_x : max_x + 1][inside]
    if values.size == 0:
        return 0.0
    return float(np.sum(values, dtype=np.float64) / values.size)


def _points_in_polygon(px: np.ndarray, py: np.ndarray, polygon) -> np.ndarray:
    """``point_in_polygon`` for arrays of points: points on an edge count as inside (even-odd rule otherwise)."""
    poly = [(float(p[0]), float(p[1])) for p in polygon]
    on_edge = np.zeros(px.shape, dtype=bool)
    inside = np.zeros(px.shape, dtype=bool)
    previous = poly[-1]
    for current in poly:
        ax, ay = previous
        bx, by = current
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        on_edge |= (np.abs(cross) < 1e-3) & (px >= min(ax, bx)) & (px <= max(ax, bx)) & (py >= min(ay, by)) & (py <= max(ay, by))
        if ay != by:
            crosses = ((by > py) != (ay > py)) & (px < (ax - bx) * (py - by) / (ay - by) + bx)
            inside ^= crosses
        previous = current
    return on_edge | inside


# ---- line grouping (numpy) --------------------------------------------------------------------------


@dataclass
class _WorkingBlock:
    """``WorkingBlock``: a growing group of line quads with its direction and font size."""

    bbox: tuple[float, float, float, float]
    lines: list[Quad]
    scores: list[float]
    vertical: bool
    angle: float = 0.0
    font_size: float = 0.0
    vector: Point = (0.0, 0.0)
    norm: float = 0.0
    merged: bool = False

    @classmethod
    def from_line(cls, quad: Quad, score: float) -> "_WorkingBlock":
        quad, vertical = _sort_line_quad(quad)
        block = cls(bbox=_quad_bbox(quad), lines=[quad], scores=[score], vertical=vertical)
        block.recalculate()
        return block

    def center(self) -> Point:
        return ((self.bbox[0] + self.bbox[2]) * 0.5, (self.bbox[1] + self.bbox[3]) * 0.5)

    def recalculate(self) -> None:
        self.bbox = _lines_bbox(self.lines)
        vertical_vector = [0.0, 0.0]
        horizontal_vector = [0.0, 0.0]
        for line in self.lines:
            top = _midpoint(line[0], line[1])
            right = _midpoint(line[1], line[2])
            bottom = _midpoint(line[2], line[3])
            left = _midpoint(line[3], line[0])
            vertical_vector[0] += bottom[0] - top[0]
            vertical_vector[1] += bottom[1] - top[1]
            horizontal_vector[0] += right[0] - left[0]
            horizontal_vector[1] += right[1] - left[1]
        vertical_norm = _vector_norm(vertical_vector)
        horizontal_norm = _vector_norm(horizontal_vector)
        count = len(self.lines)
        if self.vertical:
            self.vector = (vertical_vector[0], vertical_vector[1])
            self.norm = vertical_norm
            self.font_size = _round_half_away(horizontal_norm / count)
            self.angle = math.trunc(math.degrees(math.atan2(vertical_vector[1], vertical_vector[0]))) - 90.0
        else:
            self.vector = (horizontal_vector[0], horizontal_vector[1])
            self.norm = horizontal_norm
            self.font_size = _round_half_away(vertical_norm / count)
            self.angle = float(math.trunc(math.degrees(math.atan2(horizontal_vector[1], horizontal_vector[0]))))
        if abs(self.angle) < 3.0:
            self.angle = 0.0

    def into_public(self) -> TextBlock:
        return TextBlock(
            xyxy=tuple(int(math.trunc(value)) for value in self.bbox),
            lines=[np.array([[int(math.trunc(x)), int(math.trunc(y))] for x, y in line], dtype=np.int32) for line in self.lines],
            language="unknown",
            vertical=self.vertical,
            angle=int(math.trunc(self.angle)),
            detected_font_size=float(self.font_size),
            line_scores=list(self.scores),
        )


def group_text_lines(lines: Sequence[LinePolygon], mask: np.ndarray) -> list[TextBlock]:
    """``group_text_lines``: drop lines with little mask under them, group the rest into text blocks.

    ``lines`` are the ``LinePolygon`` candidates in destination pixels; ``mask`` is the ``(H, W)``
    uint8 map those quads are measured against. Returns blocks in reading order, with blocks mostly
    contained in an earlier block removed.
    """
    mask = np.asarray(mask)
    horizontal: list[_WorkingBlock] = []
    vertical: list[_WorkingBlock] = []
    for line in lines:
        quad = _as_quad(line.quad)
        bbox = _quad_bbox(quad)
        if _mean_mask(mask, bbox) < MASK_SCORE_THRESHOLD:
            continue
        block = _WorkingBlock.from_line(quad, float(line.score))
        (vertical if block.vertical else horizontal).append(block)
    horizontal.sort(key=lambda block: block.center()[1])
    vertical.sort(key=lambda block: -block.center()[0])

    grouped = _merge_text_lines(horizontal, 2.0)
    grouped.extend(_merge_text_lines(vertical, 1.7))
    grouped = _sort_regions(grouped)
    kept: list[_WorkingBlock] = []
    for block in grouped:
        area = max(_bbox_area(block.bbox), FLOAT_EPSILON)
        contained = any(_bbox_intersection(block.bbox, existing.bbox) / area > 0.9 for existing in kept)
        if not contained:
            kept.append(block)
    return [block.into_public() for block in kept]


def _merge_text_lines(blocks: list[_WorkingBlock], font_size_tolerance: float) -> list[_WorkingBlock]:
    if len(blocks) < 2:
        return blocks
    merged: list[_WorkingBlock] = []
    for index in range(len(blocks)):
        if blocks[index].merged:
            continue
        for other in range(index + 1, len(blocks)):
            _try_merge_text_line(blocks[index], blocks[other], font_size_tolerance)
        blocks[index].bbox = _lines_bbox(blocks[index].lines)
        merged.append(blocks[index])
    return merged


def _try_merge_text_line(block: _WorkingBlock, other: _WorkingBlock, font_size_tolerance: float) -> bool:
    if other.merged or block.font_size <= 0.0 or other.font_size <= 0.0:
        return False
    first_count = len(block.lines)
    other_count = len(other.lines)
    average_font_size = (block.font_size * first_count + other.font_size * other_count) / (first_count + other_count)
    cosine = _dot(block.vector, other.vector) / max(block.norm * other.norm, FLOAT_EPSILON)
    vector_sum = (block.vector[0] + other.vector[0], block.vector[1] + other.vector[1])
    first_bbox = _quad_bbox(block.lines[-1])
    other_bbox = _quad_bbox(other.lines[0])
    distance_x = max(first_bbox[0], other_bbox[0]) - min(first_bbox[2], other_bbox[2])
    distance_y = max(first_bbox[1], other_bbox[1]) - min(first_bbox[3], other_bbox[3])
    first_width = first_bbox[2] - first_bbox[0]
    other_width = other_bbox[2] - other_bbox[0]
    first_height = first_bbox[3] - first_bbox[1]
    other_height = other_bbox[3] - other_bbox[1]

    if not _quads_intersect(block.lines[-1], other.lines[0]):
        if block.vertical:
            if (
                distance_y > 0.0
                or distance_x > average_font_size * 0.8
                or abs(distance_y) / max(min(first_height, other_height), 1.0) < 0.4
            ):
                return False
        else:
            threshold = 0.6 if average_font_size < 24.0 else 0.5
            if (
                distance_x > 0.0
                or distance_y > average_font_size * threshold
                or abs(distance_x) / max(min(first_width, other_width), 1.0) < 0.3
            ):
                return False
        ratio = block.font_size / other.font_size
        if ratio > font_size_tolerance or (1.0 / ratio) > font_size_tolerance or abs(cosine) < 0.866:
            return False

    block.lines.extend(other.lines)
    block.scores.extend(other.scores)
    block.vector = vector_sum
    block.norm = _vector_norm(vector_sum)
    block.font_size = average_font_size
    block.angle = float(_round_half_away(math.degrees(math.atan2(vector_sum[1], vector_sum[0]))))
    if block.vertical:
        block.angle -= 90.0
    block.bbox = _lines_bbox(block.lines)
    other.merged = True
    return True


def _sort_regions(regions: list[_WorkingBlock]) -> list[_WorkingBlock]:
    right_to_left = any(region.vertical for region in regions)
    ordered = sorted(regions, key=lambda region: region.center()[1])
    output: list[_WorkingBlock] = []
    for region in ordered:
        insertion = None
        for index, existing in enumerate(output):
            if region.center()[1] > existing.bbox[3]:
                continue
            if region.center()[1] < existing.bbox[1]:
                insertion = min(index + 1, len(output))
                break
            if (right_to_left and region.center()[0] > existing.center()[0]) or (
                not right_to_left and region.center()[0] < existing.center()[0]
            ):
                insertion = index
                break
        if insertion is not None:
            output.insert(insertion, region)
        else:
            output.append(region)
    return output


def _sort_line_quad(quad: Quad) -> tuple[Quad, bool]:
    edge_a = _distance(quad[0], quad[1])
    edge_b = _distance(quad[1], quad[2])
    square = abs(edge_a - edge_b) < 1e-3
    if edge_a >= edge_b:
        long_vector = (quad[1][0] - quad[0][0], quad[1][1] - quad[0][1])
    else:
        long_vector = (quad[2][0] - quad[1][0], quad[2][1] - quad[1][1])
    vertical = not square and abs(long_vector[0]) * 1.2 <= abs(long_vector[1])
    if vertical:
        points = sorted(quad, key=lambda p: p[1])
        top = sorted(points[:2], key=lambda p: p[0])
        bottom = sorted(points[2:], key=lambda p: -p[0])
        return (top[0], top[1], bottom[0], bottom[1]), True
    points = sorted(quad, key=lambda p: p[0])
    left = sorted(points[:2], key=lambda p: p[1])
    right = sorted(points[2:], key=lambda p: p[1])
    return (left[0], right[0], right[1], left[1]), False


def _quads_intersect(a: Quad, b: Quad) -> bool:
    """Separating-axis test over the edges of both quads."""
    for polygon in (a, b):
        for index in range(4):
            p0 = polygon[index]
            p1 = polygon[(index + 1) % 4]
            edge = (p1[0] - p0[0], p1[1] - p0[1])
            axis = (-edge[1], edge[0])
            a_min, a_max = _project_quad(a, axis)
            b_min, b_max = _project_quad(b, axis)
            if not (a_max >= b_min and b_max >= a_min):
                return False
    return True


def _project_quad(quad: Quad, axis: Point) -> tuple[float, float]:
    values = [point[0] * axis[0] + point[1] * axis[1] for point in quad]
    return min(values), max(values)


def _mean_mask(mask: np.ndarray, bbox) -> float:
    x1 = max(int(math.floor(bbox[0])), 0)
    y1 = max(int(math.floor(bbox[1])), 0)
    x2 = int(min(max(math.ceil(bbox[2]), 0), mask.shape[1]))
    y2 = int(min(max(math.ceil(bbox[3]), 0), mask.shape[0]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = mask[y1:y2, x1:x2]
    return float(np.sum(region, dtype=np.float64) / region.size / 255.0)


def _as_quad(points) -> Quad:
    return tuple((float(p[0]), float(p[1])) for p in points)  # type: ignore[return-value]


def _midpoint(a, b) -> Point:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def _distance(a, b) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _vector_norm(vector) -> float:
    return math.hypot(vector[0], vector[1])


def _quad_bbox(quad) -> tuple[float, float, float, float]:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return (min(xs), min(ys), max(xs), max(ys))


def _lines_bbox(lines) -> tuple[float, float, float, float]:
    boxes = [_quad_bbox(line) for line in lines]
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _bbox_area(bbox) -> float:
    return max(bbox[2] - bbox[0], 0.0) * max(bbox[3] - bbox[1], 0.0)


def _bbox_intersection(a, b) -> float:
    return max(min(a[2], b[2]) - max(a[0], b[0]), 0.0) * max(min(a[3], b[3]) - max(a[1], b[1]), 0.0)


# ---- text mask refinement (numpy) -------------------------------------------------------------------


def refine_mask(image, predicted_mask: np.ndarray, blocks: Sequence[TextBlock]) -> np.ndarray:
    """``refine_mask``: per block, re-derive the text pixels from image colour and keep the candidate closest to the prediction.

    ``image`` is the source RGB image; ``predicted_mask`` the ``(H, W)`` uint8 mask. For each block the
    window is its bbox grown by ``REFINE_WINDOW_RATIO``; inside it, colour candidates are merged and OR-ed
    into the result. Returns ``(H, W)`` uint8, 0 or 255.
    """
    rgb = _as_rgb(image)
    predicted = np.asarray(predicted_mask, dtype=np.uint8)
    height, width = predicted.shape
    refined = np.zeros((height, width), dtype=np.uint8)
    for block in blocks:
        x1, y1, x2, y2 = _enlarge_window([float(v) for v in block.xyxy], width, height, REFINE_WINDOW_RATIO)
        if x2 <= x1 or y2 <= y1:
            continue
        image_crop = rgb[y1:y2, x1:x2]
        mask_crop = predicted[y1:y2, x1:x2]
        candidates = mask_candidates(image_crop, mask_crop)
        merged = merge_mask_candidates(candidates, mask_crop)
        region = refined[y1:y2, x1:x2]
        region[merged != 0] = 255
    return refined


def _enlarge_window(bbox, image_width: int, image_height: int, ratio: float) -> tuple[int, int, int, int]:
    """``enlarge_window``: grow a box by a margin derived from its area so that the grown area is ``ratio`` times the original."""
    x1 = int(min(max(math.floor(bbox[0]), 0), image_width))
    y1 = int(min(max(math.floor(bbox[1]), 0), image_height))
    x2 = int(min(max(math.ceil(bbox[2]), 0), image_width))
    y2 = int(min(max(math.ceil(bbox[3]), 0), image_height))
    width = float(max(x2 - x1, 0))
    height = float(max(y2 - y1, 0))
    if width <= 0.0 or height <= 0.0:
        return (0, 0, 0, 0)
    b = width + height
    c = (1.0 - ratio) * width * height
    root = (-b + math.sqrt(b * b - 4.0 * c)) * 0.5
    requested = max(int(_round_half_away(root * 0.5)), 0)
    delta_x = max(min(requested, x1, image_width - x2), 0)
    delta_y = max(min(requested, y1, image_height - y2), 0)
    return (x1 - delta_x, y1 - delta_y, x2 + delta_x, y2 + delta_y)


@dataclass
class _MaskCandidate:
    mask: np.ndarray
    xor: int


def mask_candidates(image: np.ndarray, predicted: np.ndarray) -> list[_MaskCandidate]:
    """``mask_candidates``: colour-range masks from the palette, plus the best Otsu threshold per channel."""
    candidates = top_color_masks(image, predicted)
    otsu = []
    for channel in range(3):
        values = image[:, :, channel]
        level = otsu_level(values)
        thresholded = np.where(values > level, 255, 0).astype(np.uint8)
        otsu.append(best_polarity(thresholded, predicted))
    otsu.sort(key=lambda candidate: candidate.xor)
    if otsu:
        candidates.append(otsu[0])
    return candidates


def top_color_masks(image: np.ndarray, predicted: np.ndarray) -> list[_MaskCandidate]:
    """``top_color_masks``: histogram of luma under the eroded prediction; up to three dominant bands, 60 levels wide."""
    red = image[:, :, 0].astype(np.int64)
    green = image[:, :, 1].astype(np.int64)
    blue = image[:, :, 2].astype(np.int64)
    gray = ((2126 * red + 7152 * green + 722 * blue) // 10000).astype(np.uint8)
    eroded = erode_gray(predicted, 1, "square")
    candidate_pixels = gray[eroded > 127].astype(np.float64)
    if candidate_pixels.size == 0:
        return []
    minimum = float(candidate_pixels.min())
    maximum = float(candidate_pixels.max())
    if minimum == maximum:
        minimum -= 0.5
        maximum += 0.5
    bin_width = (maximum - minimum) / 255.0
    bins = np.minimum(np.floor((candidate_pixels - minimum) / bin_width).astype(np.int64), 254)
    histogram = np.bincount(bins, minlength=255)[:255].tolist()
    order = sorted(range(255), key=lambda index: (-histogram[index], index))
    tolerance = float(sum(histogram)) * 0.001
    selected: list[float] = []
    for index in order:
        color = minimum + index * bin_width
        if all(abs(existing - color) > 10.0 for existing in selected):
            selected.append(color)
        if len(selected) >= 3 or float(histogram[index]) < tolerance:
            break
    results = []
    for color in selected:
        top = min(color + 30.0, 255.0)
        bottom = top - 60.0
        thresholded = np.where((gray >= bottom) & (gray <= top), 255, 0).astype(np.uint8)
        results.append(best_polarity(thresholded, predicted))
    return results


def best_polarity(mask: np.ndarray, predicted: np.ndarray) -> _MaskCandidate:
    """``best_polarity``: keep the mask or its inverse, whichever differs less from the prediction (xor sum)."""
    normal_xor = int(np.sum(np.bitwise_xor(mask, predicted), dtype=np.int64))
    inverted = (np.uint8(255) - mask).astype(np.uint8)
    inverted_xor = int(np.sum(np.bitwise_xor(inverted, predicted), dtype=np.int64))
    if inverted_xor < normal_xor:
        return _MaskCandidate(mask=inverted, xor=inverted_xor)
    return _MaskCandidate(mask=mask, xor=normal_xor)


def otsu_level(values: np.ndarray) -> int:
    """imageproc ``otsu_level``: the threshold maximising between-class variance (first maximum wins)."""
    histogram = np.bincount(values.ravel(), minlength=256).tolist()
    total_weight = values.size
    total_pixel_sum = float(sum(t * count for t, count in enumerate(histogram)))
    background_pixel_sum = 0.0
    background_weight = 0
    largest_variance = 0.0
    best = 0
    for threshold, count in enumerate(histogram):
        background_weight += count
        if background_weight == 0:
            continue
        foreground_weight = total_weight - background_weight
        if foreground_weight == 0:
            break
        background_pixel_sum += threshold * count
        foreground_pixel_sum = total_pixel_sum - background_pixel_sum
        background_mean = background_pixel_sum / background_weight
        foreground_mean = foreground_pixel_sum / foreground_weight
        variance = background_weight * foreground_weight * (background_mean - foreground_mean) ** 2
        if variance > largest_variance:
            largest_variance = variance
            best = threshold
    return best


def merge_mask_candidates(candidates: list[_MaskCandidate], predicted: np.ndarray) -> np.ndarray:
    """``merge_mask_candidates``: add the components of each candidate that agree with the prediction, then fill small holes."""
    candidates = sorted(candidates, key=lambda candidate: candidate.xor)
    eroded = erode_gray(predicted, 1, "ellipse")
    predicted_binary = np.where(eroded > 60, 255, 0).astype(np.uint8)
    merged = np.zeros(predicted_binary.shape, dtype=np.uint8)
    for candidate in candidates:
        for component in connected_components(candidate.mask, foreground_white=True, diagonal=True):
            bbox_area = (component.max_x - component.min_x + 1) * (component.max_y - component.min_y + 1)
            if bbox_area >= 3:
                accept_component(merged, predicted_binary, component)
    merged = dilate_binary(merged, 2, "square")

    holes = connected_components(merged, foreground_white=False, diagonal=True)
    areas = sorted(component.size for component in holes)
    if len(areas) > 1:
        area_threshold = areas[-2]
    else:
        area_threshold = areas[-1] if areas else 0
    for component in holes:
        if component.size < area_threshold:
            accept_component(merged, predicted_binary, component)
    return merged


@dataclass(frozen=True)
class Component:
    mask: np.ndarray  # bool, shaped like the bounding box
    min_x: int
    min_y: int
    max_x: int
    max_y: int
    size: int


def connected_components(image: np.ndarray, foreground_white: bool, diagonal: bool) -> list[Component]:
    """Components of nonzero (``foreground_white``) or zero pixels, in raster order of their first pixel."""
    selected = (image != 0) if foreground_white else (image == 0)
    structure = np.ones((3, 3), dtype=bool) if diagonal else ndimage.generate_binary_structure(2, 1)
    labels, count = ndimage.label(selected, structure=structure)
    components = []
    if count == 0:
        return components
    slices = ndimage.find_objects(labels)
    for label_index, box in enumerate(slices, start=1):
        if box is None:
            continue
        local = labels[box] == label_index
        components.append(
            Component(
                mask=local,
                min_x=box[1].start,
                min_y=box[0].start,
                max_x=box[1].stop - 1,
                max_y=box[0].stop - 1,
                size=int(np.count_nonzero(local)),
            )
        )
    return components


def accept_component(merged: np.ndarray, predicted: np.ndarray, component: Component) -> None:
    """``accept_component``: paint the component into ``merged`` when that lowers the xor against the prediction."""
    region = (slice(component.min_y, component.max_y + 1), slice(component.min_x, component.max_x + 1))
    current = merged[region]
    in_component = component.mask
    candidate = np.where((current != 0) | in_component, 255, 0).astype(np.uint8)
    reference = predicted[region]
    before = int(np.sum(np.bitwise_xor(current, reference), dtype=np.int64))
    after = int(np.sum(np.bitwise_xor(candidate, reference), dtype=np.int64))
    if after < before:
        current[in_component] = 255


def erode_gray(image: np.ndarray, radius: int, shape: str) -> np.ndarray:
    """Grayscale erosion (minimum over the kernel; pixels outside the image do not count)."""
    if radius == 0:
        return image.copy()
    return ndimage.grey_erosion(image, footprint=_kernel(radius, shape), mode="constant", cval=255)


def dilate_binary(image: np.ndarray, radius: int, shape: str) -> np.ndarray:
    """Binary dilation of nonzero pixels; the result is 0 or 255."""
    if radius == 0:
        return image.copy()
    dilated = ndimage.binary_dilation(image != 0, structure=_kernel(radius, shape), border_value=0)
    return np.where(dilated, 255, 0).astype(np.uint8)


def _kernel(radius: int, shape: str) -> np.ndarray:
    """``kernel_offsets`` as a boolean footprint: ``"square"`` or ``"ellipse"``."""
    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype=bool)
    for dy in range(-radius, radius + 1):
        if shape == "square":
            half_width = radius
        else:
            if radius == 0:
                half_width = 0
            else:
                half_width = int(_round_half_away(radius * math.sqrt(max(1.0 - (dy * dy) / (radius * radius), 0.0))))
        for dx in range(-radius, radius + 1):
            kernel[dy + radius, dx + radius] = abs(dx) <= half_width
    return kernel
