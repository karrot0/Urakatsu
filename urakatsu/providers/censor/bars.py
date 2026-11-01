"""Censorship-bar geometry for Er0mangaSeg scores. Not in Koharu's network; ported from the Koharu fork's ``bars.rs``.

The checkpoint scores a quarter-resolution logit grid that is interpolated back onto the page, so
confident pixels often cover only part of a bar, while thick black panel borders score as highly as
real bars. Thresholding the map and keeping every dark pixel would either lose partly scored bars
or need a strict bar-shape test, and would still mask borders. The refinement works on three levels
per pixel and only on dark ink the model supports:

1. Hysteresis: dark pixels with weak support join the confident ink they touch, so a bar scored only
   around its centre is recovered to its ends.
2. An opening with a small disk detaches strokes thinner than a bar (hair lines, outlines).
3. Stacks: elongated components lying side by side, roughly parallel and a few bar widths apart, form
   a stack. Three or more are censorship. Fragments of one collinear border and borders far apart
   never qualify.
4. Completion: rings, ladders and caps drawn from touching bars fail the elongation test, so
   components of comparable stroke width within a few bar widths of an accepted stack join it.
5. The accepted core regrows over the ink the opening removed and over its anti-aliased rim, so the
   mask covers whole bars.

The fork stops at stacks, so a page whose censor is one bar comes back empty. Here a lone component
also counts when it is unmistakably one: confidently scored, a solid rectangle, stubby and thick for
the page. Each limit keeps out something the model scores as highly on real pages: panel gutters are
far longer, spiked black balloons are not convex, and bold lettering is thinner.

Levels are 0 (unscored), ``SUPPORTED`` (above the weak threshold) or ``CONFIDENT`` (above the
caller's threshold). ``bar_mask`` returns a full-page uint8 mask, or None when the page carries no
bars. Pixel work is NumPy and SciPy on whole arrays; Python loops only visit the handful of
components that survive the shape tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

# Pixels the model scored above the weak threshold, and above the caller's threshold.
SUPPORTED = 1
CONFIDENT = 2

SUPPORT_RATIO = 0.5
SUPPORT_FLOOR = 0.2

DARK_LUMA = 80
HALO_LUMA = 128
HALO_RADIUS = 2
MIN_CONFIDENT_PIXELS = 16

MIN_COMPONENT_AREA = 20
MIN_THICKNESS = 2.0
MIN_ELONGATION = 2.2

# The opening radius grows with the page so the same stroke thickness ratio is removed at every
# resolution; bars stay far wider than the disk.
OPENING_EXTENT = 900.0
MAX_OPENING_RADIUS = 4

MIN_STACK_BARS = 3
# Combined bar length a stack needs on a page whose longest side is REFERENCE_EXTENT; scaled
# linearly (within 0.5..2.0) for other pages.
MIN_STACK_LENGTH = 128.0
REFERENCE_EXTENT = 1800.0
# Centre separation across the bar axes, in bar widths. The lower bound rejects collinear fragments
# of one border.
MIN_SEPARATION = 0.6
MAX_SEPARATION = 5.0
# Centre offset along the axes, as a fraction of the mean bar length.
MAX_OFFSET = 0.75
# Neighbouring bars may diverge as a fan: the tolerance is the angle at which two bars drift apart by
# DIVERGENCE widths over the shorter bar's length, kept within these degrees.
MIN_ANGLE_TOLERANCE = 12.0
MAX_ANGLE_TOLERANCE = 30.0
DIVERGENCE = 3.5

# A lone bar, not in the fork. Sizes are fractions of the page's longest side; the fill compares the
# component with its convex hull, and the seeds are the share of it scored confident.
LONE_MIN_THICKNESS = 0.02
LONE_MIN_LENGTH = 0.06
LONE_MAX_ELONGATION = 10.0
LONE_MIN_FILL = 0.95
LONE_MIN_SEEDS = 0.9

# Completion limits, relative to the accepted stack.
ATTACH_GAP = 4.0
ATTACH_LENGTH = 2.5
ATTACH_AREA = 4.0
WIDTH_BAND = (0.5, 2.0)
# Junctions widen a stroke by at most this factor (a perpendicular crossing reaches about 1.41);
# anything wider is a solid region, not a bar network.
JUNCTION_BULGE = 1.6

_EIGHT = np.ones((3, 3), dtype=bool)
_U32_MAX = int(np.iinfo(np.uint32).max)


def support_threshold(threshold: float) -> float:
    """Weak threshold derived from the caller's confident threshold.

    Computed in float32, as the Rust does, so comparisons against float32 scores agree exactly.
    """
    value = np.float32(threshold) * np.float32(SUPPORT_RATIO)
    value = max(value, np.float32(SUPPORT_FLOOR))
    return float(min(value, np.float32(threshold)))


def bar_mask(rgb: np.ndarray, levels: np.ndarray) -> Optional[np.ndarray]:
    """Full-page censorship-bar mask (HxW uint8, 255 inside accepted bars), or None without a stack.

    ``rgb`` is the original page (HxWx3 uint8); ``levels`` holds ``SUPPORTED`` / ``CONFIDENT`` per
    pixel and 0 elsewhere.
    """
    rgb = np.asarray(rgb)
    levels = np.asarray(levels)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must be HxWx3, got shape {rgb.shape}")
    if levels.shape != rgb.shape[:2]:
        raise ValueError(f"levels must match the page: {levels.shape} != {rgb.shape[:2]}")
    height, width = levels.shape

    # Work only in the window that holds every scored pixel; the rest of the page is background.
    window = _score_window(levels)
    if window is None:
        return None
    top, left, h, w = window
    scores = levels[top : top + h, left : left + w]
    luma = _luma(rgb[top : top + h, left : left + w])

    # Dark pixels the model supports, and the confident seeds among them.
    supported = (scores >= SUPPORTED) & (luma <= DARK_LUMA)
    confident = supported & (scores >= CONFIDENT)
    if int(np.count_nonzero(confident)) < MIN_CONFIDENT_PIXELS:
        return None

    # Hysteresis: keep supported ink components that hold enough seeds.
    ink_labels, ink_count = ndimage.label(supported, structure=_EIGHT)
    seeds = np.bincount(ink_labels[confident], minlength=ink_count + 1)
    keep = seeds >= MIN_CONFIDENT_PIXELS
    keep[0] = False
    if not keep.any():
        return None
    ink = keep[ink_labels]

    # The radius comes from the full page, not the window, as in the Rust.
    extent = max(width, height)
    radius = _opening_radius(extent)
    structure = _disk(radius)
    # Erosion treats pixels outside the window as background, so ink touching the window edge is
    # eroded too (Koharu's behaviour).
    core = ndimage.binary_dilation(
        ndimage.binary_erosion(ink, structure, border_value=0), structure, border_value=0
    )
    labels, count = ndimage.label(core, structure=_EIGHT)
    shapes = _measure(labels, count)
    minimum_length = MIN_STACK_LENGTH * min(max(extent / REFERENCE_EXTENT, 0.5), 2.0)
    stacks = _find_stacks(shapes, minimum_length)
    lone = _lone_bars(labels, shapes, confident, extent)
    if not stacks and not lone:
        return None

    accepted = np.zeros(count + 1, dtype=bool)
    for stack in stacks:
        accepted[shapes.label[np.asarray(stack, dtype=np.intp)]] = True
    for stack in stacks:
        _attach_neighbours(labels, shapes, stack, accepted)
    accepted[shapes.label[np.asarray(lone, dtype=np.intp)]] = True

    # Regrow the accepted core over the ink the opening removed, then over the anti-aliased rim.
    # The halo is taken from the painted set before it grows, so the rim does not cascade.
    painted = ndimage.binary_dilation(accepted[labels], _disk(radius + 1), border_value=0) & ink
    halo = ndimage.binary_dilation(painted, _disk(HALO_RADIUS), border_value=0) & (luma <= HALO_LUMA)
    painted |= halo

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top : top + h, left : left + w] = np.where(painted, 255, 0)
    return mask


def squared_distance(seeds: np.ndarray) -> np.ndarray:
    """Exact squared Euclidean distance from every pixel to the nearest seed (uint32, saturating).

    With no seed at all every value saturates at ``u32::MAX``, as in the Rust.
    """
    seeds = np.asarray(seeds, dtype=bool)
    if not seeds.any():
        return np.full(seeds.shape, _U32_MAX, dtype=np.uint32)
    # Distance is measured to the nearest zero of the input, so the inverted seeds give the seeds.
    nearest = ndimage.distance_transform_edt(~seeds, return_distances=False, return_indices=True)
    rows, cols = np.indices(seeds.shape)
    distance = (cols - nearest[1]) ** 2 + (rows - nearest[0]) ** 2
    return np.minimum(distance, _U32_MAX).astype(np.uint32)


@dataclass(frozen=True)
class _Shapes:
    """Components of at least MIN_COMPONENT_AREA pixels, measured as the Rust ``Shape`` does.

    Arrays are indexed by shape position; ``label`` is the component's label in the core image.
    """

    label: np.ndarray
    area: np.ndarray
    min_x: np.ndarray
    min_y: np.ndarray
    max_x: np.ndarray
    max_y: np.ndarray
    center_x: np.ndarray
    center_y: np.ndarray
    length: np.ndarray
    thickness: np.ndarray
    cos: np.ndarray
    sin: np.ndarray
    is_bar: np.ndarray


def _score_window(levels: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """(top, left, height, width) of the smallest window holding every scored pixel."""
    rows = np.flatnonzero(levels.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(levels.any(axis=0))
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    return top, left, bottom - top, right - left


def _luma(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luma with the integer arithmetic the pipeline has always used (floor division)."""
    red = rgb[..., 0].astype(np.int64)
    green = rgb[..., 1].astype(np.int64)
    blue = rgb[..., 2].astype(np.int64)
    return (red * 299 + green * 587 + blue * 114) // 1000


def _round_half_away(value: float) -> int:
    """``f64::round``: halves go away from zero (Python's ``round`` goes to even)."""
    magnitude = abs(value)
    base = math.floor(magnitude)
    if magnitude - base >= 0.5:
        base += 1
    return int(math.copysign(base, value))


def _opening_radius(extent: int) -> int:
    return min(max(_round_half_away(extent / OPENING_EXTENT), 1), MAX_OPENING_RADIUS)


def _disk(radius: int) -> np.ndarray:
    """Digital disk ``dx^2 + dy^2 <= r(r + 1)`` as a boolean structuring element."""
    span = np.arange(-radius, radius + 1)
    dy, dx = np.meshgrid(span, span, indexing="ij")
    return dx * dx + dy * dy <= radius * (radius + 1)


def _median(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64))
    count = values.size
    if count == 0:
        return 0.0
    middle = count // 2
    if count % 2 == 1:
        return float(values[middle])
    return float((values[middle - 1] + values[middle]) / 2.0)


def _measure(labels: np.ndarray, count: int) -> _Shapes:
    """Shape moments of every component with at least MIN_COMPONENT_AREA pixels.

    The Rust accumulates integer sums and converts them; float64 sums of integers are exact at
    page scale, so the same covariance arithmetic follows.
    """
    width = labels.shape[1]
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    flat = np.flatnonzero(labels)
    owners = labels.ravel()[flat]
    xs = (flat % width).astype(np.float64)
    ys = (flat // width).astype(np.float64)
    sum_x = np.bincount(owners, weights=xs, minlength=count + 1)
    sum_y = np.bincount(owners, weights=ys, minlength=count + 1)
    sum_xx = np.bincount(owners, weights=xs * xs, minlength=count + 1)
    sum_yy = np.bincount(owners, weights=ys * ys, minlength=count + 1)
    sum_xy = np.bincount(owners, weights=xs * ys, minlength=count + 1)

    ids = np.flatnonzero(areas[1:] >= MIN_COMPONENT_AREA) + 1
    boxes = ndimage.find_objects(labels, max_label=count)
    box_list = [boxes[index - 1] for index in ids]
    min_y = np.array([box[0].start for box in box_list], dtype=np.int64)
    min_x = np.array([box[1].start for box in box_list], dtype=np.int64)
    max_y = np.array([box[0].stop - 1 for box in box_list], dtype=np.int64)
    max_x = np.array([box[1].stop - 1 for box in box_list], dtype=np.int64)

    area = areas[ids].astype(np.float64)
    center_x = sum_x[ids] / area
    center_y = sum_y[ids] / area
    cov_xx = sum_xx[ids] / area - center_x * center_x
    cov_yy = sum_yy[ids] / area - center_y * center_y
    cov_xy = sum_xy[ids] / area - center_x * center_y
    trace = cov_xx + cov_yy
    spread = np.sqrt((cov_xx - cov_yy) * (cov_xx - cov_yy) + 4.0 * (cov_xy * cov_xy))
    major = np.maximum((trace + spread) / 2.0, 0.0)
    minor = np.maximum((trace - spread) / 2.0, 0.0)
    # A rectangle of length L and width T has second moments L^2/12 and T^2/12, so these
    # square-root forms recover L and T.
    angle = 0.5 * np.arctan2(2.0 * cov_xy, cov_xx - cov_yy)
    length = np.sqrt(12.0 * major)
    thickness = np.sqrt(12.0 * minor)
    return _Shapes(
        label=ids.astype(np.int64),
        area=area,
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        center_x=center_x,
        center_y=center_y,
        length=length,
        thickness=thickness,
        cos=np.cos(angle),
        sin=np.sin(angle),
        is_bar=(thickness >= MIN_THICKNESS) & (length >= MIN_ELONGATION * thickness),
    )


def _side_by_side(
    first: int,
    cos: np.ndarray,
    sin: np.ndarray,
    center_x: np.ndarray,
    center_y: np.ndarray,
    thickness: np.ndarray,
    length: np.ndarray,
) -> np.ndarray:
    """``side_by_side`` of bar ``first`` against every bar, as a boolean row.

    Vectorised over the second bar, with the Rust's operation order kept so the thresholds fall
    on the same side.
    """
    alignment = cos[first] * cos + sin[first] * sin
    sign = np.where(alignment >= 0.0, 1.0, -1.0)
    axis_x = cos[first] + sign * cos
    axis_y = sin[first] + sign * sin
    norm = np.hypot(axis_x, axis_y)
    usable = norm >= 1e-9
    # Unusable rows are masked out below; the placeholder keeps the division finite.
    norm = np.where(usable, norm, 1.0)
    axis_x = axis_x / norm
    axis_y = axis_y / norm
    dx = center_x - center_x[first]
    dy = center_y - center_y[first]
    along = np.abs(dx * axis_x + dy * axis_y)
    across = np.abs(-dx * axis_y + dy * axis_x)
    widest = np.maximum(thickness[first], thickness)
    shorter = np.minimum(length[first], length)
    tolerance = np.arcsin(np.minimum(DIVERGENCE * widest / shorter, 1.0))
    tolerance = np.clip(
        tolerance, math.radians(MIN_ANGLE_TOLERANCE), math.radians(MAX_ANGLE_TOLERANCE)
    )
    angle = np.arccos(np.minimum(np.abs(alignment), 1.0))
    return (
        usable
        & (angle <= tolerance)
        & (across >= MIN_SEPARATION * widest)
        & (across <= MAX_SEPARATION * widest)
        & (along <= MAX_OFFSET * (length[first] + length) / 2.0)
    )


def _find_stacks(shapes: _Shapes, minimum_length: float) -> list[list[int]]:
    """Chains of side-by-side bars that are numerous and long enough. Members are shape positions.

    The relation is symmetric, so the chain is a connected component; the depth-first walk keeps
    the Rust's visiting order so the length sum is added in the same sequence.
    """
    bars = np.flatnonzero(shapes.is_bar)
    cos, sin = shapes.cos[bars], shapes.sin[bars]
    center_x, center_y = shapes.center_x[bars], shapes.center_y[bars]
    thickness, length = shapes.thickness[bars], shapes.length[bars]
    visited = np.zeros(bars.size, dtype=bool)
    stacks: list[list[int]] = []
    for start in range(bars.size):
        if visited[start]:
            continue
        visited[start] = True
        members: list[int] = []
        pending = [start]
        while pending:
            current = pending.pop()
            members.append(int(bars[current]))
            row = _side_by_side(current, cos, sin, center_x, center_y, thickness, length)
            linked = np.flatnonzero(row & ~visited)
            visited[linked] = True
            pending.extend(linked.tolist())
        total = 0.0
        for member in members:
            total += float(shapes.length[member])
        if len(members) >= MIN_STACK_BARS and total >= minimum_length:
            stacks.append(members)
    return stacks


def _lone_bars(labels: np.ndarray, shapes: _Shapes, confident: np.ndarray, extent: int) -> list[int]:
    """Shape positions of components that are censor bars on their own (see the module notes)."""
    candidates = np.flatnonzero(
        shapes.is_bar
        & (shapes.thickness >= LONE_MIN_THICKNESS * extent)
        & (shapes.length >= LONE_MIN_LENGTH * extent)
        & (shapes.length <= LONE_MAX_ELONGATION * shapes.thickness)
    )
    found = []
    for candidate in candidates.tolist():
        rows, cols = np.nonzero(labels == shapes.label[candidate])
        hull = ConvexHull(np.column_stack((cols, rows))).volume  # in two dimensions, the area
        if rows.size >= LONE_MIN_FILL * hull and np.count_nonzero(confident[rows, cols]) >= LONE_MIN_SEEDS * rows.size:
            found.append(candidate)
    return found


def _attach_neighbours(
    labels: np.ndarray, shapes: _Shapes, stack: list[int], accepted: np.ndarray
) -> None:
    """Accepts components that continue a stack: bars of comparable width close to it, and bar
    networks (rings, ladders, caps) whose strokes match its width."""
    index = np.asarray(stack, dtype=np.intp)
    stack_width = _median(shapes.thickness[index])
    stack_length = _median(shapes.length[index])
    stack_area = float(shapes.area[index].sum())
    gap = ATTACH_GAP * stack_width
    reach = math.ceil(gap)
    narrowest = WIDTH_BAND[0] * stack_width / 2.0
    widest = WIDTH_BAND[1] * stack_width / 2.0

    member = np.zeros(accepted.size, dtype=bool)
    member[shapes.label[index]] = True
    left = int(shapes.min_x[index].min())
    top = int(shapes.min_y[index].min())
    right = int(shapes.max_x[index].max())
    bottom = int(shapes.max_y[index].max())

    # Chebyshev gap between each shape's box and the stack's box. Doing this for all shapes at once
    # leaves only a few candidates for the per-shape tests.
    horizontal = np.maximum(shapes.min_x - right, left - shapes.max_x)
    vertical = np.maximum(shapes.min_y - bottom, top - shapes.max_y)
    box_gap = np.maximum(np.maximum(horizontal, vertical), 0)
    candidates = np.flatnonzero(~accepted[shapes.label] & (box_gap <= reach))

    for candidate in candidates:
        if shapes.is_bar[candidate]:
            if shapes.length[candidate] > ATTACH_LENGTH * stack_length:
                continue
            half_width = float(shapes.thickness[candidate]) / 2.0
        else:
            if float(shapes.area[candidate]) > ATTACH_AREA * stack_area:
                continue
            ridge, bulge = _stroke_profile(labels, shapes, candidate)
            if bulge > JUNCTION_BULGE * ridge:
                continue
            half_width = ridge
        if half_width < narrowest or half_width > widest:
            continue
        nearest = _nearest_squared(labels, shapes, candidate, member, reach)
        if nearest is not None and nearest <= gap * gap:
            accepted[shapes.label[candidate]] = True


def _nearest_squared(
    labels: np.ndarray, shapes: _Shapes, candidate: int, member: np.ndarray, reach: int
) -> Optional[int]:
    """Smallest squared distance from the candidate to any stack pixel, or None when no stack pixel
    lies within ``reach``. Only the region reachable within ``reach`` is searched; a closer pair
    always lies inside it, so the test against the gap is exact."""
    h, w = labels.shape
    left = max(int(shapes.min_x[candidate]) - reach, 0)
    top = max(int(shapes.min_y[candidate]) - reach, 0)
    right = min(int(shapes.max_x[candidate]) + reach + 1, w)
    bottom = min(int(shapes.max_y[candidate]) + reach + 1, h)
    region = labels[top:bottom, left:right]
    seeds = member[region]
    if not seeds.any():
        return None
    distance = squared_distance(seeds)
    own = region == shapes.label[candidate]
    return int(distance[own].min())


def _stroke_profile(labels: np.ndarray, shapes: _Shapes, candidate: int) -> tuple[float, float]:
    """(median half-width of the strokes, widest inscribed radius) of a component."""
    x0, x1 = int(shapes.min_x[candidate]), int(shapes.max_x[candidate]) + 1
    y0, y1 = int(shapes.min_y[candidate]), int(shapes.max_y[candidate]) + 1
    inside = labels[y0:y1, x0:x1] == shapes.label[candidate]
    inner_h, inner_w = inside.shape
    # A one-pixel frame of background lets the distance reach the component's edge everywhere.
    outside = np.ones((inner_h + 2, inner_w + 2), dtype=bool)
    outside[1:-1, 1:-1] = ~inside
    distance = squared_distance(outside)
    inner = distance[1:-1, 1:-1]
    widest = int(inner[inside].max())
    # A ridge is a local maximum of the distance (3x3, ties allowed) at least 3 squared units deep.
    is_ridge = inside & (inner >= 3)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neighbour = distance[1 + dy : 1 + dy + inner_h, 1 + dx : 1 + dx + inner_w]
            is_ridge &= neighbour <= inner
    ridge = np.sqrt(inner[is_ridge].astype(np.float64))
    return _median(ridge), math.sqrt(widest)
