"""White-stroke censorship candidates for manga pages.

Ported from the Koharu fork's ``white_bars.rs`` (Apache-2.0). Some releases hide anatomy with strokes
painted in paper white instead of black bars or mosaic. A white stroke lying on white paper is
invisible by construction, so what stays observable is where a stroke interrupts ink: a straight,
constant-width, pure-white band whose long edges end tone dots or line work. Natural white shapes with
the same silhouette (hair gaps, speed-line gaps, panel gutters, lettering rows) are far more common
than strokes, so one band proves nothing; a mask is reported only for a stack of such bands.

This is Koharu's own heuristic and is not part of any network. Pixel sizes below are stated for a page
whose longest side is ``REFERENCE_EXTENT`` and scale linearly with the page.

1. Large white regions (speech bubbles, paper margins) are located with a distance transform; bands
   near them are never strokes.
2. For ``ORIENTATIONS`` directions the page is rotated so the candidate direction is horizontal, and
   the white mask is opened with a thin rectangle. What survives and is not near a large region is a
   band: a long, thin, straight run of white.
3. A band is kept as a strip only when it is not wider than a brush stroke and its long edges are
   straight, both long edges meet ink that is textured, and the page around it carries ink.
4. A stroke lying between two directions yields strips in both. Strips that cover the same area are
   merged into a bar; strips that merely touch or cross are not.
5. Three or more bars side by side, roughly parallel and a few widths apart, form a stack and are
   reported. Isolated bars, pairs, rows packed closer than three widths (lettering) and bars far longer
   than a stroke never are.

A white stroke that is not part of a stack, one drawn entirely over paper, and one over tone that the
source has resampled into flat grey are not detected.

Where the Rust code runs per pixel, this module runs on whole-page arrays. Integer decisions stay
integer, the Rust's left-to-right float sums are reproduced with ``cumsum`` (numpy's pairwise ``sum``
would round differently), and rotations use the same expressions and rounding as the Rust.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# Longest page side for which the pixel lengths below are stated.
REFERENCE_EXTENT = 1808.0
# Pages smaller than this on a side (after downscaling) carry no stack.
MIN_EXTENT = 48

# Every channel of a paper pixel is at least this bright.
WHITE_LEVEL = 240
# Luma below which a pixel counts as ink.
INK_LUMA = 190

# A white region whose inscribed circle reaches this radius is paper, not a stroke (strokes are at
# most MAX_THICKNESS wide).
LARGE_RADIUS = 18.0
# Bands closer than this to a large region are its rim, not strokes.
LARGE_MARGIN = 6.0

# Directions tried, evenly spread over 180 degrees. The step bounds how far a stroke can sit from the
# nearest direction, and the rectangle below fits a thin stroke only up to that tilt.
ORIENTATIONS = 24
RECT_LENGTH = 48.0
RECT_THICKNESS = 10.0
# Shortest band, measured along its direction.
MIN_LENGTH = 56.0
# Widest band: the median thickness along its length.
MAX_THICKNESS = 32.0

# How far from a band's edge to look for the first ink pixel.
FLANK_REACH = 6.0
# Rows sampled outward from that first ink pixel.
FLANK_DEPTH = 3.0
# Longest unbroken stretch of ink along an edge, as a share of the band. Outlines and neighbouring
# panel borders run alongside a band for most of its length; tone and cut line work do not.
FLANK_MAX_RUN = 0.4
# Share of sampled flank pixels that are ink, on each side (inclusive).
FLANK_INK = (0.15, 0.75)
# Ink/paper changes per column along an edge, at least.
FLANK_MIN_TRANSITIONS = 0.08
# The page band from this distance to CONTEXT_TO beyond each edge must carry ink on average. Gaps
# between sparse speed lines are bordered by thin lines and surrounded by paper.
CONTEXT_FROM = 4.0
CONTEXT_TO = 16.0
MIN_CONTEXT_INK = 0.15
# Largest 80th-percentile deviation of an edge from its best straight line. Hand-drawn gaps wander; a
# brushed stroke does not.
MAX_EDGE_DEVIATION = 3.0

# Share of the smaller strip that another must cover to be the same bar.
MIN_OVERLAP = 0.5
# Merged strips are bars when their equal-moment rectangle is within these bounds. The longest bar is
# a share of the page's longest side.
MIN_BAR_WIDTH = 8.0
MAX_BAR_WIDTH = 30.0
MIN_ELONGATION = 3.0
MAX_BAR_LENGTH = 0.30

MIN_STACK_BARS = 3
# Centre separation across the bars' axes, in bar widths. The lower bound rejects rows of lettering;
# the upper bound keeps stacks local.
MIN_SEPARATION = 3.0
MAX_SEPARATION = 8.0
# Centre offset along the axes as a share of the mean bar length.
MAX_OFFSET = 0.6
# Strokes of one object fan out. The tolerated angle is the one at which two bars drift apart by
# DIVERGENCE widths over the shorter bar, clamped to these degrees.
MIN_ANGLE_TOLERANCE = 12.0
MAX_ANGLE_TOLERANCE = 30.0
DIVERGENCE = 3.5
WIDTH_RATIO = (0.5, 2.0)

# Growth of the reported mask, which covers the strokes' anti-aliased rim.
GROW = 2.0

WHITE = 1
INK = 2
LARGE = 4
# Canvas pixels outside the page: paper that no stroke can be part of.
OUTSIDE = WHITE | LARGE

# The Rust labels are u16; components beyond this count per direction are dropped.
MAX_LABELS = 65535
EIGHT_CONNECTED = np.ones((3, 3), dtype=np.int32)
# Worker threads for the directions; each holds a few page-sized canvases, so the count stays modest.
_WORKERS = max(1, min(4, os.cpu_count() or 1))


def _round(value: float) -> int:
    """Rust ``f64::round`` for the non-negative values used here (half away from zero)."""
    return int(math.floor(value + 0.5))


def _sequential_sum(values: np.ndarray) -> float:
    """Left-to-right sum. numpy's ``sum`` is pairwise and rounds differently from the Rust loops."""
    return float(np.cumsum(values)[-1]) if values.size else 0.0


# ---------------------------------------------------------------------------------------------------
# Planes: the page pooled into flag bytes at analysis resolution.
# ---------------------------------------------------------------------------------------------------


def _pool_flags(rgb: np.ndarray, factor: int) -> np.ndarray:
    """Pools every ``factor`` x ``factor`` block: paper only when all of it is paper, ink when any of
    it is ink. Returns one flag byte per block."""
    height, width = rgb.shape[:2]
    planes_h = -(-height // factor)
    planes_w = -(-width // factor)
    red = rgb[..., 0].astype(np.int32)
    green = rgb[..., 1].astype(np.int32)
    blue = rgb[..., 2].astype(np.int32)
    white = np.minimum(np.minimum(red, green), blue) >= WHITE_LEVEL
    # Integer luma, compared as the Rust does: ``luma / 1000 < INK_LUMA`` on u32.
    ink = (299 * red + 587 * green + 114 * blue) // 1000 < INK_LUMA
    # Partial blocks at the bottom and right are padded so they cannot change the result: paper for
    # the all-white test, not ink for the any-ink test.
    pad = ((0, planes_h * factor - height), (0, planes_w * factor - width))
    shape = (planes_h, factor, planes_w, factor)
    white = np.pad(white, pad, constant_values=True).reshape(shape).all(axis=(1, 3))
    ink = np.pad(ink, pad, constant_values=False).reshape(shape).any(axis=(1, 3))
    return white.astype(np.uint8) * WHITE | ink.astype(np.uint8) * INK


def _squared_distance_to(mask: np.ndarray) -> np.ndarray:
    """Squared distance from each pixel to the nearest True pixel (0 on True pixels).

    This is imageproc's ``euclidean_squared_distance_transform``, which measures to the nearest
    non-zero pixel. It is inf when the mask is empty, as the Rust returns for a foreground-free image.
    """
    if not mask.any():
        return np.full(mask.shape, np.inf)
    if mask.all():
        return np.zeros(mask.shape)
    # scipy measures to the nearest zero, so the True pixels of ``mask`` are the zeros of the input.
    # The exact nearest-site indices give exact integer squared distances.
    near_rows, near_cols = ndimage.distance_transform_edt(
        ~mask, return_distances=False, return_indices=True
    )
    rows, cols = np.indices(mask.shape)
    return ((rows - near_rows) ** 2 + (cols - near_cols) ** 2).astype(np.float64)


def _mark_large_regions(flags: np.ndarray, scale: float) -> np.ndarray:
    """Sets LARGE on every pixel within the rim of a large white region."""
    non_white = (flags & WHITE) == 0
    clearance = _squared_distance_to(non_white)
    radius = LARGE_RADIUS * scale
    # Core: white pixels whose inscribed circle reaches the radius. Non-white pixels have clearance 0.
    core = clearance >= radius * radius
    if not core.any():
        return flags
    distance = _squared_distance_to(core)
    reach = (LARGE_RADIUS + LARGE_MARGIN) * scale
    return flags | np.where(distance <= reach * reach, LARGE, 0).astype(np.uint8)


# ---------------------------------------------------------------------------------------------------
# Limits: pixel sizes at the analysis scale.
# ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Limits:
    rect_length: int
    rect_thickness: int
    min_length: int
    max_thickness: int
    reach: int
    depth: int
    context_from: int
    context_to: int
    deviation: float

    @classmethod
    def new(cls, scale: float) -> _Limits:
        def scaled(length: float, minimum: int) -> int:
            return max(_round(length * scale), minimum)

        context_from = scaled(CONTEXT_FROM, 2)
        return cls(
            rect_length=scaled(RECT_LENGTH, 3),
            rect_thickness=scaled(RECT_THICKNESS, 3),
            min_length=scaled(MIN_LENGTH, 4),
            max_thickness=scaled(MAX_THICKNESS, 4),
            reach=scaled(FLANK_REACH, 3),
            depth=scaled(FLANK_DEPTH, 2),
            context_from=context_from,
            context_to=max(scaled(CONTEXT_TO, 0), context_from + 4),
            deviation=MAX_EDGE_DEVIATION * scale,
        )


# ---------------------------------------------------------------------------------------------------
# Frame: the page rotated onto a canvas that holds all of it.
# ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Frame:
    cos: float
    sin: float
    width: int
    height: int
    source_cx: float
    source_cy: float
    cx: float
    cy: float

    @classmethod
    def new(cls, source_width: int, source_height: int, degrees: float) -> _Frame:
        # ``math.radians`` is ``x * (pi / 180)``, the same expression as Rust's ``to_radians``.
        sin = math.sin(math.radians(degrees))
        cos = math.cos(math.radians(degrees))
        sw = float(source_width)
        sh = float(source_height)
        width = math.ceil(sw * abs(cos) + sh * abs(sin)) + 2
        height = math.ceil(sw * abs(sin) + sh * abs(cos)) + 2
        return cls(
            cos=cos,
            sin=sin,
            width=width,
            height=height,
            source_cx=(sw - 1.0) / 2.0,
            source_cy=(sh - 1.0) / 2.0,
            cx=(width - 1.0) / 2.0,
            cy=(height - 1.0) / 2.0,
        )

    def rotate(self, flags: np.ndarray) -> np.ndarray:
        """Nearest-neighbour rotation of the flags; canvas pixels off the page are OUTSIDE."""
        planes_h, planes_w = flags.shape
        # Each term is computed in the Rust's order: (dx * cos - dy * sin + source_cx) + 0.5, then the
        # Rust floors it. floor(u) lies in [0, n) exactly when u does, so the bounds are tested on u.
        dx = np.arange(self.width, dtype=np.float64) - self.cx
        dy = np.arange(self.height, dtype=np.float64) - self.cy
        u = dx * self.cos - dy[:, None] * self.sin
        u += self.source_cx + 0.5
        v = dx * self.sin + dy[:, None] * self.cos
        v += self.source_cy + 0.5
        inside = (u >= 0.0) & (u < planes_w) & (v >= 0.0) & (v < planes_h)
        columns = np.clip(u, 0.0, planes_w - 1).astype(np.intp)
        rows = np.clip(v, 0.0, planes_h - 1).astype(np.intp)
        sampled = flags.ravel().take(rows * planes_w + columns)
        return np.where(inside, sampled, np.uint8(OUTSIDE))

    def canvas_index(self, planes_w: int, planes_h: int) -> np.ndarray:
        """Flat canvas index that each page pixel (row-major) projects to, or -1 off the canvas."""
        dx = np.arange(planes_w, dtype=np.float64) - self.source_cx
        dy = np.arange(planes_h, dtype=np.float64) - self.source_cy
        # The inverse rotation, with the same order of operations as the Rust ``to_canvas``.
        u = dx * self.cos + dy[:, None] * self.sin
        u += self.cx + 0.5
        v = -dx * self.sin + dy[:, None] * self.cos
        v += self.cy + 0.5
        inside = (u >= 0.0) & (u < self.width) & (v >= 0.0) & (v < self.height)
        columns = np.clip(u, 0.0, self.width - 1).astype(np.intp)
        rows = np.clip(v, 0.0, self.height - 1).astype(np.intp)
        return np.where(inside, rows * self.width + columns, -1).ravel()


# ---------------------------------------------------------------------------------------------------
# Opening with a thin rectangle.
# ---------------------------------------------------------------------------------------------------


def _erode_forward(cells: np.ndarray, length: int, axis: int) -> np.ndarray:
    """A cell survives when the ``length`` cells from it onward along ``axis`` are all set. Cells past
    the edge count as set, so a band touching the edge is not shortened by it."""
    # A minimum filter over [i, i + length) is the window above; ``origin`` moves scipy's centred
    # window to start at i.
    return ndimage.minimum_filter1d(
        cells.view(np.uint8), size=length, axis=axis, origin=-(length // 2), mode="constant", cval=1
    ).view(bool)


def _dilate_backward(cells: np.ndarray, length: int, axis: int) -> np.ndarray:
    """A cell is set when a survivor lies within the ``length`` cells up to and including it.
    Cells before the edge count as unset."""
    # A maximum filter over (i - length, i] is the window above.
    return ndimage.maximum_filter1d(
        cells.view(np.uint8), size=length, axis=axis, origin=(length - 1) - length // 2, mode="constant", cval=0
    ).view(bool)


def _open_rectangle(cells: np.ndarray, length: int, thickness: int) -> np.ndarray:
    """Morphological opening of a boolean canvas by a ``length`` x ``thickness`` rectangle, in the
    Rust's pass order: erosion by rows then columns, dilation by rows then columns."""
    cells = _erode_forward(cells, length, axis=1)
    cells = _erode_forward(cells, thickness, axis=0)
    cells = _dilate_backward(cells, length, axis=1)
    return _dilate_backward(cells, thickness, axis=0)


# ---------------------------------------------------------------------------------------------------
# Strips: bands measured on the rotated canvas.
# ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Flank:
    ink: float
    longest_run: float
    transitions: float
    context: float


def _textured(flank: _Flank) -> bool:
    return (
        flank.longest_run <= FLANK_MAX_RUN
        and FLANK_INK[0] <= flank.ink <= FLANK_INK[1]
        and flank.transitions >= FLANK_MIN_TRANSITIONS
    )


def _longest_true_run(bits: np.ndarray) -> int:
    if not bits.any():
        return 0
    padded = np.concatenate(([False], bits, [False])).astype(np.int8)
    change = np.diff(padded)
    starts = np.flatnonzero(change == 1)
    ends = np.flatnonzero(change == -1)
    return int((ends - starts).max())


def _edge_deviation(columns: np.ndarray, edge: np.ndarray) -> float:
    """80th percentile of the distance of ``edge`` from its least-squares line."""
    x = columns.astype(np.float64)
    y = edge.astype(np.float64)
    count = float(x.size)
    mean_x = _sequential_sum(x) / count
    mean_y = _sequential_sum(y) / count
    dx = x - mean_x
    variance = _sequential_sum(dx * dx)
    covariance = _sequential_sum(dx * (y - mean_y))
    slope = covariance / variance if variance > 0.0 else 0.0
    residuals = np.sort(np.abs((y - mean_y) - slope * (x - mean_x)))
    return float(residuals[int(math.floor((residuals.size - 1) * 0.8 + 0.5))])


class _Strip:
    """A candidate band measured column by column on its frame's canvas."""

    def __init__(self, pixels: np.ndarray, canvas_width: int) -> None:
        x = pixels % canvas_width
        y = pixels // canvas_width
        self.left = int(x.min())
        self.span = int(x.max()) - self.left + 1
        self.rows = int(y.max()) - int(y.min()) + 1
        offset = x - self.left
        top = np.full(self.span, np.iinfo(np.int64).max, dtype=np.int64)
        bottom = np.full(self.span, -1, dtype=np.int64)
        np.minimum.at(top, offset, y)
        np.maximum.at(bottom, offset, y)
        occupied = bottom >= 0
        self.columns = np.flatnonzero(occupied).astype(np.int64) + self.left
        self.top = top[occupied]
        self.bottom = bottom[occupied]

    def median_thickness(self) -> int:
        thickness = np.sort(self.bottom - self.top + 1)
        return int(thickness[thickness.size // 2])


def _accepts(limits: _Limits, canvas: np.ndarray, width: int, height: int, pixels: np.ndarray) -> bool:
    """Whether the band made of ``pixels`` (canvas indices) is a stroke candidate."""
    if pixels.size < limits.min_length * limits.rect_thickness:
        return False
    strip = _Strip(pixels, width)
    if (
        strip.span < limits.min_length
        or strip.rows < limits.rect_thickness
        or strip.median_thickness() > limits.max_thickness
    ):
        return False
    upper = _flank(limits, canvas, width, height, strip, upward=True)
    lower = _flank(limits, canvas, width, height, strip, upward=False)
    return (
        _textured(upper)
        and _textured(lower)
        and 0.5 * (upper.context + lower.context) >= MIN_CONTEXT_INK
        and max(_edge_deviation(strip.columns, strip.top), _edge_deviation(strip.columns, strip.bottom))
        <= limits.deviation
    )


def _flank(
    limits: _Limits, canvas: np.ndarray, width: int, height: int, strip: _Strip, upward: bool
) -> _Flank:
    """Samples the page beyond the upper or lower edge of ``strip``, starting at the first pixel that
    is not paper within reach of the band."""
    edges = strip.top if upward else strip.bottom
    step = -1 if upward else 1
    last_row = height - 1
    columns = strip.columns[:, None]
    count = strip.columns.size

    # First non-paper pixel within reach + 1 rows. Leaving the canvas stops the search at its edge row.
    # Where nothing is found, the boundary sits at the full reach, unclamped, as in the Rust.
    tried_distance = np.arange(1, limits.reach + 2)
    tried = edges[:, None] + step * tried_distance[None, :]
    off_canvas = (tried < 0) | (tried > last_row)
    clipped = np.clip(tried, 0, last_row)
    non_white = (canvas[clipped * width + columns] & WHITE) == 0
    stop = off_canvas | non_white
    hit = stop.any(axis=1)
    first_hit = stop.argmax(axis=1)
    boundary = np.where(hit, clipped[np.arange(count), first_hit], tried[:, -1])

    def ink_bits(offsets: np.ndarray) -> np.ndarray:
        rows = np.clip(boundary[:, None] + step * offsets[None, :], 0, last_row)
        return (canvas[rows * width + columns] & INK) != 0

    near = ink_bits(np.arange(limits.depth))
    far = ink_bits(np.arange(limits.context_from, limits.context_to))
    any_ink = near.any(axis=1)
    return _Flank(
        ink=float(int(near.sum())) / float(limits.depth * count),
        longest_run=float(_longest_true_run(any_ink)) / count,
        transitions=float(np.count_nonzero(any_ink[1:] != any_ink[:-1])) / count,
        context=float(int(far.sum())) / float((limits.context_to - limits.context_from) * count),
    )


def _directional_strips(
    flags: np.ndarray, planes_w: int, planes_h: int, degrees: float, limits: _Limits
) -> list[np.ndarray]:
    """The strips whose direction is ``degrees``, each as the page pixels (row-major) it covers."""
    frame = _Frame.new(planes_w, planes_h, degrees)
    canvas = frame.rotate(flags)

    cells = _open_rectangle((canvas & WHITE).astype(bool), limits.rect_length, limits.rect_thickness)
    cells &= (canvas & LARGE) == 0

    labels, count = ndimage.label(cells, structure=EIGHT_CONNECTED)
    if count == 0:
        return []
    flat_labels = labels.ravel()
    # Components smaller than the shortest band are rejected before any measurement, as in the Rust.
    sizes = np.bincount(flat_labels, minlength=count + 1)
    candidates = np.flatnonzero(sizes >= limits.min_length * limits.rect_thickness)
    candidates = candidates[candidates > 0]
    if candidates.size == 0:
        return []

    # Pixel lists of the candidates, grouped by label, each in canvas order.
    members = np.flatnonzero(np.isin(flat_labels, candidates))
    members = members[np.argsort(flat_labels[members], kind="stable")]
    member_labels = flat_labels[members]
    starts = np.searchsorted(member_labels, candidates, side="left")
    ends = np.searchsorted(member_labels, candidates, side="right")

    canvas_flat = canvas.ravel()
    accepted: list[int] = []
    for label, start, end in zip(candidates.tolist(), starts.tolist(), ends.tolist()):
        if len(accepted) >= MAX_LABELS:
            break
        if _accepts(limits, canvas_flat, frame.width, frame.height, members[start:end]):
            accepted.append(label)
    if not accepted:
        return []

    # Carry the accepted regions back to the page grid, one pixel list per region.
    keep = np.zeros(count + 1, dtype=bool)
    keep[accepted] = True
    canvas_label = np.where(keep[flat_labels], flat_labels, 0)
    canvas_at = frame.canvas_index(planes_w, planes_h)
    on_canvas = np.flatnonzero(canvas_at >= 0)
    labels_at = canvas_label[canvas_at[on_canvas]]
    hit = labels_at > 0
    pixels = on_canvas[hit]
    labels_at = labels_at[hit]
    order = np.argsort(labels_at, kind="stable")
    pixels = pixels[order]
    labels_at = labels_at[order]
    lo = np.searchsorted(labels_at, accepted, side="left")
    hi = np.searchsorted(labels_at, accepted, side="right")
    return [pixels[a:b].astype(np.int64) for a, b in zip(lo.tolist(), hi.tolist())]


# ---------------------------------------------------------------------------------------------------
# Merging strips into bars.
# ---------------------------------------------------------------------------------------------------


def _find(parent: list[int], node: int) -> int:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def _merge_strips(strips: list[np.ndarray], page_len: int) -> list[np.ndarray]:
    """Groups strips that cover the same page area. Each group is the sorted, deduplicated union of
    its strips' pixels, in order of the group's first strip."""
    count = len(strips)
    if count == 0:
        return []
    lengths = np.array([pixels.size for pixels in strips], dtype=np.int64)
    all_pixels = np.concatenate(strips)
    all_index = np.repeat(np.arange(count, dtype=np.int64), lengths)

    # In the Rust, the first strip to cover a pixel owns it and later strips count their pixels per
    # owner. The owner of a pixel is the lowest-index strip containing it, so the overlap counts can be
    # read off in one pass: every entry whose owner is another strip is one count toward that owner.
    unique_pixels, first_entry = np.unique(all_pixels, return_index=True)
    owner_of = np.full(page_len, -1, dtype=np.int64)
    owner_of[unique_pixels] = all_index[first_entry]
    owner = owner_of[all_pixels]
    other = owner != all_index
    keys, shared = np.unique(all_index[other] * count + owner[other], return_counts=True)

    parent = list(range(count))
    for key, overlap in zip(keys.tolist(), shared.tolist()):
        index, earlier = divmod(key, count)
        smaller = min(int(lengths[index]), int(lengths[earlier]))
        if overlap >= MIN_OVERLAP * smaller:
            parent[_find(parent, index)] = _find(parent, earlier)

    group_of: dict[int, int] = {}
    members: list[list[np.ndarray]] = []
    for index in range(count):
        leader = _find(parent, index)
        group = group_of.get(leader)
        if group is None:
            group = len(members)
            group_of[leader] = group
            members.append([])
        members[group].append(strips[index])
    return [np.unique(np.concatenate(group)) for group in members]


class _Bar:
    """Merged strips, measured as a rectangle of equal second moments."""

    __slots__ = ("cx", "cy", "ax", "ay", "length", "width", "pixels")

    def __init__(self, pixels: np.ndarray, planes_w: int) -> None:
        xs = (pixels % planes_w).astype(np.float64)
        ys = (pixels // planes_w).astype(np.float64)
        count = float(pixels.size)
        self.cx = _sequential_sum(xs) / count
        self.cy = _sequential_sum(ys) / count
        dx = xs - self.cx
        dy = ys - self.cy
        xx = _sequential_sum(dx * dx) / count
        xy = _sequential_sum(dx * dy) / count
        yy = _sequential_sum(dy * dy) / count
        mean = 0.5 * (xx + yy)
        spread = math.sqrt(0.25 * (xx - yy) * (xx - yy) + xy * xy)
        angle = 0.5 * math.atan2(2.0 * xy, xx - yy)
        self.ax = math.cos(angle)
        self.ay = math.sin(angle)
        self.length = math.sqrt(12.0 * (mean + spread))
        self.width = math.sqrt(12.0 * max(mean - spread, 1e-9))
        self.pixels = pixels


def _select_bars(
    groups: list[np.ndarray], planes_w: int, planes_h: int, scale: float, limits: _Limits
) -> list[_Bar]:
    """The merged strips that could be brush strokes."""
    longest = float(max(planes_w, planes_h))
    min_area = limits.min_length * limits.rect_thickness
    low = MIN_BAR_WIDTH * scale
    high = MAX_BAR_WIDTH * scale
    bars: list[_Bar] = []
    for pixels in groups:
        if pixels.size < min_area:
            continue
        bar = _Bar(pixels, planes_w)
        if (
            bar.length >= MIN_LENGTH * scale
            and bar.length <= MAX_BAR_LENGTH * longest
            and low <= bar.width <= high
            and bar.length >= MIN_ELONGATION * bar.width
        ):
            bars.append(bar)
    return bars


# ---------------------------------------------------------------------------------------------------
# Stacks of bars.
# ---------------------------------------------------------------------------------------------------


def _angle_tolerance(first: _Bar, second: _Bar) -> float:
    width = 0.5 * (first.width + second.width)
    shorter = min(first.length, second.length)
    return min(max(math.degrees(math.atan(DIVERGENCE * width / shorter)), MIN_ANGLE_TOLERANCE), MAX_ANGLE_TOLERANCE)


class _Pair:
    """How two bars sit relative to each other, along their mean axis."""

    def __init__(self, first: _Bar, second: _Bar) -> None:
        dot = first.ax * second.ax + first.ay * second.ay
        sign = 1.0 if dot >= 0.0 else -1.0
        ax = first.ax + sign * second.ax
        ay = first.ay + sign * second.ay
        norm = math.hypot(ax, ay)
        if norm < 1e-9:
            ax, ay = first.ax, first.ay
        else:
            ax, ay = ax / norm, ay / norm
        self.axis = (ax, ay)
        self.normal = (-ay, ax)
        offset_x = second.cx - first.cx
        offset_y = second.cy - first.cy
        self.along = abs(offset_x * ax + offset_y * ay)
        self.across = abs(offset_x * -ay + offset_y * ax)
        self.degrees = math.degrees(math.acos(min(abs(dot), 1.0)))


@dataclass
class _BarArrays:
    """The bars as arrays, for the scan over every bar in ``_neighbours``."""

    cx: np.ndarray
    cy: np.ndarray
    ax: np.ndarray
    ay: np.ndarray
    width: np.ndarray
    length: np.ndarray

    @classmethod
    def of(cls, bars: list[_Bar]) -> _BarArrays:
        return cls(
            cx=np.array([bar.cx for bar in bars]),
            cy=np.array([bar.cy for bar in bars]),
            ax=np.array([bar.ax for bar in bars]),
            ay=np.array([bar.ay for bar in bars]),
            width=np.array([bar.width for bar in bars]),
            length=np.array([bar.length for bar in bars]),
        )


def _neighbours(bars: list[_Bar], arrays: _BarArrays, first: int, second: int) -> bool:
    """Whether two bars are neighbours in a stack: roughly parallel and alike in width, overlapping
    along their axes, a few widths apart, and with no other parallel bar between them."""
    a = bars[first]
    b = bars[second]
    pair = _Pair(a, b)
    width = 0.5 * (a.width + b.width)
    length = 0.5 * (a.length + b.length)
    if (
        pair.degrees > _angle_tolerance(a, b)
        or not (WIDTH_RATIO[0] <= a.width / b.width <= WIDTH_RATIO[1])
        or pair.along > MAX_OFFSET * length
        or pair.across < MIN_SEPARATION * width
        or pair.across > MAX_SEPARATION * width
    ):
        return False

    normal_x, normal_y = pair.normal
    axis_x, axis_y = pair.axis
    near = min(a.cx * normal_x + a.cy * normal_y, b.cx * normal_x + b.cy * normal_y)
    far = max(a.cx * normal_x + a.cy * normal_y, b.cx * normal_x + b.cy * normal_y)
    middle_x = 0.5 * (a.cx + b.cx)
    middle_y = 0.5 * (a.cy + b.cy)

    # The Rust scans every other bar here; this is the same test over the arrays. A bar lies between
    # the pair when its projection is strictly inside the gap, and it blocks only when it is also
    # along the pair and parallel to ``a``.
    at = arrays.cx * normal_x + arrays.cy * normal_y
    between = (at > near + 0.5 * width) & (at < far - 0.5 * width)
    along = np.abs((arrays.cx - middle_x) * axis_x + (arrays.cy - middle_y) * axis_y)
    dot = a.ax * arrays.ax + a.ay * arrays.ay
    degrees = np.degrees(np.arccos(np.minimum(np.abs(dot), 1.0)))
    tolerance = np.clip(
        np.degrees(np.arctan(DIVERGENCE * (0.5 * (a.width + arrays.width)) / np.minimum(a.length, arrays.length))),
        MIN_ANGLE_TOLERANCE,
        MAX_ANGLE_TOLERANCE,
    )
    blocking = between & (along <= MAX_OFFSET * length) & (degrees <= tolerance)
    blocking[first] = False
    blocking[second] = False
    return not blocking.any()


def _find_stacks(bars: list[_Bar]) -> list[list[int]]:
    """Connected groups of neighbouring bars with at least MIN_STACK_BARS members, as indices."""
    count = len(bars)
    if count < MIN_STACK_BARS:
        return []
    arrays = _BarArrays.of(bars)
    adjacent: list[list[int]] = [[] for _ in range(count)]
    for first in range(count):
        for second in range(first + 1, count):
            if _neighbours(bars, arrays, first, second):
                adjacent[first].append(second)
                adjacent[second].append(first)

    seen = [False] * count
    stacks: list[list[int]] = []
    for start in range(count):
        if seen[start]:
            continue
        seen[start] = True
        members: list[int] = []
        pending = [start]
        while pending:
            bar = pending.pop()
            members.append(bar)
            for following in adjacent[bar]:
                if not seen[following]:
                    seen[following] = True
                    pending.append(following)
        if len(members) >= MIN_STACK_BARS:
            stacks.append(members)
    return stacks


# ---------------------------------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------------------------------


def white_bar_mask(rgb: np.ndarray) -> np.ndarray:
    """Marks the strokes of every white-stroke stack on ``rgb``.

    ``rgb`` is HxWx3 uint8. The result is an HxW uint8 mask with 255 on stroke pixels and 0 elsewhere;
    it is empty when the page carries no stack.
    """
    height, width = rgb.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    longest = float(max(width, height))
    # Pages are analysed at a resolution where the longest side is about REFERENCE_EXTENT.
    factor = max(_round(longest / REFERENCE_EXTENT), 1)

    flags = _pool_flags(rgb, factor)
    planes_h, planes_w = flags.shape
    if min(planes_w, planes_h) < MIN_EXTENT:
        return mask
    scale = max(planes_w, planes_h) / REFERENCE_EXTENT
    flags = _mark_large_regions(flags, scale)
    limits = _Limits.new(scale)

    # A band is opened from white pixels outside every large region. Canvas pixels only ever copy page
    # flags, so a page without such pixels has no band in any direction.
    if not ((flags & (WHITE | LARGE)) == WHITE).any():
        return mask

    # Collected in direction order, so the merge below is deterministic. The directions are
    # independent; numpy releases the GIL in the array operations that dominate, so they run on threads.
    def strips_for(index: int) -> list[np.ndarray]:
        return _directional_strips(flags, planes_w, planes_h, 180.0 * index / ORIENTATIONS, limits)

    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        per_direction = list(pool.map(strips_for, range(ORIENTATIONS)))
    strips = [strip for strips_of in per_direction for strip in strips_of]
    bars = _select_bars(_merge_strips(strips, planes_w * planes_h), planes_w, planes_h, scale, limits)
    stacks = _find_stacks(bars)
    if not stacks:
        return mask

    stroke = np.zeros((planes_h, planes_w), dtype=bool)
    for stack in stacks:
        for index in stack:
            stroke.ravel()[bars[index].pixels] = True
    page = np.repeat(np.repeat(stroke, factor, axis=0), factor, axis=1)[:height, :width]

    # Grow by the square neighbourhood (imageproc's LInf dilation) to cover the anti-aliased rim.
    grow = min(max(_round(GROW * longest / REFERENCE_EXTENT), 1), 255)
    return ndimage.maximum_filter(page.astype(np.uint8) * 255, size=2 * grow + 1, mode="constant", cval=0)
