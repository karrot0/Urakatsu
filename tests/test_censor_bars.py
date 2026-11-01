"""Censorship-bar geometry: the tests of the Koharu fork's bars.rs, ported one for one.

The Canvas draws the same anti-aliased bars, blurred scores and solid blocks as the Rust helper.
The last test is the bar-refinement case from the fork's eromanga_seg.rs, which calls bar_mask.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from urakatsu.providers.censor import bars

U32_MAX = int(np.iinfo(np.uint32).max)


def _round(value: float) -> int:
    """Rust's ``f64::round`` for the non-negative coordinates used here."""
    return int(math.floor(value + 0.5))


@dataclass(frozen=True)
class Rod:
    cx: float
    cy: float
    length: float
    thickness: float
    degrees: float


def rod(cx: float, cy: float, length: float, thickness: float, degrees: float) -> Rod:
    return Rod(cx, cy, length, thickness, degrees)


class Canvas:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.rgb = np.full((height, width, 3), 255, dtype=np.uint8)
        self.levels = np.zeros((height, width), dtype=np.uint8)

    def bounds(self, r: Rod, margin: float) -> tuple[int, int, int, int]:
        extent = (r.length + r.thickness) / 2.0 + margin + 2.0
        x0 = max(math.floor(r.cx - extent), 0)
        y0 = max(math.floor(r.cy - extent), 0)
        x1 = min(max(math.ceil(r.cx + extent), 0), self.width)
        y1 = min(max(math.ceil(r.cy + extent), 0), self.height)
        return x0, y0, x1, y1

    def bar(self, r: Rod) -> None:
        """Paints an anti-aliased black bar with 4x4 supersampling."""
        sin, cos = math.sin(math.radians(r.degrees)), math.cos(math.radians(r.degrees))
        x0, y0, x1, y1 = self.bounds(r, 0.0)
        offsets = (np.arange(4) + 0.5) / 4.0
        # Axes: [pixel, subsample]; the subsample grid is 4x4 per pixel as in the Rust.
        dx = (np.arange(x0, x1)[:, None] + offsets) - r.cx
        dy = (np.arange(y0, y1)[:, None] + offsets) - r.cy
        along = dx[None, :, :, None] * cos + dy[:, None, None, :] * sin
        across = -dx[None, :, :, None] * sin + dy[:, None, None, :] * cos
        inside = (np.abs(along) <= r.length / 2.0) & (np.abs(across) <= r.thickness / 2.0)
        covered = inside.sum(axis=(2, 3))
        luma = np.floor(255.0 * (1.0 - covered / 16.0) + 0.5).astype(np.uint8)
        target = np.where(covered > 0, luma, np.uint8(255))[..., None]
        region = self.rgb[y0:y1, x0:x1]
        np.minimum(region, target, out=region)

    def block(self, left: int, top: int, right: int, bottom: int) -> None:
        self.rgb[top:bottom, left:right] = 0

    def score(self, r: Rod, level: int, margin: float) -> None:
        """Raises the score of every pixel centre inside the grown rod."""
        sin, cos = math.sin(math.radians(r.degrees)), math.cos(math.radians(r.degrees))
        x0, y0, x1, y1 = self.bounds(r, margin)
        dx = (np.arange(x0, x1) + 0.5) - r.cx
        dy = (np.arange(y0, y1) + 0.5) - r.cy
        along = dx[None, :] * cos + dy[:, None] * sin
        across = -dx[None, :] * sin + dy[:, None] * cos
        inside = (np.abs(along) <= r.length / 2.0 + margin) & (np.abs(across) <= r.thickness / 2.0 + margin)
        region = self.levels[y0:y1, x0:x1]
        np.maximum(region, np.where(inside, np.uint8(level), np.uint8(0)), out=region)

    def predicted_bar(self, r: Rod) -> None:
        """A bar plus the blurry, centre-weighted scores the checkpoint gives it."""
        self.bar(r)
        self.score(replace(r, length=r.length * 0.8), bars.CONFIDENT, 2.0)
        self.score(r, bars.SUPPORTED, 6.0)

    def mask(self) -> np.ndarray:
        found = bars.bar_mask(self.rgb, self.levels)
        if found is None:
            return np.zeros((self.height, self.width), dtype=np.uint8)
        return found

    def set(self, mask: np.ndarray, x: int, y: int) -> bool:
        return bool(mask[y, x] != 0)


def stack_of(canvas: Canvas, count: int, length: float, thickness: float, pitch: float) -> None:
    for index in range(count):
        canvas.predicted_bar(rod(canvas.width / 2.0, 60.0 + index * pitch, length, thickness, 0.0))


def test_squared_distance_matches_exhaustive_search():
    width, height = 23, 17
    flat = np.arange(width * height)
    seeds = ((flat * 7 + flat // 5) % 31 == 0).reshape(height, width)
    distance = bars.squared_distance(seeds)
    seed_points = [(x, y) for y in range(height) for x in range(width) if seeds[y, x]]
    assert seed_points
    for y in range(height):
        for x in range(width):
            expected = min((x - sx) ** 2 + (y - sy) ** 2 for sx, sy in seed_points)
            assert distance[y, x] == expected, f"at ({x}, {y})"
    assert np.all(bars.squared_distance(np.zeros((3, 4), dtype=bool)) >= U32_MAX // 2)


def test_support_threshold_tracks_the_confident_threshold():
    assert bars.support_threshold(0.6) == np.float32(0.3)
    assert bars.support_threshold(0.5) == np.float32(0.25)
    assert bars.support_threshold(0.3) == np.float32(0.2)
    assert bars.support_threshold(0.1) == np.float32(0.1)
    assert bars.support_threshold(0.0) == np.float32(0.0)


def test_rotated_anti_aliased_stack_is_retained_with_its_rim():
    canvas = Canvas(1400, 420)
    sin, cos = math.sin(math.radians(25.0)), math.cos(math.radians(25.0))
    centers = []
    for index in range(5):
        offset = (index - 2.0) * 38.0
        centers.append((700.0 - offset * sin, 210.0 + offset * cos))
    for cx, cy in centers:
        canvas.predicted_bar(rod(cx, cy, 120.0, 14.0, 25.0))
    mask = canvas.mask()

    for cx, cy in centers:
        assert canvas.set(mask, _round(cx), _round(cy))
    assert not canvas.set(mask, 5, 5)
    # Between two bars: a 24 px gap is far from every rim.
    assert not canvas.set(mask, int(700.0 - 19.0 * sin), int(210.0 + 19.0 * cos))

    # Anti-aliased edge pixels (luma 81..=128) are covered, light ones are not.
    luma = canvas.rgb[:, :, 0]
    covered = mask != 0
    rim = np.argwhere((luma > 80) & (luma <= 128) & ~covered)
    assert rim.size == 0, f"rim pixel ({rim[0][1]}, {rim[0][0]}) luma {luma[tuple(rim[0])]}"
    light = np.argwhere((luma > 200) & covered)
    assert light.size == 0, f"light pixel ({light[0][1]}, {light[0][0]}) luma {luma[tuple(light[0])]}"


def test_fan_with_wide_angle_steps_is_one_stack():
    canvas = Canvas(520, 520)
    centers = []
    for index in range(7):
        phi = math.radians(200.0 + index * 12.0)
        centers.append((260.0 + 150.0 * math.cos(phi), 260.0 + 150.0 * math.sin(phi)))
    for index, (cx, cy) in enumerate(centers):
        canvas.predicted_bar(rod(cx, cy, 100.0, 13.0, 8.0 + index * 17.0))
    mask = canvas.mask()
    for cx, cy in centers:
        assert canvas.set(mask, _round(cx), _round(cy))
    assert not canvas.set(mask, 5, 5)


def test_rings_and_ladders_join_the_stack_they_touch():
    canvas = Canvas(520, 600)
    stack_of(canvas, 4, 160.0, 18.0, 44.0)
    corners = [
        (160.0, 270.0),
        (360.0, 270.0),
        (400.0, 360.0),
        (360.0, 450.0),
        (160.0, 450.0),
        (120.0, 360.0),
    ]
    for index in range(len(corners)):
        x0, y0 = corners[index]
        x1, y1 = corners[(index + 1) % len(corners)]
        canvas.predicted_bar(
            rod(
                (x0 + x1) / 2.0,
                (y0 + y1) / 2.0,
                math.hypot(x1 - x0, y1 - y0),
                22.0,
                math.degrees(math.atan2(y1 - y0, x1 - x0)),
            )
        )
    canvas.predicted_bar(rod(260.0, 360.0, 260.0, 20.0, 0.0))
    mask = canvas.mask()

    for x, y in [(260, 60), (260, 192), (260, 270), (400, 360), (260, 360), (122, 360)]:
        assert canvas.set(mask, x, y), f"({x}, {y})"
    assert not canvas.set(mask, 260, 315), "the ring's hole stays open"
    assert not canvas.set(mask, 5, 590)


def test_a_ring_without_a_stack_is_not_a_bar_network():
    canvas = Canvas(520, 400)
    corners = [(160.0, 100.0), (360.0, 100.0), (360.0, 300.0), (160.0, 300.0)]
    for index in range(len(corners)):
        x0, y0 = corners[index]
        x1, y1 = corners[(index + 1) % len(corners)]
        canvas.predicted_bar(
            rod(
                (x0 + x1) / 2.0,
                (y0 + y1) / 2.0,
                math.hypot(x1 - x0, y1 - y0) + 22.0,
                22.0,
                math.degrees(math.atan2(y1 - y0, x1 - x0)),
            )
        )
    assert not canvas.mask().any()


def test_weakly_scored_bar_ends_grow_from_confident_centres():
    canvas = Canvas(420, 300)
    for index in range(4):
        bar = rod(210.0, 60.0 + index * 50.0, 220.0, 16.0, 0.0)
        canvas.bar(bar)
        canvas.score(replace(bar, length=66.0), bars.CONFIDENT, 2.0)
        canvas.score(bar, bars.SUPPORTED, 6.0)
    mask = canvas.mask()
    for index in range(4):
        y = 60 + index * 50
        assert canvas.set(mask, 104, y), f"left end of bar {index}"
        assert canvas.set(mask, 210, y), f"centre of bar {index}"
        assert canvas.set(mask, 316, y), f"right end of bar {index}"

    # Without weak support the confident centres alone stay as scored.
    canvas = Canvas(420, 300)
    for index in range(4):
        bar = rod(210.0, 60.0 + index * 50.0, 220.0, 16.0, 0.0)
        canvas.bar(bar)
        canvas.score(replace(bar, length=66.0), bars.CONFIDENT, 2.0)
    mask = canvas.mask()
    assert canvas.set(mask, 210, 60)
    assert not canvas.set(mask, 104, 60)


def test_thin_strokes_crossing_a_stack_neither_join_bars_nor_enter_the_mask():
    # A 1400 px page opens with a radius of two, which removes 2 px hair.
    canvas = Canvas(1400, 320)
    for index in range(3):
        canvas.predicted_bar(rod(300.0, 80.0 + index * 60.0, 220.0, 18.0, 0.0))
    canvas.block(299, 30, 301, 250)
    canvas.score(rod(300.0, 140.0, 220.0, 2.0, 90.0), bars.SUPPORTED, 3.0)
    mask = canvas.mask()
    for y in [80, 140, 200]:
        assert canvas.set(mask, 200, y)
    for y in [40, 110, 170, 240]:
        assert not canvas.set(mask, 300, y), f"stroke at y={y}"


def test_borders_frames_and_blocks_are_not_censorship():
    # One long diagonal gutter plus a heavy rectangular frame.
    canvas = Canvas(600, 600)
    canvas.predicted_bar(rod(300.0, 300.0, 420.0, 14.0, 40.0))
    for cx, cy, length, degrees in [
        (300.0, 66.0, 480.0, 0.0),
        (300.0, 534.0, 480.0, 0.0),
        (66.0, 300.0, 480.0, 90.0),
        (534.0, 300.0, 480.0, 90.0),
    ]:
        canvas.predicted_bar(rod(cx, cy, length, 12.0, degrees))
    assert not canvas.mask().any()

    # A solid black block is not a bar however confidently it scores.
    canvas = Canvas(300, 300)
    canvas.block(90, 90, 210, 210)
    canvas.score(rod(150.0, 150.0, 130.0, 130.0, 0.0), bars.CONFIDENT, 2.0)
    canvas.score(rod(150.0, 150.0, 130.0, 130.0, 0.0), bars.SUPPORTED, 6.0)
    assert not canvas.mask().any()

    # Two bars, or three far apart, are neither stacks.
    canvas = Canvas(420, 300)
    stack_of(canvas, 2, 200.0, 16.0, 50.0)
    assert not canvas.mask().any()
    canvas = Canvas(420, 800)
    stack_of(canvas, 3, 200.0, 16.0, 220.0)
    assert not canvas.mask().any()


def test_collinear_fragments_of_one_border_are_not_a_stack():
    # Four pieces of a single 58 degree gutter (a page the previous "parallel and near" rule
    # masked as a bar group).
    canvas = Canvas(900, 900)
    sin, cos = math.sin(math.radians(58.0)), math.cos(math.radians(58.0))
    for offset, length in [(-340.0, 230.0), (-120.0, 212.0), (100.0, 161.0), (250.0, 137.0)]:
        canvas.predicted_bar(
            rod(450.0 + offset * cos, 450.0 + offset * sin, length, 14.0, 58.0)
        )
    assert not canvas.mask().any()


def test_unsupported_or_barely_scored_ink_is_ignored():
    canvas = Canvas(420, 300)
    for index in range(4):
        canvas.bar(rod(210.0, 60.0 + index * 50.0, 220.0, 16.0, 0.0))
    assert not canvas.mask().any()

    for index in range(4):
        canvas.score(rod(210.0, 60.0 + index * 50.0, 220.0, 16.0, 0.0), bars.SUPPORTED, 6.0)
    assert not canvas.mask().any(), "weak support alone"

    # Thin parallel lines never survive the opening.
    canvas = Canvas(300, 300)
    for index in range(5):
        canvas.predicted_bar(rod(150.0, 80.0 + index * 10.0, 160.0, 2.0, 0.0))
    assert not canvas.mask().any()


def test_a_lone_solid_bar_is_censorship_when_the_model_is_sure_of_it():
    # Not in the fork, which needs three bars: a page's only bar, measured from a real page.
    canvas = Canvas(1008, 1433)
    bar = rod(345.0, 741.0, 183.0, 38.0, -10.0)
    canvas.bar(bar)
    canvas.score(bar, bars.CONFIDENT, 2.0)
    canvas.score(bar, bars.SUPPORTED, 6.0)
    assert canvas.set(canvas.mask(), 345, 741)


def test_a_lone_bar_the_model_barely_scores_is_left_alone():
    canvas = Canvas(1008, 1433)
    bar = rod(345.0, 741.0, 183.0, 38.0, 0.0)
    canvas.bar(bar)
    canvas.score(replace(bar, length=60.0), bars.CONFIDENT, 0.0)
    canvas.score(bar, bars.SUPPORTED, 6.0)
    assert not canvas.mask().any()


def test_bold_lettering_is_not_a_lone_bar():
    # A heavy katakana dash and an exclamation mark, sized as they scored on a real page.
    canvas = Canvas(1000, 1500)
    for bar in (rod(300.0, 400.0, 117.0, 21.0, 4.0), rod(700.0, 900.0, 71.0, 18.0, 80.0)):
        canvas.bar(bar)
        canvas.score(bar, bars.CONFIDENT, 2.0)
    assert not canvas.mask().any()


def test_a_spiked_balloon_is_not_a_lone_bar():
    # A black burst behind speech: elongated and confident, but no rectangle.
    canvas = Canvas(1000, 1500)
    rows, cols = np.mgrid[0:1500, 0:1000]
    angle = np.arctan2((rows - 700) / 2.5, cols - 500.0)
    radius = np.hypot((rows - 700) / 2.5, cols - 500.0)
    burst = radius <= 50.0 + 22.0 * (np.cos(angle * 16.0) > 0)
    canvas.rgb[burst] = 0
    canvas.levels[burst] = bars.CONFIDENT
    assert not canvas.mask().any()


def test_page_size_scales_the_required_stack_length():
    # Three 40 px bars total 120 px: a 640 px page asks for 64 px of bar, a 3600 px page for
    # 256 px. Bars stay wider than the opening disk.
    small = Canvas(640, 200)
    stack_of(small, 3, 40.0, 12.0, 36.0)
    assert small.set(small.mask(), 320, 60)

    large = Canvas(3600, 200)
    stack_of(large, 3, 40.0, 12.0, 36.0)
    assert not large.mask().any()


def test_placement_merge_keeps_the_larger_value():
    # The Rust merged a window-sized Placement into the page with a max. bar_mask now returns the
    # full page, so the same 2x2 window is written into a 4x4 mask at (left 1, top 1) and merged
    # with np.maximum, which is what the caller does.
    placement = np.zeros((4, 4), dtype=np.uint8)
    placement[1, 1] = 255
    placement[1, 2] = 0
    placement[2, 1] = 0
    placement[2, 2] = 255
    page = np.zeros((4, 4), dtype=np.uint8)
    page[1, 1] = 7
    page[1, 2] = 7
    merged = np.maximum(page, placement)
    assert merged[1, 1] == 255
    assert merged[1, 2] == 7
    assert merged[2, 1] == 0
    assert merged[2, 2] == 255
    assert merged[0, 0] == 0


def test_bar_refinement_keeps_stacked_bars_and_rejects_isolated_dark_shapes():
    # From the fork's eromanga_seg.rs tests: every page pixel is scored CONFIDENT.
    image = np.full((512, 512, 3), 255, dtype=np.uint8)
    for top in (50, 75, 100):
        image[top : top + 8, 40:120] = 0
    image[400:408, 300:380] = 0
    image[50:70, 300:320] = 0

    levels = np.full((512, 512), bars.CONFIDENT, dtype=np.uint8)
    mask = bars.bar_mask(image, levels)
    assert mask is not None, "three stacked bars are censorship"
    for y in (53, 78, 103):
        assert mask[y, 80] == 255
    assert mask[403, 340] == 0
    assert mask[60, 310] == 0
