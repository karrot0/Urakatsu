"""Tests for the white-stroke censorship candidates, ported from the Koharu fork's white_bars.rs tests."""

from __future__ import annotations

import math

import numpy as np

from urakatsu.providers.censor.white_bars import white_bar_mask

WIDTH = 640
HEIGHT = 900

# Pixel coordinates of the test page, as float grids (x across, y down).
GRID_Y, GRID_X = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)


class Stroke:
    """A painted white stroke: its centre, size and tilt."""

    def __init__(self, x: float, y: float, length: float, thickness: float, degrees: float) -> None:
        self.x = x
        self.y = y
        self.length = length
        self.thickness = thickness
        self.degrees = degrees

    def relative(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Position of pixels relative to the stroke: along it, across it."""
        sin, cos = math.sin(math.radians(self.degrees)), math.cos(math.radians(self.degrees))
        dx = x - self.x
        dy = y - self.y
        return dx * cos + dy * sin, -dx * sin + dy * cos


def paper() -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)


def solid_ink(image: np.ndarray, left: int, top: int, right: int, bottom: int) -> None:
    image[top:bottom, left:right] = 0


def tone(image: np.ndarray, left: int, top: int, right: int, bottom: int) -> None:
    """Halftone dots: 2 x 2 ink on a 3 px grid."""
    ys = np.arange(top, bottom)
    xs = np.arange(left, right)
    dots = ((xs % 3 < 2)[None, :]) & ((ys % 3 < 2)[:, None])
    image[top:bottom, left:right][dots] = 0


def paint(image: np.ndarray, stroke: Stroke) -> None:
    along, across = stroke.relative(GRID_X, GRID_Y)
    inside = (np.abs(along) <= stroke.length / 2.0) & (np.abs(across) <= stroke.thickness / 2.0)
    image[inside] = 255


def ladder(count: int) -> tuple[np.ndarray, list[Stroke]]:
    """Halftone with ``count`` tilted white strokes 60 px apart across it."""
    image = paper()
    tone(image, 120, 200, 520, 700)
    tilts = [0.0, 5.0, -4.0]
    strokes = [
        Stroke(
            x=320.0,
            y=450.0 + 60.0 * (index - (count - 1.0) / 2.0),
            length=160.0,
            thickness=10.0,
            degrees=tilts[index % len(tilts)],
        )
        for index in range(count)
    ]
    for stroke in strokes:
        paint(image, stroke)
    return image, strokes


def marked(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask > 0))


def test_a_stack_of_white_strokes_across_tone_is_marked_and_nothing_else() -> None:
    image, strokes = ladder(3)
    mask = white_bar_mask(image)
    assert mask.shape == image.shape[:2]

    for stroke in strokes:
        along, across = stroke.relative(GRID_X, GRID_Y)
        core_region = (np.abs(along) <= stroke.length / 2.0 - 20.0) & (np.abs(across) <= 2.0)
        core = int(np.count_nonzero(core_region))
        hit = int(np.count_nonzero(mask[core_region] > 0))
        assert hit * 10 >= core * 9, f"{hit} of {core} stroke pixels marked"

    marked_y, marked_x = np.nonzero(mask)
    near = np.zeros(marked_x.shape, dtype=bool)
    for stroke in strokes:
        along, across = stroke.relative(marked_x.astype(np.float64), marked_y.astype(np.float64))
        gap_along = np.maximum(np.abs(along) - stroke.length / 2.0, 0.0)
        gap_across = np.maximum(np.abs(across) - stroke.thickness / 2.0, 0.0)
        near |= np.hypot(gap_along, gap_across) <= 4.0
    stray = np.flatnonzero(~near)
    assert stray.size == 0, (
        f"stray mask pixel at ({marked_x[stray[0]]}, {marked_y[stray[0]]})" if stray.size else ""
    )


def test_two_strokes_are_not_a_stack() -> None:
    image, _ = ladder(2)
    assert marked(white_bar_mask(image)) == 0


def test_bubbles_gutters_and_lettering_rows_are_left_alone() -> None:
    # Two toned panels with borders and a gutter, and an outlined bubble.
    panels = paper()
    for top, bottom in [(20, 24), (400, 404), (440, 444), (880, 884)]:
        solid_ink(panels, 20, top, 620, bottom)
    tone(panels, 30, 30, 610, 390)
    tone(panels, 30, 452, 610, 870)
    dx = (GRID_X[90:310, 200:440] - 320.0) / 90.0
    dy = (GRID_Y[90:310, 200:440] - 200.0) / 60.0
    radius = dx * dx + dy * dy
    bubble = panels[90:310, 200:440]
    bubble[(radius >= 1.0) & (radius < 1.07)] = 0
    bubble[radius < 1.0] = 255
    assert marked(white_bar_mask(panels)) == 0

    # Eight rows of glyph-sized ink blocks on a 30 px pitch.
    lettering = paper()
    for row in range(8):
        for column in range(200, 400, 18):
            top = 200 + row * 30
            solid_ink(lettering, column, top, column + 12, top + 18)
    assert marked(white_bar_mask(lettering)) == 0


def test_blank_and_tiny_pages_have_empty_masks() -> None:
    assert marked(white_bar_mask(paper())) == 0
    tiny = np.full((20, 30, 3), 255, dtype=np.uint8)
    mask = white_bar_mask(tiny)
    assert mask.shape == (20, 30)
    assert marked(mask) == 0
