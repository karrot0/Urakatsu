"""Webtoon stitching and slicing (an Urakatsu addition; Koharu translates page by page only).

Webtoons, manhwa and manhua are one long vertical strip that sites serve cut into images at
arbitrary heights, often through a speech bubble or a face. Restitching joins a chapter's
images into the strip and cuts it again where a cut hurts nothing, so every page the pipeline
sees holds whole bubbles.

The cut finder follows SmartStitch's pixel comparison detector
(https://github.com/MechTechnology/SmartStitch, MIT): a row may be cut when no two neighbouring
pixels in it differ by more than ``255 * (1 - sensitivity / 100)``, so rows through line art,
lettering or bubble outlines are refused while rows through gutters and flat backgrounds pass.
Differences from SmartStitch:

- The whole strip is classified at once (numpy) instead of scanning row by row.
- Within the allowed height window, wider quiet runs win over narrow ones near the target:
  the gap between two lines of borderless lettering is quiet too, a panel gutter is wider.
  A cut sits inside its run, away from the art on either side.
- ``avoid`` bands (text and bubbles the detector already found) are never cut, and the
  detection model can be asked where text is around each cut before it is made.
- Pages stay within ``[MIN_FRACTION, MAX_FRACTION]`` of the target height. With no quiet row
  in that window the least busy row is cut, where SmartStitch would keep scanning down. Only
  when every row there is text does a page grow, up to STRETCH_FRACTION.
- Images of differing widths are scaled to the most common width before stitching.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable, Literal, Optional, Sequence

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

# The page height when ``StitchSettings.height`` is 0, relative to the strip width: detection
# models resize a page to a square, so taller pages shrink their lettering vertically.
AUTO_HEIGHT_RATIO = 1.5
MIN_FRACTION = 0.5
MAX_FRACTION = 1.5
# A page may grow to this many target heights when that is the only way not to cut text.
STRETCH_FRACTION = 3.0
# A quiet run this tall (relative to the target height) earns the full bonus, worth being
# GUTTER_WEIGHT target heights away from the target.
GUTTER_FRACTION = 0.1
GUTTER_WEIGHT = 0.5
# A cut keeps this many rows between itself and the art at the edge of its quiet run.
CUT_PADDING = 16
# Detection runs on square tiles centred on the cuts (detection models resize to a square);
# only cuts are checked, so a chapter costs about one tile per page. A cut is settled once it
# lies away from the ends of a detected tile, where a bubble may be clipped; a cut that moves
# elsewhere to avoid what was found is looked at again, at most DETECT_ROUNDS times.
TILE_TRUST = 0.2
DETECT_ROUNDS = 4
# Rows kept clear above and below detected text, bubbles and sound effects.
AVOID_PADDING = 8
# Detection labels a cut may cross: panels span most of the strip.
CUTTABLE_LABELS = frozenset({"panel"})

Format = Literal["manga", "webtoon"]


class StitchSettings(BaseModel):
    """How a webtoon chapter is cut into pages; SmartStitch's settings, with its defaults."""

    # Target page height in pixels; 0 picks AUTO_HEIGHT_RATIO times the strip width.
    height: int = Field(default=0, ge=0, le=20000)
    # 0 cuts anywhere, 100 cuts only perfectly flat rows.
    sensitivity: int = Field(default=90, ge=0, le=100)
    # Rows the scan steps by; cuts land on multiples of it from the window start.
    scan_step: int = Field(default=5, ge=1, le=100)
    # Columns at each side ignored when comparing (borders, scrollbars, watermarks at the edge).
    margin: int = Field(default=0, ge=0, le=1000)
    # Run the detection model over the strip and never cut through what it finds.
    detect: bool = True


class ProjectFormat(BaseModel):
    """What a project holds: manga pages, or webtoon/manhwa/manhua strips that are restitched."""

    kind: Format = "manga"
    stitch: StitchSettings = Field(default_factory=StitchSettings)


def common_width(sizes: Iterable[tuple[int, int]]) -> int:
    """The width covering the most rows; ties go to the narrower width (no upscaling)."""
    rows: Counter[int] = Counter()
    for width, height in sizes:
        rows[width] += height
    if not rows:
        raise ValueError("no images to stitch")
    return max(rows, key=lambda width: (rows[width], -width))


def scaled_height(size: tuple[int, int], width: int) -> int:
    """An image's height once scaled to ``width``."""
    return size[1] if size[0] == width else max(1, round(size[1] * width / size[0]))


def stitch(images: Sequence[Image.Image], width: Optional[int] = None) -> Image.Image:
    """``images`` joined top to bottom, each scaled to ``width`` (default: :func:`common_width`)."""
    if not images:
        raise ValueError("no images to stitch")
    width = width or common_width(image.size for image in images)
    mode = "L" if all(image.mode == "L" for image in images) else "RGB"
    scaled = []
    for image in images:
        image = image if image.mode == mode else image.convert(mode)
        if image.width != width:
            image = image.resize((width, scaled_height(image.size, width)), Image.Resampling.LANCZOS)
        scaled.append(image)
    strip = Image.new(mode, (width, sum(image.height for image in scaled)))
    top = 0
    for image in scaled:
        strip.paste(image, (0, top))
        top += image.height
    return strip


def target_height(settings: StitchSettings, width: int) -> int:
    return settings.height or max(1, round(width * AUTO_HEIGHT_RATIO))


def row_activity(strip: Image.Image, settings: StitchSettings) -> tuple[np.ndarray, np.ndarray]:
    """Per row: whether it may be cut (``quiet``), and how many neighbour differences break the
    threshold (``busy``, to pick the least harmful row when nothing is quiet)."""
    gray = np.asarray(strip.convert("L"), dtype=np.int16)
    margin = min(settings.margin, max(0, (gray.shape[1] - 2) // 2))
    if margin:
        gray = gray[:, margin : gray.shape[1] - margin]
    threshold = int(255 * (1 - settings.sensitivity / 100))
    if gray.shape[1] < 2:
        busy = np.zeros(gray.shape[0], dtype=np.int32)
    else:
        busy = (np.abs(np.diff(gray, axis=1)) > threshold).sum(axis=1, dtype=np.int32)
    return busy == 0, busy


def _runs(quiet: np.ndarray) -> list[tuple[int, int]]:
    """``[start, end)`` of every run of quiet rows."""
    padded = np.concatenate(([False], quiet, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def find_cuts(
    strip: Image.Image,
    settings: StitchSettings | None = None,
    avoid: Iterable[tuple[float, float]] = (),
) -> list[int]:
    """Rows to cut ``strip`` at, starting with 0 and ending with its height.

    ``avoid`` lists ``(top, bottom)`` bands (detected text and bubbles) that no cut may cross.
    """
    settings = settings or StitchSettings()
    end = strip.height
    target = target_height(settings, strip.width)
    quiet, busy = row_activity(strip, settings)
    blocked = np.zeros(end, dtype=bool)
    for top, bottom in avoid:
        blocked[max(0, int(top)) : min(end, int(np.ceil(bottom)) + 1)] = True
    quiet &= ~blocked
    runs = _runs(quiet)
    full_bonus = max(1.0, GUTTER_FRACTION * target)
    step = settings.scan_step

    def quiet_cut(low: int, high: int, ideal: int) -> Optional[int]:
        """The best quiet row in ``[low, high]``: near ``ideal``, in a wide run."""
        best: Optional[tuple[float, int]] = None
        for start, stop in runs:
            if stop <= low or start > high:
                continue
            length = stop - start
            padding = min(CUT_PADDING, (length - 1) // 2)
            first, final = max(start + padding, low), min(stop - 1 - padding, high)
            # The scan grid: SmartStitch only looks at rows scan_step apart.
            first = low + -(-(first - low) // step) * step
            if first > final:
                continue
            row = min(max(ideal, first), final)
            row = first + (row - first) // step * step
            cost = abs(row - ideal) / target - GUTTER_WEIGHT * min(1.0, length / full_bonus)
            if best is None or cost < best[0]:
                best = (cost, row)
        return None if best is None else best[1]

    def least_busy(low: int, high: int, ideal: int, cross: bool = False) -> Optional[int]:
        """The least busy row in ``[low, high]``, nearest ``ideal`` on a tie; one in a band only with ``cross``."""
        if high < low:
            return None
        rows = np.arange(low, high + 1)
        score = busy[low : high + 1].astype(np.float64) + np.abs(rows - ideal) / (10 * target)
        if not cross:
            score[blocked[low : high + 1]] = np.inf
        index = int(np.argmin(score))
        return None if np.isinf(score[index]) else int(rows[index])

    cuts = [0]
    while end - cuts[-1] > MAX_FRACTION * target:
        last = cuts[-1]
        ideal = last + target
        low = last + max(1, int(MIN_FRACTION * target))
        high = min(last + int(MAX_FRACTION * target), end - int(MIN_FRACTION * target))
        if high < low:
            break
        # Cutting through art is harmless (readers show the pages edge to edge) and keeps pages
        # small; text is never cut while a taller page, up to STRETCH_FRACTION, avoids it.
        wide = min(last + int(STRETCH_FRACTION * target), end - int(MIN_FRACTION * target))
        for find in (
            lambda: quiet_cut(low, high, ideal),
            lambda: least_busy(low, high, ideal),
            lambda: quiet_cut(high + 1, wide, ideal),
            lambda: least_busy(high + 1, wide, ideal),
            lambda: least_busy(low, high, ideal, cross=True),
        ):
            row = find()
            if row is not None:
                break
        cuts.append(row)
    cuts.append(end)
    return cuts


def slice_strip(strip: Image.Image, cuts: Sequence[int]) -> list[Image.Image]:
    return [strip.crop((0, top, strip.width, bottom)) for top, bottom in zip(cuts, cuts[1:]) if bottom > top]


def tile_around(strip: Image.Image, row: int) -> tuple[int, Image.Image]:
    """``(top, tile)``: the square of ``strip`` centred on ``row``, kept inside the strip."""
    size = min(strip.width, strip.height)
    top = min(max(0, row - size // 2), strip.height - size)
    return top, strip.crop((0, top, strip.width, top + size))


def detected_bands(tops: Sequence[int], results: Sequence) -> list[tuple[float, float]]:
    """Strip rows covered by what detection found (``LayoutDetections`` per tile), padded."""
    bands = []
    for top, result in zip(tops, results, strict=True):
        for detection in result.detections:
            if detection.label in CUTTABLE_LABELS:
                continue
            _, y0, _, y1 = detection.bbox
            bands.append((top + y0 - AVOID_PADDING, top + y1 + AVOID_PADDING))
    return bands


# Runs the detection model on tiles; one ``LayoutDetections`` per tile.
Detector = Callable[[list[Image.Image]], Sequence]


def restitch(
    images: Sequence[Image.Image],
    settings: StitchSettings | None = None,
    avoid: Sequence[Iterable[tuple[float, float]]] = (),
    detect: Optional[Detector] = None,
) -> list[Image.Image]:
    """Joins ``images`` and cuts the strip again at quiet rows.

    ``avoid`` gives, per image, bands in that image's rows that must not be cut (text the
    project already detected on it). ``detect`` runs detection over the strip; with
    ``settings.detect`` its findings are avoided too.
    """
    settings = settings or StitchSettings()
    width = common_width(image.size for image in images)
    strip = stitch(images, width)
    bands: list[tuple[float, float]] = []
    top = 0
    for index, image in enumerate(images):
        scale = width / image.width
        for y0, y1 in avoid[index] if index < len(avoid) else ():
            bands.append((top + y0 * scale - AVOID_PADDING, top + y1 * scale + AVOID_PADDING))
        top += scaled_height(image.size, width)
    cuts = find_cuts(strip, settings, bands)
    if detect is not None and settings.detect:
        seen: list[tuple[int, int]] = []  # rows detection has settled
        for _ in range(DETECT_ROUNDS):
            unseen = [cut for cut in cuts[1:-1] if not any(low <= cut <= high for low, high in seen)]
            if not unseen:
                break
            tiles = [tile_around(strip, cut) for cut in unseen]
            bands += detected_bands([top for top, _ in tiles], detect([tile for _, tile in tiles]))
            for top, tile in tiles:
                margin = int(tile.height * TILE_TRUST)
                seen.append((top + margin if top else 0, top + tile.height - margin if top + tile.height < strip.height else strip.height))
            cuts = find_cuts(strip, settings, bands)
    return slice_strip(strip, cuts)
