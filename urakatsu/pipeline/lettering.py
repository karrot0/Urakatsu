"""Which lettering style a translation is set in, and how heavy.

Port of Koharu 0.100's ``stages/translate/placement/kernel.rs``, with its constants and the
measurements behind them: a text takes its source's style, as the font reader read it
(``providers/lettering``) and pooled with the project's readings of its kind, when the reader is
sure of its group; a speech balloon's text read as antique or mincho takes the project's dialogue
face of the two, as a book letters its dialogue in one and the font reader cannot tell them
apart; and a text is bold when its source is heavier than the book's dialogue, as a shout's is.

Difference from Koharu: the font reader's own weight (the weights of the fonts it named,
``providers/lettering/font_names.py``) guards the stroke. Koharu calls a source bold when its
glyph stroke is 1.3 times the dialogue's, and so does this; but a stroke measured from a box that
holds a balloon's outline or line art comes out thick, so a source whose fonts read ``LIGHTER``
than the dialogue's is not bold whatever its stroke, and one with no stroke measured is bold when
its fonts read ``HEAVIER``. The fonts' weight does not decide beside a measured stroke: weights
named in different families do not compare (on rendered pages a dialogue of bold mincho kana read
700 and a Heavy gothic shout 770 to 900, with strokes 1.5 times apart), so asking the two to agree
missed real shouts.

Difference from Koharu: the glyph stroke is measured here (``measure_stroke``), from the page's
pixels inside the text's box, since Urakatsu's detection keeps no glyph stroke.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median
from typing import Optional

import cv2
import numpy as np

from urakatsu.core.styles import DEFAULT_STYLE, group

# The least probability of a source's likeliest group of styles that letters it in that group's
# likeliest style rather than antique's. A group's, since the reader splits a face among the
# styles of its look.
SURE = 0.6
# The least probability of the mincho styles that letters a text as them. Antique kana with gothic
# kanji, the commonest dialogue face, reads as them often past 0.9, as the reader knows 21 antique
# fonts against 1,189 mincho ones (Koharu: of Manga109 texts read at 0.995 and up, 87 were in a
# mincho face and 27 in an antique one).
MINCHO = 0.995
# How much heavier than the book's dialogue a source's stroke is to be bold.
HEAVY = 1.3
# On the 100 to 900 scale of the fonts the reader named: how far under the dialogue's a source's
# fonts read for its stroke not to count, and how far over for a source with no stroke measured
# to be bold.
LIGHTER = 150.0
HEAVIER = 200.0
BOLD = 700
# How far a block's reading leans to its project's reading of its kind: the power the others' mean
# takes in their product.
POOL = 0.5
# The share of Han among a text's Han and kana from which it is kanji-heavy, the least such
# dialogue texts, and their least mean reading as the mincho styles, that set a project's dialogue
# in mincho. Antique sets its kanji in gothic, so its kanji-heavy texts seldom read as mincho.
KANJI = 0.4
KANJI_TEXTS = 5
MINCHO_DIALOGUE = 0.7
# The least share of a balloon's outline, simplified to within a fiftieth of its diagonal, running
# within SQUARE degrees of horizontal or vertical that makes it a box. Narration and monologue sit
# in boxes, where speech balloons curve and a shout's spikes run diagonal.
BOX = 0.8
SQUARE = 10.0

# What a block is, which a project letters alike.
BALLOON = "balloon"
BOXED = "box"
FREE = "free"
SOUND_EFFECT = "sound-effect"


@dataclass
class Reading:
    """A block's reading: how likely each style is to be its source's, how heavy it is by its
    stroke over its size and by the fonts named, what it is, and whether its source is kanji-heavy."""

    kernels: dict[str, float] = field(default_factory=dict)
    stroke: Optional[float] = None
    weight: Optional[float] = None
    kind: str = FREE
    kanji: bool = False

    def top(self) -> Optional[tuple[str, float, str]]:
        """Its likeliest group, the probability its styles sum to, and the likeliest style in it."""
        groups: dict[str, float] = {}
        for style, probability in self.kernels.items():
            groups[group(style)] = groups.get(group(style), 0.0) + probability
        if not groups:
            return None
        best = max(groups, key=groups.get)  # type: ignore[arg-type]
        members = {style: probability for style, probability in self.kernels.items() if group(style) == best}
        return best, groups[best], max(members, key=members.get)  # type: ignore[arg-type]


def reading(kernels: dict[str, float], text: str, kind: str, stroke: Optional[float] = None, weight: Optional[float] = None) -> Reading:
    """``read``: a block's reading from what the font reader found and its source text. The mincho
    styles, which kana alone cannot tell from antique, count as antique when the source has no Han."""
    han, kana = count_han_kana(text)
    merged: dict[str, float] = {}
    for style, probability in kernels.items():
        if han == 0 and group(style) == "mincho":
            style = DEFAULT_STYLE
        merged[style] = merged.get(style, 0.0) + probability
    return Reading(kernels=merged, stroke=stroke, weight=weight, kind=kind, kanji=han > 0 and han >= KANJI * (han + kana))


def count_han_kana(text: str) -> tuple[int, int]:
    """How many Han characters and how many kana ``text`` holds."""
    han = kana = 0
    for character in text:
        code = ord(character)
        if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0xF900 <= code <= 0xFAFF or code in (0x3005, 0x3007):
            han += 1
        elif (0x3041 <= code <= 0x309F or 0x30A1 <= code <= 0x30FF) and code not in (0x30FB, 0x30FC):
            kana += 1
    return han, kana


def _confused(style: str) -> bool:
    """Whether ``style``'s group is antique or mincho, which the font reader confuses."""
    return group(style) in ("antique", "mincho")


class Placement:
    """``Kernels``: each block's reading, the project's dialogue face, and how heavy its dialogue
    is, read once for a project. The dialogue is its texts in speech balloons, or every text
    outside them when none is in one."""

    def __init__(self, readings: dict[str, Reading]) -> None:
        self.readings = {key: Reading(dict(r.kernels), r.stroke, r.weight, r.kind, r.kanji) for key, r in readings.items()}
        in_balloons = any(r.kind == BALLOON for r in self.readings.values())

        def dialogue(r: Reading) -> bool:
            return r.kind == BALLOON or (r.kind in (BOXED, FREE) and not in_balloons)

        spoken = [r for r in self.readings.values() if dialogue(r)]
        self.face = dialogue_face(spoken)
        pool(self.readings)
        plain = [r for r in spoken if (top := r.top()) is not None and _confused(top[0])]
        strokes = [r.stroke for r in plain if r.stroke is not None]
        weights = [r.weight for r in plain if r.weight is not None]
        # The medians of the dialogue texts read as antique or mincho.
        self.stroke = median(strokes) if strokes else None
        self.weight = median(weights) if weights else None

    def of(self, key: str) -> tuple[str, Optional[int]]:
        """``Kernels::of``: the style block ``key`` is lettered in, and the weight it is set at
        when its source is heavier than the book's dialogue."""
        r = self.readings.get(key)
        if r is None:
            return DEFAULT_STYLE, None
        top = r.top()
        if top is None:
            style = DEFAULT_STYLE
        elif top[0] == "mincho" and top[1] < MINCHO:
            style = DEFAULT_STYLE
        elif top[1] >= SURE:
            style = top[2]
        else:
            style = DEFAULT_STYLE
        if r.kind == BALLOON and _confused(style):
            style = self.face
        return style, (BOLD if self.heavy(r) else None)

    def heavy(self, r: Reading) -> bool:
        """Whether a source is heavier than the book's dialogue: by its stroke, unless the fonts the
        reader named are lighter than the dialogue's; by those fonts when no stroke is measured."""
        stroke = r.stroke / self.stroke if r.stroke is not None and self.stroke else None
        weight = r.weight - self.weight if r.weight is not None and self.weight is not None else None
        if stroke is None:
            return weight is not None and weight >= HEAVIER
        return stroke >= HEAVY and (weight is None or weight > -LIGHTER)


def dialogue_face(dialogue: list[Reading]) -> str:
    """``face``: mincho when a project's kanji-heavy dialogue reads as the mincho styles, else antique."""
    kanji = [sum(p for style, p in r.kernels.items() if group(style) == "mincho") for r in dialogue if r.kanji]
    if len(kanji) >= KANJI_TEXTS and sum(kanji) / len(kanji) >= MINCHO_DIALOGUE:
        return "mincho"
    return DEFAULT_STYLE


def pool(readings: dict[str, Reading]) -> None:
    """``pool``: each reading times the mean of its project's other readings of its kind, to the
    power ``POOL``, so a reading the crop swings does not stand alone."""
    sums: dict[str, tuple[dict[str, float], float]] = {}
    for r in readings.values():
        if not r.kernels:
            continue
        total, count = sums.get(r.kind, ({}, 0.0))
        for style, probability in r.kernels.items():
            total[style] = total.get(style, 0.0) + probability
        sums[r.kind] = (total, count + 1.0)
    for r in readings.values():
        if r.kind not in sums or sums[r.kind][1] < 2.0 or not r.kernels:
            continue
        total, count = sums[r.kind]
        for style, probability in r.kernels.items():
            others = (total[style] - probability) / (count - 1.0)
            r.kernels[style] = probability * math.pow(max(others, 0.0) + 1e-3, POOL)
        whole = sum(r.kernels.values())
        if whole > 0.0:
            for style in r.kernels:
                r.kernels[style] /= whole


def boxed(points: list[tuple[float, float]]) -> bool:
    """``boxed``: whether a balloon's outline is a box: ``BOX`` or more of it, simplified, runs
    within ``SQUARE`` degrees of horizontal or vertical."""
    if len(points) < 4:
        return False
    outline = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    xs, ys = outline[:, 0, 0], outline[:, 0, 1]
    epsilon = math.hypot(float(xs.max() - xs.min()), float(ys.max() - ys.min())) / 50.0
    ring = cv2.approxPolyDP(outline, epsilon, True).reshape(-1, 2)
    if len(ring) < 3:
        return False
    total = square = 0.0
    for start, end in zip(ring, np.roll(ring, -1, axis=0)):
        across, down = abs(float(end[0] - start[0])), abs(float(end[1] - start[1]))
        length = math.hypot(across, down)
        angle = math.degrees(math.atan2(down, across))
        total += length
        if angle <= SQUARE or angle >= 90.0 - SQUARE:
            square += length
    return total > 0.0 and square >= BOX * total


# measure_stroke: the least ink pixels and the least glyph size, in pixels, worth measuring.
MIN_INK_PIXELS = 24
MIN_GLYPH_SIZE = 6.0


def measure_stroke(rgb: np.ndarray, vertical: bool) -> Optional[float]:
    """A lettering's glyph stroke over its glyph size, which compares its weight with another's at
    any size, from the pixels of its box; None when it holds too little ink to tell.

    The ink is the smaller side of an Otsu threshold. The stroke is the ink's area over half the
    length of its outline, as Leptonica's ``pixFindStrokeWidth`` (which Koharu measures with)
    takes it: a stroke's two sides make up its outline, so their mean length is the stroke's own.
    It is not rounded to pixels, where the depth of a stroke's centre line is, and small text is
    two to four pixels thick. The size is the median breadth of the columns (of the lines, for
    horizontal text) the ink stands in.
    """
    if rgb.ndim != 3 or rgb.shape[0] < 2 or rgb.shape[1] < 2:
        return None
    gray = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = binary == 0
    if ink.mean() > 0.5:
        ink = ~ink
    area = int(ink.sum())
    if area < MIN_INK_PIXELS:
        return None
    # The outline: ink with paper above, below or beside it (the box's edge counts as paper).
    padded = np.pad(ink, 1).astype(np.uint8)
    inner = cv2.erode(padded, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))[1:-1, 1:-1].astype(bool)
    outline = area - int(inner.sum())
    size = _glyph_size(ink, vertical)
    if size is None or size < MIN_GLYPH_SIZE or outline == 0:
        return None
    return round(area / (outline / 2.0) / size, 3)


def _glyph_size(ink: np.ndarray, vertical: bool) -> Optional[float]:
    """The median breadth of the columns (rows of text, when horizontal) the ink stands in."""
    occupied = ink.any(axis=0 if vertical else 1)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], occupied.astype(np.int8), [0]))))
    runs = (edges[1::2] - edges[0::2]).astype(np.float64)
    if len(runs) == 0:
        return None
    # Specks (a dakuten cut off by the box, a stray dot of screentone) are not columns.
    kept = runs[runs >= max(runs.max() * 0.4, 2.0)]
    return float(np.median(kept)) if len(kept) else None
