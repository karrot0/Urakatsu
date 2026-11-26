"""QA's layout half: what is wrong with a page's lettering, from the renderer's own measurements.

No model and no image: ``Renderer.measure_text`` lays out the page's visible text layers exactly
as a render does, without reading a page raster, and every check below reads that layout.

- ``overflow`` and ``small_font`` are the renderer's own ``text_overflow`` and
  ``text_below_readable_size`` diagnostics. Both can be fixed by wording, so the block also gets a
  length (``max_chars``) the proofreader is told to keep to: the characters that would fit at the
  same size (overflow) or at the readable size (small font), from the area the text needs.
- ``off_page``: the ink leaves the page.
- ``overlap``: two blocks' ink boxes share more than ``OVERLAP_SHARE`` of the smaller one.
- ``crosses_contour`` / ``touches_contour`` (a block that flows in a balloon): sample points on each
  glyph, placed from its pen position and the font size, lie outside the balloon's polygon, or
  within ``CONTOUR_MARGIN`` font sizes of its edge. The samples are an estimate of the ink, not
  its outline, so a glyph's far edge may cross a little before this sees it.
- ``off_centre`` (a block in a balloon): the ink box's centre is more than ``OFF_CENTRE`` of the
  balloon's size away from the balloon box's centre.
- ``font_outlier``: font size against the square root of the block's frame area, compared with
  the page's other blocks (at least ``OUTLIER_MINIMUM`` of them, sound effects left out). Only
  for a block with a set size: auto-fit picks each block's size from its text and frame, so a
  short line lettered large is the renderer's choice, not a finding. Often deliberate (a shout, a
  whisper), so it is only information.
- ``overlong`` (an auto-fit block): auto-fit never overflows, it shrinks, so an overlong text shows
  only as small lettering, and a page's sizes vary too much from bubble to bubble to tell that
  from the size alone. What tells it is the text against its source: a translation from a CJK
  source into a Latin script runs about 1 to 6 letters per source character (median 2.3 on the
  QA benchmark's official English), so one longer than ``EXPANSION`` per source character plus
  ``EXPANSION_SLACK`` letters was padded, and auto-fit letters it smaller than its bubble's usual
  size by about the square root of the excess. The proofreader gets the length that would fit.

What a reader sees (round two of the layout review: a blind judge preferred the official MANGA Plus
balloon in 23 of 60 crops, for these reasons), from the block's lines as laid out (``_lines``: the
glyph runs grouped by baseline, matched to the text's line starts) and its balloon's polygon:

- ``underfilled``: ``UNDERFILL_MIN_LINES`` or more lines at a size under ``UNDERFILL_SIZE`` of the
  square root of the balloon box's area: small letters in many short lines leave the balloon
  looking empty; the message gives the usual size (``USUAL_SIZE``).
- ``narrow_column``: ``NARROW_MIN_LINES`` or more lines whose ink box is far narrower for its height
  than the balloon (its width-to-height ratio under ``NARROW_COLUMN`` of the balloon's).
- ``stranded_line``: a line that is only punctuation or a word of ``STRANDED_LETTERS`` letters or
  fewer ("I", "?!") while a neighbouring line has room for it.
- ``hyphenation``: a word broken across two lines that fits whole on one of them at the same size.
- ``crowds_outline``: the ink comes within ``CROWD_MARGIN`` font sizes of the balloon's outline on
  some side (the closer ``touches_contour`` and ``crosses_contour`` take precedence).

These five are ``info``, hints rather than warnings: on the held-out pages a second blind judge
preferred the official balloon in 11 of the 25 they flagged and ours in 10, so they describe the
lettering without predicting that a reader will find it worse. ``UNDERFILL_SIZE`` is at the
stricter of the settings tried for the same reason.

"Room" is measured against the balloon: the chord of its polygon at a line's baseline, times the
largest share of its chord any line of the block already uses (the renderer's own inset).
Text drawn over art inside the box is not checked: that needs the cleaned page's pixels, which
the review never decodes.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Optional

from urakatsu.core.config import TypesettingConfig
from urakatsu.renderer.frame import Layer, RenderBounds, RenderDiagnostic, affine_apply
from urakatsu.renderer.renderer import Renderer
from urakatsu.renderer.text_renderer import GlyphRun
from urakatsu.renderer.types import WritingMode
from urakatsu.scene import components as c
from urakatsu.scene.components import QaIssue
from urakatsu.scene.scene import Snapshot

from ..stages.sound_effects import is_sound_effect

# Kinds this half writes; everything else on a ``QaReview`` came from proofreading.
KINDS = frozenset(
    {
        "overflow", "small_font", "overlong", "font_outlier", "off_centre", "touches_contour", "crosses_contour", "overlap", "off_page",
        "underfilled", "narrow_column", "stranded_line", "hyphenation", "crowds_outline",
    }
)
# Kinds that shorter wording fixes, so the proofreader hears about them.
WORDING = frozenset({"overflow", "small_font", "overlong"})

OFF_PAGE_TOLERANCE = 1.0  # pixels
OVERLAP_SHARE = 0.15
CONTOUR_MARGIN = 0.1  # font sizes
OFF_CENTRE = 0.2  # of the balloon's width or height
OUTLIER_RATIO = 1.75
OUTLIER_MINIMUM = 3
LENGTH_SLACK = 0.9  # a shorter text wraps less evenly than the area says
EXPANSION = 4  # letters per CJK source character beyond which a Latin translation is padded
EXPANSION_SLACK = 20  # letters: short sources ("はあ？" is "WHAT THE HELL?") run long
USUAL_EXPANSION = 2.5  # letters per source character a usual translation runs
# Where a glyph's ink is, relative to its pen position, in font sizes (horizontal text).
GLYPH_SAMPLES = ((0.0, -0.05), (0.0, -0.65), (0.45, -0.05), (0.45, -0.65))
# What a reader sees (see the module notes; tuned on the layout bench's dev pages).
# underfilled: letters small for the balloon (font size against the square root of the balloon
# box's area; the bench's median is 0.12) on a block of several lines, which a bigger size and a
# wider wrap would fill. On the dev pages the ink box's share of the balloon did not tell the
# judge's "too small" losses from wins (one-line shouts in tall balloons were ties), this did.
UNDERFILL_SIZE = 0.095
UNDERFILL_MIN_LINES = 4
USUAL_SIZE = 0.12  # the suggested size, against the same square root
INSCRIBED = 0.62  # share of a balloon's area lettering can use (a rectangle in an oval, with margins)
NARROW_COLUMN = 0.5  # the ink box's width/height against the balloon's
NARROW_MIN_LINES = 3
STRANDED_LETTERS = 2
CROWD_MARGIN = 0.35  # font sizes
FIT_TOLERANCE = 1.02  # a line may run this much past the widest share the block already uses
ADVANCE = 0.6  # a glyph's advance in font sizes, for the last glyph of a line


@dataclass
class BlockLayout:
    """One text layer's layout and what is wrong with it."""

    layer: str
    content: str
    text: str
    font_size: float
    ink: RenderBounds  # page coordinates
    frame: RenderBounds
    balloon: Optional[list[tuple[float, float]]] = None  # the balloon it flows in, page coordinates
    issues: list[QaIssue] = field(default_factory=list)
    # The length that would fix a wording-fixable finding; None when there is none.
    max_chars: Optional[int] = None
    auto_fit: bool = True
    # What the reader checks measured (fill, column, crowding, lines), for benchmarks and debugging.
    measures: dict = field(default_factory=dict)

    def wording_hint(self) -> str:
        """What the proofreader is told about this block's length, or ""."""
        kinds = [issue.kind for issue in self.issues if issue.kind in WORDING]
        if not kinds:
            return ""
        if "overflow" in kinds:
            return "overflows its bubble"
        if "overlong" in kinds:
            return "far longer than its source; shrunk to fit its bubble"
        return "lettered below the readable size"


_renderers: dict[tuple[str, ...], Renderer] = {}


def renderer_for(typesetting: Optional[TypesettingConfig]) -> Renderer:
    """One renderer per font family list, kept: its font scan is the slow part."""
    typesetting = typesetting or TypesettingConfig()
    key = tuple(typesetting.font_families)
    if key not in _renderers:
        _renderers[key] = Renderer(typesetting)
    return _renderers[key]


def check_layout(snapshot: Snapshot, page: str, renderer: Optional[Renderer] = None, typesetting: Optional[TypesettingConfig] = None) -> dict[str, BlockLayout]:
    """Every visible text block of ``page`` and its layout findings, by text content id."""
    renderer = renderer or renderer_for(typesetting)
    layers, diagnostics = renderer.measure_text(snapshot, page)
    page_value = snapshot.component(page, c.Page)
    width, height = (page_value.width, page_value.height) if page_value is not None else (math.inf, math.inf)
    by_entity: dict[str, list[RenderDiagnostic]] = {}
    for diagnostic in diagnostics:
        by_entity.setdefault(diagnostic.entity, []).append(diagnostic)

    blocks: dict[str, BlockLayout] = {}
    for layer in layers:
        meta = layer.kind
        content = snapshot.layer_content(layer.entity)
        block = BlockLayout(
            layer=layer.entity,
            content=content,
            text=meta.text,
            font_size=meta.font_size,
            ink=meta.rendered_bounds,
            frame=meta.layout_bounds,
            balloon=_balloon(snapshot, layer.entity),
        )
        _diagnosed(block, by_entity.get(layer.entity, []))
        typography = snapshot.component(layer.entity, c.Typography)
        block.auto_fit = typography is None or typography.auto_fit
        if block.auto_fit and not any(issue.kind == "overflow" for issue in block.issues):
            source = snapshot.component(content, c.SourceText)
            _overlong(block, source.text.value if source is not None else "")
        _off_page(block, width, height)
        if block.balloon:
            _contour(block, layer, meta.writing_mode)
            _off_centre(block)
            if not meta.writing_mode.is_vertical():
                _as_read(block, layer, meta.line_starts)
        blocks[content] = block
    _overlaps(list(blocks.values()))
    lettered = [block for block in blocks.values() if not is_sound_effect(snapshot, block.content)]
    _outliers([block for block in lettered if not block.auto_fit], lettered)
    return blocks


def _balloon(snapshot: Snapshot, layer: str) -> Optional[list[tuple[float, float]]]:
    target = snapshot.balloon_target(layer)
    geometry = snapshot.component(target, c.Geometry) if target else None
    if geometry is None or len(geometry.points) < 3:
        return None
    return [(point.x, point.y) for point in geometry.points]


def _diagnosed(block: BlockLayout, diagnostics: list[RenderDiagnostic]) -> None:
    length = len(block.text.strip())
    for diagnostic in diagnostics:
        if diagnostic.kind == "text_overflow":
            available = diagnostic.available
            needed = diagnostic.actual_width * diagnostic.actual_height
            share = (available.width * available.height) / needed if available is not None and needed > 0 else 1.0
            fits = max(1, int(length * min(share, 1.0) * LENGTH_SLACK))
            block.max_chars = fits if block.max_chars is None else min(block.max_chars, fits)
            block.issues.append(QaIssue(kind="overflow", severity="error", message=f"The text does not fit its box at {diagnostic.font_size:.0f} px; about {fits} characters would."))
        elif diagnostic.kind == "text_below_readable_size":
            share = (diagnostic.font_size / diagnostic.minimum_font_size) ** 2 if diagnostic.minimum_font_size > 0 else 1.0
            fits = max(1, int(length * share * LENGTH_SLACK))
            block.max_chars = fits if block.max_chars is None else min(block.max_chars, fits)
            block.issues.append(
                QaIssue(
                    kind="small_font",
                    severity="warning",
                    message=f"Lettered at {diagnostic.font_size:.1f} px, below the readable {diagnostic.minimum_font_size:.0f} px; about {fits} characters would fit at that size.",
                )
            )


def _off_page(block: BlockLayout, width: float, height: float) -> None:
    ink = block.ink
    if ink.x < -OFF_PAGE_TOLERANCE or ink.y < -OFF_PAGE_TOLERANCE or ink.x + ink.width > width + OFF_PAGE_TOLERANCE or ink.y + ink.height > height + OFF_PAGE_TOLERANCE:
        block.issues.append(QaIssue(kind="off_page", severity="error", message="The text runs off the page."))


def _ink_points(layer: Layer, writing_mode: WritingMode) -> list[tuple[float, float, float]]:
    """``(x, y, font size)`` samples of where the layer's glyphs put ink, in page coordinates."""
    samples = GLYPH_SAMPLES if not writing_mode.is_vertical() else ((0.0, 0.0),)
    points = []
    for run in layer.scene.runs if layer.scene is not None else []:
        if not isinstance(run, GlyphRun):
            continue
        size = run.font_size
        for _, x, y in run.glyphs:
            for dx, dy in samples:
                local = affine_apply(run.transform, x + dx * size, y + dy * size)
                page = affine_apply(layer.placement, *local)
                points.append((page[0], page[1], size))
    return points


def _contour(block: BlockLayout, layer: Layer, writing_mode: WritingMode) -> None:
    polygon = block.balloon or []
    crossing = touching = False
    for x, y, size in _ink_points(layer, writing_mode):
        distance = _edge_distance(polygon, x, y)
        if not _inside(polygon, x, y):
            if distance > CONTOUR_MARGIN * size:
                crossing = True
                break
            touching = True
        elif distance < CONTOUR_MARGIN * size:
            touching = True
    if crossing:
        block.issues.append(QaIssue(kind="crosses_contour", severity="error", message="The text crosses the balloon's outline."))
    elif touching:
        block.issues.append(QaIssue(kind="touches_contour", severity="warning", message="The text touches the balloon's outline."))


def _off_centre(block: BlockLayout) -> None:
    xs = [x for x, _ in block.balloon or []]
    ys = [y for _, y in block.balloon or []]
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    if width <= 0 or height <= 0:
        return
    dx = (block.ink.x + block.ink.width / 2 - (min(xs) + max(xs)) / 2) / width
    dy = (block.ink.y + block.ink.height / 2 - (min(ys) + max(ys)) / 2) / height
    if abs(dx) > OFF_CENTRE or abs(dy) > OFF_CENTRE:
        side = ("right" if dx > 0 else "left") if abs(dx) >= abs(dy) else ("low" if dy > 0 else "high")
        block.issues.append(QaIssue(kind="off_centre", severity="warning", message=f"The text sits {side} in its balloon, off its centre."))


@dataclass
class _Line:
    text: str
    baseline: float  # page y
    left: float
    right: float
    size: float


def _lines(layer: Layer, text: str, line_starts: tuple[int, ...]) -> Optional[list[_Line]]:
    """The block's lines as laid out: glyphs grouped by baseline, in reading order, each with its
    slice of ``text``; None when the glyph rows and the line starts do not agree."""
    rows: list[list[float]] = []  # [baseline, left, right, size]
    for run in layer.scene.runs if layer.scene is not None else []:
        if not isinstance(run, GlyphRun):
            continue
        for _, x, y in run.glyphs:
            px, py = affine_apply(layer.placement, *affine_apply(run.transform, x, y))
            row = next((r for r in rows if abs(r[0] - py) < 0.3 * run.font_size), None)
            if row is None:
                rows.append([py, px, px + ADVANCE * run.font_size, run.font_size])
            else:
                row[1], row[2] = min(row[1], px), max(row[2], px + ADVANCE * run.font_size)
    rows.sort(key=lambda r: r[0])
    starts = list(line_starts) or [0]
    pieces = [text[start:end] for start, end in zip(starts, starts[1:] + [len(text)])]
    pieces = [piece for piece in pieces if piece.strip()]
    if not rows or len(rows) != len(pieces):
        return None
    return [_Line(piece, row[0], row[1], row[2], row[3]) for piece, row in zip(pieces, rows)]


def _chord(polygon: list[tuple[float, float]], y: float, x: float) -> Optional[tuple[float, float]]:
    """The span of the polygon's interior along the horizontal line ``y`` that holds ``x`` (or the
    widest span)."""
    crossings = []
    for i, (x0, y0) in enumerate(polygon):
        x1, y1 = polygon[(i + 1) % len(polygon)]
        if (y0 > y) != (y1 > y):
            crossings.append(x0 + (y - y0) * (x1 - x0) / (y1 - y0))
    crossings.sort()
    spans = list(zip(crossings[0::2], crossings[1::2]))
    if not spans:
        return None
    return next((span for span in spans if span[0] <= x <= span[1]), max(spans, key=lambda s: s[1] - s[0]))


def _area(polygon: list[tuple[float, float]]) -> float:
    return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + polygon[:1]))) / 2


def _as_read(block: BlockLayout, layer: Layer, line_starts: tuple[int, ...]) -> None:
    """The reader checks (see the module notes) of a horizontal block in a balloon."""
    polygon = block.balloon or []
    xs, ys = [x for x, _ in polygon], [y for _, y in polygon]
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    area = _area(polygon)
    ink_area = block.ink.width * block.ink.height
    if width <= 0 or height <= 0 or area <= 0 or ink_area <= 0:
        return
    lines = _lines(layer, block.text, line_starts) or []
    taken = {issue.kind for issue in block.issues}
    fill = ink_area / area
    growth = math.sqrt(INSCRIBED * area / ink_area)
    aspect = (block.ink.width / block.ink.height) / (width / height)
    relative = block.font_size / math.sqrt(width * height)
    block.measures.update({"fill": round(fill, 3), "growth": round(growth, 3), "aspect_share": round(aspect, 3), "lines": len(lines), "relative_size": round(relative, 4)})
    # Overflow, small text and an overlong line already say why the letters are small.
    if not taken & {"overflow", "small_font", "crosses_contour", "overlong"} and relative < UNDERFILL_SIZE and len(lines) >= UNDERFILL_MIN_LINES:
        usual = USUAL_SIZE * math.sqrt(width * height)
        block.issues.append(
            QaIssue(
                kind="underfilled",
                severity="info",
                message=f"{len(lines)} lines at {block.font_size:.0f} px leave the balloon looking empty; about {usual:.0f} px on fewer, wider lines is usual for its size.",
            )
        )
    if len(lines) >= NARROW_MIN_LINES and aspect < NARROW_COLUMN:
        block.issues.append(
            QaIssue(
                kind="narrow_column",
                severity="info",
                message=f"{len(lines)} lines in a column {block.ink.width / width:.0%} of its balloon's width; a wider wrap would need fewer lines.",
            )
        )
    # The widest share of its chord any line already takes: what the renderer allows here.
    chords = [_chord(polygon, line.baseline - 0.35 * line.size, (line.left + line.right) / 2) for line in lines]
    shares = [(line.right - line.left) / (chord[1] - chord[0]) for line, chord in zip(lines, chords) if chord and chord[1] > chord[0]]
    allowed = max(shares) * FIT_TOLERANCE if shares else 0.0

    def fits(index: int, extra_chars: int) -> bool:
        line, chord = lines[index], chords[index]
        if not chord or allowed <= 0:
            return False
        advance = (line.right - line.left) / max(len(line.text.strip()), 1)
        return (line.right - line.left) + extra_chars * advance <= allowed * (chord[1] - chord[0])

    stranded = hyphenated = False
    for index, line in enumerate(lines):
        piece = line.text.strip()
        letters = sum(character.isalpha() for character in piece)
        if len(lines) > 1 and len(piece.split()) <= 1 and letters <= STRANDED_LETTERS:
            neighbours = [i for i in (index - 1, index + 1) if 0 <= i < len(lines)]
            if any(fits(i, len(piece) + 1) for i in neighbours):
                stranded = True
        if index + 1 < len(lines):
            here, after = line.text.rstrip(), lines[index + 1].text.lstrip()
            broken = here and after and after[0].isalpha() and (here[-1].isalpha() and line.text == here or here.endswith("-") and here[-2:-1].isalpha())
            if broken:
                head = here.split()[-1].rstrip("-") if here.split() else ""
                tail = after.split()[0] if after.split() else ""
                if fits(index, len(tail)) or fits(index + 1, len(head) + 1):
                    hyphenated = True
    block.measures.update({"stranded": stranded, "hyphenated": hyphenated})
    if stranded:
        block.issues.append(QaIssue(kind="stranded_line", severity="info", message="A line holds only a short word or punctuation that the line next to it has room for."))
    if hyphenated:
        block.issues.append(QaIssue(kind="hyphenation", severity="info", message="A word is broken across two lines although it fits whole on one of them."))
    if not taken & {"touches_contour", "crosses_contour"}:
        closest = min((_edge_distance(polygon, x, y) / size for x, y, size in _ink_points(layer, WritingMode.HORIZONTAL)), default=math.inf)
        block.measures["crowd"] = round(closest, 3)
        if closest < CROWD_MARGIN:
            block.issues.append(QaIssue(kind="crowds_outline", severity="info", message=f"The text comes within {closest:.2f} of its size of the balloon's outline."))


def _overlaps(blocks: list[BlockLayout]) -> None:
    for index, first in enumerate(blocks):
        for second in blocks[index + 1 :]:
            shared = _intersection(first.ink, second.ink)
            smaller = min(first.ink.width * first.ink.height, second.ink.width * second.ink.height)
            if smaller > 0 and shared > OVERLAP_SHARE * smaller:
                for block in (first, second):
                    if not any(issue.kind == "overlap" for issue in block.issues):
                        block.issues.append(QaIssue(kind="overlap", severity="warning", message="The text overlaps another text block."))


def _overlong(block: BlockLayout, source: str) -> None:
    """``overlong`` (see the module notes)."""
    wide = sum(1 for character in source if character.isalpha() and _is_cjk(character))
    letters = [character for character in block.text if character.isalpha()]
    if wide == 0 or not letters or any(_is_cjk(character) for character in letters):
        return
    allowed = EXPANSION * wide + EXPANSION_SLACK
    if len(letters) <= allowed:
        return
    usual = USUAL_EXPANSION * wide + EXPANSION_SLACK / 2
    fits = max(1, int(len(block.text.strip()) * usual / len(letters)))
    usual_size = block.font_size * math.sqrt(len(letters) / usual)
    block.max_chars = fits if block.max_chars is None else min(block.max_chars, fits)
    block.issues.append(
        QaIssue(
            kind="overlong",
            severity="warning",
            message=f"{len(letters)} letters for {wide} source characters: auto-fit letters it at {block.font_size:.0f} px, where a usual-length line would get about {usual_size:.0f} px. About {fits} characters would do.",
        )
    )


def _is_cjk(character: str) -> bool:
    code = ord(character)
    return 0x3040 <= code <= 0x30FF or 0x3400 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF or 0xAC00 <= code <= 0xD7AF or 0xFF66 <= code <= 0xFF9F


def _outliers(blocks: list[BlockLayout], page: Optional[list[BlockLayout]] = None) -> None:
    """Flags ``blocks`` whose size stands out from ``page``'s (``blocks`` itself when not given)."""
    ratios = {}
    for block in page if page is not None else blocks:
        area = block.frame.width * block.frame.height
        if area > 0 and block.font_size > 0:
            ratios[block.content] = block.font_size / math.sqrt(area)
    if len(ratios) < OUTLIER_MINIMUM:
        return
    typical = statistics.median(ratios.values())
    for block in blocks:
        ratio = ratios.get(block.content)
        if ratio is None or typical <= 0:
            continue
        if ratio > typical * OUTLIER_RATIO or ratio < typical / OUTLIER_RATIO:
            bigger = ratio > typical
            block.issues.append(
                QaIssue(
                    kind="font_outlier",
                    severity="info",
                    message=f"At {block.font_size:.0f} px the text is much {'larger' if bigger else 'smaller'} than the page's other text for its bubble's size.",
                )
            )


def _intersection(a: RenderBounds, b: RenderBounds) -> float:
    width = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)
    height = min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
    return width * height if width > 0 and height > 0 else 0.0


def _inside(polygon: list[tuple[float, float]], x: float, y: float) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _edge_distance(polygon: list[tuple[float, float]], x: float, y: float) -> float:
    best = math.inf
    for i, (x0, y0) in enumerate(polygon):
        x1, y1 = polygon[(i + 1) % len(polygon)]
        dx, dy = x1 - x0, y1 - y0
        length = dx * dx + dy * dy
        t = 0.0 if length == 0 else max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / length))
        best = min(best, math.hypot(x - (x0 + t * dx), y - (y0 + t * dy)))
    return best
