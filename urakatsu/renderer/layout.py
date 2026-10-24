"""Unicode-aware text shaping, line breaking, and layout.

Port of ``layout.rs``. Function and constant names follow the
Rust source; comments cite the Rust item they mirror.

Divergences:
- Offsets (``LayoutLine.range``, cluster values) are code-point indices, not UTF-8 bytes.
- Bidi levels come from ``python-bidi`` (UAX #9) instead of ``unicode-bidi``.
- Unicode NFKC normalization uses ``unicodedata.normalize`` instead of ICU's ``new_nfkc`` normalizer.
"""

from __future__ import annotations

import copy
import math
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence, TypeVar

from bidi.algorithm import (  # UAX #9 resolution steps, in the order python-bidi's get_display runs them
    explicit_embed_and_overrides,
    get_base_level,
    get_embedding_levels,
    get_empty_storage,
    resolve_implicit_levels,
    resolve_neutral_types,
    resolve_weak_types,
)

from .fonts import Font
from .script import shaping_direction_for_text
from .segment import (
    HyphenationOptions,
    LineBreakSuffix,
    LineBreaker,
    hyphenation_bounds,
    hyphenation_lang_from_tag,
)
from .shape import PositionedGlyph, ShapedRun, ShapingOptions, TextShaper, shape_script_runs
from .types import TextAlign, WritingMode

HYPHENATION_MIN_WORD_LEN = 5
COMPACT_HYPHENATION_FRAGMENT_LEN = 2
LINE_BREAK_HYPHEN_PENALTY = 2_000.0
LINE_BREAK_OVERFLOW_MULTIPLIER = 10_000.0
COMIC_LINE_OVERFLOW_PENALTY = 1_000_000.0
COMIC_MAX_LINES = 64


class HyphenationPolicy(Enum):
    """``HyphenationPolicy`` (layout.rs)."""

    DISABLED = "disabled"
    # Use a discretionary hyphen only when the unhyphenated text overflows.
    LAST_RESORT = "last_resort"
    # Consider discretionary hyphens during normal line optimization (default).
    NORMAL = "normal"


@dataclass
class LayoutLine:
    """``LayoutLine``: positioned glyphs for one line plus its baseline."""

    glyphs: list[PositionedGlyph] = field(default_factory=list)
    range: tuple[int, int] = (0, 0)
    advance: float = 0.0
    baseline: tuple[float, float] = (0.0, 0.0)


@dataclass
class LayoutRun:
    """``LayoutRun``: laid-out lines in local layout coordinates."""

    lines: list[LayoutLine]
    width: float
    height: float
    font_size: float
    overflowed_flag: bool
    placement_offset_x: float
    placement_offset_y: float

    def overflowed(self) -> bool:
        return self.overflowed_flag


@dataclass
class LineRun:
    shaped: ShapedRun
    level: int


@dataclass
class ShapedBreakSuffix:
    runs: list[LineRun]
    advance: float


@dataclass
class ShapedSegment:
    range: tuple[int, int]
    visible_end: int
    next_offset: int
    is_mandatory: bool
    runs: list[LineRun]
    advance: float
    trailing_advance: float
    break_penalty: float
    break_suffix: Optional[ShapedBreakSuffix]


@dataclass(frozen=True)
class LineBreakMeasure:
    advance: float
    trailing_advance: float
    break_suffix_advance: float
    break_penalty: float
    is_mandatory: bool


@dataclass(frozen=True)
class LineProfile:
    """Available extent on the inline axis: X for horizontal text, Y for vertical text."""

    width: float
    center_offset: float
    block_baseline: float


@dataclass(frozen=True)
class InkBand:
    before: float
    after: float

    def thickness(self) -> float:
        return self.before + self.after


@dataclass
class LineBreakResult:
    breaks: list[int]
    profiles: list[LineProfile]
    overflowed: bool
    cost: float


T = TypeVar("T")


def largest_fitting_font_size(
    minimum: float,
    maximum: float,
    layout_at: Callable[[float], T],
    fits: Callable[[T], bool],
) -> Optional[T]:
    """``largest_fitting_font_size``: eight coarse probes from the top, then bisection."""
    PROBES = 8
    if maximum - minimum <= 1.1920929e-07:  # f32::EPSILON
        candidate = layout_at(maximum)
        return candidate if fits(candidate) else None
    step = (maximum - minimum) / PROBES
    larger_non_fit: Optional[float] = None
    for probe in range(PROBES + 1):
        size = minimum if probe == PROBES else maximum - step * probe
        candidate = layout_at(size)
        if fits(candidate):
            high = larger_non_fit
            if high is None:
                return candidate
            low = size
            best = candidate
            iterations = 0
            while high - low > 0.01 and iterations < 10:
                iterations += 1
                midpoint = (low + high) * 0.5
                candidate = layout_at(midpoint)
                if fits(candidate):
                    best = candidate
                    low = midpoint
                else:
                    high = midpoint
            return best
        larger_non_fit = size
    return None


class TextLayout:
    """``TextLayout``: builder for one text block. ``with_*`` methods mutate and return self."""

    def __init__(self, font: Font) -> None:
        self.writing_mode: WritingMode = WritingMode.HORIZONTAL
        self.center_vertical_punctuation: bool = True
        self.hyphenation_lang: Optional[str] = "English"
        self.hyphenation_policy: HyphenationPolicy = HyphenationPolicy.NORMAL
        self.comic_balloon: Optional[ComicBalloon] = None
        self.font: Font = font
        self.fallback_fonts: Sequence[Font] = ()
        self.font_size: Optional[float] = None
        self.min_font_size: Optional[float] = None
        self.max_font_size: Optional[float] = None
        self.max_width: Optional[float] = None
        self.max_height: Optional[float] = None
        self.alignment: Optional[TextAlign] = None
        self.line_height: Optional[float] = None
        self.letter_spacing: float = 0.0
        self.word_spacing: float = 0.0
        self.cjk_punctuation_layout: bool = False
        # Rich text: each character's face key (bold, italic, family) with a face list per key, and its size
        # relative to the layout's.
        self.style_keys: Optional[list[tuple]] = None
        self.style_fonts: dict[tuple, list[Font]] = {}
        self.style_scales: Optional[list[float]] = None

    def with_font_size(self, size: float) -> "TextLayout":
        self.font_size = size
        self.min_font_size = None
        self.max_font_size = None
        return self

    def with_max_font_size(self, size: float) -> "TextLayout":
        """Automatically fit the text to its bounds without growing beyond ``size``."""
        self.font_size = None
        self.max_font_size = size
        return self

    def with_min_font_size(self, size: float) -> "TextLayout":
        """Prevent automatically fitted text from becoming unreadably small."""
        self.font_size = None
        self.min_font_size = size
        return self

    def with_writing_mode(self, mode: WritingMode) -> "TextLayout":
        self.writing_mode = mode
        return self

    def with_hyphenation_language_tag(self, tag: str) -> "TextLayout":
        self.hyphenation_lang = hyphenation_lang_from_tag(tag)
        return self

    def with_hyphenation_policy(self, policy: HyphenationPolicy) -> "TextLayout":
        self.hyphenation_policy = policy
        return self

    def with_comic_balloon(
        self,
        width: float,
        height: float,
        contour: list[tuple[float, float]],
        minimum_air: float,
    ) -> "TextLayout":
        self.comic_balloon = ComicBalloon(width, height, contour, max(minimum_air, 0.0))
        return self

    def with_fallback_fonts(self, fonts: Sequence[Font]) -> "TextLayout":
        self.fallback_fonts = list(fonts)
        return self

    def with_max_width(self, width: float) -> "TextLayout":
        self.max_width = width
        return self

    def with_max_height(self, height: float) -> "TextLayout":
        self.max_height = height
        return self

    def with_alignment(self, alignment: TextAlign) -> "TextLayout":
        self.alignment = alignment
        return self

    def with_line_height(self, ratio: float) -> "TextLayout":
        self.line_height = ratio
        return self

    def with_spacing(self, letter: float, word: float) -> "TextLayout":
        self.letter_spacing = letter
        self.word_spacing = word
        return self

    def with_cjk_punctuation_layout(self, enabled: bool) -> "TextLayout":
        self.cjk_punctuation_layout = enabled
        return self

    def with_styles(self, keys: list[tuple], fonts: dict[tuple, list[Font]], scales: Optional[list[float]] = None) -> "TextLayout":
        """Rich text (not in Koharu): character i is shaped with ``fonts[keys[i]]`` (its bold, italic or
        family) instead of the layout's faces, at ``scales[i]`` times the layout's size. Those faces and
        sizes have their own advances, so they take part in line breaking and the fitted size."""
        self.style_keys = keys
        self.style_fonts = fonts
        self.style_scales = scales
        return self

    def scale_at(self, index: int) -> float:
        if self.style_scales is None or not 0 <= index < len(self.style_scales):
            return 1.0
        return self.style_scales[index]

    def fonts_at(self, index: int) -> list[Font]:
        default = [self.font, *self.fallback_fonts]
        if self.style_keys is None or not 0 <= index < len(self.style_keys):
            return default
        return self.style_fonts.get(self.style_keys[index], default)

    def style_pieces(self, start: int, end: int) -> list[tuple[int, int]]:
        """``[start, end)`` split where the faces or sizes change."""
        if self.style_keys is None and self.style_scales is None:
            return [(start, end)]
        pieces = []
        piece_start = start
        for index in range(start + 1, end):
            key_changes = self.style_keys is not None and self.style_keys[index] != self.style_keys[index - 1]
            if key_changes or self.scale_at(index) != self.scale_at(index - 1):
                pieces.append((piece_start, index))
                piece_start = index
        pieces.append((piece_start, end))
        return pieces

    def run(self, text: str) -> LayoutRun:
        if self.font_size is not None:
            return self.run_with_size(text, self.font_size)
        return self.run_auto(text)

    def run_auto(self, text: str) -> LayoutRun:
        """``TextLayout::run_auto``: automatic size search (largest fitting size)."""
        max_height = self.max_height if self.max_height is not None else math.inf
        max_width = self.max_width if self.max_width is not None else math.inf
        maximum = max(self.max_font_size if self.max_font_size is not None else 300.0, 0.5)
        minimum = min(
            max(self.min_font_size if self.min_font_size is not None else min(maximum, 1.0), 0.5),
            maximum,
        )
        eps = 1.1920929e-07

        def fits(layout: LayoutRun) -> bool:
            return (
                not layout.overflowed()
                and layout.width <= max_width + eps
                and layout.height <= max_height + eps
            )

        if self.comic_balloon is not None:
            # A balloon's usable width changes when the text reflows to a different number of
            # lines, so a smaller font can fail even though a larger one fits. Search from
            # largest to smallest instead of assuming monotonicity.
            if self.hyphenation_policy == HyphenationPolicy.LAST_RESORT:
                unhyphenated = self.clone()
                unhyphenated.hyphenation_policy = HyphenationPolicy.DISABLED
                clean = largest_fitting_font_size(
                    minimum, maximum, lambda size: unhyphenated.run_with_size(text, size), fits
                )
                # Avoiding a word break is no longer useful once it pins the text to the
                # configured readability floor. Compare raster-size buckets so hyphenation must
                # recover a visible pixel, not a tuned percentage.
                if clean is not None and math.floor(clean.font_size) > math.floor(minimum):
                    return clean
                hyphenated = largest_fitting_font_size(
                    minimum, maximum, lambda size: self.run_with_size(text, size), fits
                )
                if clean is not None and hyphenated is not None:
                    if math.floor(hyphenated.font_size) > math.floor(clean.font_size):
                        return hyphenated
                    return clean
                if clean is not None:
                    return clean
                if hyphenated is not None:
                    return hyphenated
            else:
                best = largest_fitting_font_size(
                    minimum, maximum, lambda size: self.run_with_size(text, size), fits
                )
                if best is not None:
                    return best
            return self.run_with_size(text, minimum)

        maximum_layout = self.run_with_size(text, maximum)
        if fits(maximum_layout):
            return maximum_layout

        best = self.run_with_size(text, minimum)
        if not fits(best):
            return best

        low = minimum
        high = maximum
        iterations = 0
        while high - low > 0.01 and iterations < 16:
            iterations += 1
            size = (low + high) * 0.5
            layout = self.run_with_size(text, size)
            if fits(layout):
                best = layout
                low = size
            else:
                high = size
        return best

    def clone(self) -> "TextLayout":
        import copy

        return copy.copy(self)

    def run_with_size(self, text: str, font_size: float) -> LayoutRun:
        """``TextLayout::run_with_size``: shape, break, balance lines, position baselines."""
        shaper = TextShaper()
        line_breaker = LineBreaker()
        if (
            self.writing_mode is not WritingMode.VERTICAL_RL
            and self.hyphenation_policy != HyphenationPolicy.DISABLED
            and self.hyphenation_lang is not None
        ):
            minimum_prefix_length, minimum_suffix_length = _bounds_of(self.hyphenation_lang)
            options = HyphenationOptions.new(self.hyphenation_lang, HYPHENATION_MIN_WORD_LEN)
            if self.comic_balloon is not None:
                options = options.with_fragment_bounds(
                    min(minimum_prefix_length, COMPACT_HYPHENATION_FRAGMENT_LEN),
                    min(minimum_suffix_length, COMPACT_HYPHENATION_FRAGMENT_LEN),
                )
            line_breaker = line_breaker.with_hyphenation(options)

        # Use real font metrics for consistent line sizing across modes.
        ascent, descent_negative, leading = self.font.metrics(font_size)
        descent = -descent_negative
        if self.line_height is not None:
            line_height = font_size * self.line_height
        else:
            line_height = max(ascent + descent + leading, font_size)

        levels = _embedding_levels(text)
        # Rich text sizes: balloons and vertical columns fit on one line pitch, so it grows with the largest
        # run; horizontal lines each grow with their own largest run (below, at the baselines).
        largest_scale = max(self.style_scales) if self.style_scales else 1.0
        uniform_pitch = self.comic_balloon is not None or self.writing_mode.is_vertical()
        if uniform_pitch and largest_scale != 1.0:
            line_height *= largest_scale
            ascent *= largest_scale
            descent *= largest_scale

        direction, script = shaping_direction_for_text(text, self.writing_mode)
        features = {"vert": 1, "vrt2": 1} if self.writing_mode.is_vertical() else {}
        options = ShapingOptions(direction=direction, script=script, font_size=font_size, features=features)
        max_extent = (self.max_height if self.max_height is not None else math.inf) if self.writing_mode.is_vertical() else (
            self.max_width if self.max_width is not None else math.inf
        )

        def shape_break_suffix(suffix: LineBreakSuffix, level: int, cluster: int) -> ShapedBreakSuffix:
            scale = self.scale_at(cluster - 1)
            suffix_options = ShapingOptions(
                direction="rtl" if level % 2 == 1 else "ltr",
                script=options.script,
                font_size=options.font_size * scale,
                features=options.features,
            )
            runs: list[LineRun] = []
            advance = 0.0
            # A hyphen takes the formatting of the character it follows.
            for shaped in shape_script_runs(shaper, suffix.as_str(), self.fonts_at(cluster - 1), suffix_options):
                for glyph in shaped.glyphs:
                    glyph.cluster += cluster
                    glyph.scale = scale
                advance += abs(shaped.x_advance)
                runs.append(LineRun(shaped=shaped, level=level))
            return ShapedBreakSuffix(runs=runs, advance=advance)

        shaped_segments: list[ShapedSegment] = []
        for segment in line_breaker.line_segments(text):
            seg_start, seg_end = segment.range
            segment_text = text[seg_start:seg_end]
            visible_end = seg_start + len(segment_text.rstrip())

            segment_runs: list[LineRun] = []
            segment_advance = 0.0

            if segment_text:
                # Subdivide segment into constant BiDi level runs.
                index = seg_start
                while index < seg_end:
                    run_start = index
                    level = levels[run_start]
                    run_end = seg_end
                    scan = index + 1
                    while scan < seg_end:
                        if levels[scan] != level:
                            run_end = scan
                            break
                        scan += 1
                    index = run_end

                    for piece_start, piece_end in self.style_pieces(run_start, run_end):
                        run_text = text[piece_start:piece_end]
                        scale = self.scale_at(piece_start)
                        piece_size = font_size * scale
                        if self.writing_mode.is_vertical():
                            run_direction = "ttb"
                        elif level % 2 == 1:
                            run_direction = "rtl"
                        else:
                            run_direction = "ltr"
                        run_options = ShapingOptions(
                            direction=run_direction,
                            script=options.script,
                            font_size=piece_size,
                            features=options.features,
                        )

                        normalized_punctuation = (
                            normalize_cjk_emphasis_punctuation(run_text)
                            if self.cjk_punctuation_layout
                            else None
                        )
                        shaping_text = normalized_punctuation[0] if normalized_punctuation else run_text
                        script_runs = shape_script_runs(shaper, shaping_text, self.fonts_at(piece_start), run_options)
                        for shaped in script_runs:
                            self.apply_spacing(shaping_text, shaped)
                            if self.writing_mode.is_vertical() and self.center_vertical_punctuation:
                                self.center_vertical_punctuation_glyphs(piece_size, shaping_text, shaped.glyphs)
                            if self.cjk_punctuation_layout:
                                self.layout_cjk_emphasis_runs(piece_size, shaping_text, shaped)
                            if normalized_punctuation is not None:
                                cluster_map = normalized_punctuation[1]
                                for glyph in shaped.glyphs:
                                    if 0 <= glyph.cluster < len(cluster_map):
                                        glyph.cluster = cluster_map[glyph.cluster]

                            for glyph in shaped.glyphs:
                                glyph.cluster += piece_start
                                glyph.scale = scale

                            if self.writing_mode.is_vertical():
                                segment_advance += abs(shaped.y_advance)
                            else:
                                segment_advance += abs(shaped.x_advance)

                            segment_runs.append(LineRun(shaped=shaped, level=level))

            segment_break_suffix = None
            if segment.break_suffix is not None and segment_runs:
                segment_break_suffix = shape_break_suffix(
                    segment.break_suffix, segment_runs[-1].level, seg_end
                )

            trailing_advance = 0.0
            for run in segment_runs:
                for glyph in run.shaped.glyphs:
                    if glyph.cluster >= visible_end:
                        trailing_advance += (
                            abs(glyph.y_advance) if self.writing_mode.is_vertical() else abs(glyph.x_advance)
                        )

            shaped_segments.append(
                ShapedSegment(
                    range=segment.range,
                    visible_end=visible_end,
                    next_offset=segment.next_offset,
                    is_mandatory=segment.is_mandatory,
                    runs=segment_runs,
                    advance=segment_advance,
                    trailing_advance=trailing_advance,
                    break_penalty=comic_break_penalty(text, segment.next_offset)
                    if self.comic_balloon is not None
                    else 0.0,
                    break_suffix=segment_break_suffix,
                )
            )

        if self.writing_mode.is_vertical():
            fallback_ink = InkBand(before=line_height * 0.5, after=line_height * 0.5)
        else:
            fallback_ink = InkBand(before=ascent, after=descent)
        if self.comic_balloon is not None:
            line_ink = self.shaped_block_extents(font_size, shaped_segments) or fallback_ink
        else:
            line_ink = fallback_ink
        balloon_air = self.comic_balloon.air(line_ink.thickness()) if self.comic_balloon is not None else 0.0

        lines: list[LayoutLine] = []
        line_profiles: list[LineProfile] = []
        contour_overflowed = False
        line_offset = [0]
        if self.comic_balloon is not None:
            contour_overflowed = self.append_balanced_segment_lines(
                shaped_segments,
                line_offset,
                len(text),
                False,
                max_extent,
                line_height,
                line_ink,
                balloon_air,
                balloon_air,
                levels,
                lines,
                line_profiles,
            )
        else:
            paragraph_start = 0
            for index, segment in enumerate(shaped_segments):
                if not segment.is_mandatory:
                    continue
                contour_overflowed |= self.append_balanced_segment_lines(
                    shaped_segments[paragraph_start : index + 1],
                    line_offset,
                    segment.next_offset,
                    True,
                    max_extent,
                    line_height,
                    line_ink,
                    balloon_air,
                    balloon_air,
                    levels,
                    lines,
                    line_profiles,
                )
                paragraph_start = index + 1
            if paragraph_start < len(shaped_segments):
                contour_overflowed |= self.append_balanced_segment_lines(
                    shaped_segments[paragraph_start:],
                    line_offset,
                    len(text),
                    False,
                    max_extent,
                    line_height,
                    line_ink,
                    balloon_air,
                    balloon_air,
                    levels,
                    lines,
                    line_profiles,
                )

        # BASELINES (layout.rs: per-line baseline from line index and metrics; vertical X absolute).
        line_count = len(lines)
        effective_alignment = self.alignment if self.alignment is not None else TextAlign.LEFT
        line_top = 0.0
        for i, line in enumerate(lines):
            if self.writing_mode.is_vertical():
                if self.comic_balloon is not None and i < len(line_profiles):
                    x = line_profiles[i].block_baseline
                else:
                    x = (max(line_count - 1, 0) - i) * line_height + line_height * 0.5
                line.baseline = (x, ascent)
            else:
                if self.comic_balloon is not None and i < len(line_profiles):
                    y = line_profiles[i].block_baseline
                elif uniform_pitch or largest_scale == 1.0:
                    y = ascent + i * line_height
                else:
                    # A line is as tall as its largest run.
                    line_scale = max((glyph.scale for glyph in line.glyphs), default=1.0)
                    y = line_top + ascent * line_scale
                    line_top += line_height * line_scale
                line.baseline = (0.0, y)

        if effective_alignment == TextAlign.JUSTIFY and not self.writing_mode.is_vertical():
            justify_lines(text, lines, max_extent, line_profiles)

        # Alignment is an inline-axis operation inside one intrinsic text block. Balloon contours
        # constrain line breaking and block placement, but do not bend the text axis.
        PAD = 1.0
        inline_extent = 0.0
        for line in lines:
            bounds = self.ink_bounds(font_size, [line])
            if bounds is None:
                continue
            min_x, min_y, max_x, max_y = bounds
            extent = (max_y - min_y) if self.writing_mode.is_vertical() else (max_x - min_x)
            inline_extent = max(inline_extent, extent)
        inline_box = None
        if inline_extent > 0.0:
            extent = inline_extent + PAD * 2.0
            inline_box = (extent, PAD, extent * 0.5, extent - PAD)
        if inline_box is not None:
            _, inline_start, inline_center, inline_end = inline_box
            for line in lines:
                bounds = self.ink_bounds(font_size, [line])
                if bounds is None:
                    continue
                min_x, min_y, max_x, max_y = bounds
                if self.writing_mode.is_vertical():
                    minimum, center, maximum = min_y, (min_y + max_y) * 0.5, max_y
                else:
                    minimum, center, maximum = min_x, (min_x + max_x) * 0.5, max_x
                if effective_alignment in (TextAlign.LEFT, TextAlign.JUSTIFY):
                    offset = inline_start - minimum
                elif effective_alignment == TextAlign.CENTER:
                    offset = inline_center - center
                else:
                    offset = inline_end - maximum
                bx, by = line.baseline
                if self.writing_mode.is_vertical():
                    line.baseline = (bx, by + offset)
                else:
                    line.baseline = (bx + offset, by)

        # Tight ink bounding box from per-glyph bounds, then normalize only the block axis.
        # The inline axis remains in the fixed layout box.
        width = 0.0
        height = 0.0
        placement_offset_x = 0.0
        placement_offset_y = 0.0
        ink = self.ink_bounds(font_size, lines)
        if ink is not None:
            min_x, min_y, max_x, max_y = ink
            # Keep a tiny safety pad for hinting/AA differences.
            min_x -= PAD
            min_y -= PAD
            max_x += PAD
            max_y += PAD

            for line in lines:
                bx, by = line.baseline
                if inline_box is None or self.writing_mode.is_vertical():
                    bx -= min_x
                if inline_box is None or not self.writing_mode.is_vertical():
                    by -= min_y
                line.baseline = (bx, by)
            actual_width = max(max_x - min_x, 0.0)
            actual_height = max(max_y - min_y, 0.0)
            if self.writing_mode.is_vertical():
                width = actual_width
                height = inline_box[0] if inline_box is not None else actual_height
            else:
                height = actual_height
                width = inline_box[0] if inline_box is not None else actual_width

            if self.comic_balloon is not None:
                balloon = self.comic_balloon
                inline_offset = line_profiles[0].center_offset if line_profiles else 0.0
                if self.writing_mode.is_vertical():
                    default_left = (balloon.width - width) * 0.5
                    placement_offset_x = min_x - default_left
                    placement_offset_y = inline_offset
                else:
                    default_top = (balloon.height - height) * 0.5
                    placement_offset_x = inline_offset
                    placement_offset_y = min_y - default_top

        eps = 1.1920929e-07
        overflowed = (
            contour_overflowed
            or (self.max_width is not None and width > self.max_width + eps)
            or (self.max_height is not None and height > self.max_height + eps)
        )

        return LayoutRun(
            lines=lines,
            width=width,
            height=height,
            font_size=font_size,
            overflowed_flag=overflowed,
            placement_offset_x=placement_offset_x,
            placement_offset_y=placement_offset_y,
        )

    def shaped_block_extents(self, font_size: float, segments: Sequence[ShapedSegment]) -> Optional[InkBand]:
        """``shaped_block_extents``: block-axis ink extents of all shaped glyphs (balloon fit)."""
        minimum = math.inf
        maximum = -math.inf
        for segment in segments:
            suffix_runs = segment.break_suffix.runs if segment.break_suffix is not None else []
            for run in [*segment.runs, *suffix_runs]:
                for glyph in run.shaped.glyphs:
                    size = font_size * glyph.scale
                    bounds = glyph.font.glyph_bounds(glyph.glyph_id, size)
                    if bounds is None:
                        continue
                    x_min, y_min, x_max, y_max = bounds
                    if self.writing_mode.is_vertical():
                        synthetic_pad = (size * 0.25 if glyph.font.synthetic_skew() is not None else 0.0) + (
                            size * 0.05 if glyph.font.synthetic_bold() else 0.0
                        )
                        minimum = min(minimum, glyph.x_offset + x_min - synthetic_pad)
                        maximum = max(maximum, glyph.x_offset + x_max + synthetic_pad)
                    else:
                        minimum = min(minimum, -glyph.y_offset - y_max)
                        maximum = max(maximum, -glyph.y_offset - y_min)
        if math.isfinite(minimum):
            return InkBand(before=max(-minimum, 0.0), after=max(maximum, 0.0))
        return None

    def append_balanced_segment_lines(
        self,
        segments: Sequence[ShapedSegment],
        line_offset: list[int],
        final_next_offset: int,
        force_final_line: bool,
        max_extent: float,
        line_height: float,
        line_ink: InkBand,
        balloon_air_x: float,
        balloon_air_y: float,
        levels: Sequence[int],
        lines: list[LayoutLine],
        line_profiles: list[LineProfile],
    ) -> bool:
        """``append_balanced_segment_lines``: choose breaks, then emit the chosen lines."""
        if not segments:
            if force_final_line:
                line_offset[0] = self.push_layout_line(
                    [], line_offset[0], line_offset[0], final_next_offset, None, True, levels, lines
                )
                line_profiles.append(LineProfile(width=max_extent, center_offset=0.0, block_baseline=0.0))
            return False

        measures = [
            LineBreakMeasure(
                advance=segment.advance,
                trailing_advance=segment.trailing_advance,
                break_suffix_advance=segment.break_suffix.advance if segment.break_suffix is not None else 0.0,
                break_penalty=segment.break_penalty,
                is_mandatory=segment.is_mandatory,
            )
            for segment in segments
        ]
        if self.comic_balloon is not None:
            result = comic_line_breaks(
                measures,
                self.comic_balloon,
                self.writing_mode,
                line_height,
                line_ink,
                (balloon_air_x, balloon_air_y),
                self.hyphenation_policy,
            )
        elif math.isfinite(max_extent) and max_extent > 0.0:
            result = line_breaks_with_policy(measures, max_extent, self.hyphenation_policy)
        else:
            result = LineBreakResult(
                breaks=[len(segments)],
                profiles=[
                    LineProfile(
                        width=sum(measure.advance for measure in measures),
                        center_offset=0.0,
                        block_baseline=0.0,
                    )
                ],
                overflowed=False,
                cost=0.0,
            )

        start = 0
        for line_index, end in enumerate(result.breaks):
            if end <= start or end > len(segments):
                continue
            final_line = end == len(segments)
            mandatory_line = segments[end - 1].is_mandatory
            visible_end = segments[end - 1].visible_end
            if mandatory_line:
                next_offset = segments[end - 1].next_offset
            elif final_line:
                next_offset = final_next_offset
            else:
                next_offset = segments[end].range[0]
            break_suffix = None if (final_line or mandatory_line) else segments[end - 1].break_suffix
            runs: list[LineRun] = []
            for segment in segments[start : end - 1]:
                runs.extend(_copy_run(run) for run in segment.runs)
            final_runs = [_copy_run(run) for run in segments[end - 1].runs]
            for run in final_runs:
                run.shaped.glyphs = [glyph for glyph in run.shaped.glyphs if glyph.cluster < visible_end]
            runs.extend(final_runs)
            line_offset[0] = self.push_layout_line(
                runs,
                line_offset[0],
                visible_end,
                next_offset,
                break_suffix,
                mandatory_line or (force_final_line and final_line),
                levels,
                lines,
            )
            if line_index < len(result.profiles):
                line_profiles.append(result.profiles[line_index])
            else:
                line_profiles.append(LineProfile(width=max_extent, center_offset=0.0, block_baseline=0.0))
            start = end
        return result.overflowed

    def push_layout_line(
        self,
        runs: list[LineRun],
        offset: int,
        visible_end: int,
        next_offset: int,
        break_suffix: Optional[ShapedBreakSuffix],
        force_push: bool,
        _levels: Sequence[int],
        lines: list[LayoutLine],
    ) -> int:
        """``push_layout_line``: visual reordering of runs, then one ``LayoutLine``."""
        if not runs and not force_push:
            return next_offset

        if break_suffix is not None:
            runs.extend(break_suffix.runs)

        visual_indices = reorder_visual([run.level for run in runs])

        line = LayoutLine(range=(offset, visible_end))
        pen_x = 0.0
        pen_y = 0.0
        for idx in visual_indices:
            run = runs[idx]
            line.glyphs.extend(run.shaped.glyphs)
            run.shaped.glyphs = []
            if self.writing_mode.is_vertical():
                pen_y -= run.shaped.y_advance
            else:
                pen_x += run.shaped.x_advance

        line.advance = abs(pen_y) if self.writing_mode.is_vertical() else pen_x
        lines.append(line)
        return next_offset

    def ink_bounds(self, font_size: float, lines: Sequence[LayoutLine]) -> Optional[tuple[float, float, float, float]]:
        """``ink_bounds``: tight box of all glyph outlines in layout coordinates (Y down)."""
        min_x = math.inf
        min_y = math.inf
        max_x = -math.inf
        max_y = -math.inf
        for line in lines:
            x, y = line.baseline
            for g in line.glyphs:
                size = font_size * g.scale
                bounds = g.font.glyph_bounds(g.glyph_id, size)
                if bounds is not None:
                    b_x_min, b_y_min, b_x_max, b_y_max = bounds
                    x0 = x + g.x_offset + b_x_min
                    x1 = x + g.x_offset + b_x_max
                    synthetic_pad = (size * 0.25 if g.font.synthetic_skew() is not None else 0.0) + (
                        size * 0.05 if g.font.synthetic_bold() else 0.0
                    )
                    # Font bounds are Y-up; layout coordinates are Y-down, so subtract.
                    y0 = (y - g.y_offset) - b_y_max
                    y1 = (y - g.y_offset) - b_y_min
                    min_x = min(min_x, x0 - synthetic_pad, x1 - synthetic_pad)
                    max_x = max(max_x, x0 + synthetic_pad, x1 + synthetic_pad)
                    min_y = min(min_y, y0, y1)
                    max_y = max(max_y, y0, y1)
                x += g.x_advance
                y -= g.y_advance
        if math.isfinite(min_x):
            return (min_x, min_y, max_x, max_y)
        return None

    def center_vertical_punctuation_glyphs(self, font_size: float, segment: str, glyphs: list[PositionedGlyph]) -> None:
        """``center_vertical_punctuation``: centre punctuation glyphs in their vertical cell."""
        if not segment or not glyphs:
            return
        for glyph in glyphs:
            cluster = glyph.cluster
            if not (0 <= cluster < len(segment)):
                continue
            ch = segment[cluster]
            if not is_punctuation(ch):
                continue
            bounds = glyph.font.glyph_bounds(glyph.glyph_id, font_size)
            if bounds is None:
                continue
            x_min, y_min, x_max, y_max = bounds
            glyph.x_offset = centered_x_offset(x_min, x_max)
            glyph.y_offset = glyph.y_advance * 0.5 - (y_min + y_max) * 0.5

    def layout_cjk_emphasis_runs(self, font_size: float, text: str, shaped: ShapedRun) -> None:
        """``layout_cjk_emphasis_runs``: lays out runs of emphasis marks (sesame dots) as a group."""
        GAP_EM = 0.04
        glyphs = shaped.glyphs

        def is_emphasis(glyph: PositionedGlyph) -> bool:
            cluster = glyph.cluster
            if not (0 <= cluster < len(text)):
                return False
            return cjk_emphasis_mark(text[cluster]) is not None

        start = 0
        while start < len(glyphs):
            if not is_emphasis(glyphs[start]):
                start += 1
                continue
            end = start + 1
            while end < len(glyphs) and end - start < 3 and is_emphasis(glyphs[end]):
                end += 1
            if end - start < 2:
                start = end
                continue

            bounds = []
            for glyph in glyphs[start:end]:
                b = glyph.font.glyph_bounds(glyph.glyph_id, font_size)
                if b is None:
                    bounds = None
                    break
                bounds.append(b)
            if bounds is None:
                start = end
                continue

            gap = font_size * GAP_EM
            if self.writing_mode.is_vertical():
                cell_advance = -max(
                    [font_size] + [abs(glyph.y_advance) for glyph in glyphs[start:end]]
                )
                group_width = sum(x_max - x_min for x_min, _, x_max, _ in bounds) + gap * (len(bounds) - 1)
                cursor = -group_width * 0.5
                last = end - start - 1
                for index, (glyph, (x_min, y_min, x_max, y_max)) in enumerate(zip(glyphs[start:end], bounds)):
                    glyph.x_offset = cursor - x_min
                    glyph.y_offset = cell_advance * 0.5 - (y_min + y_max) * 0.5
                    glyph.x_advance = 0.0
                    glyph.y_advance = cell_advance if index == last else 0.0
                    cursor += x_max - x_min + gap
            else:
                last = end - start - 1
                for index, (glyph, (x_min, _, x_max, _)) in enumerate(zip(glyphs[start:end], bounds)):
                    glyph.x_offset = -x_min
                    if index == last:
                        glyph.x_advance = extend_advance(x_max - x_min, self.letter_spacing)
                    else:
                        glyph.x_advance = x_max - x_min + gap
            start = end

        shaped.x_advance = sum(glyph.x_advance for glyph in glyphs)
        shaped.y_advance = sum(glyph.y_advance for glyph in glyphs)

    def apply_spacing(self, text: str, shaped: ShapedRun) -> None:
        """``apply_spacing``: letter spacing on every glyph, word spacing on whitespace."""
        if self.letter_spacing == 0.0 and self.word_spacing == 0.0:
            return
        for glyph in shaped.glyphs:
            cluster = glyph.cluster
            character = text[cluster] if 0 <= cluster < len(text) else None
            extra = self.letter_spacing + (
                self.word_spacing if character is not None and character.isspace() else 0.0
            )
            if self.writing_mode.is_vertical():
                glyph.y_advance = extend_advance(glyph.y_advance, extra)
            else:
                glyph.x_advance = extend_advance(glyph.x_advance, extra)
        shaped.x_advance = sum(glyph.x_advance for glyph in shaped.glyphs)
        shaped.y_advance = sum(glyph.y_advance for glyph in shaped.glyphs)


def extend_advance(advance: float, extra: float) -> float:
    """``extend_advance``: grows the magnitude of an advance, keeping its sign."""
    if advance < 0.0:
        return advance - extra
    return advance + extra


def is_punctuation(character: str) -> bool:
    """``GeneralCategoryGroup::Punctuation`` (Pc, Pd, Ps, Pe, Pi, Pf, Po)."""
    return unicodedata.category(character).startswith("P")


def _copy_run(run: LineRun) -> LineRun:
    glyphs = [copy.copy(glyph) for glyph in run.shaped.glyphs]
    return LineRun(
        shaped=ShapedRun(glyphs=glyphs, x_advance=run.shaped.x_advance, y_advance=run.shaped.y_advance),
        level=run.level,
    )


_PARAGRAPH_SEPARATORS = frozenset("\n\r\x1c\x1d\x1e\x85 ")  # Bidi_Class B


def _embedding_levels(text: str) -> list[int]:
    """Per-code-point resolved embedding levels (UAX #9 through W, N and I rules).

    Mirrors ``BidiInfo::new(text, None)`` followed by ``levels``, including L1. Like unicode-bidi,
    each paragraph (ending after a Bidi_Class B character) resolves its own base level (P2/P3).
    Reordering (L2) is applied by ``reorder_visual`` on the runs, so the logical-order levels are
    returned here.
    """
    levels: list[int] = []
    start = 0
    for index, ch in enumerate(text):
        if ch in _PARAGRAPH_SEPARATORS:
            levels.extend(_paragraph_levels(text[start : index + 1]))
            start = index + 1
    levels.extend(_paragraph_levels(text[start:]))
    return levels


def _paragraph_levels(text: str) -> list[int]:
    if not text:
        return []
    storage = get_empty_storage()
    base_level = get_base_level(text)
    storage["base_level"] = base_level
    storage["base_dir"] = ("L", "R")[base_level]
    get_embedding_levels(text, storage, False, False)
    explicit_embed_and_overrides(storage, False)
    resolve_weak_types(storage, False)
    resolve_neutral_types(storage, False)
    resolve_implicit_levels(storage, False)
    # UAX #9 L1: separators and trailing whitespace/isolates take the paragraph level.
    chars = storage["chars"]
    trailing = True
    for ch in reversed(chars):
        original = ch["orig"]
        if original in ("B", "S"):
            ch["level"] = base_level
            trailing = True
        elif original in ("WS", "LRI", "RLI", "FSI", "PDI") and trailing:
            ch["level"] = base_level
        else:
            trailing = False
    return [int(ch["level"]) for ch in chars]


def _bounds_of(language: str) -> tuple[int, int]:
    return hyphenation_bounds(language)


def justify_lines(text: str, lines: list[LayoutLine], max_width: float, profiles: Sequence[LineProfile]) -> None:
    """``justify_lines``: distributes slack over the spaces of every non-final, unbroken line."""
    if not profiles and (not math.isfinite(max_width) or max_width <= 0.0):
        return
    last = len(lines) - 1
    for index, line in enumerate(lines):
        end = line.range[1]
        if index == last or (0 <= end < len(text) and text[end] in "\n\r\u0085  "):
            continue
        target_width = profiles[index].width if index < len(profiles) else max_width
        if not math.isfinite(target_width) or target_width <= 0.0:
            continue

        def is_space(glyph: PositionedGlyph) -> bool:
            return 0 <= glyph.cluster < len(text) and text[glyph.cluster].isspace()

        spaces = sum(1 for glyph in line.glyphs if is_space(glyph))
        if spaces == 0 or line.advance >= target_width:
            continue
        extra = (target_width - line.advance) / spaces
        for glyph in line.glyphs:
            if is_space(glyph):
                glyph.x_advance = extend_advance(glyph.x_advance, extra)
        line.advance = target_width


def line_breaks_with_policy(
    segments: Sequence[LineBreakMeasure], max_extent: float, policy: HyphenationPolicy
) -> LineBreakResult:
    """``line_breaks_with_policy``: last-resort hyphens only if the clean breaking overflows."""
    if policy == HyphenationPolicy.LAST_RESORT:
        without_hyphens = optimal_uniform_line_breaks(segments, max_extent, False)
        if not without_hyphens.overflowed:
            return without_hyphens
    return optimal_uniform_line_breaks(segments, max_extent, policy != HyphenationPolicy.DISABLED)


def line_break_badness(line_advance: float, max_extent: float) -> float:
    """Cubic badness; overflow is penalised by ``LINE_BREAK_OVERFLOW_MULTIPLIER``."""
    if line_advance <= max_extent:
        return (max_extent - line_advance) ** 3
    return (line_advance - max_extent) ** 3 * LINE_BREAK_OVERFLOW_MULTIPLIER


def optimal_uniform_line_breaks(
    segments: Sequence[LineBreakMeasure], max_extent: float, allow_hyphenation: bool
) -> LineBreakResult:
    """Knuth-style minimum-badness DP over segment boundaries with a uniform line width."""
    length = len(segments)
    if length == 0:
        return LineBreakResult(breaks=[], profiles=[], overflowed=False, cost=0.0)
    if not math.isfinite(max_extent) or max_extent <= 0.0:
        return LineBreakResult(
            breaks=[length],
            profiles=[LineProfile(width=max_extent, center_offset=0.0, block_baseline=0.0)],
            overflowed=False,
            cost=0.0,
        )

    dp = [math.inf] * (length + 1)
    prev: list[Optional[int]] = [None] * (length + 1)
    dp[0] = 0.0

    for start in range(length):
        if not math.isfinite(dp[start]):
            continue
        advance = 0.0
        for end in range(start + 1, length + 1):
            advance += segments[end - 1].advance
            suffix_advance = segments[end - 1].break_suffix_advance if end < length else 0.0
            line_advance = advance - segments[end - 1].trailing_advance + suffix_advance
            hyphenated_break = end < length and suffix_advance > 0.0
            if hyphenated_break and not allow_hyphenation:
                continue
            cost = dp[start] + line_break_badness(line_advance, max_extent)
            if hyphenated_break:
                cost += LINE_BREAK_HYPHEN_PENALTY
            if end < length:
                cost += segments[end - 1].break_penalty
            if cost < dp[end]:
                dp[end] = cost
                prev[end] = start
            if segments[end - 1].is_mandatory or advance > max_extent:
                break

    if not math.isfinite(dp[length]):
        total = sum(segment.advance for segment in segments)
        trailing = segments[-1].trailing_advance if segments else 0.0
        return LineBreakResult(
            breaks=[length],
            profiles=[LineProfile(width=max_extent, center_offset=0.0, block_baseline=0.0)],
            overflowed=(total - trailing) > max_extent,
            cost=math.inf,
        )

    breaks: list[int] = []
    index = length
    while index > 0:
        breaks.append(index)
        previous = prev[index]
        if previous is None:
            return LineBreakResult(
                breaks=[length],
                profiles=[LineProfile(width=max_extent, center_offset=0.0, block_baseline=0.0)],
                overflowed=True,
                cost=math.inf,
            )
        index = previous
    breaks.reverse()
    overflowed = breaks_overflow(segments, breaks, [max_extent])
    profiles = [LineProfile(width=max_extent, center_offset=0.0, block_baseline=0.0) for _ in breaks]
    return LineBreakResult(breaks=breaks, profiles=profiles, overflowed=overflowed, cost=dp[length])


def comic_line_breaks(
    segments: Sequence[LineBreakMeasure],
    balloon: "ComicBalloon",
    writing_mode: WritingMode,
    line_height: float,
    line_ink: InkBand,
    air: tuple[float, float],
    policy: HyphenationPolicy,
) -> LineBreakResult:
    """``comic_line_breaks``: exact profiled breaks inside a balloon contour, best line count."""
    air_x, air_y = air
    block_extent = balloon.width if writing_mode.is_vertical() else balloon.height
    block_air = air_x if writing_mode.is_vertical() else air_y
    available_block_extent = max(block_extent - block_air * 2.0, 0.0)
    if line_ink.thickness() <= available_block_extent:
        maximum_lines = 1 + int(math.floor((available_block_extent - line_ink.thickness()) / line_height))
    else:
        maximum_lines = 0
    maximum_lines = min(maximum_lines, COMIC_MAX_LINES, len(segments))
    if maximum_lines == 0:
        fallback = line_breaks_with_policy(segments, balloon.inline_extent(writing_mode, air_x, air_y), policy)
        fallback.overflowed = True
        return fallback

    def select(allow_hyphenation: bool) -> Optional[LineBreakResult]:
        candidates: list[LineBreakResult] = []
        for line_count in range(1, maximum_lines + 1):
            for profiles in balloon.line_profile_candidates(
                writing_mode, line_count, line_height, line_ink, air_x, air_y
            ):
                result = exact_profiled_line_breaks(segments, profiles, allow_hyphenation)
                if result is not None:
                    candidates.append(result)
        if not candidates:
            return None
        # Rust ``min_by``: the first of several equal minima wins.
        return min(
            candidates,
            key=lambda result: (result.overflowed, len(result.profiles), result.cost),
        )

    if policy == HyphenationPolicy.LAST_RESORT:
        without_hyphens = select(False)
        if without_hyphens is not None and not without_hyphens.overflowed:
            return without_hyphens

    chosen = select(policy != HyphenationPolicy.DISABLED)
    if chosen is not None:
        return chosen
    fallback = line_breaks_with_policy(segments, balloon.inline_extent(writing_mode, air_x, air_y), policy)
    fallback.overflowed = True
    return fallback


def exact_profiled_line_breaks(
    segments: Sequence[LineBreakMeasure], profiles: list[LineProfile], allow_hyphenation: bool
) -> Optional[LineBreakResult]:
    """DP over (line index, segment index) with the per-line widths from the contour."""
    length = len(segments)
    line_count = len(profiles)
    if line_count == 0 or line_count > length:
        return None
    dp = [[math.inf] * (length + 1) for _ in range(line_count + 1)]
    previous: list[list[Optional[int]]] = [[None] * (length + 1) for _ in range(line_count + 1)]
    dp[0][0] = 0.0

    for line in range(line_count):
        remaining_lines = line_count - line - 1
        for start in range(line, length):
            if not math.isfinite(dp[line][start]):
                continue
            advance = 0.0
            last_end = length - remaining_lines
            for end in range(start + 1, last_end + 1):
                advance += segments[end - 1].advance
                suffix = segments[end - 1].break_suffix_advance if end < length else 0.0
                hyphenated_break = end < length and suffix > 0.0
                if hyphenated_break and not allow_hyphenation:
                    continue
                line_advance = advance - segments[end - 1].trailing_advance + suffix
                width = max(profiles[line].width, 1.0)
                overflow = max(line_advance - width, 0.0) / width
                slack = max(width - line_advance, 0.0) / width
                cost = dp[line][start] + slack * slack * 1_000.0 + overflow * overflow * COMIC_LINE_OVERFLOW_PENALTY
                if hyphenated_break:
                    cost += LINE_BREAK_HYPHEN_PENALTY
                if end < length:
                    cost += segments[end - 1].break_penalty
                if cost < dp[line + 1][end]:
                    dp[line + 1][end] = cost
                    previous[line + 1][end] = start
                if segments[end - 1].is_mandatory or advance > width:
                    break

    cost = dp[line_count][length]
    if not math.isfinite(cost):
        return None
    cost = cost / line_count + line_count * 8.0
    breaks: list[int] = []
    end = length
    for line in range(line_count, 0, -1):
        breaks.append(end)
        found = previous[line][end]
        if found is None:
            return None
        end = found
    if end != 0:
        return None
    breaks.reverse()
    widths = [profile.width for profile in profiles]
    overflowed = breaks_overflow(segments, breaks, widths)
    return LineBreakResult(breaks=breaks, profiles=profiles, overflowed=overflowed, cost=cost)


def breaks_overflow(segments: Sequence[LineBreakMeasure], breaks: Sequence[int], widths: Sequence[float]) -> bool:
    start = 0
    for line, end in enumerate(breaks):
        advance = sum(segment.advance for segment in segments[start:end])
        advance -= segments[end - 1].trailing_advance
        if end < len(segments):
            advance += segments[end - 1].break_suffix_advance
        if line < len(widths):
            width = widths[line]
        elif widths:
            width = widths[0]
        else:
            return True
        if advance > width + 1.1920929e-07:
            return True
        start = end
    return False


def polygon_inline_spans(
    contour: Sequence[tuple[float, float]], writing_mode: WritingMode, block: float
) -> list[tuple[float, float]]:
    """Even-odd scanline intersections of a polygon at one block coordinate, paired up."""
    if len(contour) < 3:
        return []
    intersections: list[float] = []
    for index in range(len(contour)):
        first = contour[index]
        second = contour[(index + 1) % len(contour)]
        if writing_mode.is_vertical():
            first_block, first_inline, second_block, second_inline = first[0], first[1], second[0], second[1]
        else:
            first_block, first_inline, second_block, second_inline = first[1], first[0], second[1], second[0]
        if (first_block <= block < second_block) or (second_block <= block < first_block) or (
            (first_block <= block and second_block > block) or (second_block <= block and first_block > block)
        ):
            fraction = (block - first_block) / (second_block - first_block)
            intersections.append(first_inline + (second_inline - first_inline) * fraction)
    intersections.sort()
    return [(intersections[i], intersections[i + 1]) for i in range(0, len(intersections) - 1, 2)]


def comic_break_penalty(text: str, boundary: int) -> float:
    """Heuristic break cost for balloon text: punctuation, conjunctions and articles."""
    boundary = min(boundary, len(text))
    before = text[:boundary].rstrip()
    after = text[boundary:].lstrip()
    if before:
        last = before[-1]
        if last in ".!?…‼⁇⁈⁉":
            return 0.0
        if last in ",;:—–":
            return 20.0
    next_word = re.split(r"[^A-Za-z]", after)[0]
    if next_word.lower() in ("and", "but", "or", "so", "because", "although", "while", "then"):
        return 40.0
    previous_word = re.split(r"[^A-Za-z]", before)[-1]
    if previous_word.lower() in ("a", "an", "the", "to", "of", "for", "in", "on", "at", "with", "from"):
        return 300.0
    return 100.0


def centered_x_offset(x_min: float, x_max: float) -> float:
    return -((x_min + x_max) * 0.5)


def cjk_emphasis_mark(character: str) -> Optional[str]:
    """Full-width form of ``!`` / ``?`` after NFKC, or None (``cjk_emphasis_mark``)."""
    normalized = unicodedata.normalize("NFKC", character)
    if len(normalized) != 1:
        return None
    if normalized == "!":
        return "！"
    if normalized == "?":
        return "？"
    return None


def normalize_cjk_emphasis_punctuation(text: str) -> Optional[tuple[str, list[int]]]:
    """Maps half-width emphasis marks to full-width; returns (text, code-point cluster map)."""
    if not any((mark := cjk_emphasis_mark(ch)) is not None and mark != ch for ch in text):
        return None
    output = []
    cluster_map: list[int] = []
    for source_offset, ch in enumerate(text):
        mapped = cjk_emphasis_mark(ch) or ch
        output.append(mapped)
        cluster_map.append(source_offset)
    return "".join(output), cluster_map


def reorder_visual(levels: Sequence[int]) -> list[int]:
    """UAX #9 L2: reverse every maximal run at each level from the highest down to the lowest odd."""
    indices = list(range(len(levels)))
    if not levels:
        return indices
    max_level = max(levels)
    odd = [level for level in levels if level % 2 != 0]
    if not odd:
        return indices
    min_odd_level = min(odd)
    for level in range(max_level, min_odd_level - 1, -1):
        i = 0
        while i < len(levels):
            if levels[i] >= level:
                j = i
                while j < len(levels) and levels[j] >= level:
                    j += 1
                indices[i:j] = reversed(indices[i:j])
                i = j
            else:
                i += 1
    return indices


class ComicBalloon:
    """``ComicBalloon``: balloon geometry used to constrain line profiles."""

    def __init__(self, width: float, height: float, contour: list[tuple[float, float]], minimum_air: float):
        self.width = width
        self.height = height
        self.contour = contour
        self.minimum_air = minimum_air

    def air(self, ink_height: float) -> float:
        return max(self.minimum_air, ink_height)

    def inline_extent(self, writing_mode: WritingMode, air_x: float, air_y: float) -> float:
        if writing_mode.is_vertical():
            return max(self.height - air_y * 2.0, 1.0)
        return max(self.width - air_x * 2.0, 1.0)

    def line_profile_candidates(
        self,
        writing_mode: WritingMode,
        line_count: int,
        line_height: float,
        line_ink: InkBand,
        air_x: float,
        air_y: float,
    ) -> list[list[LineProfile]]:
        vertical = writing_mode.is_vertical()
        if vertical:
            block_extent, inline_extent, block_air, inline_air = self.width, self.height, air_x, air_y
        else:
            block_extent, inline_extent, block_air, inline_air = self.height, self.width, air_y, air_x
        inline_radius = inline_extent * 0.5 - inline_air
        block_size = line_ink.thickness() + max(line_count - 1, 0) * line_height
        if inline_radius <= 0.0:
            return []
        block_origin = self.centered_block_origin(writing_mode, block_extent, block_air, block_size)
        if block_origin is None:
            return []
        profiles = self.line_profiles_at_origin(
            writing_mode,
            line_count,
            line_height,
            line_ink,
            block_extent,
            inline_extent,
            block_air,
            inline_air,
            block_origin,
        )
        return [profiles] if profiles is not None else []

    def line_profiles_at_origin(
        self,
        writing_mode: WritingMode,
        line_count: int,
        line_height: float,
        line_ink: InkBand,
        block_extent: float,
        inline_extent: float,
        block_air: float,
        inline_air: float,
        block_origin: float,
    ) -> Optional[list[LineProfile]]:
        inline_center = inline_extent * 0.5
        spans: list[tuple[float, float, float]] = []
        for line in range(line_count):
            block_index = line_count - line - 1 if writing_mode == WritingMode.VERTICAL_RL else line
            baseline = block_origin + line_ink.before + block_index * line_height
            band_start = baseline - line_ink.before
            left = -math.inf
            right = math.inf
            # Intersect the contour across the glyph ink band. Leading belongs between baselines
            # and must not consume inline space at a balloon's tapered ends.
            for sample in range(5):
                block = band_start + line_ink.thickness() * sample / 4.0
                span = self.inline_span(writing_mode, block, block_extent, inline_extent, block_air, inline_air)
                if span is None:
                    return None
                left = max(left, span[0])
                right = min(right, span[1])
            if right <= left:
                return None
            spans.append((left, right, baseline))

        # A phrase has one visual axis: the contour controls each line's usable width while a
        # common centre keeps the setting coherent instead of a zig-zag paragraph.
        common_left = max(left for left, _, _ in spans)
        common_right = min(right for _, right, _ in spans)
        if common_right <= common_left:
            return None
        contour_center = sum((left + right) * 0.5 for left, right, _ in spans) / len(spans)
        shared_center = min(max(contour_center, common_left), common_right)
        profiles: list[LineProfile] = []
        for left, right, block_baseline in spans:
            half_width = min(shared_center - left, right - shared_center)
            if not half_width > 0.0:
                return None
            profiles.append(
                LineProfile(width=half_width * 2.0, center_offset=shared_center - inline_center, block_baseline=block_baseline)
            )
        return profiles

    def centered_block_origin(
        self, writing_mode: WritingMode, block_extent: float, block_air: float, block_size: float
    ) -> Optional[float]:
        first = block_air
        last = block_extent - block_air
        if len(self.contour) >= 3:
            values = [x if writing_mode.is_vertical() else y for x, y in self.contour]
            minimum = min(values)
            maximum = max(values)
            first = max(first, minimum + block_air)
            last = min(last, maximum - block_air)
        if (last - first) + 1.1920929e-07 >= block_size:
            return first + (last - first - block_size) * 0.5
        return None

    def inline_span(
        self,
        writing_mode: WritingMode,
        block: float,
        block_extent: float,
        inline_extent: float,
        block_air: float,
        inline_air: float,
    ) -> Optional[tuple[float, float]]:
        if len(self.contour) >= 3:
            return self.contour_inline_span(writing_mode, block, inline_air)
        block_radius = block_extent * 0.5 - block_air
        inline_radius = inline_extent * 0.5 - inline_air
        normalized_block = min(max((block - block_extent * 0.5) / block_radius, -1.0), 1.0)
        half_extent = inline_radius * math.sqrt(max(1.0 - normalized_block * normalized_block, 0.0))
        if half_extent > 0.0:
            return (inline_extent * 0.5 - half_extent, inline_extent * 0.5 + half_extent)
        return None

    def contour_inline_span(self, writing_mode: WritingMode, block: float, inline_air: float) -> Optional[tuple[float, float]]:
        spans = []
        for left, right in polygon_inline_spans(self.contour, writing_mode, block):
            span = (left + inline_air, right - inline_air)
            if span[1] > span[0]:
                spans.append(span)
        if not spans:
            return None
        # Rust ``max_by``: the last of several equal maxima wins.
        best = spans[0]
        for span in spans[1:]:
            if (span[1] - span[0]) >= (best[1] - best[0]):
                best = span
        return best
