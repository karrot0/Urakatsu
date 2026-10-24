"""Text layout and portable glyph recording.

Port of ``text_renderer.rs``.

Rust records glyph runs into a Vello ``PreparedScene``. Here a run is a ``GlyphRun`` with the
same fields (font, size, glyph ids and positions, transform, synthetic skew, embolden amount,
colour); ``frame.py`` rasterizes them with Skia paths built from the font outlines.

Divergence (not in Koharu): rich text. ``TextNodeDescriptor.styles`` override bold, italic, family, size
(relative to the layer's), colour, underline and strikethrough for runs of the text. Runs with other faces
or sizes are shaped with them
(``TextLayout.with_styles``); underline and strikethrough are ``DecorationRun`` bars recorded among the
glyph runs, so the border pass outlines them too.

Divergence (not in Koharu): gradient fill and border (Photoshop's Gradient Overlay and gradient Stroke).
A ``GradientSpec`` is laid over the text's layout bounds as a ``GradientPaint`` carried by the runs; a
run with a rich text colour of its own keeps that colour.

Divergence: Rust's dilation for the border pass is ``embolden`` inside the rasterizer. Here it is
a Skia stroke of the glyph outline (round joins) drawn before the fill, which gives the same
outward-only dilation except at sharp corners where the round join differs from a mitred offset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .fonts import Font, Fonts
from .frame import RenderBounds, RenderDiagnostic
from .layout import HyphenationPolicy, LayoutRun, TextLayout
from .script import is_chinese_or_japanese_text
from .types import FontStyle, TextAlign, WritingMode

Rgba = tuple[int, int, int, int]
Affine = tuple[float, float, float, float, float, float]  # kurbo coefficients (a, b, c, d, e, f)

MINIMUM_FONT_SIZE = 9.0


@dataclass(frozen=True)
class GradientSpec:
    """A gradient as authored: linear at ``angle`` degrees (Photoshop's convention) or radial, with stops."""

    kind: str  # "linear" | "radial"
    angle: float
    stops: tuple[tuple[float, Rgba], ...]  # (offset 0..1, colour)


@dataclass(frozen=True)
class GradientPaint:
    """A gradient placed in layout coordinates: from ``start`` to ``end`` (linear) or around ``start``
    with radius ``radius`` (radial)."""

    kind: str
    start: tuple[float, float]
    end: tuple[float, float]
    radius: float
    colors: tuple[Rgba, ...]
    positions: tuple[float, ...]


def place_gradient(spec: GradientSpec, x: float, y: float, width: float, height: float) -> GradientPaint:
    """Photoshop's "Align with Layer": the gradient spans the bounds along its angle (y is down here, so
    90 degrees runs from the bottom up)."""
    stops = sorted(spec.stops, key=lambda stop: stop[0])
    colors = tuple(color for _, color in stops)
    positions = tuple(min(max(offset, 0.0), 1.0) for offset, _ in stops)
    cx, cy = x + width / 2, y + height / 2
    if spec.kind == "radial":
        return GradientPaint("radial", (cx, cy), (cx, cy), max(width, height) / 2 or 1.0, colors, positions)
    theta = math.radians(spec.angle)
    dx, dy = math.cos(theta), -math.sin(theta)
    half = abs(width / 2 * dx) + abs(height / 2 * dy) or 1.0
    return GradientPaint("linear", (cx - dx * half, cy - dy * half), (cx + dx * half, cy + dy * half), 0.0, colors, positions)


@dataclass(frozen=True)
class StrokeOptions:
    color: Rgba
    width_px: float


@dataclass(frozen=True)
class TextStyleRun:
    """Rich text formatting over ``[start, end)`` of the text (code points)."""

    start: int
    end: int
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikethrough: bool = False
    color: Optional[Rgba] = None
    font: Optional[str] = None  # a family of its own
    scale: float = 1.0  # size relative to the layer's


@dataclass
class TextNodeDescriptor:
    """``TextNodeDescriptor``: every input that determines one text layer's output."""

    entity: str
    text: str
    language: Optional[str]
    width: float
    height: float
    balloon_contour: Optional[list[tuple[float, float]]]
    preferred_font: Optional[str]
    font_families: list[str]
    font_weight: Optional[int]
    font_style: Optional[FontStyle]
    font_size: Optional[float]
    minimum_font_size: float
    auto_fit: bool
    alignment: TextAlign
    writing_mode: WritingMode
    foreground_color: Rgba
    stroke: Optional[StrokeOptions]
    line_height: float
    letter_spacing: float
    word_spacing: float
    text_inset: tuple[float, float, float, float]
    point_text: bool
    styles: list[TextStyleRun] = field(default_factory=list)
    fill_gradient: Optional[GradientSpec] = None
    stroke_gradient: Optional[GradientSpec] = None


@dataclass
class GlyphRun:
    """One ``PreparedGlyphRun``: glyphs of one font, positioned in layout coordinates."""

    font: Font
    font_size: float
    glyphs: list[tuple[int, float, float]]  # (glyph id, x, y) with y down
    transform: Affine
    glyph_skew: Optional[float]  # tan-space skew of the glyph outline (synthetic italic)
    embolden: float  # extra outline thickness in pixels (synthetic bold and border dilation)
    color: Rgba
    gradient: Optional[GradientPaint] = None  # paints instead of ``color``


@dataclass
class DecorationRun:
    """An underline or strikethrough bar: the rectangle ``(x0, y0, x1, y1)`` in layout coordinates."""

    rect: tuple[float, float, float, float]
    transform: Affine
    dilation: float  # outward growth in pixels (the border pass)
    color: Rgba
    gradient: Optional[GradientPaint] = None


@dataclass
class TextScene:
    runs: list[GlyphRun | DecorationRun] = field(default_factory=list)


@dataclass
class RenderedTextMetadata:
    rendered_bounds: RenderBounds
    layout_bounds: RenderBounds
    post_script_fonts: list[str]
    font_size: float
    color: Rgba


@dataclass
class RenderedTextNode:
    scene: TextScene
    local_bounds: RenderBounds
    metadata: RenderedTextMetadata
    diagnostics: list[RenderDiagnostic]


@dataclass
class TextRenderOptions:
    color: Rgba = (0, 0, 0, 255)
    padding: float = 0.0
    baseline_shift: float = 0.0
    stroke: Optional[StrokeOptions] = None
    fill_gradient: Optional[GradientSpec] = None
    stroke_gradient: Optional[GradientSpec] = None


@dataclass
class _GlyphPaint:
    color: Rgba
    dilation_px: float
    gradient: Optional[GradientPaint] = None


def translate(x: float, y: float) -> Affine:
    return (1.0, 0.0, 0.0, 1.0, x, y)


class TextRenderer:
    """``TextRenderer``: shapes text and records the resulting glyph runs."""

    def layout(self, builder: TextLayout, text: str) -> LayoutRun:
        return builder.run(text)

    def render(
        self,
        scene: TextScene,
        layout: LayoutRun,
        writing_mode: WritingMode,
        options: TextRenderOptions,
        transform: Affine,
        styles: Optional[list[Optional[TextStyleRun]]] = None,
    ) -> None:
        """Border first as an outward-only dilation of the outline, then the fill on top.

        Unlike a centred stroke, a dilation never grows inward, so it cannot punch a hole through
        small features once the border width exceeds their radius.
        """
        pad = options.padding
        fill_gradient = place_gradient(options.fill_gradient, pad, pad, layout.width, layout.height) if options.fill_gradient else None
        stroke_gradient = place_gradient(options.stroke_gradient, pad, pad, layout.width, layout.height) if options.stroke_gradient else None
        if options.stroke is not None and options.stroke.width_px > 0.0 and (options.stroke.color[3] > 0 or stroke_gradient is not None):
            _draw_layout(
                scene,
                layout,
                writing_mode,
                options,
                transform,
                _GlyphPaint(color=options.stroke.color, dilation_px=options.stroke.width_px, gradient=stroke_gradient),
                styles,
            )
        _draw_layout(
            scene,
            layout,
            writing_mode,
            options,
            transform,
            _GlyphPaint(color=options.color, dilation_px=0.0, gradient=fill_gradient),
            styles,
        )

    def render_descriptor(self, descriptor: TextNodeDescriptor, fonts: Fonts) -> RenderedTextNode:
        is_bubble_text = descriptor.balloon_contour is not None
        from .bubble import LayoutBox

        frame = LayoutBox(x=0.0, y=0.0, width=descriptor.width, height=descriptor.height)
        bounds = _inset(frame, descriptor.text_inset) if is_bubble_text else frame
        if bounds.width <= 0.0 or bounds.height <= 0.0:
            raise ValueError(f"text inset leaves no layout area for entity {descriptor.entity}")

        font_list = fonts.resolve(
            descriptor.preferred_font,
            descriptor.font_weight,
            descriptor.font_style,
            descriptor.font_families,
            descriptor.text,
            descriptor.language,
        )

        def automatic_maximum_value() -> float:
            return _automatic_maximum(descriptor, bounds, is_bubble_text)

        if descriptor.auto_fit:
            maximum = automatic_maximum_value()
        else:
            maximum = descriptor.font_size if descriptor.font_size is not None else automatic_maximum_value()
        minimum = min(descriptor.minimum_font_size, maximum)

        layout = (
            TextLayout(font_list[0])
            .with_fallback_fonts(font_list[1:])
            .with_writing_mode(descriptor.writing_mode)
            .with_alignment(descriptor.alignment)
            .with_line_height(descriptor.line_height)
            .with_spacing(descriptor.letter_spacing, descriptor.word_spacing)
            .with_cjk_punctuation_layout(
                is_chinese_or_japanese_text(descriptor.text)
                or (descriptor.language is not None and _is_chinese_or_japanese_language(descriptor.language))
            )
        )
        if not descriptor.point_text:
            layout = layout.with_max_width(bounds.width).with_max_height(bounds.height)
        if descriptor.balloon_contour is not None:
            top, _, _, left = descriptor.text_inset
            contour = [(x - left, y - top) for x, y in descriptor.balloon_contour]
            layout = layout.with_comic_balloon(bounds.width, bounds.height, contour, max(descriptor.text_inset))
        # Rich text: each character's formatting, a face list per (bold, italic, family) in use, and sizes.
        per_char = _styles_per_character(descriptor.styles, len(descriptor.text))
        if any(style is not None for style in per_char):
            plain = (False, False, None)
            keys = [(style.bold, style.italic, style.font) if style is not None else plain for style in per_char]
            scales = [style.scale if style is not None else 1.0 for style in per_char]
            bold_weight = 700 if (descriptor.font_weight or 400) < 700 else 900
            style_fonts = {
                key: fonts.resolve(
                    key[2] or descriptor.preferred_font,
                    bold_weight if key[0] else descriptor.font_weight,
                    FontStyle.ITALIC if key[1] else descriptor.font_style,
                    descriptor.font_families,
                    descriptor.text,
                    descriptor.language,
                )
                for key in set(keys)
                if key != plain
            }
            layout = layout.with_styles(keys, style_fonts, scales if any(scale != 1.0 for scale in scales) else None)
        if descriptor.language is not None:
            layout = layout.with_hyphenation_language_tag(descriptor.language)
        if is_bubble_text and descriptor.writing_mode == WritingMode.HORIZONTAL:
            layout = layout.with_hyphenation_policy(HyphenationPolicy.LAST_RESORT)
        if descriptor.auto_fit and not descriptor.point_text:
            layout = layout.with_max_font_size(maximum).with_min_font_size(minimum)
        else:
            layout = layout.with_font_size(descriptor.font_size if descriptor.font_size is not None else maximum)

        run = self.layout(layout, descriptor.text)
        if descriptor.point_text:
            x, y = bounds.x, bounds.y
        else:
            x, y = _placement(bounds, run.width, run.height)
        x += run.placement_offset_x
        y += run.placement_offset_y
        transform = translate(x, y)
        color = descriptor.foreground_color
        options = TextRenderOptions(color=color, stroke=descriptor.stroke, fill_gradient=descriptor.fill_gradient, stroke_gradient=descriptor.stroke_gradient)
        scene = TextScene()
        self.render(scene, run, descriptor.writing_mode, options, transform, per_char)

        diagnostics: list[RenderDiagnostic] = []
        if run.font_size + 1.1920929e-07 < descriptor.minimum_font_size:
            diagnostics.append(
                RenderDiagnostic(
                    kind="text_below_readable_size",
                    entity=descriptor.entity,
                    font_size=run.font_size,
                    minimum_font_size=descriptor.minimum_font_size,
                )
            )
        if run.overflowed():
            diagnostics.append(
                RenderDiagnostic(
                    kind="text_overflow",
                    entity=descriptor.entity,
                    available=RenderBounds(bounds.x, bounds.y, bounds.width, bounds.height),
                    actual_width=run.width,
                    actual_height=run.height,
                    font_size=run.font_size,
                )
            )

        rendered_bounds = RenderBounds(x=x, y=y, width=run.width, height=run.height)
        stroke_padding = max(descriptor.stroke.width_px, 0.0) if descriptor.stroke is not None else 0.0
        left, top = rendered_bounds.x, rendered_bounds.y
        right, bottom = left + rendered_bounds.width, top + rendered_bounds.height
        # An underline can sit below the glyphs' ink, so it widens the drawn area.
        for item in scene.runs:
            if isinstance(item, DecorationRun):
                x0, y0, x1, y1 = item.rect
                left, top = min(left, x + x0), min(top, y + y0)
                right, bottom = max(right, x + x1), max(bottom, y + y1)
        local_bounds = RenderBounds(
            x=left - stroke_padding,
            y=top - stroke_padding,
            width=right - left + stroke_padding * 2.0,
            height=bottom - top + stroke_padding * 2.0,
        )
        layout_bounds = rendered_bounds if descriptor.point_text else RenderBounds(bounds.x, bounds.y, bounds.width, bounds.height)
        return RenderedTextNode(
            scene=scene,
            local_bounds=local_bounds,
            metadata=RenderedTextMetadata(
                rendered_bounds=rendered_bounds,
                layout_bounds=layout_bounds,
                post_script_fonts=[font.post_script_name for font in font_list],
                font_size=run.font_size,
                color=color,
            ),
            diagnostics=diagnostics,
        )


def _automatic_maximum(descriptor: TextNodeDescriptor, bounds, is_bubble_text: bool) -> float:
    if is_bubble_text:
        return bounds.height if descriptor.writing_mode.is_vertical() else bounds.width
    return 24.0


def _is_chinese_or_japanese_language(language: str) -> bool:
    """``is_chinese_or_japanese_language``: primary subtag ``ja`` or ``zh``."""
    primary = language.replace("_", "-").split("-")[0].lower()
    return primary in ("ja", "zh")


def _inset(rect, insets):
    from .bubble import LayoutBox

    top, right, bottom, left = insets
    return LayoutBox(
        x=rect.x + left,
        y=rect.y + top,
        width=max(rect.width - left - right, 0.0),
        height=max(rect.height - top - bottom, 0.0),
    )


def _placement(rect, width: float, height: float) -> tuple[float, float]:
    x = rect.x + (rect.width - width) * 0.5
    remaining = rect.height - height
    y = rect.y + remaining * 0.5
    return x, y


def _styles_per_character(styles: list[TextStyleRun], length: int) -> list[Optional[TextStyleRun]]:
    per_char: list[Optional[TextStyleRun]] = [None] * length
    for style in styles:
        for index in range(max(style.start, 0), min(style.end, length)):
            per_char[index] = style
    return per_char


def _draw_layout(
    scene: TextScene,
    layout: LayoutRun,
    writing_mode: WritingMode,
    options: TextRenderOptions,
    transform: Affine,
    paint: _GlyphPaint,
    styles: Optional[list[Optional[TextStyleRun]]] = None,
) -> None:
    """``draw_layout``: groups consecutive glyphs by font (and rich text colour) and records one run
    per group, then the line's underline and strikethrough bars."""
    border = paint.dilation_px > 0.0

    def style_of(cluster: int) -> Optional[TextStyleRun]:
        if not styles:
            return None
        return styles[min(max(cluster, 0), len(styles) - 1)]

    def gradient_of(cluster: int) -> Optional[GradientPaint]:
        # A run's own colour wins over the fill gradient; the border's gradient covers everything.
        style = style_of(cluster)
        return paint.gradient if border or style is None or style.color is None else None

    def color_of(cluster: int) -> Rgba:
        style = style_of(cluster)
        # The border keeps its own colour; a run's colour is its fill.
        return style.color if style is not None and style.color is not None and not border else paint.color

    vertical = writing_mode.is_vertical()
    for line in layout.lines:
        baseline_x, baseline_y = line.baseline
        pen_x = 0.0
        pen_y = 0.0
        start = 0
        glyphs = line.glyphs
        # Open bars by kind: [inline start, inline end, colour, font].
        bars: dict[str, Optional[list]] = {}
        while start < len(glyphs):
            font = glyphs[start].font
            color = color_of(glyphs[start].cluster)
            scale = glyphs[start].scale
            gradient = gradient_of(glyphs[start].cluster)
            end = start + 1
            while (
                end < len(glyphs)
                and glyphs[end].font is font
                and color_of(glyphs[end].cluster) == color
                and glyphs[end].scale == scale
                and gradient_of(glyphs[end].cluster) is gradient
            ):
                end += 1
            positioned: list[tuple[int, float, float]] = []
            for glyph in glyphs[start:end]:
                positioned.append(
                    (
                        glyph.glyph_id,
                        options.padding + baseline_x + pen_x + glyph.x_offset,
                        options.padding + baseline_y + pen_y - glyph.y_offset - options.baseline_shift,
                    )
                )
                inline_start = pen_y if vertical else pen_x
                pen_x += glyph.x_advance
                pen_y -= glyph.y_advance
                inline_end = pen_y if vertical else pen_x
                style = style_of(glyph.cluster)
                for kind, wanted in (
                    ("underline", style is not None and style.underline),
                    ("strikethrough", style is not None and style.strikethrough),
                ):
                    bar = bars.get(kind)
                    if wanted and bar is not None and bar[2] == color and bar[4] == scale and bar[5] is gradient:
                        bar[1] = inline_end
                        continue
                    if bar is not None:
                        _record_bar(scene, kind, bar, line, layout.font_size, vertical, options, transform, paint)
                    bars[kind] = [inline_start, inline_end, color, glyph.font, scale, gradient] if wanted else None

            synthetic_bold = 1.0 if font.synthetic_bold() else 0.0
            skew = font.synthetic_skew()
            scene.runs.append(
                GlyphRun(
                    font=font,
                    font_size=layout.font_size * scale,
                    glyphs=positioned,
                    transform=transform,
                    glyph_skew=math.tan(math.radians(skew)) if skew is not None else None,
                    embolden=synthetic_bold + paint.dilation_px,
                    color=color,
                    gradient=gradient,
                )
            )
            start = end
        for kind, bar in bars.items():
            if bar is not None:
                _record_bar(scene, kind, bar, line, layout.font_size, vertical, options, transform, paint)


def _record_bar(
    scene: TextScene,
    kind: str,
    bar: list,
    line,
    font_size: float,
    vertical: bool,
    options: TextRenderOptions,
    transform: Affine,
    paint: _GlyphPaint,
) -> None:
    """An underline below the baseline, or a strikethrough at the font's strikeout height. In vertical
    text the underline is the Japanese sideline, right of the column, and the strikethrough runs down its
    middle."""
    inline_start, inline_end, color, font, scale, gradient = bar
    if inline_end == inline_start:
        return
    font_size *= scale
    baseline_x, baseline_y = line.baseline
    under_offset, under_thickness, strike_height, strike_thickness = font.decoration_metrics(font_size)
    thickness = max(under_thickness if kind == "underline" else strike_thickness, 1.0)
    low, high = sorted((inline_start, inline_end))
    pad = options.padding
    if vertical:
        center = baseline_x + (font_size * 0.5 + under_offset if kind == "underline" else 0.0)
        top = baseline_y + low - options.baseline_shift
        bottom = baseline_y + high - options.baseline_shift
        rect = (pad + center - thickness / 2, pad + top, pad + center + thickness / 2, pad + bottom)
    else:
        center = baseline_y + (under_offset if kind == "underline" else -strike_height) - options.baseline_shift
        rect = (pad + baseline_x + low, pad + center - thickness / 2, pad + baseline_x + high, pad + center + thickness / 2)
    scene.runs.append(DecorationRun(rect=rect, transform=transform, dilation=paint.dilation_px, color=color, gradient=gradient))
