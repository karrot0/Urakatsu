"""Text layout and portable glyph recording.

Port of ``text_renderer.rs``.

Rust records glyph runs into a Vello ``PreparedScene``. Here a run is a ``GlyphRun`` with the
same fields (font, size, glyph ids and positions, transform, synthetic skew, embolden amount,
colour); ``frame.py`` rasterizes them with Skia paths built from the font outlines.

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
class StrokeOptions:
    color: Rgba
    width_px: float


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


@dataclass
class TextScene:
    runs: list[GlyphRun] = field(default_factory=list)


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


@dataclass
class _GlyphPaint:
    color: Rgba
    dilation_px: float


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
    ) -> None:
        """Border first as an outward-only dilation of the outline, then the fill on top.

        Unlike a centred stroke, a dilation never grows inward, so it cannot punch a hole through
        small features once the border width exceeds their radius.
        """
        if options.stroke is not None and options.stroke.width_px > 0.0 and options.stroke.color[3] > 0:
            _draw_layout(
                scene,
                layout,
                writing_mode,
                options,
                transform,
                _GlyphPaint(color=options.stroke.color, dilation_px=options.stroke.width_px),
            )
        _draw_layout(
            scene,
            layout,
            writing_mode,
            options,
            transform,
            _GlyphPaint(color=options.color, dilation_px=0.0),
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
        options = TextRenderOptions(color=color, stroke=descriptor.stroke)
        scene = TextScene()
        self.render(scene, run, descriptor.writing_mode, options, transform)

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
        local_bounds = RenderBounds(
            x=rendered_bounds.x - stroke_padding,
            y=rendered_bounds.y - stroke_padding,
            width=rendered_bounds.width + stroke_padding * 2.0,
            height=rendered_bounds.height + stroke_padding * 2.0,
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


def _draw_layout(
    scene: TextScene,
    layout: LayoutRun,
    writing_mode: WritingMode,
    options: TextRenderOptions,
    transform: Affine,
    paint: _GlyphPaint,
) -> None:
    """``draw_layout``: groups consecutive glyphs by font and records one run per group."""
    for line in layout.lines:
        baseline_x, baseline_y = line.baseline
        pen_x = 0.0
        pen_y = 0.0
        start = 0
        glyphs = line.glyphs
        while start < len(glyphs):
            font = glyphs[start].font
            end = start + 1
            while end < len(glyphs) and glyphs[end].font is font:
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
                pen_x += glyph.x_advance
                pen_y -= glyph.y_advance

            synthetic_bold = 1.0 if font.synthetic_bold() else 0.0
            skew = font.synthetic_skew()
            scene.runs.append(
                GlyphRun(
                    font=font,
                    font_size=layout.font_size,
                    glyphs=positioned,
                    transform=transform,
                    glyph_skew=math.tan(math.radians(skew)) if skew is not None else None,
                    embolden=synthetic_bold + paint.dilation_px,
                    color=paint.color,
                )
            )
            start = end
