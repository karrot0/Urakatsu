"""HarfRust shaping and grapheme-safe font fallback.

Port of ``shape.rs``.

Divergences:
- HarfRust is replaced by HarfBuzz through ``uharfbuzz`` (harfrust is a HarfBuzz port).
  Positions are read in font units and scaled by ``font_size / upem`` exactly as in
  ``TextShaper::shape``. ``point_size`` maps to ``hb_font.ptem``.
- Cluster values are code-point indices into the Python ``str`` rather than UTF-8 byte
  offsets. Every consumer in this package uses the same convention.
- Extended grapheme clusters come from ``uniseg`` (UAX #29) instead of ICU's
  ``GraphemeClusterSegmenter``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import uharfbuzz as hb
from uniseg.graphemecluster import grapheme_clusters

from .script import COMMON, INHERITED, _script_of, harfbuzz_script

if TYPE_CHECKING:  # pragma: no cover
    from .fonts import Font


@dataclass
class PositionedGlyph:
    """A glyph positioned for layout while retaining its source font and text cluster."""

    glyph_id: int
    cluster: int
    font: "Font"
    x_advance: float
    y_advance: float
    x_offset: float
    y_offset: float
    # Rich text: the run's size relative to the layout's (the glyph was shaped at font_size * scale).
    scale: float = 1.0


@dataclass
class ShapedRun:
    glyphs: list[PositionedGlyph]
    x_advance: float
    y_advance: float


@dataclass
class ShapingOptions:
    direction: str  # "ltr" | "rtl" | "ttb" | "btt"
    script: Optional[str]  # ISO 15924 tag, e.g. "Hani"
    font_size: float
    features: dict[str, int] = field(default_factory=dict)


class TextShaper:
    """``TextShaper``: shapes one run with a single font."""

    def shape(self, text: str, font: "Font", options: ShapingOptions) -> ShapedRun:
        buffer = hb.Buffer()
        buffer.add_str(text)
        buffer.guess_segment_properties()
        buffer.direction = options.direction
        if options.script is not None:
            buffer.script = options.script

        hb_font = font.hb_font
        hb_font.ptem = options.font_size
        features = {tag: value for tag, value in options.features.items()} if options.features else None
        hb.shape(hb_font, buffer, features or {})

        scale = options.font_size / font.units_per_em
        glyphs: list[PositionedGlyph] = []
        x_total = 0.0
        y_total = 0.0
        for info, pos in zip(buffer.glyph_infos, buffer.glyph_positions):
            glyphs.append(
                PositionedGlyph(
                    glyph_id=info.codepoint,
                    cluster=info.cluster,
                    font=font,
                    x_offset=pos.x_offset * scale,
                    y_offset=pos.y_offset * scale,
                    x_advance=pos.x_advance * scale,
                    y_advance=pos.y_advance * scale,
                )
            )
            x_total += pos.x_advance * scale
            y_total += pos.y_advance * scale
        return ShapedRun(glyphs=glyphs, x_advance=x_total, y_advance=y_total)


def shape_script_runs(
    shaper: TextShaper,
    text: str,
    fonts: Sequence["Font"],
    options: ShapingOptions,
) -> list[ShapedRun]:
    """``shape_script_runs``: split by script, then by font coverage of grapheme clusters."""
    if not text or not fonts:
        return []

    runs: list[ShapedRun] = []
    n = len(text)
    i = 0
    while i < n:
        start = i
        script = _script_of(text[i])
        end = i + 1
        j = i + 1
        while j < n:
            next_script = _script_of(text[j])
            if next_script == script or next_script in (COMMON, INHERITED):
                j += 1
                end = j
            elif script in (COMMON, INHERITED):
                script = next_script
                j += 1
                end = j
            else:
                break
        i = end

        script_run_text = text[start:end]
        run_opts = ShapingOptions(
            direction=options.direction,
            script=harfbuzz_script(script),
            font_size=options.font_size,
            features=options.features,
        )

        if fonts[0].covers(script_run_text):
            _push_font_run(shaper, script_run_text, fonts[0], run_opts, start, runs)
            continue

        # No single face covers the script run: keep extended grapheme clusters together so
        # combining marks, variation selectors and emoji ZWJ sequences are not split.
        clusters = list(grapheme_clusters(script_run_text))
        group_start = 0
        group_font = 0
        offset = 0
        for index, cluster_text in enumerate(clusters):
            font_index = next((k for k, f in enumerate(fonts) if f.covers(cluster_text)), 0)
            if index == 0:
                group_font = font_index
            elif font_index != group_font:
                _push_font_run(
                    shaper,
                    script_run_text[group_start:offset],
                    fonts[group_font],
                    run_opts,
                    start + group_start,
                    runs,
                )
                group_start = offset
                group_font = font_index
            offset += len(cluster_text)
        _push_font_run(
            shaper,
            script_run_text[group_start:],
            fonts[group_font],
            run_opts,
            start + group_start,
            runs,
        )

    return runs


def _push_font_run(
    shaper: TextShaper,
    text: str,
    font: "Font",
    options: ShapingOptions,
    text_offset: int,
    runs: list[ShapedRun],
) -> None:
    if not text:
        return
    shaped = shaper.shape(text, font, options)
    for glyph in shaped.glyphs:
        glyph.cluster += text_offset
    runs.append(shaped)
