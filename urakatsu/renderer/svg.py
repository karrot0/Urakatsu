"""The page's text as vector graphics, for the editor.

Koharu's editor draws its prepared frame (glyph runs as outlines) with Vello on WebGPU at screen
resolution, so text stays sharp at any zoom while the export is rasterized at the page's own size.
Urakatsu's editor is a browser canvas over server-rendered images, so the same split is made here:
``split_index`` cuts the frame's stack at its first visible text layer, everything below is one
raster image (the page's art), and ``layers_svg`` writes everything from there up as SVG, glyphs
as outline paths, which the browser draws at whatever size the page is shown.

The drawing mirrors ``frame._draw_layer_content`` run for run (the same outlines, transforms,
synthetic bold strokes, gradients and decoration bars), so the vector text is the export's text,
only not yet rasterized. A raster layer stacked above text (rare: a paint layer over the lettering)
is embedded as a PNG at its own place so the stacking order holds.
"""

from __future__ import annotations

import base64
import io
import math
from typing import Iterable, Optional
from xml.sax.saxutils import quoteattr

from .frame import Affine, Frame, Layer, _rounded_out, affine_apply, affine_mul, affine_translate


def split_index(frame: Frame) -> int:
    """Index of the first layer drawn as vectors: the first visible text layer (the layer count when
    the page shows no text, so the whole page is the raster part)."""
    for index, layer in enumerate(frame.layers):
        if layer.is_text() and layer.presentation.visible and layer.presentation.opacity > 0.0:
            return index
    return len(frame.layers)


def layers_svg(frame: Frame, layers: Iterable[Layer]) -> bytes:
    """``layers`` (in stacking order) as one SVG the size of the page, transparent elsewhere."""
    writer = _Writer()
    body: list[str] = []
    for layer in layers:
        if not layer.presentation.visible or layer.presentation.opacity <= 0.0:
            continue
        opacity = layer.presentation.opacity
        body.append(f'<g data-layer={quoteattr(layer.entity)}' + (f' opacity="{_n(opacity)}"' if opacity < 1.0 else "") + ">")
        if layer.raster is not None:
            body.append(_raster(layer))
        elif layer.scene is not None:
            for run in layer.scene.runs:
                body.append(writer.decoration(run, layer.placement) if hasattr(run, "rect") else writer.glyph_run(run, layer.placement))
        body.append("</g>")
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{frame.width}" height="{frame.height}" viewBox="0 0 {frame.width} {frame.height}">'
    )
    defs = f"<defs>{''.join(writer.defs)}</defs>" if writer.defs else ""
    return (head + defs + "".join(body) + "</svg>").encode("utf-8")


# ---- numbers and colours ------------------------------------------------------------------------


def _n(value: float, places: int = 3) -> str:
    text = f"{value:.{places}f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _matrix(transform: Affine) -> str:
    a, b, c, d, e, f = transform
    return f"matrix({_n(a, 6)} {_n(b, 6)} {_n(c, 6)} {_n(d, 6)} {_n(e, 4)} {_n(f, 4)})"


def _paint(color: tuple[int, int, int, int], prefix: str) -> str:
    r, g, b, a = color
    attrs = f' {prefix}="#{r:02x}{g:02x}{b:02x}"'
    if a < 255:
        attrs += f' {prefix}-opacity="{_n(a / 255.0, 4)}"'
    return attrs


# ---- outlines -----------------------------------------------------------------------------------


def _outline(font, glyph_id: int, transform: Optional[Affine] = None, places: int = 2) -> str:
    """A glyph's outline as SVG path data, in font units (y up), or moved by ``transform``. Built from
    the same pen operations as ``frame._glyph_path``."""

    def p(point) -> str:
        x, y = point if transform is None else affine_apply(transform, point[0], point[1])
        return f"{_n(x, places)} {_n(y, places)}"

    parts: list[str] = []
    started = False
    for op, args in font.glyph_recording(glyph_id):
        if op == "moveTo":
            parts.append("M" + p(args[0]))
            started = True
            continue
        if not started and op in ("lineTo", "qCurveTo", "curveTo"):
            parts.append("M" + p((0.0, 0.0)))  # Skia starts a path without a move at the origin
            started = True
        if op == "lineTo":
            parts.append("L" + p(args[0]))
        elif op == "qCurveTo":
            points = [point for point in args if point is not None]
            if not points:
                continue
            control, end = points[:-1], points[-1]
            if len(control) == 1:
                parts.append("Q" + p(control[0]) + " " + p(end))
            else:
                for index in range(len(control) - 1):
                    c0, c1 = control[index], control[index + 1]
                    parts.append("Q" + p(c0) + " " + p(((c0[0] + c1[0]) * 0.5, (c0[1] + c1[1]) * 0.5)))
                parts.append("Q" + p(control[-1]) + " " + p(end))
        elif op == "curveTo":
            parts.append("C" + p(args[0]) + " " + p(args[1]) + " " + p(args[2]))
        elif op in ("closePath", "endPath"):
            parts.append("Z")
            started = False
    return "".join(parts)


def _glyph_matrix(run) -> Affine:
    """Glyph space to layout space without the origin: scale to pixels, flip to y down, then the
    synthetic italic skew (as ``frame._draw_glyph_run``)."""
    scale = run.font_size / run.font.units_per_em
    matrix: Affine = (scale, 0.0, 0.0, -scale, 0.0, 0.0)
    if run.glyph_skew is not None:
        matrix = affine_mul((1.0, 0.0, -run.glyph_skew, 1.0, 0.0, 0.0), matrix)
    return matrix


class _Writer:
    """Collects shared definitions (one outline per font glyph, gradients) while runs are written."""

    def __init__(self) -> None:
        self.defs: list[str] = []
        self._glyphs: dict[tuple[int, int], Optional[str]] = {}
        self._fonts: dict[int, int] = {}
        self._gradients = 0

    def _glyph(self, font, glyph_id: int) -> Optional[str]:
        key = (id(font), glyph_id)
        if key not in self._glyphs:
            data = _outline(font, glyph_id)
            if not data:
                self._glyphs[key] = None
            else:
                number = self._fonts.setdefault(id(font), len(self._fonts))
                ident = f"f{number}g{glyph_id}"
                self.defs.append(f'<path id="{ident}" d="{data}"/>')
                self._glyphs[key] = ident
        return self._glyphs[key]

    def _gradient(self, gradient) -> str:
        self._gradients += 1
        ident = f"gr{self._gradients}"
        stops = "".join(
            f'<stop offset="{_n(position, 4)}" stop-color="#{r:02x}{g:02x}{b:02x}"' + (f' stop-opacity="{_n(a / 255.0, 4)}"' if a < 255 else "") + "/>"
            for position, (r, g, b, a) in zip(gradient.positions, gradient.colors)
        )
        if gradient.kind == "radial":
            (cx, cy), radius = gradient.start, gradient.radius
            self.defs.append(f'<radialGradient id="{ident}" gradientUnits="userSpaceOnUse" cx="{_n(cx)}" cy="{_n(cy)}" r="{_n(radius)}">{stops}</radialGradient>')
        else:
            (x1, y1), (x2, y2) = gradient.start, gradient.end
            self.defs.append(
                f'<linearGradient id="{ident}" gradientUnits="userSpaceOnUse" x1="{_n(x1)}" y1="{_n(y1)}" x2="{_n(x2)}" y2="{_n(y2)}">{stops}</linearGradient>'
            )
        return ident

    def glyph_run(self, run, placement: Affine) -> str:
        base = affine_mul(placement, run.transform)
        glyph = _glyph_matrix(run)
        if getattr(run, "gradient", None) is not None:
            # The gradient is laid out in layout coordinates, so the outlines are moved there instead
            # of each glyph getting its own coordinate system (as ``_draw_gradient_glyph_run``).
            ident = self._gradient(run.gradient)
            data = "".join(_outline(run.font, glyph_id, affine_mul(affine_translate(x, y), glyph)) for glyph_id, x, y in run.glyphs)
            if not data:
                return ""
            stroke = (
                f' stroke="url(#{ident})" stroke-width="{_n(2.0 * run.embolden, 4)}" stroke-linejoin="round" stroke-linecap="round"'
                if run.embolden > 0.0
                else ""
            )
            return f'<path transform="{_matrix(base)}" fill="url(#{ident})"{stroke} d="{data}"/>'
        scale = run.font_size / run.font.units_per_em
        paint = _paint(run.color, "fill")
        if run.embolden > 0.0:
            # Outward by ``embolden`` pixels: a centred stroke twice that wide, in glyph units.
            paint += _paint(run.color, "stroke") + f' stroke-width="{_n(2.0 * run.embolden / scale, 3)}" stroke-linejoin="round" stroke-linecap="round"'
        uses = []
        for glyph_id, x, y in run.glyphs:
            ident = self._glyph(run.font, glyph_id)
            if ident is None:
                continue
            uses.append(f'<use xlink:href="#{ident}" transform="{_matrix(affine_mul(base, affine_mul(affine_translate(x, y), glyph)))}"/>')
        return f"<g{paint}>{''.join(uses)}</g>" if uses else ""

    def decoration(self, run, placement: Affine) -> str:
        x0, y0, x1, y1 = run.rect
        grow = run.dilation
        fill = f' fill="url(#{self._gradient(run.gradient)})"' if getattr(run, "gradient", None) is not None else _paint(run.color, "fill")
        return (
            f'<rect transform="{_matrix(affine_mul(placement, run.transform))}" x="{_n(x0 - grow)}" y="{_n(y0 - grow)}" '
            f'width="{_n(x1 - x0 + 2 * grow)}" height="{_n(y1 - y0 + 2 * grow)}"{fill}/>'
        )


def _raster(layer: Layer) -> str:
    """A raster layer as an embedded PNG at its bounds (its pixels as the compositor draws them)."""
    left, top, right, bottom = _rounded_out(layer.bounds)
    if right <= left or bottom <= top or not all(math.isfinite(v) for v in (left, top, right, bottom)):
        return ""
    buffer = io.BytesIO()
    layer.image().save(buffer, format="PNG")
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<image x="{left}" y="{top}" width="{right - left}" height="{bottom - top}" preserveAspectRatio="none" xlink:href="data:image/png;base64,{data}"/>'
