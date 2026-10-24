"""Immutable retained page output and synchronous raster access.

Port of ``frame.rs``. A ``Frame`` is one prepared page: its layers
(source, cleanup and paint rasters, embedded images, and visible translated text) in stacking
order, with the metadata the editor and exporters read. ``Frame.composite`` flattens the page and
``Layer.image`` returns one layer's RGBA pixels, which is what a PSD export stores per layer.

Divergences:
- Rust hands the layers to the rasterizer (Vello) as a prepared bundle. Here every visual
  is rasterized with Skia (``skia-python``): text from glyph outline paths, raster images as
  premultiplied-aware RGBA images, and the page composite by source-over on RGBA.
- ``Layer.image`` is the layer's pixels cropped to ``bounds``. Rust's ``PreparedFrameBundle`` (the
  GPU-side tile and resource format) has no Python counterpart and is not produced.
- Affine transforms use kurbo's six-coefficient convention ``(a, b, c, d, e, f)``, with
  ``mul(a, b)`` applying ``b`` first, as ``vello::kurbo::Affine`` does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

import numpy as np
import skia
from PIL import Image as PILImage

from ..scene.components import Geometry
from .types import TextAlign, WritingMode

if TYPE_CHECKING:  # pragma: no cover
    from .text_renderer import TextScene

Affine = tuple[float, float, float, float, float, float]

MAX_SURFACE_DIMENSION = 32_768
MAX_SURFACE_PIXELS = 268_435_456


# ---- affine helpers (kurbo convention) ---------------------------------------------------------


def affine_identity() -> Affine:
    return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def affine_mul(left: Affine, right: Affine) -> Affine:
    """``left * right``: applies ``right`` first, then ``left``."""
    a1, b1, c1, d1, e1, f1 = left
    a2, b2, c2, d2, e2, f2 = right
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def affine_translate(x: float, y: float) -> Affine:
    return (1.0, 0.0, 0.0, 1.0, x, y)


def affine_scale_non_uniform(sx: float, sy: float) -> Affine:
    return (sx, 0.0, 0.0, sy, 0.0, 0.0)


def affine_rotate(radians: float) -> Affine:
    s, c = math.sin(radians), math.cos(radians)
    return (c, s, -s, c, 0.0, 0.0)


def affine_then_translate(transform: Affine, x: float, y: float) -> Affine:
    return affine_mul(affine_translate(x, y), transform)


def affine_apply(transform: Affine, x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = transform
    return (a * x + c * y + e, b * x + d * y + f)


def affine_bbox(transform: Affine, x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
    corners = [affine_apply(transform, x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return min(xs), min(ys), max(xs), max(ys)


# ---- value types -------------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderBounds:
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0


@dataclass(frozen=True)
class Presentation:
    visible: bool
    opacity: float


class ImageKind(str, Enum):
    SOURCE = "source"
    CLEANUP = "cleanup"
    PAINT = "paint"
    EMBEDDED = "embedded"


@dataclass(frozen=True)
class ImageMetadata:
    name: Optional[str]
    kind: ImageKind


@dataclass(frozen=True)
class TextMetadata:
    text: str
    language: Optional[str]
    rendered_bounds: RenderBounds
    layout_bounds: RenderBounds
    post_script_fonts: list[str]
    font_size: float
    color: tuple[int, int, int, int]
    alignment: TextAlign
    writing_mode: WritingMode
    angle_degrees: float


@dataclass(frozen=True)
class RetentionStats:
    candidate_layers: int = 0
    reused_layers: int = 0
    rebuilt_layers: int = 0


@dataclass(frozen=True)
class RenderDependency:
    """What a layer was built from: ``kind`` is one of entity, hierarchy, component, relation,
    relation_query, relation_target_query, blob, font (mirrors ``RenderDependency``)."""

    kind: str
    key: str


@dataclass(frozen=True)
class RenderDiagnostic:
    """``RenderDiagnostic``. ``kind`` is ``missing_asset``, ``text_overflow`` or
    ``text_below_readable_size``; the other fields are set per kind."""

    kind: str
    entity: str
    role: Optional[str] = None
    available: Optional[RenderBounds] = None
    actual_width: float = 0.0
    actual_height: float = 0.0
    font_size: float = 0.0
    minimum_font_size: float = 0.0


@dataclass
class RasterImage:
    """A decoded source/cleanup/paint/embedded image: RGBA8 pixels, row-major, unpremultiplied."""

    blob: str
    width: int
    height: int
    media_type: str
    encoded: bytes
    pixels: np.ndarray  # shape (height, width, 4), uint8

    def size(self) -> tuple[int, int]:
        return (self.width, self.height)


@dataclass
class Layer:
    """One stacking entry of a frame (``Layer`` / ``LayerData``)."""

    entity: str
    geometry: Geometry
    bounds: RenderBounds  # page coordinates, transformed local bounds
    presentation: Presentation
    ancestry: list[str]
    kind: object  # ImageMetadata | TextMetadata
    placement: Affine
    local_bounds: RenderBounds
    raster: Optional[RasterImage] = None
    scene: Optional["TextScene"] = None
    dependencies: list[RenderDependency] = field(default_factory=list)
    element_angle: float = 0.0

    def raster_image(self) -> Optional[RasterImage]:
        return self.raster

    def is_text(self) -> bool:
        return isinstance(self.kind, TextMetadata)

    def image(self) -> PILImage.Image:
        """The layer's pixels, cropped to ``bounds`` (rounded outward), straight-alpha RGBA.

        The layer's own presentation (visibility, opacity) is not applied here; that is the
        compositor's job, as in ``Frame::cropped``.
        """
        left, top, right, bottom = _rounded_out(self.bounds)
        width = max(right - left, 1)
        height = max(bottom - top, 1)
        surface = skia.Surface(width, height)
        canvas = surface.getCanvas()
        canvas.clear(skia.ColorTRANSPARENT)
        # Local-to-canvas: placement, then shift the crop origin to (0, 0).
        matrix = affine_mul(affine_translate(-left, -top), self.placement)
        _draw_layer_content(canvas, self, matrix)
        return _surface_to_pil(surface)


@dataclass
class Frame:
    """An immutable, fully prepared page (``Frame``)."""

    revision: int
    page: str
    width: int
    height: int
    origin: tuple[int, int]
    layers: list[Layer]
    dependencies: list[RenderDependency]
    diagnostics: list[RenderDiagnostic]
    stats: RetentionStats = field(default_factory=RetentionStats)

    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def layer(self, entity: str) -> Optional[Layer]:
        for layer in self.layers:
            if layer.entity == entity:
                return layer
        return None

    def composite(self) -> PILImage.Image:
        """Flattens the page: source, then each visible layer in order (source-over)."""
        canvas = PILImage.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        for layer in self.layers:
            if not layer.presentation.visible or layer.presentation.opacity <= 0.0:
                continue
            image = layer.image()
            if layer.presentation.opacity < 1.0:
                alpha = np.asarray(image.getchannel("A"), dtype=np.float32) * layer.presentation.opacity
                image = image.copy()
                image.putalpha(PILImage.fromarray(np.clip(alpha, 0, 255).astype(np.uint8), mode="L"))
            left, top, right, bottom = _rounded_out(layer.bounds)
            # Layer images are cropped to their own bounds, so paste at the same origin.
            overlay = PILImage.new("RGBA", canvas.size, (0, 0, 0, 0))
            overlay.paste(image, (left, top))
            canvas = PILImage.alpha_composite(canvas, overlay)
        return canvas

    def cropped(self, entity: str) -> Optional["Frame"]:
        """One entity normalized into a tightly cropped frame, drawn at full opacity."""
        layer = self.layer(entity)
        if layer is None:
            return None
        bounds = layer.bounds
        if not all(math.isfinite(v) for v in (bounds.x, bounds.y, bounds.width, bounds.height)):
            raise ValueError(f"visual layer bounds are not finite for entity {entity}")
        if bounds.width < 0.0 or bounds.height < 0.0:
            raise ValueError(f"visual layer bounds are negative for entity {entity}")
        left, top, right, bottom = _rounded_out(bounds)
        width = max(right - left, 1)
        height = max(bottom - top, 1)
        if width > MAX_SURFACE_DIMENSION or height > MAX_SURFACE_DIMENSION or width * height > MAX_SURFACE_PIXELS:
            raise ValueError(f"cropped surface {width}x{height} exceeds renderer limits")
        isolated = Layer(
            entity=layer.entity,
            geometry=layer.geometry,
            bounds=RenderBounds(bounds.x - left, bounds.y - top, bounds.width, bounds.height),
            presentation=Presentation(visible=True, opacity=1.0),
            ancestry=list(layer.ancestry),
            kind=layer.kind,
            placement=affine_mul(affine_translate(-left, -top), layer.placement),
            local_bounds=layer.local_bounds,
            raster=layer.raster,
            scene=layer.scene,
            dependencies=list(layer.dependencies),
            element_angle=layer.element_angle,
        )
        return Frame(
            revision=self.revision,
            page=self.page,
            width=width,
            height=height,
            origin=(left, top),
            layers=[isolated],
            dependencies=list(self.dependencies),
            diagnostics=list(self.diagnostics),
            stats=self.stats,
        )


# ---- drawing -----------------------------------------------------------------------------------


def _rounded_out(bounds: RenderBounds) -> tuple[int, int, int, int]:
    """``Frame::cropped`` edge rule: floor the origin, ceil the far edge."""
    left = math.floor(bounds.x)
    top = math.floor(bounds.y)
    right = math.ceil(bounds.x + bounds.width)
    bottom = math.ceil(bounds.y + bounds.height)
    return int(left), int(top), int(right), int(bottom)


def _surface_to_pil(surface: skia.Surface) -> PILImage.Image:
    array = surface.makeImageSnapshot().toarray(
        colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kUnpremul_AlphaType
    )
    return PILImage.fromarray(np.ascontiguousarray(array), mode="RGBA")


def _to_skia_color(color: tuple[int, int, int, int]) -> int:
    r, g, b, a = color
    return skia.Color(r, g, b, a)


def _draw_layer_content(canvas: skia.Canvas, layer: Layer, matrix: Affine) -> None:
    if layer.raster is not None:
        _draw_raster(canvas, layer.raster, layer, matrix)
    elif layer.scene is not None:
        for run in layer.scene.runs:
            if hasattr(run, "rect"):
                _draw_decoration(canvas, run, matrix)
            else:
                _draw_glyph_run(canvas, run, matrix)


def _gradient_shader(gradient):
    """A Skia shader for a ``GradientPaint`` (layout coordinates; draw under the layout's matrix)."""
    colors = [_to_skia_color(color) for color in gradient.colors]
    positions = list(gradient.positions)
    if gradient.kind == "radial":
        return skia.GradientShader.MakeRadial(center=gradient.start, radius=gradient.radius, colors=colors, positions=positions, mode=skia.TileMode.kClamp)
    return skia.GradientShader.MakeLinear(points=[gradient.start, gradient.end], colors=colors, positions=positions, mode=skia.TileMode.kClamp)


def _draw_decoration(canvas: skia.Canvas, run, matrix: Affine) -> None:
    """An underline or strikethrough bar (rich text), grown by ``dilation`` for the border pass."""
    x0, y0, x1, y1 = run.rect
    grow = run.dilation
    paint = skia.Paint()
    paint.setAntiAlias(True)
    paint.setColor(_to_skia_color(run.color))
    if getattr(run, "gradient", None) is not None:
        paint.setShader(_gradient_shader(run.gradient))
    canvas.save()
    canvas.concat(_skia_matrix(affine_mul(matrix, run.transform)))
    canvas.drawRect(skia.Rect.MakeLTRB(x0 - grow, y0 - grow, x1 + grow, y1 + grow), paint)
    canvas.restore()


def _skia_matrix(transform: Affine) -> skia.Matrix:
    a, b, c, d, e, f = transform
    # skia Matrix layout: [scaleX, skewX, transX, skewY, scaleY, transY, persp0, persp1, persp2]
    return skia.Matrix.MakeAll(a, c, e, b, d, f, 0.0, 0.0, 1.0)


def _draw_raster(canvas: skia.Canvas, image: RasterImage, layer: Layer, matrix: Affine) -> None:
    if image.width <= 0 or image.height <= 0:
        return
    skia_image = skia.Image.fromarray(
        np.ascontiguousarray(image.pixels), colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kUnpremul_AlphaType
    )
    canvas.save()
    canvas.concat(_skia_matrix(matrix))
    paint = skia.Paint()
    paint.setAntiAlias(True)
    sampling = skia.SamplingOptions(skia.FilterMode.kLinear, skia.MipmapMode.kNone)
    canvas.drawImageRect(
        skia_image,
        skia.Rect.MakeWH(float(image.width), float(image.height)),
        sampling,
        paint,
    )
    canvas.restore()


def _glyph_path(font, glyph_id: int) -> skia.Path:
    """Glyph outline in font units, y up, as a Skia path (cached on the font)."""
    cache = font.__dict__.setdefault("_skia_glyph_paths", {})
    path = cache.get(glyph_id)
    if path is not None:
        return path
    path = skia.Path()
    for op, args in font.glyph_recording(glyph_id):
        if op == "moveTo":
            path.moveTo(*args[0])
        elif op == "lineTo":
            path.lineTo(*args[0])
        elif op == "qCurveTo":
            # fontTools decomposes to quadratic segments; a trailing None means a closed contour.
            points = [p for p in args if p is not None]
            if not points:
                continue
            control = points[:-1]
            end = points[-1]
            if len(control) == 1:
                path.quadTo(*control[0], *end)
            else:
                # Implied on-curve midpoints between consecutive off-curve points.
                for index in range(len(control) - 1):
                    c0 = control[index]
                    c1 = control[index + 1]
                    mid = ((c0[0] + c1[0]) * 0.5, (c0[1] + c1[1]) * 0.5)
                    path.quadTo(*c0, *mid)
                path.quadTo(*control[-1], *end)
        elif op == "curveTo":
            path.cubicTo(*args[0], *args[1], *args[2])
        elif op in ("closePath", "endPath"):
            path.close()
    cache[glyph_id] = path
    return path


def _draw_gradient_glyph_run(canvas: skia.Canvas, run, matrix: Affine) -> None:
    """A glyph run painted with a gradient. The gradient lives in layout coordinates, so the canvas keeps
    the layout's matrix and each outline is moved into layout space instead of the canvas into glyph space."""
    shader = _gradient_shader(run.gradient)
    scale = run.font_size / run.font.units_per_em
    fill = skia.Paint()
    fill.setAntiAlias(True)
    fill.setShader(shader)
    fill.setStyle(skia.Paint.kFill_Style)
    stroke = None
    if run.embolden > 0.0:
        stroke = skia.Paint()
        stroke.setAntiAlias(True)
        stroke.setShader(shader)
        stroke.setStyle(skia.Paint.kStroke_Style)
        stroke.setStrokeWidth(2.0 * run.embolden)  # layout pixels: outward by ``embolden``
        stroke.setStrokeJoin(skia.Paint.kRound_Join)
        stroke.setStrokeCap(skia.Paint.kRound_Cap)
    canvas.save()
    canvas.concat(_skia_matrix(affine_mul(matrix, run.transform)))
    for glyph_id, x, y in run.glyphs:
        outline = _glyph_path(run.font, glyph_id)
        if outline.isEmpty():
            continue
        glyph_matrix = (scale, 0.0, 0.0, -scale, 0.0, 0.0)
        if run.glyph_skew is not None:
            glyph_matrix = affine_mul((1.0, 0.0, -run.glyph_skew, 1.0, 0.0, 0.0), glyph_matrix)
        placed = skia.Path(outline)
        placed.transform(_skia_matrix(affine_mul(affine_translate(x, y), glyph_matrix)))
        if stroke is not None:
            canvas.drawPath(placed, stroke)
        canvas.drawPath(placed, fill)
    canvas.restore()


def _draw_glyph_run(canvas: skia.Canvas, run, matrix: Affine) -> None:
    if getattr(run, "gradient", None) is not None:
        return _draw_gradient_glyph_run(canvas, run, matrix)
    a, b, c, d, e, f = run.transform
    base = affine_mul(matrix, (a, b, c, d, e, f))
    scale = run.font_size / run.font.units_per_em
    color = _to_skia_color(run.color)
    fill = skia.Paint()
    fill.setAntiAlias(True)
    fill.setColor(color)
    fill.setStyle(skia.Paint.kFill_Style)
    stroke = None
    if run.embolden > 0.0:
        # Outward offset of ``embolden`` pixels: a centred stroke of twice that width. The stroke is
        # drawn in glyph units (the canvas matrix carries the font-size scale), so divide by it.
        stroke = skia.Paint()
        stroke.setAntiAlias(True)
        stroke.setColor(color)
        stroke.setStyle(skia.Paint.kStroke_Style)
        stroke.setStrokeWidth(2.0 * run.embolden / scale)
        stroke.setStrokeJoin(skia.Paint.kRound_Join)
        stroke.setStrokeCap(skia.Paint.kRound_Cap)
    skew = run.glyph_skew
    for glyph_id, x, y in run.glyphs:
        path = _glyph_path(run.font, glyph_id)
        if path.isEmpty():
            continue
        # Glyph space to layout space: scale to pixels, flip to y down, skew (synthetic italic)
        # in y-down pixels, then place at the glyph origin.
        sx = scale
        glyph_matrix = (sx, 0.0, 0.0, -sx, 0.0, 0.0)
        if skew is not None:
            glyph_matrix = affine_mul((1.0, 0.0, -skew, 1.0, 0.0, 0.0), glyph_matrix)
        placed = affine_mul(affine_translate(x, y), glyph_matrix)
        total = affine_mul(base, placed)
        canvas.save()
        canvas.concat(_skia_matrix(total))
        if stroke is not None:
            canvas.drawPath(path, stroke)
        canvas.drawPath(path, fill)
        canvas.restore()
