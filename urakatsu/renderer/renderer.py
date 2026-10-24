"""Renderer ownership, scene interpretation, and frame assembly.

Port of ``renderer.rs`` (full render path). ``Renderer.render`` compiles
one page of a scene ``Snapshot`` into layer drafts, builds each text layer and image node, and
assembles an immutable ``Frame``.

Not ported (Rust-only machinery with no effect on the output):
- retained node caches, ``update`` with ``AffectedDependencies`` and ``RenderDependency`` tracking:
  every ``render`` is a full render, which is the output Koharu's incremental path must match;
- async image loading, rayon worker pools, font preview WebP generation;
- ``Renderer::font_preview``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


from ..core.config import TypesettingConfig
from ..scene import components as c
from ..scene.scene import Snapshot
from .bubble import GeometryFrame, LayoutBox, contour, flow_cells, geometry_bounds, geometry_frame
from .fonts import Fonts
from .images import ImageCache, decode
from .frame import (
    Affine,
    Frame,
    ImageKind,
    ImageMetadata,
    Layer,
    Presentation,
    RasterImage,
    RenderBounds,
    RenderDiagnostic,
    RetentionStats,
    TextMetadata,
    affine_bbox,
    affine_mul,
    affine_scale_non_uniform,
    affine_then_translate,
    affine_translate,
    affine_rotate,
)
from .script import is_chinese_or_japanese_text, shaping_direction_for_text
from .text_renderer import MINIMUM_FONT_SIZE, GradientSpec, StrokeOptions, TextNodeDescriptor, TextRenderer, TextStyleRun
from .types import FontStyle, TextAlign, WritingMode

MAX_SURFACE_DIMENSION = 32_768
MAX_SURFACE_PIXELS = 268_435_456
SOURCE_ROLE = "source"


class RendererError(Exception):
    pass


@dataclass
class ImageNodeDescriptor:
    blob: str
    media_type: str
    expected_size: Optional[tuple[int, int]]
    require_size: Optional[tuple[int, int]]
    data: bytes


@dataclass
class ResolvedPlacement:
    geometry: c.Geometry
    frame: GeometryFrame
    balloon_contour: Optional[list[tuple[float, float]]]


@dataclass
class LayerDraft:
    entity: str
    geometry: c.Geometry
    frame: GeometryFrame
    presentation: Presentation
    ancestry: list[str]
    descriptor: object  # ImageNodeDescriptor | TextNodeDescriptor
    metadata: object  # ImageMetadata | TextDraftMetadata


@dataclass
class TextDraftMetadata:
    text: str
    language: Optional[str]
    alignment: TextAlign
    writing_mode: WritingMode
    angle_degrees: float


@dataclass
class CompiledPage:
    revision: int
    page: str
    width: int
    height: int
    layers: list[LayerDraft]
    diagnostics: list[RenderDiagnostic]


def _surface_size(page: c.Page) -> tuple[int, int]:
    """``surface_size``: finite, positive page dimensions rounded up, within renderer limits."""
    if not all(math.isfinite(v) for v in (page.width, page.height)) or page.width <= 0.0 or page.height <= 0.0:
        raise RendererError("page dimensions must be finite and positive")
    width = int(math.ceil(page.width))
    height = int(math.ceil(page.height))
    if width > MAX_SURFACE_DIMENSION or height > MAX_SURFACE_DIMENSION or width * height > MAX_SURFACE_PIXELS:
        raise RendererError(f"page surface {width}x{height} exceeds renderer limits")
    return width, height


def _transform_bounds(bounds: RenderBounds, transform: Affine) -> RenderBounds:
    x0, y0, x1, y1 = affine_bbox(transform, bounds.x, bounds.y, bounds.x + bounds.width, bounds.y + bounds.height)
    return RenderBounds(x=x0, y=y0, width=x1 - x0, height=y1 - y0)


def _placement_for_text(frame: GeometryFrame, width: float, height: float) -> Affine:
    """Centre-rotate-corner placement of a text box inside its (possibly rotated) frame."""
    bounds = frame.bounds
    centre = affine_translate(bounds.x + bounds.width * 0.5, bounds.y + bounds.height * 0.5)
    rotation = affine_rotate(math.radians(frame.angle_degrees))
    corner = affine_translate(-width * 0.5, -height * 0.5)
    return affine_mul(affine_mul(centre, rotation), corner)


class Renderer:
    """``Renderer``: stateless apart from font and decoded-image caches."""

    def __init__(self, typesetting: Optional[TypesettingConfig] = None) -> None:
        self.typesetting = typesetting or TypesettingConfig()
        self.fonts = Fonts()
        self._images = ImageCache()
        self._text = TextRenderer()

    def _decode(self, descriptor: "ImageNodeDescriptor") -> RasterImage:
        """Decoded image for a blob, with the expected and required size checks of ``build_node``."""
        decoded = self._images.get(descriptor.blob)
        if decoded is None:
            try:
                decoded = decode(descriptor.blob, descriptor.data, descriptor.expected_size)
            except ValueError as error:
                raise RendererError(str(error)) from error
            self._images.insert(descriptor.blob, decoded)
        size = (decoded.width, decoded.height)
        if descriptor.require_size is not None and descriptor.require_size != size:
            raise RendererError(
                f"base image blob {descriptor.blob} is {size[0]}x{size[1]}, expected {descriptor.require_size}"
            )
        return RasterImage(
            blob=descriptor.blob,
            width=decoded.width,
            height=decoded.height,
            media_type=descriptor.media_type,
            encoded=decoded.encoded,
            pixels=decoded.pixels,
        )

    def available_fonts(self) -> list:
        """``Renderer::available_fonts``: system families merged with bundled ones."""
        return self.fonts.families()

    def render(self, snapshot: Snapshot, page: str) -> Frame:
        """Completely renders one page (``Renderer::render``)."""
        compiled = self._compile(snapshot, page)
        return self._finish(compiled)

    # ---- compile -----------------------------------------------------------------------------

    def _compile(self, snapshot: Snapshot, page: str) -> CompiledPage:
        page_value = snapshot.component(page, c.Page)
        if page_value is None:
            raise RendererError(f"page {page} has no page component")
        width, height = _surface_size(page_value)
        flow_placements = _resolve_balloon_flows(snapshot, page)
        traversal = _Traversal(
            snapshot=snapshot,
            page=page,
            width=width,
            height=height,
            font_families=list(self.typesetting.font_families),
            balloon_flows=flow_placements,
        )
        source = snapshot.asset(page, SOURCE_ROLE)
        if source is not None:
            geometry = c.Geometry.rectangle(0.0, 0.0, float(width), float(height))
            traversal.layers.append(
                _image_draft(
                    page,
                    geometry,
                    ImageNodeDescriptor(
                        blob=source.blob,
                        media_type=source.media_type,
                        expected_size=_asset_size(source),
                        require_size=(width, height),
                        data=snapshot.asset_bytes(page, SOURCE_ROLE),
                    ),
                    ImageMetadata(name=None, kind=ImageKind.SOURCE),
                    Presentation(visible=True, opacity=1.0),
                    [],
                )
            )
        else:
            traversal.diagnostics.append(RenderDiagnostic(kind="missing_asset", entity=page, role=SOURCE_ROLE))
        traversal.visit_children(page, Presentation(visible=True, opacity=1.0), [])
        return CompiledPage(
            revision=snapshot.revision,
            page=page,
            width=width,
            height=height,
            layers=traversal.layers,
            diagnostics=traversal.diagnostics,
        )

    # ---- finish ------------------------------------------------------------------------------

    def _finish(self, compiled: CompiledPage) -> Frame:
        layers: list[Layer] = []
        diagnostics = list(compiled.diagnostics)
        for draft in compiled.layers:
            if isinstance(draft.descriptor, ImageNodeDescriptor):
                layer, extra = self._build_image_layer(draft)
            else:
                layer, extra = self._build_text_layer(draft)
            layers.append(layer)
            diagnostics.extend(extra)
        count = len(layers)
        return Frame(
            revision=compiled.revision,
            page=compiled.page,
            width=compiled.width,
            height=compiled.height,
            origin=(0, 0),
            layers=layers,
            dependencies=[],
            diagnostics=diagnostics,
            stats=RetentionStats(candidate_layers=count, reused_layers=0, rebuilt_layers=count),
        )

    def _build_image_layer(self, draft: LayerDraft) -> tuple[Layer, list[RenderDiagnostic]]:
        descriptor: ImageNodeDescriptor = draft.descriptor  # type: ignore[assignment]
        image = self._decode(descriptor)
        local = RenderBounds(0.0, 0.0, float(image.width), float(image.height))
        frame = draft.frame.bounds
        placement = affine_then_translate(
            affine_scale_non_uniform(frame.width / local.width, frame.height / local.height),
            frame.x,
            frame.y,
        )
        bounds = _transform_bounds(local, placement)
        return (
            Layer(
                entity=draft.entity,
                geometry=draft.geometry,
                bounds=bounds,
                presentation=draft.presentation,
                ancestry=list(draft.ancestry),
                kind=draft.metadata,
                placement=placement,
                local_bounds=local,
                raster=image,
                scene=None,
                dependencies=[],
                element_angle=0.0,
            ),
            [],
        )

    def _build_text_layer(self, draft: LayerDraft) -> tuple[Layer, list[RenderDiagnostic]]:
        descriptor: TextNodeDescriptor = draft.descriptor  # type: ignore[assignment]
        meta: TextDraftMetadata = draft.metadata  # type: ignore[assignment]
        rendered = self._text.render_descriptor(descriptor, self.fonts)
        # The layout box is the frame's own size; its local origin maps onto the rotated frame.
        box = draft.frame.bounds
        placement = _placement_for_text(draft.frame, box.width, box.height)
        bounds = _transform_bounds(rendered.local_bounds, placement)
        metadata = TextMetadata(
            text=meta.text,
            language=meta.language,
            rendered_bounds=_transform_bounds(rendered.metadata.rendered_bounds, placement),
            layout_bounds=_transform_bounds(rendered.metadata.layout_bounds, placement),
            post_script_fonts=list(rendered.metadata.post_script_fonts),
            font_size=rendered.metadata.font_size,
            color=rendered.metadata.color,
            alignment=meta.alignment,
            writing_mode=meta.writing_mode,
            angle_degrees=meta.angle_degrees,
        )
        diagnostics = []
        for diagnostic in rendered.diagnostics:
            if diagnostic.kind == "text_overflow" and diagnostic.available is not None:
                diagnostic = RenderDiagnostic(
                    kind=diagnostic.kind,
                    entity=diagnostic.entity,
                    available=_transform_bounds(diagnostic.available, placement),
                    actual_width=diagnostic.actual_width,
                    actual_height=diagnostic.actual_height,
                    font_size=diagnostic.font_size,
                )
            diagnostics.append(diagnostic)
        return (
            Layer(
                entity=draft.entity,
                geometry=draft.geometry,
                bounds=bounds,
                presentation=draft.presentation,
                ancestry=list(draft.ancestry),
                kind=metadata,
                placement=placement,
                local_bounds=rendered.local_bounds,
                raster=None,
                scene=rendered.scene,
                dependencies=[],
                element_angle=draft.frame.angle_degrees,
            ),
            diagnostics,
        )


def _asset_size(asset: c.Asset) -> Optional[tuple[int, int]]:
    if asset.metadata.width is not None and asset.metadata.height is not None:
        return (asset.metadata.width, asset.metadata.height)
    return None


def _image_draft(
    entity: str,
    geometry: c.Geometry,
    descriptor: ImageNodeDescriptor,
    metadata: ImageMetadata,
    presentation: Presentation,
    ancestry: list[str],
) -> LayerDraft:
    bounds = geometry_bounds(geometry)
    if bounds is None:
        raise RendererError(f"invalid image geometry for entity {entity}")
    return LayerDraft(
        entity=entity,
        geometry=geometry,
        frame=GeometryFrame(bounds=bounds, angle_degrees=0.0),
        presentation=presentation,
        ancestry=list(ancestry),
        descriptor=descriptor,
        metadata=metadata,
    )


# ---- scene interpretation ------------------------------------------------------------------------


class _Traversal:
    def __init__(
        self,
        snapshot: Snapshot,
        page: str,
        width: int,
        height: int,
        font_families: list[str],
        balloon_flows: dict[str, ResolvedPlacement],
    ) -> None:
        self.snapshot = snapshot
        self.page = page
        self.width = width
        self.height = height
        self.font_families = font_families
        self.balloon_flows = balloon_flows
        self.layers: list[LayerDraft] = []
        self.diagnostics: list[RenderDiagnostic] = []

    def visit_children(self, parent: str, inherited: Presentation, ancestry: list[str]) -> None:
        for entity in self.snapshot.children(parent):
            visibility = self.snapshot.component(entity, c.Visibility)
            if visibility is None:
                visibility = c.Visibility(visible=True, opacity=1.0)
            presentation = Presentation(
                visible=inherited.visible and visibility.visible,
                opacity=inherited.opacity * visibility.opacity,
            )
            if self.snapshot.has(entity, c.Group):
                self.visit_children(entity, presentation, [*ancestry, entity])
                continue
            raster = self.snapshot.component(entity, c.RasterLayer)
            if raster is not None:
                asset = self.snapshot.asset(entity, SOURCE_ROLE)
                if asset is not None:
                    kind = ImageKind.CLEANUP if raster.kind == c.RasterLayerKind.CLEANUP else ImageKind.PAINT
                    self.layers.append(
                        _image_draft(
                            entity,
                            c.Geometry.rectangle(0.0, 0.0, float(self.width), float(self.height)),
                            ImageNodeDescriptor(
                                blob=asset.blob,
                                media_type=asset.media_type,
                                expected_size=_asset_size(asset),
                                require_size=None,
                                data=self.snapshot.asset_bytes(entity, SOURCE_ROLE),
                            ),
                            ImageMetadata(name=raster.name, kind=kind),
                            presentation,
                            ancestry,
                        )
                    )
                else:
                    self.diagnostics.append(RenderDiagnostic(kind="missing_asset", entity=entity, role=SOURCE_ROLE))
                continue
            layout = self.snapshot.component(entity, c.TextLayout)
            if layout is not None:
                draft = self.text_draft(entity, layout, presentation, ancestry)
                if draft is not None:
                    self.layers.append(draft)
                continue
            if self.snapshot.has(entity, c.Region):
                continue
            geometry = self.snapshot.component(entity, c.Geometry)
            asset = self.snapshot.asset(entity, SOURCE_ROLE)
            if geometry is not None and asset is not None:
                self.layers.append(
                    _image_draft(
                        entity,
                        geometry,
                        ImageNodeDescriptor(
                            blob=asset.blob,
                            media_type=asset.media_type,
                            expected_size=_asset_size(asset),
                            require_size=None,
                            data=self.snapshot.asset_bytes(entity, SOURCE_ROLE),
                        ),
                        ImageMetadata(name=None, kind=ImageKind.EMBEDDED),
                        presentation,
                        ancestry,
                    )
                )

    def text_draft(
        self,
        entity: str,
        layout: c.TextLayout,
        presentation: Presentation,
        ancestry: list[str],
    ) -> Optional[LayerDraft]:
        presents = self.snapshot.relation_from(entity, c.Relations.PRESENTS)
        if presents is None:
            return None
        content = presents.target
        translation = self.snapshot.component(content, c.Translation)
        if translation is None:
            return None
        text = translation.text.value
        if not text.strip():
            return None
        authored = self.snapshot.component(entity, c.Geometry)
        placement = self.balloon_flows.get(entity)
        if placement is None:
            placement = self.balloon_flow(entity)
        if placement is None:
            placement = self.fit(entity)

        if authored is not None:
            frame = geometry_frame(authored, layout.angle_degrees)
            if frame is None:
                return None
            geometry = authored
            balloon = contour(authored, frame) if placement is not None and placement.balloon_contour is not None else None
        else:
            if placement is None:
                return None
            geometry = placement.geometry
            frame = placement.frame
            balloon = placement.balloon_contour

        typography = self.snapshot.component(entity, c.Typography)
        analysis = None
        recognized = self.snapshot.relation_from(content, c.Relations.RECOGNIZED_FROM)
        if recognized is not None:
            analysis = self.snapshot.component(recognized.target, c.OcrAnalysis)
        writing_mode = _resolve_writing_mode(text, frame.bounds, typography, analysis)
        direction, _ = shaping_direction_for_text(text, writing_mode)
        alignment = _resolve_alignment(
            typography.alignment if typography is not None else None,
            writing_mode,
            direction == "rtl",
        )
        preferred_font = typography.preferred_font if typography is not None else None
        is_bubble = balloon is not None
        descriptor = TextNodeDescriptor(
            entity=entity,
            text=text,
            language=translation.language,
            width=frame.bounds.width,
            height=frame.bounds.height,
            balloon_contour=balloon,
            preferred_font=preferred_font,
            font_families=list(self.font_families),
            font_weight=typography.font_weight if typography is not None else None,
            font_style=_font_style(typography.font_style) if typography is not None else None,
            font_size=typography.size if typography is not None else None,
            minimum_font_size=MINIMUM_FONT_SIZE,
            auto_fit=typography.auto_fit if typography is not None else True,
            alignment=alignment,
            writing_mode=writing_mode,
            foreground_color=tuple(typography.color) if typography is not None and typography.color else (0, 0, 0, 255),
            stroke=_resolve_stroke(typography),
            line_height=1.2,
            letter_spacing=0.0,
            word_spacing=0.0,
            text_inset=(4.0, 4.0, 4.0, 4.0),
            point_text=not is_bubble and layout.kind == c.TextLayoutKind.POINT,
            styles=[
                TextStyleRun(
                    start=style.start,
                    end=style.end,
                    bold=style.bold,
                    italic=style.italic,
                    underline=style.underline,
                    strikethrough=style.strikethrough,
                    color=tuple(style.color) if style.color else None,
                    font=style.font,
                    scale=style.scale or 1.0,
                )
                for style in c.normalize_text_styles(translation.styles, len(text))
            ],
            fill_gradient=_gradient_spec(typography.fill_gradient if typography is not None else None),
            stroke_gradient=_gradient_spec(typography.stroke_gradient if typography is not None else None),
        )
        return LayerDraft(
            entity=entity,
            geometry=geometry,
            frame=frame,
            presentation=presentation,
            ancestry=list(ancestry),
            descriptor=descriptor,
            metadata=TextDraftMetadata(
                text=text,
                language=translation.language,
                alignment=alignment,
                writing_mode=writing_mode,
                angle_degrees=frame.angle_degrees,
            ),
        )

    def fit(self, entity: str) -> Optional[ResolvedPlacement]:
        relation = self.snapshot.relation_from(entity, c.Relations.FITS_TO)
        if relation is None:
            return None
        target = relation.target
        if not _belongs_to_page(self.snapshot, target, self.page):
            return None
        geometry = self.snapshot.component(target, c.Geometry)
        if geometry is None:
            raise RendererError(f"analysis region {target} has no geometry")
        frame = geometry_frame(geometry, None)
        if frame is None:
            return None
        return ResolvedPlacement(geometry=geometry, frame=frame, balloon_contour=None)

    def balloon_flow(self, entity: str) -> Optional[ResolvedPlacement]:
        relation = self.snapshot.relation_from(entity, c.Relations.FLOWS_IN)
        if relation is None:
            return None
        target = relation.target
        if not _belongs_to_page(self.snapshot, target, self.page):
            return None
        geometry = self.snapshot.component(target, c.Geometry)
        if geometry is None:
            raise RendererError(f"analysis region {target} has no geometry")
        frame = geometry_frame(geometry, None)
        if frame is None:
            return None
        return ResolvedPlacement(geometry=geometry, frame=frame, balloon_contour=contour(geometry, frame))


def _belongs_to_page(snapshot: Snapshot, entity: str, page: str) -> bool:
    current = entity
    while True:
        if current == page:
            return True
        parent = snapshot.parent(current)
        if parent is None:
            return False
        current = parent


def _resolve_balloon_flows(snapshot: Snapshot, page: str) -> dict[str, ResolvedPlacement]:
    """``resolve_balloon_flows``: text layers that flow into one balloon share its lobes."""
    @dataclass
    class FlowSeed:
        layer: str
        anchor: Optional[tuple[float, float]]

    groups: dict[str, list[FlowSeed]] = {}
    group = snapshot.text_group(page)
    layers = snapshot.text_layers(page) if group is not None else []
    for layer in layers:
        relation = snapshot.relation_from(layer, c.Relations.FLOWS_IN)
        if relation is None:
            continue
        balloon = relation.target
        if not _belongs_to_page(snapshot, balloon, page):
            continue
        presents = snapshot.relation_from(layer, c.Relations.PRESENTS)
        if presents is None:
            continue
        content = presents.target
        translation = snapshot.component(content, c.Translation)
        if translation is None or not translation.text.value.strip():
            continue
        anchor = _flow_anchor(snapshot, content)
        groups.setdefault(balloon, []).append(FlowSeed(layer=layer, anchor=anchor))

    placements: dict[str, ResolvedPlacement] = {}
    for balloon in sorted(groups):
        seeds = groups[balloon]
        geometry = snapshot.component(balloon, c.Geometry)
        if geometry is None:
            raise RendererError(f"analysis region {balloon} has no geometry")
        frame = geometry_frame(geometry, None)
        if frame is None:
            continue
        balloon_contour = contour(geometry, frame)
        centre = (
            frame.bounds.x + frame.bounds.width * 0.5,
            frame.bounds.y + frame.bounds.height * 0.5,
        )
        anchors = [seed.anchor if seed.anchor is not None else centre for seed in seeds]
        cells = flow_cells(frame, balloon_contour, anchors) if len(seeds) > 1 else None
        for index, seed in enumerate(seeds):
            seed_geometry = geometry
            if cells is not None and index < len(cells):
                radians = math.radians(frame.angle_degrees)
                sin, cos = math.sin(radians), math.cos(radians)
                half_width = frame.bounds.width * 0.5
                half_height = frame.bounds.height * 0.5
                points = []
                for x, y in cells[index]:
                    lx = x - half_width
                    ly = y - half_height
                    points.append(
                        c.Point(
                            x=frame.bounds.x + half_width + lx * cos - ly * sin,
                            y=frame.bounds.y + half_height + lx * sin + ly * cos,
                        )
                    )
                seed_geometry = c.Geometry(origin=geometry.origin, points=points)
            seed_frame = geometry_frame(seed_geometry, frame.angle_degrees)
            if seed_frame is None:
                continue
            placements[seed.layer] = ResolvedPlacement(
                geometry=seed_geometry,
                frame=seed_frame,
                balloon_contour=contour(seed_geometry, seed_frame),
            )
    return placements


def _flow_anchor(snapshot: Snapshot, content: str) -> Optional[tuple[float, float]]:
    recognized = snapshot.relation_from(content, c.Relations.RECOGNIZED_FROM)
    if recognized is None:
        return None
    geometry = snapshot.component(recognized.target, c.Geometry)
    if geometry is None:
        return None
    bounds = geometry_bounds(geometry)
    if bounds is None:
        return None
    return (bounds.x + bounds.width * 0.5, bounds.y + bounds.height * 0.5)


def _resolve_writing_mode(text: str, bounds: LayoutBox, typography, analysis) -> WritingMode:
    """``resolve_writing_mode``: authored mode, then CJK detection, then OCR direction and shape."""
    typography_mode = None
    if typography is not None and typography.writing_mode is not None:
        typography_mode = (
            WritingMode.HORIZONTAL if typography.writing_mode == c.WritingMode.HORIZONTAL else WritingMode.VERTICAL_RL
        )
    if typography is not None and isinstance(typography.origin, c.UserOrigin) and typography_mode is not None:
        return typography_mode
    if not is_chinese_or_japanese_text(text):
        return WritingMode.HORIZONTAL
    if typography_mode is not None:
        return typography_mode
    direction = analysis.direction if analysis is not None else None
    if direction == c.TextDirection.VERTICAL:
        return WritingMode.VERTICAL_RL
    if direction == c.TextDirection.HORIZONTAL:
        return WritingMode.HORIZONTAL
    return WritingMode.VERTICAL_RL if bounds.height > bounds.width else WritingMode.HORIZONTAL


def _resolve_alignment(alignment: Optional[c.TextAlignment], writing_mode: WritingMode, rtl: bool) -> TextAlign:
    if alignment is None:
        alignment = c.TextAlignment.START if writing_mode.is_vertical() else c.TextAlignment.CENTER
    vertical = writing_mode.is_vertical()
    if alignment == c.TextAlignment.START and vertical:
        return TextAlign.LEFT
    if alignment == c.TextAlignment.END and vertical:
        return TextAlign.RIGHT
    if alignment == c.TextAlignment.START and rtl:
        return TextAlign.RIGHT
    if alignment == c.TextAlignment.START:
        return TextAlign.LEFT
    if alignment == c.TextAlignment.CENTER:
        return TextAlign.CENTER
    if alignment == c.TextAlignment.END and rtl:
        return TextAlign.LEFT
    if alignment == c.TextAlignment.END:
        return TextAlign.RIGHT
    return TextAlign.JUSTIFY


def _gradient_spec(gradient: Optional[c.Gradient]) -> Optional[GradientSpec]:
    if gradient is None or len(gradient.stops) < 2:
        return None
    return GradientSpec(
        kind=gradient.kind.value,
        angle=gradient.angle,
        stops=tuple((stop.offset, tuple(stop.color)) for stop in gradient.stops),
    )


def _resolve_stroke(typography) -> Optional[StrokeOptions]:
    if typography is None:
        return None
    if typography.stroke_width is None or not typography.stroke_width > 0.0:
        return None
    color = tuple(typography.stroke_color) if typography.stroke_color is not None else (255, 255, 255, 255)
    return StrokeOptions(color=color, width_px=float(typography.stroke_width))


def _font_style(style: Optional[c.FontStyle]) -> Optional[FontStyle]:
    if style is None:
        return None
    return FontStyle(style.value)
