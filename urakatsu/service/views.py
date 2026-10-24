"""Read-only views of a scene page for the UI and APIs (flattened document views).

A text block is one text layer joined with the semantic content it presents and
the analysis regions it is bound to (``TextLayerRef`` / ``TextContentRef`` /
``AnalysisRegionRef``).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from urakatsu.scene import components as c
from urakatsu.scene.scene import Snapshot


class PointView(BaseModel):
    x: float
    y: float


class BoxView(BaseModel):
    x: float
    y: float
    width: float
    height: float


class GradientStopView(BaseModel):
    offset: float
    color: str  # #rrggbbaa


class GradientView(BaseModel):
    kind: str = "linear"
    angle: float = 90.0
    stops: list[GradientStopView] = Field(default_factory=list)


class TypographyView(BaseModel):
    preferred_font: Optional[str] = None
    font_weight: Optional[int] = None
    font_style: Optional[str] = None
    size: Optional[float] = None
    auto_fit: bool = True
    color: Optional[str] = None  # #rrggbbaa
    stroke_color: Optional[str] = None
    stroke_width: Optional[float] = None
    alignment: Optional[str] = None
    writing_mode: Optional[str] = None
    fill_gradient: Optional[GradientView] = None
    stroke_gradient: Optional[GradientView] = None
    authored: bool = False  # the user set this style; a pipeline-made one only hints its writing mode


class TextBlockView(BaseModel):
    layer_id: str
    content_id: str
    region_id: Optional[str] = None
    balloon_id: Optional[str] = None
    layout_kind: str
    frame: list[PointView] = Field(default_factory=list)
    bounds: Optional[BoxView] = None
    angle: float = 0.0
    source_text: Optional[str] = None
    source_origin: Optional[str] = None
    source_language: Optional[str] = None
    translation: Optional[str] = None
    translation_origin: Optional[str] = None
    translation_language: Optional[str] = None
    translation_styles: list[TextStyleView] = Field(default_factory=list)
    ocr_direction: Optional[str] = None
    typography: Optional[TypographyView] = None
    visible: bool = True
    opacity: float = 1.0
    role: Optional[str] = None


class RegionView(BaseModel):
    id: str
    kind: str
    label: Optional[str] = None
    polygon: list[PointView] = Field(default_factory=list)
    bounds: Optional[BoxView] = None
    confidence: Optional[float] = None


class RasterLayerView(BaseModel):
    id: str
    name: str
    kind: str
    visible: bool = True
    opacity: float = 1.0


class TextStyleView(BaseModel):
    start: int
    end: int
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikethrough: bool = False
    color: Optional[str] = None  # #rrggbbaa
    font: Optional[str] = None
    scale: Optional[float] = None


class StageStatusView(BaseModel):
    stage: str
    done: bool
    model: Optional[str] = None


class PageView(BaseModel):
    id: str
    name: str
    width: int
    height: int
    blocks: list[TextBlockView] = Field(default_factory=list)
    regions: list[RegionView] = Field(default_factory=list)
    raster_layers: list[RasterLayerView] = Field(default_factory=list)
    assets: list[str] = Field(default_factory=list)
    stages: list[StageStatusView] = Field(default_factory=list)
    revision: int = 0
    # Cache key per asset kind (source, thumbnail, cleanup, layer:<id>, render); see page_view.
    asset_versions: dict[str, str] = Field(default_factory=dict)


def origin_name(origin: Optional[c.Origin]) -> Optional[str]:
    if origin is None:
        return None
    return "user" if isinstance(origin, c.UserOrigin) else "generated"


def origin_model(origin: Optional[c.Origin]) -> Optional[str]:
    if isinstance(origin, c.GeneratedOrigin):
        return origin.generation.model
    return None


def rgba_hex(value: Optional[tuple[int, int, int, int]]) -> Optional[str]:
    if value is None:
        return None
    return "#" + "".join(f"{int(v):02x}" for v in value)


def parse_rgba(value: Optional[str]) -> Optional[tuple[int, int, int, int]]:
    if not value:
        return None
    text = value.strip().lstrip("#")
    if len(text) == 6:
        text += "ff"
    if len(text) != 8:
        raise ValueError(f"colour {value!r} must be #rrggbb or #rrggbbaa")
    return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4, 6))  # type: ignore[return-value]


def _points(geometry: Optional[c.Geometry]) -> list[PointView]:
    return [PointView(x=p.x, y=p.y) for p in geometry.points] if geometry else []


def _bounds(geometry: Optional[c.Geometry]) -> Optional[BoxView]:
    if geometry is None or not geometry.points:
        return None
    x0, y0, x1, y1 = geometry.bounds()
    return BoxView(x=x0, y=y0, width=x1 - x0, height=y1 - y0)


def gradient_view(value: Optional[c.Gradient]) -> Optional[GradientView]:
    if value is None:
        return None
    return GradientView(kind=value.kind.value, angle=value.angle, stops=[GradientStopView(offset=s.offset, color=rgba_hex(s.color)) for s in value.stops])


def typography_view(value: Optional[c.Typography]) -> Optional[TypographyView]:
    if value is None:
        return None
    return TypographyView(
        preferred_font=value.preferred_font,
        font_weight=value.font_weight,
        font_style=value.font_style.value if value.font_style else None,
        size=value.size,
        auto_fit=value.auto_fit,
        color=rgba_hex(value.color),
        stroke_color=rgba_hex(value.stroke_color),
        stroke_width=value.stroke_width,
        fill_gradient=gradient_view(value.fill_gradient),
        stroke_gradient=gradient_view(value.stroke_gradient),
        alignment=value.alignment.value if value.alignment else None,
        writing_mode=value.writing_mode.value if value.writing_mode else None,
        authored=isinstance(value.origin, c.UserOrigin),
    )


def page_view(snapshot: Snapshot, page: str, extra_assets: list[str] | None = None) -> PageView:
    meta = snapshot.page(page)
    blocks: list[TextBlockView] = []
    for layer in snapshot.text_layers(page):
        layout = snapshot.component(layer, c.TextLayout)
        if layout is None:
            continue
        try:
            content = snapshot.layer_content(layer)
        except Exception:
            continue
        region = snapshot.source_region(content)
        source = snapshot.component(content, c.SourceText)
        translation = snapshot.component(content, c.Translation)
        ocr = snapshot.component(region, c.OcrAnalysis) if region else None
        visibility = snapshot.component(layer, c.Visibility)
        role = snapshot.component(content, c.TextRole)
        frame = snapshot.frame(layer)
        blocks.append(
            TextBlockView(
                layer_id=layer,
                content_id=content,
                region_id=region,
                balloon_id=snapshot.balloon_target(layer),
                layout_kind=layout.kind.value,
                frame=_points(frame),
                bounds=_bounds(frame),
                angle=layout.angle_degrees or 0.0,
                source_text=source.text.value if source else None,
                source_origin=origin_name(source.text.origin) if source else None,
                source_language=source.language if source else None,
                translation=translation.text.value if translation else None,
                translation_origin=origin_name(translation.text.origin) if translation else None,
                translation_language=translation.language if translation else None,
                translation_styles=[
                    TextStyleView(
                        start=style.start,
                        end=style.end,
                        bold=style.bold,
                        italic=style.italic,
                        underline=style.underline,
                        strikethrough=style.strikethrough,
                        color=rgba_hex(style.color),
                        font=style.font,
                        scale=style.scale,
                    )
                    for style in c.normalize_text_styles(translation.styles, len(translation.text.value))
                ]
                if translation
                else [],
                ocr_direction=ocr.direction.value if ocr else None,
                typography=typography_view(snapshot.component(layer, c.Typography)),
                visible=visibility.visible if visibility else True,
                opacity=visibility.opacity if visibility else 1.0,
                role=role.role if role else None,
            )
        )

    regions: list[RegionView] = []
    for entity in snapshot.descendants(page):
        region = snapshot.component(entity, c.Region)
        if region is None:
            continue
        geometry = snapshot.component(entity, c.Geometry)
        detection = snapshot.component(entity, c.DetectionAnalysis)
        confidence = max((label.confidence for label in detection.labels), default=None) if detection else None
        regions.append(
            RegionView(id=entity, kind=region.kind, label=region.label, polygon=_points(geometry), bounds=_bounds(geometry), confidence=confidence)
        )

    raster_layers: list[RasterLayerView] = []
    for entity in snapshot.raster_layers(page):
        layer = snapshot.component(entity, c.RasterLayer)
        visibility = snapshot.component(entity, c.Visibility)
        raster_layers.append(
            RasterLayerView(
                id=entity,
                name=layer.name,
                kind=layer.kind.value,
                visible=visibility.visible if visibility else True,
                opacity=visibility.opacity if visibility else 1.0,
            )
        )

    assets = list(snapshot.state.entity(page).assets.keys())
    if any(r.kind == "cleanup" for r in raster_layers):
        assets.append("cleanup")
    assets += ["thumbnail", "render"] + list(extra_assets or [])

    # A blob's content address changes exactly when its bytes do, so an edit to one layer leaves every
    # other asset's URL (and the browser's copy of it) valid. Thumbnails are made from the source alone;
    # the render is drawn from the whole scene, so it follows the revision.
    versions = {name: asset.blob for name, asset in snapshot.state.entity(page).assets.items()}
    if "source" in versions:
        versions["thumbnail"] = versions["source"]
    for raster in raster_layers:
        asset = snapshot.asset(raster.id, "source")
        if asset is not None:
            versions[f"layer:{raster.id}"] = asset.blob
            if raster.kind == "cleanup":
                versions.setdefault("cleanup", asset.blob)  # the first cleanup layer, as Workspace.asset serves it
    versions["render"] = str(snapshot.revision)

    return PageView(
        id=page,
        name=meta.label,
        width=int(meta.width),
        height=int(meta.height),
        blocks=blocks,
        regions=regions,
        raster_layers=raster_layers,
        assets=sorted(set(assets), key=assets.index),
        stages=stage_status(snapshot, page, blocks, regions, raster_layers),
        revision=snapshot.revision,
        asset_versions=versions,
    )


def stage_status(snapshot: Snapshot, page: str, blocks, regions, raster_layers) -> list[StageStatusView]:
    def generated_model(entities, cls) -> Optional[str]:
        for entity in entities:
            value = snapshot.component(entity, cls)
            if value is not None:
                model = origin_model(value.get_origin())
                if model:
                    return model
        return None

    text_regions = [r.id for r in regions if r.kind == c.RegionKinds.TEXT]
    contents = [b.content_id for b in blocks]
    cleanup = [r.id for r in raster_layers if r.kind == "cleanup"]
    return [
        StageStatusView(stage="detection", done=bool(text_regions), model=generated_model(text_regions, c.Region)),
        StageStatusView(stage="ocr", done=any(b.source_text for b in blocks), model=generated_model(contents, c.SourceText)),
        StageStatusView(stage="translation", done=any(b.translation for b in blocks), model=generated_model(contents, c.Translation)),
        StageStatusView(stage="inpainting", done=bool(cleanup), model=generated_model(cleanup, c.RasterLayer)),
    ]
