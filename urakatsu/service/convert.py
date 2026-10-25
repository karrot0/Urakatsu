"""View/config <-> protobuf conversions: the only place the wire format is spelled out."""

from __future__ import annotations

import json
from typing import Any, Optional

from urakatsu import __version__
from urakatsu.core.config import SOUND_EFFECT_MODES, GenerationConfig, ModelSelection, PipelineConfig, SoundEffectsConfig, TypesettingConfig
from urakatsu.core.progress import Progress
from urakatsu.core.stage import Operation, Stage
from urakatsu.runtimes.device import Device
from urakatsu.runtimes.manager import RuntimeReport

from .gen import urakatsu_pb2 as pb
from .views import BoxView, PageView, PointView, TextBlockView, TypographyView


def _opt(message, field: str, value) -> None:
    if value is not None:
        setattr(message, field, value)


# ---- documents ----------------------------------------------------------------------------------


def points_to_proto(points: list[PointView]) -> list[pb.Point]:
    return [pb.Point(x=p.x, y=p.y) for p in points]


def box_to_proto(box: Optional[BoxView]) -> Optional[pb.Box]:
    return pb.Box(x=box.x, y=box.y, width=box.width, height=box.height) if box else None


def text_style_to_proto(style) -> pb.TextStyle:
    message = pb.TextStyle(
        start=style.start,
        end=style.end,
        bold=style.bold,
        italic=style.italic,
        underline=style.underline,
        strikethrough=style.strikethrough,
    )
    if style.color:
        message.color = style.color
    if style.font:
        message.font = style.font
    if style.scale is not None:
        message.scale = style.scale
    if style.gradient is not None:
        message.gradient.CopyFrom(_gradient_to_proto(style.gradient))
    return message


def text_style_from_proto(message: pb.TextStyle) -> dict[str, Any]:
    return {
        "start": message.start,
        "end": message.end,
        "bold": message.bold,
        "italic": message.italic,
        "underline": message.underline,
        "strikethrough": message.strikethrough,
        "color": message.color if message.HasField("color") else None,
        "font": message.font or None,
        "scale": message.scale if message.HasField("scale") else None,
        "gradient": _gradient_dict(message.gradient) if message.HasField("gradient") else None,
    }


def typography_to_proto(view: Optional[TypographyView]) -> Optional[pb.Typography]:
    if view is None:
        return None
    message = pb.Typography(
        preferred_font=view.preferred_font or "",
        font_style=view.font_style or "",
        auto_fit=view.auto_fit,
        alignment=view.alignment or "",
        writing_mode=view.writing_mode or "",
        authored=view.authored,
    )
    _opt(message, "font_weight", view.font_weight)
    _opt(message, "size", view.size)
    _opt(message, "color", view.color)
    _opt(message, "stroke_color", view.stroke_color)
    _opt(message, "stroke_width", view.stroke_width)
    for key in ("fill_gradient", "stroke_gradient"):
        gradient = getattr(view, key)
        if gradient is not None:
            getattr(message, key).CopyFrom(_gradient_to_proto(gradient))
    return message


def _gradient_to_proto(gradient) -> pb.Gradient:
    return pb.Gradient(kind=gradient.kind, angle=gradient.angle, stops=[pb.GradientStop(offset=s.offset, color=s.color) for s in gradient.stops])


def _gradient_dict(gradient: pb.Gradient) -> dict[str, Any]:
    return {"kind": gradient.kind or "linear", "angle": gradient.angle, "stops": [{"offset": s.offset, "color": s.color} for s in gradient.stops]}


def _gradient_from_proto(message: pb.Typography, key: str) -> Optional[dict[str, Any]]:
    if not message.HasField(key):
        return None
    return _gradient_dict(getattr(message, key))


def typography_from_proto(message: pb.Typography) -> dict[str, Any]:
    return {
        "preferred_font": message.preferred_font or None,
        "font_weight": message.font_weight if message.HasField("font_weight") else None,
        "font_style": message.font_style or None,
        "size": message.size if message.HasField("size") else None,
        "auto_fit": message.auto_fit,
        "color": message.color if message.HasField("color") else None,
        "stroke_color": message.stroke_color if message.HasField("stroke_color") else None,
        "stroke_width": message.stroke_width if message.HasField("stroke_width") else None,
        "alignment": message.alignment or None,
        "writing_mode": message.writing_mode or None,
        "fill_gradient": _gradient_from_proto(message, "fill_gradient"),
        "stroke_gradient": _gradient_from_proto(message, "stroke_gradient"),
    }


def block_to_proto(block: TextBlockView) -> pb.TextBlock:
    message = pb.TextBlock(
        layer_id=block.layer_id,
        content_id=block.content_id,
        region_id=block.region_id or "",
        balloon_id=block.balloon_id or "",
        layout_kind=block.layout_kind,
        frame=points_to_proto(block.frame),
        angle=block.angle,
        source_text=block.source_text or "",
        source_origin=block.source_origin or "",
        source_language=block.source_language or "",
        translation=block.translation or "",
        translation_origin=block.translation_origin or "",
        translation_language=block.translation_language or "",
        translation_styles=[text_style_to_proto(style) for style in block.translation_styles],
        ocr_direction=block.ocr_direction or "",
        visible=block.visible,
        opacity=block.opacity,
        role=block.role or "",
    )
    if block.bounds:
        message.bounds.CopyFrom(box_to_proto(block.bounds))
    typography = typography_to_proto(block.typography)
    if typography is not None:
        message.typography.CopyFrom(typography)
    return message


def page_to_proto(view: PageView) -> pb.Page:
    message = pb.Page(
        id=view.id,
        name=view.name,
        width=view.width,
        height=view.height,
        blocks=[block_to_proto(b) for b in view.blocks],
        raster_layers=[
            pb.RasterLayer(id=r.id, name=r.name, kind=r.kind, visible=r.visible, opacity=r.opacity) for r in view.raster_layers
        ],
        assets=view.assets,
        stages=[pb.StageStatus(stage=s.stage, done=s.done, model=s.model or "") for s in view.stages],
        revision=view.revision,
        asset_versions=view.asset_versions,
    )
    for region in view.regions:
        item = pb.Region(id=region.id, kind=region.kind, label=region.label or "", polygon=points_to_proto(region.polygon), confidence=region.confidence or 0.0)
        if region.bounds:
            item.bounds.CopyFrom(box_to_proto(region.bounds))
        message.regions.append(item)
    return message


def project_to_proto(view) -> pb.Project:
    return pb.Project(
        id=view.id,
        name=view.name,
        created_at=view.created_at,
        updated_at=view.updated_at,
        revision=view.revision,
        pages=[page_to_proto(p) for p in view.pages],
        can_undo=view.can_undo,
        can_redo=view.can_redo,
        chapters=[pb.Chapter(id=ch.id, name=ch.name, page_ids=ch.page_ids) for ch in view.chapters],
    )


def summary_to_proto(summary) -> pb.ProjectSummary:
    return pb.ProjectSummary(
        id=summary.id,
        name=summary.name,
        pages=summary.pages,
        created_at=summary.created_at,
        updated_at=summary.updated_at,
        chapters=summary.chapters,
    )


# ---- configuration --------------------------------------------------------------------------------


def config_to_proto(config: PipelineConfig) -> pb.Config:
    selection = config.translation.model
    message = pb.Config(
        device=config.device,
        idle_unload_minutes=config.idle_unload_minutes,
        detection=config.detection.model,
        ocr=config.ocr.model,
        inpainting=config.inpainting.model,
        fill=config.fill.model,
        translation=pb.ModelSelection(
            provider=selection.provider,
            model=selection.model or "",
            quantization=selection.quantization or "",
            vision=selection.vision,
            reasoning=selection.reasoning,
        ),
        generation_json=config.translation.generation.model_dump_json(),
        target_language=config.translation.target_language,
        instructions=config.translation.instructions or "",
        font_families=list(config.typesetting.font_families),
        lossless_pages=config.storage.lossless_pages,
        sound_effects=config.sound_effects.mode,
        sound_effects_model=config.sound_effects.model,
    )
    for key, value in config.processor.items():
        message.processor_json[key] = json.dumps(value)
    for key, value in config.providers.items():
        message.providers_json[key] = json.dumps(value)
    for key, value in config.runtimes.items():
        message.runtimes[key] = value
    return message


def _json_map(entries, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, raw in entries.items():
        try:
            value = json.loads(raw) if raw else {}
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} settings for {key} are not valid JSON: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"{label} settings for {key} must be a JSON object")
        result[key] = value
    return result


def config_from_proto(message: pb.Config, base: PipelineConfig) -> PipelineConfig:
    config = base.model_copy(deep=True)
    config.device = message.device or config.device
    if message.HasField("idle_unload_minutes"):
        if message.idle_unload_minutes < 0:
            raise ValueError("idle_unload_minutes cannot be negative")
        config.idle_unload_minutes = message.idle_unload_minutes
    if message.detection:
        config.detection.model = message.detection
    if message.ocr:
        config.ocr.model = message.ocr
    if message.inpainting:
        config.inpainting.model = message.inpainting
    config.fill.model = message.fill  # empty is meaningful: the inpainting model
    if message.HasField("translation") and message.translation.provider:
        t = message.translation
        config.translation.model = ModelSelection(
            provider=t.provider,
            model=t.model or None,
            quantization=t.quantization or None,
            vision=t.vision,
            reasoning=t.reasoning,
        )
    if message.generation_json:
        config.translation.generation = GenerationConfig.model_validate_json(message.generation_json)
    if message.target_language:
        config.translation.target_language = message.target_language
    config.translation.instructions = message.instructions or None
    if message.font_families:
        config.typesetting = TypesettingConfig(font_families=list(message.font_families))
    config.storage = config.storage.model_copy(update={"lossless_pages": message.lossless_pages})
    if message.sound_effects:
        if message.sound_effects not in SOUND_EFFECT_MODES:
            raise ValueError(f"unknown sound effects mode {message.sound_effects!r}; expected one of {list(SOUND_EFFECT_MODES)}")
        config.sound_effects = SoundEffectsConfig(mode=message.sound_effects, model=message.sound_effects_model)
    config.processor = _json_map(message.processor_json, "processor")
    config.providers = _json_map(message.providers_json, "provider")
    config.runtimes = dict(message.runtimes)
    return config


# ---- capabilities ----------------------------------------------------------------------------------


def device_to_proto(device: Device) -> pb.Device:
    return pb.Device(
        index=device.index,
        name=device.name,
        description=device.description,
        backend=device.backend.value,
        device_type=device.device_type.value,
        memory_total=device.memory_total,
        memory_free=device.memory_free,
    )


def provider_to_proto(info) -> pb.ProviderInfo:
    return pb.ProviderInfo(
        id=info.id,
        name=info.name,
        stage=info.stage.value,
        description=info.description,
        runtimes=[pb.RuntimeOption(name=r.name, available=r.available.ok, reason=r.available.reason) for r in info.runtimes],
        default_runtime=info.default_runtime or "",
        available=info.available.ok,
        unavailable_reason=info.available.reason,
        weights=list(info.weights),
        settings_schema_json=json.dumps(info.settings_schema),
        defaults_json=json.dumps(info.defaults),
        prompted=info.prompted,
        instructed=info.instructed,
    )


def capabilities_to_proto(report: RuntimeReport, providers: list, loaded: dict[Stage, str], languages) -> pb.Capabilities:
    message = pb.Capabilities(
        version=__version__,
        device=device_to_proto(report.device),
        devices=[device_to_proto(d) for d in report.hardware.candidates()],
        platform=report.hardware.platform,
        runtimes_json=json.dumps(report.runtimes),
        providers=[provider_to_proto(p) for p in providers],
        stages=[stage.value for stage in Stage.all()],
        store_root=report.store_root,
        store_bytes=report.store_bytes,
        languages=[pb.Language(tag=tag, name=name) for tag, name in languages],
    )
    for stage, provider_id in loaded.items():
        message.loaded[stage.value] = provider_id
    return message


def translation_models_to_proto(models: list[dict[str, Any]]) -> pb.TranslationModelList:
    items = []
    for model in models:
        items.append(
            pb.TranslationModel(
                provider=str(model.get("provider", "")),
                provider_name=str(model.get("provider_name") or model.get("provider", "")),
                model=str(model.get("model") or ""),
                name=str(model.get("name", "")),
                quantizations=[
                    pb.Quantization(id=str(q.get("id", "")), name=str(q.get("name", "")), downloaded=bool(q.get("downloaded")))
                    for q in model.get("quantizations", [])
                ],
                vision=bool(model.get("vision")),
                reasoning=bool(model.get("reasoning")),
            )
        )
    return pb.TranslationModelList(models=items)


# ---- processing -------------------------------------------------------------------------------------


def operation_from_proto(message: Optional[pb.Operation]) -> Operation:
    if message is None or not message.kind or message.kind == "full":
        return Operation.full()
    if message.kind == "through":
        return Operation.through(Stage.parse(message.stage))
    if message.kind == "only":
        return Operation.only(Stage.parse(message.stage))
    if message.kind == "stages":
        return Operation.select([Stage.parse(s) for s in message.stages])
    raise ValueError(f"unknown operation kind {message.kind!r}")


def progress_to_proto(job_id: str, progress: Progress, page: Optional[PageView] = None) -> pb.ProgressEvent:
    message = pb.ProgressEvent(
        job_id=job_id,
        kind=progress.kind.value,
        page_id=progress.page or "",
        stage=progress.stage.value if progress.stage else "",
        model=progress.model or "",
        elapsed_ms=progress.elapsed_ms or 0.0,
        message=progress.message,
        completed=progress.completed,
        total=progress.total,
        pages=list(progress.pages),
        stages=[s.value for s in progress.stages],
        step=progress.step,
        steps=progress.steps,
        preview=progress.preview or b"",
    )
    if progress.preview_box is not None:
        x, y, width, height = progress.preview_box
        message.preview_box.CopyFrom(pb.Box(x=x, y=y, width=width, height=height))
    if page is not None:
        message.page.CopyFrom(page_to_proto(page))
    return message
