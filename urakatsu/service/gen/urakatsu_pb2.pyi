from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Device(_message.Message):
    __slots__ = ("index", "name", "description", "backend", "device_type", "memory_total", "memory_free")
    INDEX_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    DEVICE_TYPE_FIELD_NUMBER: _ClassVar[int]
    MEMORY_TOTAL_FIELD_NUMBER: _ClassVar[int]
    MEMORY_FREE_FIELD_NUMBER: _ClassVar[int]
    index: int
    name: str
    description: str
    backend: str
    device_type: str
    memory_total: int
    memory_free: int
    def __init__(self, index: _Optional[int] = ..., name: _Optional[str] = ..., description: _Optional[str] = ..., backend: _Optional[str] = ..., device_type: _Optional[str] = ..., memory_total: _Optional[int] = ..., memory_free: _Optional[int] = ...) -> None: ...

class RuntimeOption(_message.Message):
    __slots__ = ("name", "available", "reason")
    NAME_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    name: str
    available: bool
    reason: str
    def __init__(self, name: _Optional[str] = ..., available: _Optional[bool] = ..., reason: _Optional[str] = ...) -> None: ...

class ProviderInfo(_message.Message):
    __slots__ = ("id", "name", "stage", "description", "runtimes", "default_runtime", "available", "unavailable_reason", "weights", "settings_schema_json", "defaults_json", "prompted", "instructed")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    STAGE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    RUNTIMES_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_RUNTIME_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    UNAVAILABLE_REASON_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    SETTINGS_SCHEMA_JSON_FIELD_NUMBER: _ClassVar[int]
    DEFAULTS_JSON_FIELD_NUMBER: _ClassVar[int]
    PROMPTED_FIELD_NUMBER: _ClassVar[int]
    INSTRUCTED_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    stage: str
    description: str
    runtimes: _containers.RepeatedCompositeFieldContainer[RuntimeOption]
    default_runtime: str
    available: bool
    unavailable_reason: str
    weights: _containers.RepeatedScalarFieldContainer[str]
    settings_schema_json: str
    defaults_json: str
    prompted: bool
    instructed: bool
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., stage: _Optional[str] = ..., description: _Optional[str] = ..., runtimes: _Optional[_Iterable[_Union[RuntimeOption, _Mapping]]] = ..., default_runtime: _Optional[str] = ..., available: _Optional[bool] = ..., unavailable_reason: _Optional[str] = ..., weights: _Optional[_Iterable[str]] = ..., settings_schema_json: _Optional[str] = ..., defaults_json: _Optional[str] = ..., prompted: _Optional[bool] = ..., instructed: _Optional[bool] = ...) -> None: ...

class Capabilities(_message.Message):
    __slots__ = ("version", "device", "devices", "platform", "runtimes_json", "providers", "stages", "store_root", "store_bytes", "loaded", "languages")
    class LoadedEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    VERSION_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    DEVICES_FIELD_NUMBER: _ClassVar[int]
    PLATFORM_FIELD_NUMBER: _ClassVar[int]
    RUNTIMES_JSON_FIELD_NUMBER: _ClassVar[int]
    PROVIDERS_FIELD_NUMBER: _ClassVar[int]
    STAGES_FIELD_NUMBER: _ClassVar[int]
    STORE_ROOT_FIELD_NUMBER: _ClassVar[int]
    STORE_BYTES_FIELD_NUMBER: _ClassVar[int]
    LOADED_FIELD_NUMBER: _ClassVar[int]
    LANGUAGES_FIELD_NUMBER: _ClassVar[int]
    version: str
    device: Device
    devices: _containers.RepeatedCompositeFieldContainer[Device]
    platform: str
    runtimes_json: str
    providers: _containers.RepeatedCompositeFieldContainer[ProviderInfo]
    stages: _containers.RepeatedScalarFieldContainer[str]
    store_root: str
    store_bytes: int
    loaded: _containers.ScalarMap[str, str]
    languages: _containers.RepeatedCompositeFieldContainer[Language]
    def __init__(self, version: _Optional[str] = ..., device: _Optional[_Union[Device, _Mapping]] = ..., devices: _Optional[_Iterable[_Union[Device, _Mapping]]] = ..., platform: _Optional[str] = ..., runtimes_json: _Optional[str] = ..., providers: _Optional[_Iterable[_Union[ProviderInfo, _Mapping]]] = ..., stages: _Optional[_Iterable[str]] = ..., store_root: _Optional[str] = ..., store_bytes: _Optional[int] = ..., loaded: _Optional[_Mapping[str, str]] = ..., languages: _Optional[_Iterable[_Union[Language, _Mapping]]] = ...) -> None: ...

class Language(_message.Message):
    __slots__ = ("tag", "name")
    TAG_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    tag: str
    name: str
    def __init__(self, tag: _Optional[str] = ..., name: _Optional[str] = ...) -> None: ...

class ModelSelection(_message.Message):
    __slots__ = ("provider", "model", "quantization", "vision", "reasoning")
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    QUANTIZATION_FIELD_NUMBER: _ClassVar[int]
    VISION_FIELD_NUMBER: _ClassVar[int]
    REASONING_FIELD_NUMBER: _ClassVar[int]
    provider: str
    model: str
    quantization: str
    vision: bool
    reasoning: bool
    def __init__(self, provider: _Optional[str] = ..., model: _Optional[str] = ..., quantization: _Optional[str] = ..., vision: _Optional[bool] = ..., reasoning: _Optional[bool] = ...) -> None: ...

class Config(_message.Message):
    __slots__ = ("device", "detection", "ocr", "inpainting", "translation", "generation_json", "target_language", "instructions", "processor_json", "providers_json", "runtimes", "font_families", "fill", "lossless_pages")
    class ProcessorJsonEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class ProvidersJsonEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class RuntimesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    DETECTION_FIELD_NUMBER: _ClassVar[int]
    OCR_FIELD_NUMBER: _ClassVar[int]
    INPAINTING_FIELD_NUMBER: _ClassVar[int]
    TRANSLATION_FIELD_NUMBER: _ClassVar[int]
    GENERATION_JSON_FIELD_NUMBER: _ClassVar[int]
    TARGET_LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    INSTRUCTIONS_FIELD_NUMBER: _ClassVar[int]
    PROCESSOR_JSON_FIELD_NUMBER: _ClassVar[int]
    PROVIDERS_JSON_FIELD_NUMBER: _ClassVar[int]
    RUNTIMES_FIELD_NUMBER: _ClassVar[int]
    FONT_FAMILIES_FIELD_NUMBER: _ClassVar[int]
    FILL_FIELD_NUMBER: _ClassVar[int]
    LOSSLESS_PAGES_FIELD_NUMBER: _ClassVar[int]
    device: str
    detection: str
    ocr: str
    inpainting: str
    translation: ModelSelection
    generation_json: str
    target_language: str
    instructions: str
    processor_json: _containers.ScalarMap[str, str]
    providers_json: _containers.ScalarMap[str, str]
    runtimes: _containers.ScalarMap[str, str]
    font_families: _containers.RepeatedScalarFieldContainer[str]
    fill: str
    lossless_pages: bool
    def __init__(self, device: _Optional[str] = ..., detection: _Optional[str] = ..., ocr: _Optional[str] = ..., inpainting: _Optional[str] = ..., translation: _Optional[_Union[ModelSelection, _Mapping]] = ..., generation_json: _Optional[str] = ..., target_language: _Optional[str] = ..., instructions: _Optional[str] = ..., processor_json: _Optional[_Mapping[str, str]] = ..., providers_json: _Optional[_Mapping[str, str]] = ..., runtimes: _Optional[_Mapping[str, str]] = ..., font_families: _Optional[_Iterable[str]] = ..., fill: _Optional[str] = ..., lossless_pages: _Optional[bool] = ...) -> None: ...

class TranslationModel(_message.Message):
    __slots__ = ("provider", "provider_name", "model", "name", "quantizations", "vision", "reasoning")
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    QUANTIZATIONS_FIELD_NUMBER: _ClassVar[int]
    VISION_FIELD_NUMBER: _ClassVar[int]
    REASONING_FIELD_NUMBER: _ClassVar[int]
    provider: str
    provider_name: str
    model: str
    name: str
    quantizations: _containers.RepeatedCompositeFieldContainer[Quantization]
    vision: bool
    reasoning: bool
    def __init__(self, provider: _Optional[str] = ..., provider_name: _Optional[str] = ..., model: _Optional[str] = ..., name: _Optional[str] = ..., quantizations: _Optional[_Iterable[_Union[Quantization, _Mapping]]] = ..., vision: _Optional[bool] = ..., reasoning: _Optional[bool] = ...) -> None: ...

class Quantization(_message.Message):
    __slots__ = ("id", "name", "downloaded")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DOWNLOADED_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    downloaded: bool
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., downloaded: _Optional[bool] = ...) -> None: ...

class TranslationModelList(_message.Message):
    __slots__ = ("models",)
    MODELS_FIELD_NUMBER: _ClassVar[int]
    models: _containers.RepeatedCompositeFieldContainer[TranslationModel]
    def __init__(self, models: _Optional[_Iterable[_Union[TranslationModel, _Mapping]]] = ...) -> None: ...

class Point(_message.Message):
    __slots__ = ("x", "y")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ...) -> None: ...

class Box(_message.Message):
    __slots__ = ("x", "y", "width", "height")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    width: float
    height: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., width: _Optional[float] = ..., height: _Optional[float] = ...) -> None: ...

class GradientStop(_message.Message):
    __slots__ = ("offset", "color")
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    offset: float
    color: str
    def __init__(self, offset: _Optional[float] = ..., color: _Optional[str] = ...) -> None: ...

class Gradient(_message.Message):
    __slots__ = ("kind", "angle", "stops")
    KIND_FIELD_NUMBER: _ClassVar[int]
    ANGLE_FIELD_NUMBER: _ClassVar[int]
    STOPS_FIELD_NUMBER: _ClassVar[int]
    kind: str
    angle: float
    stops: _containers.RepeatedCompositeFieldContainer[GradientStop]
    def __init__(self, kind: _Optional[str] = ..., angle: _Optional[float] = ..., stops: _Optional[_Iterable[_Union[GradientStop, _Mapping]]] = ...) -> None: ...

class Typography(_message.Message):
    __slots__ = ("preferred_font", "font_weight", "font_style", "size", "auto_fit", "color", "stroke_color", "stroke_width", "alignment", "writing_mode", "authored", "fill_gradient", "stroke_gradient")
    PREFERRED_FONT_FIELD_NUMBER: _ClassVar[int]
    FONT_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    FONT_STYLE_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    AUTO_FIT_FIELD_NUMBER: _ClassVar[int]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    STROKE_COLOR_FIELD_NUMBER: _ClassVar[int]
    STROKE_WIDTH_FIELD_NUMBER: _ClassVar[int]
    ALIGNMENT_FIELD_NUMBER: _ClassVar[int]
    WRITING_MODE_FIELD_NUMBER: _ClassVar[int]
    AUTHORED_FIELD_NUMBER: _ClassVar[int]
    FILL_GRADIENT_FIELD_NUMBER: _ClassVar[int]
    STROKE_GRADIENT_FIELD_NUMBER: _ClassVar[int]
    preferred_font: str
    font_weight: int
    font_style: str
    size: float
    auto_fit: bool
    color: str
    stroke_color: str
    stroke_width: float
    alignment: str
    writing_mode: str
    authored: bool
    fill_gradient: Gradient
    stroke_gradient: Gradient
    def __init__(self, preferred_font: _Optional[str] = ..., font_weight: _Optional[int] = ..., font_style: _Optional[str] = ..., size: _Optional[float] = ..., auto_fit: _Optional[bool] = ..., color: _Optional[str] = ..., stroke_color: _Optional[str] = ..., stroke_width: _Optional[float] = ..., alignment: _Optional[str] = ..., writing_mode: _Optional[str] = ..., authored: _Optional[bool] = ..., fill_gradient: _Optional[_Union[Gradient, _Mapping]] = ..., stroke_gradient: _Optional[_Union[Gradient, _Mapping]] = ...) -> None: ...

class TextStyle(_message.Message):
    __slots__ = ("start", "end", "bold", "italic", "underline", "strikethrough", "color", "font", "scale")
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    BOLD_FIELD_NUMBER: _ClassVar[int]
    ITALIC_FIELD_NUMBER: _ClassVar[int]
    UNDERLINE_FIELD_NUMBER: _ClassVar[int]
    STRIKETHROUGH_FIELD_NUMBER: _ClassVar[int]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    FONT_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    start: int
    end: int
    bold: bool
    italic: bool
    underline: bool
    strikethrough: bool
    color: str
    font: str
    scale: float
    def __init__(self, start: _Optional[int] = ..., end: _Optional[int] = ..., bold: _Optional[bool] = ..., italic: _Optional[bool] = ..., underline: _Optional[bool] = ..., strikethrough: _Optional[bool] = ..., color: _Optional[str] = ..., font: _Optional[str] = ..., scale: _Optional[float] = ...) -> None: ...

class TextBlock(_message.Message):
    __slots__ = ("layer_id", "content_id", "region_id", "balloon_id", "layout_kind", "frame", "bounds", "angle", "source_text", "source_origin", "source_language", "translation", "translation_origin", "translation_language", "ocr_direction", "typography", "visible", "role", "opacity", "translation_styles")
    LAYER_ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_ID_FIELD_NUMBER: _ClassVar[int]
    REGION_ID_FIELD_NUMBER: _ClassVar[int]
    BALLOON_ID_FIELD_NUMBER: _ClassVar[int]
    LAYOUT_KIND_FIELD_NUMBER: _ClassVar[int]
    FRAME_FIELD_NUMBER: _ClassVar[int]
    BOUNDS_FIELD_NUMBER: _ClassVar[int]
    ANGLE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_TEXT_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ORIGIN_FIELD_NUMBER: _ClassVar[int]
    SOURCE_LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    TRANSLATION_FIELD_NUMBER: _ClassVar[int]
    TRANSLATION_ORIGIN_FIELD_NUMBER: _ClassVar[int]
    TRANSLATION_LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    OCR_DIRECTION_FIELD_NUMBER: _ClassVar[int]
    TYPOGRAPHY_FIELD_NUMBER: _ClassVar[int]
    VISIBLE_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    OPACITY_FIELD_NUMBER: _ClassVar[int]
    TRANSLATION_STYLES_FIELD_NUMBER: _ClassVar[int]
    layer_id: str
    content_id: str
    region_id: str
    balloon_id: str
    layout_kind: str
    frame: _containers.RepeatedCompositeFieldContainer[Point]
    bounds: Box
    angle: float
    source_text: str
    source_origin: str
    source_language: str
    translation: str
    translation_origin: str
    translation_language: str
    ocr_direction: str
    typography: Typography
    visible: bool
    role: str
    opacity: float
    translation_styles: _containers.RepeatedCompositeFieldContainer[TextStyle]
    def __init__(self, layer_id: _Optional[str] = ..., content_id: _Optional[str] = ..., region_id: _Optional[str] = ..., balloon_id: _Optional[str] = ..., layout_kind: _Optional[str] = ..., frame: _Optional[_Iterable[_Union[Point, _Mapping]]] = ..., bounds: _Optional[_Union[Box, _Mapping]] = ..., angle: _Optional[float] = ..., source_text: _Optional[str] = ..., source_origin: _Optional[str] = ..., source_language: _Optional[str] = ..., translation: _Optional[str] = ..., translation_origin: _Optional[str] = ..., translation_language: _Optional[str] = ..., ocr_direction: _Optional[str] = ..., typography: _Optional[_Union[Typography, _Mapping]] = ..., visible: _Optional[bool] = ..., role: _Optional[str] = ..., opacity: _Optional[float] = ..., translation_styles: _Optional[_Iterable[_Union[TextStyle, _Mapping]]] = ...) -> None: ...

class Region(_message.Message):
    __slots__ = ("id", "kind", "label", "polygon", "bounds", "confidence")
    ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    POLYGON_FIELD_NUMBER: _ClassVar[int]
    BOUNDS_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    id: str
    kind: str
    label: str
    polygon: _containers.RepeatedCompositeFieldContainer[Point]
    bounds: Box
    confidence: float
    def __init__(self, id: _Optional[str] = ..., kind: _Optional[str] = ..., label: _Optional[str] = ..., polygon: _Optional[_Iterable[_Union[Point, _Mapping]]] = ..., bounds: _Optional[_Union[Box, _Mapping]] = ..., confidence: _Optional[float] = ...) -> None: ...

class RasterLayer(_message.Message):
    __slots__ = ("id", "name", "kind", "visible", "opacity")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    VISIBLE_FIELD_NUMBER: _ClassVar[int]
    OPACITY_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    kind: str
    visible: bool
    opacity: float
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., kind: _Optional[str] = ..., visible: _Optional[bool] = ..., opacity: _Optional[float] = ...) -> None: ...

class LayerVisibilityRequest(_message.Message):
    __slots__ = ("project_id", "page_id", "layer_ids", "visible", "opacity")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    PAGE_ID_FIELD_NUMBER: _ClassVar[int]
    LAYER_IDS_FIELD_NUMBER: _ClassVar[int]
    VISIBLE_FIELD_NUMBER: _ClassVar[int]
    OPACITY_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    page_id: str
    layer_ids: _containers.RepeatedScalarFieldContainer[str]
    visible: bool
    opacity: float
    def __init__(self, project_id: _Optional[str] = ..., page_id: _Optional[str] = ..., layer_ids: _Optional[_Iterable[str]] = ..., visible: _Optional[bool] = ..., opacity: _Optional[float] = ...) -> None: ...

class AddRasterLayerRequest(_message.Message):
    __slots__ = ("project_id", "page_id", "name", "above_layer_id")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    PAGE_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ABOVE_LAYER_ID_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    page_id: str
    name: str
    above_layer_id: str
    def __init__(self, project_id: _Optional[str] = ..., page_id: _Optional[str] = ..., name: _Optional[str] = ..., above_layer_id: _Optional[str] = ...) -> None: ...

class MoveLayerRequest(_message.Message):
    __slots__ = ("project_id", "page_id", "layer_id", "delta")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    PAGE_ID_FIELD_NUMBER: _ClassVar[int]
    LAYER_ID_FIELD_NUMBER: _ClassVar[int]
    DELTA_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    page_id: str
    layer_id: str
    delta: int
    def __init__(self, project_id: _Optional[str] = ..., page_id: _Optional[str] = ..., layer_id: _Optional[str] = ..., delta: _Optional[int] = ...) -> None: ...

class SetRasterLayerRequest(_message.Message):
    __slots__ = ("project_id", "page_id", "kind", "layer_id", "png", "patch", "x", "y")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    PAGE_ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    LAYER_ID_FIELD_NUMBER: _ClassVar[int]
    PNG_FIELD_NUMBER: _ClassVar[int]
    PATCH_FIELD_NUMBER: _ClassVar[int]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    page_id: str
    kind: str
    layer_id: str
    png: bytes
    patch: bool
    x: int
    y: int
    def __init__(self, project_id: _Optional[str] = ..., page_id: _Optional[str] = ..., kind: _Optional[str] = ..., layer_id: _Optional[str] = ..., png: _Optional[bytes] = ..., patch: _Optional[bool] = ..., x: _Optional[int] = ..., y: _Optional[int] = ...) -> None: ...

class StageStatus(_message.Message):
    __slots__ = ("stage", "done", "model")
    STAGE_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    stage: str
    done: bool
    model: str
    def __init__(self, stage: _Optional[str] = ..., done: _Optional[bool] = ..., model: _Optional[str] = ...) -> None: ...

class Page(_message.Message):
    __slots__ = ("id", "name", "width", "height", "blocks", "regions", "raster_layers", "assets", "stages", "revision", "asset_versions")
    class AssetVersionsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    BLOCKS_FIELD_NUMBER: _ClassVar[int]
    REGIONS_FIELD_NUMBER: _ClassVar[int]
    RASTER_LAYERS_FIELD_NUMBER: _ClassVar[int]
    ASSETS_FIELD_NUMBER: _ClassVar[int]
    STAGES_FIELD_NUMBER: _ClassVar[int]
    REVISION_FIELD_NUMBER: _ClassVar[int]
    ASSET_VERSIONS_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    width: int
    height: int
    blocks: _containers.RepeatedCompositeFieldContainer[TextBlock]
    regions: _containers.RepeatedCompositeFieldContainer[Region]
    raster_layers: _containers.RepeatedCompositeFieldContainer[RasterLayer]
    assets: _containers.RepeatedScalarFieldContainer[str]
    stages: _containers.RepeatedCompositeFieldContainer[StageStatus]
    revision: int
    asset_versions: _containers.ScalarMap[str, str]
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., width: _Optional[int] = ..., height: _Optional[int] = ..., blocks: _Optional[_Iterable[_Union[TextBlock, _Mapping]]] = ..., regions: _Optional[_Iterable[_Union[Region, _Mapping]]] = ..., raster_layers: _Optional[_Iterable[_Union[RasterLayer, _Mapping]]] = ..., assets: _Optional[_Iterable[str]] = ..., stages: _Optional[_Iterable[_Union[StageStatus, _Mapping]]] = ..., revision: _Optional[int] = ..., asset_versions: _Optional[_Mapping[str, str]] = ...) -> None: ...

class Project(_message.Message):
    __slots__ = ("id", "name", "created_at", "updated_at", "revision", "pages", "can_undo", "can_redo", "chapters")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    REVISION_FIELD_NUMBER: _ClassVar[int]
    PAGES_FIELD_NUMBER: _ClassVar[int]
    CAN_UNDO_FIELD_NUMBER: _ClassVar[int]
    CAN_REDO_FIELD_NUMBER: _ClassVar[int]
    CHAPTERS_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    created_at: float
    updated_at: float
    revision: int
    pages: _containers.RepeatedCompositeFieldContainer[Page]
    can_undo: bool
    can_redo: bool
    chapters: _containers.RepeatedCompositeFieldContainer[Chapter]
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., created_at: _Optional[float] = ..., updated_at: _Optional[float] = ..., revision: _Optional[int] = ..., pages: _Optional[_Iterable[_Union[Page, _Mapping]]] = ..., can_undo: _Optional[bool] = ..., can_redo: _Optional[bool] = ..., chapters: _Optional[_Iterable[_Union[Chapter, _Mapping]]] = ...) -> None: ...

class Chapter(_message.Message):
    __slots__ = ("id", "name", "page_ids")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    PAGE_IDS_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    page_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., page_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class ProjectSummary(_message.Message):
    __slots__ = ("id", "name", "pages", "created_at", "updated_at", "chapters")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    PAGES_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    CHAPTERS_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    pages: int
    created_at: float
    updated_at: float
    chapters: int
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., pages: _Optional[int] = ..., created_at: _Optional[float] = ..., updated_at: _Optional[float] = ..., chapters: _Optional[int] = ...) -> None: ...

class ProjectList(_message.Message):
    __slots__ = ("projects",)
    PROJECTS_FIELD_NUMBER: _ClassVar[int]
    projects: _containers.RepeatedCompositeFieldContainer[ProjectSummary]
    def __init__(self, projects: _Optional[_Iterable[_Union[ProjectSummary, _Mapping]]] = ...) -> None: ...

class CreateProjectRequest(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class RenameProjectRequest(_message.Message):
    __slots__ = ("project_id", "name")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    name: str
    def __init__(self, project_id: _Optional[str] = ..., name: _Optional[str] = ...) -> None: ...

class ProjectRef(_message.Message):
    __slots__ = ("project_id",)
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    def __init__(self, project_id: _Optional[str] = ...) -> None: ...

class PageRef(_message.Message):
    __slots__ = ("project_id", "page_id")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    PAGE_ID_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    page_id: str
    def __init__(self, project_id: _Optional[str] = ..., page_id: _Optional[str] = ...) -> None: ...

class CreateChapterRequest(_message.Message):
    __slots__ = ("project_id", "name", "index")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    name: str
    index: int
    def __init__(self, project_id: _Optional[str] = ..., name: _Optional[str] = ..., index: _Optional[int] = ...) -> None: ...

class RenameChapterRequest(_message.Message):
    __slots__ = ("project_id", "chapter_id", "name")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    CHAPTER_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    chapter_id: str
    name: str
    def __init__(self, project_id: _Optional[str] = ..., chapter_id: _Optional[str] = ..., name: _Optional[str] = ...) -> None: ...

class ChapterRef(_message.Message):
    __slots__ = ("project_id", "chapter_id")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    CHAPTER_ID_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    chapter_id: str
    def __init__(self, project_id: _Optional[str] = ..., chapter_id: _Optional[str] = ...) -> None: ...

class ChapterLayout(_message.Message):
    __slots__ = ("chapter_id", "page_ids")
    CHAPTER_ID_FIELD_NUMBER: _ClassVar[int]
    PAGE_IDS_FIELD_NUMBER: _ClassVar[int]
    chapter_id: str
    page_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, chapter_id: _Optional[str] = ..., page_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class ArrangeProjectRequest(_message.Message):
    __slots__ = ("project_id", "chapters")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    CHAPTERS_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    chapters: _containers.RepeatedCompositeFieldContainer[ChapterLayout]
    def __init__(self, project_id: _Optional[str] = ..., chapters: _Optional[_Iterable[_Union[ChapterLayout, _Mapping]]] = ...) -> None: ...

class TextBlockRef(_message.Message):
    __slots__ = ("project_id", "page_id", "layer_id")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    PAGE_ID_FIELD_NUMBER: _ClassVar[int]
    LAYER_ID_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    page_id: str
    layer_id: str
    def __init__(self, project_id: _Optional[str] = ..., page_id: _Optional[str] = ..., layer_id: _Optional[str] = ...) -> None: ...

class AddPageRequest(_message.Message):
    __slots__ = ("project_id", "name", "data", "index", "chapter_id")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    CHAPTER_ID_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    name: str
    data: bytes
    index: int
    chapter_id: str
    def __init__(self, project_id: _Optional[str] = ..., name: _Optional[str] = ..., data: _Optional[bytes] = ..., index: _Optional[int] = ..., chapter_id: _Optional[str] = ...) -> None: ...

class AssetRequest(_message.Message):
    __slots__ = ("project_id", "page_id", "kind")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    PAGE_ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    page_id: str
    kind: str
    def __init__(self, project_id: _Optional[str] = ..., page_id: _Optional[str] = ..., kind: _Optional[str] = ...) -> None: ...

class Asset(_message.Message):
    __slots__ = ("data", "content_type", "width", "height", "found")
    DATA_FIELD_NUMBER: _ClassVar[int]
    CONTENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    FOUND_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    content_type: str
    width: int
    height: int
    found: bool
    def __init__(self, data: _Optional[bytes] = ..., content_type: _Optional[str] = ..., width: _Optional[int] = ..., height: _Optional[int] = ..., found: _Optional[bool] = ...) -> None: ...

class UpdateTextBlockRequest(_message.Message):
    __slots__ = ("project_id", "page_id", "layer_id", "source_text", "translation", "typography", "frame", "visible", "fields", "translation_styles")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    PAGE_ID_FIELD_NUMBER: _ClassVar[int]
    LAYER_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_TEXT_FIELD_NUMBER: _ClassVar[int]
    TRANSLATION_FIELD_NUMBER: _ClassVar[int]
    TYPOGRAPHY_FIELD_NUMBER: _ClassVar[int]
    FRAME_FIELD_NUMBER: _ClassVar[int]
    VISIBLE_FIELD_NUMBER: _ClassVar[int]
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    TRANSLATION_STYLES_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    page_id: str
    layer_id: str
    source_text: str
    translation: str
    typography: Typography
    frame: _containers.RepeatedCompositeFieldContainer[Point]
    visible: bool
    fields: _containers.RepeatedScalarFieldContainer[str]
    translation_styles: _containers.RepeatedCompositeFieldContainer[TextStyle]
    def __init__(self, project_id: _Optional[str] = ..., page_id: _Optional[str] = ..., layer_id: _Optional[str] = ..., source_text: _Optional[str] = ..., translation: _Optional[str] = ..., typography: _Optional[_Union[Typography, _Mapping]] = ..., frame: _Optional[_Iterable[_Union[Point, _Mapping]]] = ..., visible: _Optional[bool] = ..., fields: _Optional[_Iterable[str]] = ..., translation_styles: _Optional[_Iterable[_Union[TextStyle, _Mapping]]] = ...) -> None: ...

class CreateTextBlockRequest(_message.Message):
    __slots__ = ("project_id", "page_id", "bounds")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    PAGE_ID_FIELD_NUMBER: _ClassVar[int]
    BOUNDS_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    page_id: str
    bounds: Box
    def __init__(self, project_id: _Optional[str] = ..., page_id: _Optional[str] = ..., bounds: _Optional[_Union[Box, _Mapping]] = ...) -> None: ...

class Operation(_message.Message):
    __slots__ = ("kind", "stage", "stages")
    KIND_FIELD_NUMBER: _ClassVar[int]
    STAGE_FIELD_NUMBER: _ClassVar[int]
    STAGES_FIELD_NUMBER: _ClassVar[int]
    kind: str
    stage: str
    stages: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, kind: _Optional[str] = ..., stage: _Optional[str] = ..., stages: _Optional[_Iterable[str]] = ...) -> None: ...

class RunRequest(_message.Message):
    __slots__ = ("project_id", "page_ids", "operation", "region", "entity_ids", "inpainting_mask", "fill_prompt")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    PAGE_IDS_FIELD_NUMBER: _ClassVar[int]
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    REGION_FIELD_NUMBER: _ClassVar[int]
    ENTITY_IDS_FIELD_NUMBER: _ClassVar[int]
    INPAINTING_MASK_FIELD_NUMBER: _ClassVar[int]
    FILL_PROMPT_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    page_ids: _containers.RepeatedScalarFieldContainer[str]
    operation: Operation
    region: Box
    entity_ids: _containers.RepeatedScalarFieldContainer[str]
    inpainting_mask: bytes
    fill_prompt: str
    def __init__(self, project_id: _Optional[str] = ..., page_ids: _Optional[_Iterable[str]] = ..., operation: _Optional[_Union[Operation, _Mapping]] = ..., region: _Optional[_Union[Box, _Mapping]] = ..., entity_ids: _Optional[_Iterable[str]] = ..., inpainting_mask: _Optional[bytes] = ..., fill_prompt: _Optional[str] = ...) -> None: ...

class ProgressEvent(_message.Message):
    __slots__ = ("job_id", "kind", "page_id", "stage", "model", "elapsed_ms", "message", "completed", "total", "pages", "stages", "page")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    PAGE_ID_FIELD_NUMBER: _ClassVar[int]
    STAGE_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    ELAPSED_MS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_FIELD_NUMBER: _ClassVar[int]
    TOTAL_FIELD_NUMBER: _ClassVar[int]
    PAGES_FIELD_NUMBER: _ClassVar[int]
    STAGES_FIELD_NUMBER: _ClassVar[int]
    PAGE_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    kind: str
    page_id: str
    stage: str
    model: str
    elapsed_ms: float
    message: str
    completed: int
    total: int
    pages: _containers.RepeatedScalarFieldContainer[str]
    stages: _containers.RepeatedScalarFieldContainer[str]
    page: Page
    def __init__(self, job_id: _Optional[str] = ..., kind: _Optional[str] = ..., page_id: _Optional[str] = ..., stage: _Optional[str] = ..., model: _Optional[str] = ..., elapsed_ms: _Optional[float] = ..., message: _Optional[str] = ..., completed: _Optional[int] = ..., total: _Optional[int] = ..., pages: _Optional[_Iterable[str]] = ..., stages: _Optional[_Iterable[str]] = ..., page: _Optional[_Union[Page, _Mapping]] = ...) -> None: ...

class JobRef(_message.Message):
    __slots__ = ("job_id",)
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    def __init__(self, job_id: _Optional[str] = ...) -> None: ...

class Job(_message.Message):
    __slots__ = ("id", "project_id", "status", "completed", "total", "started_at", "error")
    ID_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_FIELD_NUMBER: _ClassVar[int]
    TOTAL_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    id: str
    project_id: str
    status: str
    completed: int
    total: int
    started_at: float
    error: str
    def __init__(self, id: _Optional[str] = ..., project_id: _Optional[str] = ..., status: _Optional[str] = ..., completed: _Optional[int] = ..., total: _Optional[int] = ..., started_at: _Optional[float] = ..., error: _Optional[str] = ...) -> None: ...

class JobList(_message.Message):
    __slots__ = ("jobs",)
    JOBS_FIELD_NUMBER: _ClassVar[int]
    jobs: _containers.RepeatedCompositeFieldContainer[Job]
    def __init__(self, jobs: _Optional[_Iterable[_Union[Job, _Mapping]]] = ...) -> None: ...

class FontFamily(_message.Message):
    __slots__ = ("name", "faces")
    NAME_FIELD_NUMBER: _ClassVar[int]
    FACES_FIELD_NUMBER: _ClassVar[int]
    name: str
    faces: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, name: _Optional[str] = ..., faces: _Optional[_Iterable[str]] = ...) -> None: ...

class FontList(_message.Message):
    __slots__ = ("families",)
    FAMILIES_FIELD_NUMBER: _ClassVar[int]
    families: _containers.RepeatedCompositeFieldContainer[FontFamily]
    def __init__(self, families: _Optional[_Iterable[_Union[FontFamily, _Mapping]]] = ...) -> None: ...
