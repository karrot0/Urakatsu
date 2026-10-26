"""Scene components: the components of the scene model.

Every component carries the Koharu ``KIND`` string it is stored under. Components
that report ownership expose ``origin`` (or, for authored text, ``text.origin``)
so an edit can stamp and check authorship exactly like ``Edit::prepare_value``.
"""

from __future__ import annotations

from enum import Enum
from typing import ClassVar, Generic, Optional, TypeVar, Union

from pydantic import BaseModel, Field

# ---- provenance (components/provenance.rs) --------------------------------------------------------


class Generation(BaseModel):
    producer: str
    model: Optional[str] = None
    confidence: Optional[float] = None


class UserOrigin(BaseModel):
    kind: str = "user"


class GeneratedOrigin(BaseModel):
    kind: str = "generated"
    generation: Generation


Origin = Union[UserOrigin, GeneratedOrigin]


def user() -> UserOrigin:
    return UserOrigin()


def generated(generation: Generation) -> GeneratedOrigin:
    return GeneratedOrigin(generation=generation)


def is_user(origin: Origin | None) -> bool:
    return isinstance(origin, UserOrigin)


T = TypeVar("T")


class Authored(BaseModel, Generic[T]):
    value: T
    origin: Origin = Field(default_factory=UserOrigin)

    @classmethod
    def user(cls, value):
        return cls(value=value, origin=UserOrigin())

    @classmethod
    def generated(cls, value, generation: Generation):
        return cls(value=value, origin=GeneratedOrigin(generation=generation))


# ---- base ------------------------------------------------------------------------------------------


class Component(BaseModel):
    KIND: ClassVar[str] = ""
    # Structural components can only be written through dedicated scene methods.
    STRUCTURAL: ClassVar[bool] = False

    def get_origin(self) -> Optional[Origin]:
        return getattr(self, "origin", None)

    def set_origin(self, origin: Origin) -> bool:
        if "origin" in type(self).model_fields:
            self.origin = origin  # type: ignore[attr-defined]
            return True
        return False


# ---- spatial (components/spatial.rs) -------------------------------------------------------------


class Point(BaseModel):
    x: float
    y: float


class Geometry(Component):
    KIND: ClassVar[str] = "dev.koharu.geometry"
    origin: Origin = Field(default_factory=UserOrigin)
    points: list[Point]

    @classmethod
    def rectangle(cls, x: float, y: float, width: float, height: float) -> "Geometry":
        return cls(
            points=[
                Point(x=x, y=y),
                Point(x=x + width, y=y),
                Point(x=x + width, y=y + height),
                Point(x=x, y=y + height),
            ]
        )

    def bounds(self) -> tuple[float, float, float, float]:
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        return min(xs), min(ys), max(xs), max(ys)


class Visibility(Component):
    KIND: ClassVar[str] = "dev.koharu.visibility"
    origin: Origin = Field(default_factory=UserOrigin)
    visible: bool = True
    opacity: float = 1.0


# ---- structure (components/structure.rs) ---------------------------------------------------------


class Project(Component):
    KIND: ClassVar[str] = "dev.koharu.project"
    source_locale: Optional[str] = None
    target_locales: list[str] = Field(default_factory=list)


class Chapter(BaseModel):
    id: str
    name: str
    pages: list[str] = Field(default_factory=list)
    # The linked source's chapter id when the chapter was downloaded from a source.
    remote: Optional[str] = None


class Chapters(Component):
    """Urakatsu extension: the project's chapters, in order, each listing its pages in order.

    There is no chapter level. The page order stays the flattened chapter order, so every
    stage still sees one ordered page list.
    """

    KIND: ClassVar[str] = "dev.urakatsu.chapters"
    chapters: list[Chapter] = Field(default_factory=list)


class Page(Component):
    KIND: ClassVar[str] = "dev.koharu.page"
    STRUCTURAL: ClassVar[bool] = True
    label: str
    width: float
    height: float


class EntityOrigin(Component):
    KIND: ClassVar[str] = "dev.koharu.entity.origin"
    STRUCTURAL: ClassVar[bool] = True
    origin: Origin = Field(default_factory=UserOrigin)


# ---- analysis (components/analysis.rs) -----------------------------------------------------------


class TextDirection(str, Enum):
    AUTO = "Auto"
    HORIZONTAL = "Horizontal"
    VERTICAL = "Vertical"


class OcrAnalysis(Component):
    KIND: ClassVar[str] = "dev.koharu.analysis.ocr"
    origin: Origin = Field(default_factory=UserOrigin)
    direction: TextDirection = TextDirection.AUTO
    confidence: Optional[float] = None
    line_boundaries: list[list[Point]] = Field(default_factory=list)


class Region(Component):
    KIND: ClassVar[str] = "dev.koharu.analysis.region"
    origin: Origin = Field(default_factory=UserOrigin)
    kind: str
    label: Optional[str] = None


class DetectionLabel(BaseModel):
    kind: str
    confidence: float


class DetectionAnalysis(Component):
    KIND: ClassVar[str] = "dev.koharu.analysis.detection"
    origin: Origin = Field(default_factory=UserOrigin)
    labels: list[DetectionLabel] = Field(default_factory=list)


# ---- layers (components/layers.rs) ---------------------------------------------------------------


class RasterLayerKind(str, Enum):
    CLEANUP = "cleanup"
    PAINT = "paint"
    # Not in Koharu: the result of an agent image edit (``agents/image_edit.py``). Its own
    # kind keeps the brush, which paints on the newest paint layer, off it; it draws like paint.
    EDIT = "edit"
    # Not in Koharu: the watermarks stage's result, the page with repeated watermarks unblended
    # (``pipeline/stages/watermarks.py``). It sits right over the source and draws like paint.
    WATERMARK = "watermark"


class RasterLayer(Component):
    KIND: ClassVar[str] = "dev.koharu.layer.raster"
    origin: Origin = Field(default_factory=UserOrigin)
    name: str
    kind: RasterLayerKind


class TextLayoutKind(str, Enum):
    POINT = "point"
    PARAGRAPH = "paragraph"


class TextLayout(Component):
    KIND: ClassVar[str] = "dev.koharu.layer.text"
    origin: Origin = Field(default_factory=UserOrigin)
    kind: TextLayoutKind
    # Overrides the axes inferred from geometry when a frame is transformed.
    angle_degrees: Optional[float] = None


class TextAlignment(str, Enum):
    START = "Start"
    CENTER = "Center"
    END = "End"
    JUSTIFY = "Justify"


class WritingMode(str, Enum):
    HORIZONTAL = "Horizontal"
    VERTICAL = "Vertical"


class FontStyle(str, Enum):
    NORMAL = "normal"
    ITALIC = "italic"
    OBLIQUE = "oblique"


class GradientKind(str, Enum):
    LINEAR = "linear"
    RADIAL = "radial"


class GradientStop(BaseModel):
    offset: float  # 0 to 1 along the gradient
    color: tuple[int, int, int, int]


class Gradient(BaseModel):
    """Photoshop's Gradient Overlay (on the fill) and gradient Stroke, aligned with the text's bounds.

    ``angle`` is in degrees counter-clockwise from the +x axis, as Photoshop's: at 90 the first stop is
    at the bottom and the last at the top. A radial gradient runs from the centre outwards.
    """

    kind: GradientKind = GradientKind.LINEAR
    angle: float = 90.0
    stops: list[GradientStop]


class Typography(Component):
    KIND: ClassVar[str] = "dev.koharu.text.typography"
    origin: Origin = Field(default_factory=UserOrigin)
    preferred_font: Optional[str] = None
    font_weight: Optional[int] = None
    font_style: Optional[FontStyle] = None
    size: Optional[float] = None
    auto_fit: bool = True
    color: Optional[tuple[int, int, int, int]] = None
    stroke_color: Optional[tuple[int, int, int, int]] = None
    stroke_width: Optional[float] = None
    alignment: Optional[TextAlignment] = None
    writing_mode: Optional[WritingMode] = None
    # Not in Koharu: gradients over the fill and the border (Photoshop's layer styles).
    fill_gradient: Optional[Gradient] = None
    stroke_gradient: Optional[Gradient] = None
    extensions: dict[str, str] = Field(default_factory=dict)


# ---- groups (components/groups.rs) ---------------------------------------------------------------


class Group(Component):
    KIND: ClassVar[str] = "dev.koharu.group"
    origin: Origin = Field(default_factory=UserOrigin)
    name: str = ""


class TextGroup(Component):
    KIND: ClassVar[str] = "dev.koharu.text.group"
    STRUCTURAL: ClassVar[bool] = True


# ---- text (components/text.rs) -------------------------------------------------------------------


class TextContent(Component):
    KIND: ClassVar[str] = "dev.koharu.text.content"
    origin: Origin = Field(default_factory=UserOrigin)


class SourceText(Component):
    KIND: ClassVar[str] = "dev.koharu.text.source"
    text: Authored[str]
    language: Optional[str] = None

    def get_origin(self) -> Optional[Origin]:
        return self.text.origin

    def set_origin(self, origin: Origin) -> bool:
        self.text.origin = origin
        return True


# How far a run's size may stray from its layer's.
MIN_TEXT_SCALE = 0.25
MAX_TEXT_SCALE = 4.0


class TextStyle(BaseModel):
    """Character formatting over ``[start, end)`` of a text, in code points (rich text).

    Not in Koharu, whose typography applies to a whole text layer: these override it for a run.
    """

    start: int
    end: int
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikethrough: bool = False
    color: Optional[tuple[int, int, int, int]] = None
    # A family of its own, and a size relative to the layer's (which auto-fit may choose).
    font: Optional[str] = None
    scale: Optional[float] = None
    # A gradient over these characters alone (instead of the layer's fill or ``color``).
    gradient: Optional[Gradient] = None

    def formatting(self) -> tuple:
        scale = None if self.scale is None or abs(self.scale - 1.0) < 1e-6 else min(max(self.scale, MIN_TEXT_SCALE), MAX_TEXT_SCALE)
        gradient = self.gradient.model_dump_json() if self.gradient is not None and len(self.gradient.stops) >= 2 else None
        return (self.bold, self.italic, self.underline, self.strikethrough, self.color, self.font or None, scale, gradient)


def normalize_text_styles(styles: list[TextStyle], length: int) -> list[TextStyle]:
    """Clamped to the text, sorted, non-overlapping (a later span wins), without empty or plain runs,
    and with touching runs of the same formatting merged."""
    if length <= 0 or not styles:
        return []
    per_char: list[Optional[tuple]] = [None] * length
    for style in styles:
        for index in range(max(style.start, 0), min(style.end, length)):
            per_char[index] = style.formatting()
    plain = TextStyle(start=0, end=0).formatting()
    result: list[TextStyle] = []
    index = 0
    while index < length:
        formatting = per_char[index]
        end = index + 1
        while end < length and per_char[end] == formatting:
            end += 1
        if formatting is not None and formatting != plain:
            bold, italic, underline, strikethrough, color, font, scale, gradient = formatting
            result.append(
                TextStyle(
                    start=index,
                    end=end,
                    bold=bold,
                    italic=italic,
                    underline=underline,
                    strikethrough=strikethrough,
                    color=color,
                    font=font,
                    scale=scale,
                    gradient=Gradient.model_validate_json(gradient) if gradient else None,
                )
            )
        index = end
    return result


class Translation(Component):
    KIND: ClassVar[str] = "dev.koharu.text.translation"
    text: Authored[str]
    language: Optional[str] = None
    # Rich text over ``text``. Whoever replaces the text without styles drops them, so a span can
    # never point at characters it was not made for.
    styles: list[TextStyle] = Field(default_factory=list)

    def get_origin(self) -> Optional[Origin]:
        return self.text.origin

    def set_origin(self, origin: Origin) -> bool:
        self.text.origin = origin
        return True


class TextRole(Component):
    KIND: ClassVar[str] = "dev.koharu.text.role"
    origin: Origin = Field(default_factory=UserOrigin)
    role: str


# ---- assets (components/assets.rs) ---------------------------------------------------------------


class AssetMetadata(BaseModel):
    width: Optional[int] = None
    height: Optional[int] = None
    attributes: dict[str, str] = Field(default_factory=dict)


class Asset(BaseModel):
    origin: Origin = Field(default_factory=UserOrigin)
    blob: str  # content address of the stored bytes
    media_type: str
    metadata: AssetMetadata = Field(default_factory=AssetMetadata)


class AssetInput(BaseModel):
    """Bytes plus metadata handed to ``Edit.set_asset``; the scene stores and addresses them."""

    model_config = {"arbitrary_types_allowed": True}
    data: bytes
    media_type: str
    metadata: AssetMetadata = Field(default_factory=AssetMetadata)


# ---- semantics (semantics.rs) --------------------------------------------------------------------


class RegionKinds:
    TEXT = "dev.koharu.region.text"
    BUBBLE = "dev.koharu.region.bubble"
    PANEL = "dev.koharu.region.panel"


class Relations:
    """Relation kinds. All built-in relations are functional (one target per source)."""

    PRESENTS = "dev.koharu.relation.presents"
    RECOGNIZED_FROM = "dev.koharu.relation.recognized-from"
    FITS_TO = "dev.koharu.relation.fits-to"
    FLOWS_IN = "dev.koharu.relation.flows-in"
    INSIDE = "dev.koharu.relation.inside"

    FUNCTIONAL = {PRESENTS, RECOGNIZED_FROM, FITS_TO, FLOWS_IN, INSIDE}


COMPONENTS: dict[str, type[Component]] = {
    cls.KIND: cls
    for cls in (
        Geometry,
        Visibility,
        Project,
        Chapters,
        Page,
        EntityOrigin,
        OcrAnalysis,
        Region,
        DetectionAnalysis,
        RasterLayer,
        TextLayout,
        Typography,
        Group,
        TextGroup,
        TextContent,
        SourceText,
        Translation,
        TextRole,
    )
}
