"""System discovery, bundled-catalog loading, fallback, and bounded font caching.

Port of ``fonts.rs``.

Divergences (no exact Python twin for these pieces):
- fontique's system collection (DirectWrite on Windows) is replaced by a scan of the Windows
  font directories with fontTools. Family, PostScript name, weight, width, style and variable
  axis ranges come from the same OpenType name/OS/2/fvar tables fontique reads.
- fontique's per-script platform fallback lists are replaced by ``SCRIPT_FALLBACK_FAMILIES``
  (the Windows families DirectWrite uses for each script).
- skrifa is replaced by fontTools for cmap, metrics and glyph bounds. Bounds ignore the
  variation location unless the face is variable (then they come from the instanced glyph set).
- The bundled catalog is the Hugging Face dataset ``mayocream/fonts`` at the same pinned
  revision: ``index.json`` (schema 3) is downloaded the first time fonts are listed or resolved,
  and each face the first time it is used, through the model store (as the upstream
  ``HuggingFaceFile::pinned_dataset``). ``URAKATSU_FONT_CATALOG`` may name a local directory with
  the same ``index.json`` and font files instead.
- When the catalog cannot be fetched (offline, network error), only system fonts are used and the
  fetch is retried after ``CATALOG_RETRY_SECONDS``; a face that cannot be downloaded falls back
  to the next family. Koharu fails the render in both cases.
- Font previews (WebP) are not implemented; ``available_fonts`` returns the same family list.
- Synthetic emboldening and skew follow the rule "request >= 600 and face < 600" and a 14 degree
  oblique, as fontique's defaults.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
import unicodedata
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional, Sequence

import uharfbuzz as hb
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.ttLib import TTCollection, TTFont

from urakatsu.runtimes.store import HuggingFaceFile, ModelStore

from .script import fontique_scripts
from .types import FontFace, FontFamily, FontMetadata, FontRange, FontSource, FontStyle

log = logging.getLogger(__name__)

FONT_REPOSITORY = "mayocream/fonts"
# The catalog and every face it names come from one immutable snapshot (FONT_REVISION).
FONT_REVISION = "304901a868bebd5f31f95664a462e5839873d939"
FONT_INDEX = "index.json"
FONT_INDEX_SCHEMA = 3
CATALOG_RETRY_SECONDS = 60.0
DEFAULT_FONT_CACHE_BYTES = 64 * 1024 * 1024
SHAPE_PLAN_CACHE_CAPACITY = 8
SYMBOL_FALLBACK_FAMILIES = [
    "Segoe UI Symbol",
    "Segoe UI Emoji",
    "Noto Sans Symbols 2",
    "Noto Sans Symbols",
    "Noto Color Emoji",
    "Apple Symbols",
    "Apple Color Emoji",
    "Symbola",
    "Arial Unicode MS",
]

# Script -> platform families tried after the requested families (fontique's fallback key).
SCRIPT_FALLBACK_FAMILIES: dict[str, list[str]] = {
    "Hani": ["Microsoft YaHei", "Yu Gothic", "MS Gothic", "SimSun", "Meiryo", "MingLiU"],
    "Hang": ["Malgun Gothic", "Batang", "Gulim"],
    "Arab": ["Segoe UI", "Arial", "Tahoma", "Times New Roman"],
    "Hebr": ["Arial", "Segoe UI", "David"],
    "Thai": ["Leelawadee UI", "Tahoma", "Cordia New"],
    "Deva": ["Nirmala UI", "Mangal"],
    "Zsye": ["Segoe UI Emoji", "Segoe UI Symbol"],
}
DEFAULT_SCRIPT_FALLBACK = ["Segoe UI", "Arial", "Times New Roman"]

WINDOWS_FONT_DIRECTORIES = [
    Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts",
]

_FONT_EXTENSIONS = {".ttf", ".otf", ".ttc", ".otc"}


@dataclass(frozen=True)
class Attributes:
    """fontique ``Attributes``: weight (CSS 1..1000), style (normal|italic|oblique), width (%)."""

    weight: float = 400.0
    style: FontStyle = FontStyle.NORMAL
    width: float = 100.0


def _is_default_ignorable(character: str) -> bool:
    code = ord(character)
    return code in (0x200C, 0x200D) or 0xFE00 <= code <= 0xFE0F or 0xE0020 <= code <= 0xE007F or 0xE0100 <= code <= 0xE01EF


def _is_control(character: str) -> bool:
    return unicodedata.category(character) == "Cc"


def _is_ignorable_for_coverage(character: str) -> bool:
    return _is_control(character) or character.isspace() or _is_default_ignorable(character)


class Font:
    """A resolved face and variable-font instance used by shaping, metrics, and drawing."""

    def __init__(
        self,
        data: bytes,
        index: int,
        family_name: str,
        font_name: str,
        post_script_name: str,
        variations: dict[str, float],
        synthetic_bold: bool,
        synthetic_skew: Optional[float],
    ) -> None:
        self._data = data
        self.index = index
        self.family_name = family_name
        self.font_name = font_name
        self.post_script_name = post_script_name
        self.variations = dict(variations)
        self._synthetic_bold = synthetic_bold
        self._synthetic_skew = synthetic_skew

        self._face = hb.Face(data, index)
        self.hb_font = hb.Font(self._face)
        if self.variations:
            self.hb_font.set_variations(self.variations)
        self.units_per_em = self._face.upem

        self._tt = TTFont(BytesIO(data), fontNumber=index, lazy=True)
        cmap = self._tt.getBestCmap() or {}
        self._cmap = set(cmap.keys())
        subtables = {(t.platformID, t.platEncID) for t in self._tt["cmap"].tables} if "cmap" in self._tt else set()
        # A symbol-encoded face has only a (3, 0) subtable and no Unicode subtable.
        self._symbol = (3, 0) in subtables and not any(p == 0 or (p == 3 and e == 1) for p, e in subtables)
        self._glyph_set = None
        self._bounds_cache: dict[int, Optional[tuple[float, float, float, float]]] = {}

    # ---- identity ------------------------------------------------------------------------------

    @property
    def bytes(self) -> bytes:
        return self._data

    def data(self) -> bytes:
        return self._data

    def synthetic_bold(self) -> bool:
        return self._synthetic_bold

    def synthetic_skew(self) -> Optional[float]:
        return self._synthetic_skew

    # ---- coverage ------------------------------------------------------------------------------

    def has_glyph(self, character: str) -> bool:
        return ord(character) in self._cmap

    def covers(self, text: str) -> bool:
        """``Font::covers``: every visible character maps to a glyph in this face."""
        return all(
            ord(ch) in self._cmap for ch in text if not _is_ignorable_for_coverage(ch)
        )

    # ---- metrics and outlines ------------------------------------------------------------------

    def metrics(self, font_size: float) -> tuple[float, float, float]:
        """(ascent, descent, leading) at ``font_size``; descent is negative as in skrifa."""
        scale = font_size / self.units_per_em
        hhea = self._tt["hhea"]
        ascent = hhea.ascent
        descent = hhea.descent
        leading = hhea.lineGap
        os2 = self._tt["OS/2"] if "OS/2" in self._tt else None
        if os2 is not None and (os2.fsSelection & (1 << 7)):
            ascent, descent, leading = os2.sTypoAscender, os2.sTypoDescender, os2.sTypoLineGap
        return ascent * scale, descent * scale, leading * scale

    def glyph_bounds(self, glyph_id: int, font_size: float) -> Optional[tuple[float, float, float, float]]:
        """Y-up outline bounds (x_min, y_min, x_max, y_max) in pixels, or None for empty glyphs."""
        if glyph_id not in self._bounds_cache:
            self._bounds_cache[glyph_id] = self._units_bounds(glyph_id)
        units = self._bounds_cache[glyph_id]
        if units is None:
            return None
        scale = font_size / self.units_per_em
        return (units[0] * scale, units[1] * scale, units[2] * scale, units[3] * scale)

    def glyph_recording(self, glyph_id: int) -> list[tuple[str, tuple]]:
        """Outline of one glyph in font units (y up) as fontTools pen operations."""
        if self._glyph_set is None:
            self._units_bounds(glyph_id)
        if glyph_id >= len(self._tt.getGlyphOrder()):
            return []
        name = self._tt.getGlyphName(glyph_id)
    def decoration_metrics(self, font_size: float) -> tuple[float, float, float, float]:
        """(underline offset below the baseline, underline thickness, strikeout height above the
        baseline, strikeout thickness) in pixels, from the post and OS/2 tables where the face has them."""
        scale = font_size / self.units_per_em
        post = self._tt["post"] if "post" in self._tt else None
        os2 = self._tt["OS/2"] if "OS/2" in self._tt else None
        underline_position = post.underlinePosition if post is not None and post.underlinePosition else -0.1 * self.units_per_em
        underline_thickness = post.underlineThickness if post is not None and post.underlineThickness > 0 else 0.05 * self.units_per_em
        strike_position = os2.yStrikeoutPosition if os2 is not None and os2.yStrikeoutPosition > 0 else 0.25 * self.units_per_em
        strike_thickness = os2.yStrikeoutSize if os2 is not None and os2.yStrikeoutSize > 0 else underline_thickness
        return -underline_position * scale, underline_thickness * scale, strike_position * scale, strike_thickness * scale

        # Decomposing: composite glyphs are expanded into their components' outlines.
        pen = DecomposingRecordingPen(self._glyph_set)
        try:
            self._glyph_set[name].draw(pen)
        except KeyError:
            return []
        return list(pen.value)

    def _units_bounds(self, glyph_id: int) -> Optional[tuple[float, float, float, float]]:
        if self._glyph_set is None:
            location = dict(self.variations)
            self._glyph_set = self._tt.getGlyphSet(location=location) if location else self._tt.getGlyphSet()
        if glyph_id >= len(self._tt.getGlyphOrder()):
            return None
        name = self._tt.getGlyphName(glyph_id)
        pen = BoundsPen(self._glyph_set)
        try:
            self._glyph_set[name].draw(pen)
        except KeyError:
            return None
        return pen.bounds



class SystemFace:
    """A face found by scanning the installed font files."""

    def __init__(
        self,
        path: Path,
        index: int,
        family_names: list[str],
        post_script_name: str,
        weight: int,
        width: int,
        style: FontStyle,
        weight_range: Optional[FontRange],
        width_range: Optional[FontRange],
        axes: frozenset[str] = frozenset(),
    ) -> None:
        self.path = path
        self.index = index
        self.family_names = family_names
        self.family_name = family_names[0]
        self.post_script_name = post_script_name
        self.weight = weight
        self.width = width
        self.style = style
        self.weight_range = weight_range
        self.width_range = width_range
        self.axes = axes  # fvar axis tags


def _name(tt: TTFont, name_id: int) -> Optional[str]:
    record = tt["name"].getName(name_id, 3, 1, 0x409) or tt["name"].getName(name_id, 1, 0, 0)
    if record is None:
        for candidate in tt["name"].names:
            if candidate.nameID == name_id:
                return candidate.toUnicode()
        return None
    return record.toUnicode()


def _scan_font_file(path: Path) -> list[SystemFace]:
    """Faces of one font file: every member of a collection (.ttc/.otc), else the single face."""
    faces: list[SystemFace] = []
    try:
        if path.suffix.lower() in (".ttc", ".otc"):
            tables = list(TTCollection(str(path), lazy=True).fonts)
        else:
            tables = [TTFont(str(path), lazy=True)]
    except Exception:
        return faces
    index_count = len(tables)
    for index, tt in enumerate(tables):
        try:
            names = [n for n in (_name(tt, 16), _name(tt, 1)) if n]
            family_names = list(dict.fromkeys(names))
            if not family_names:
                continue
            ps_name = _name(tt, 6) or family_names[0]
            os2 = tt["OS/2"] if "OS/2" in tt else None
            weight = int(os2.usWeightClass) if os2 is not None else 400
            width = int(os2.usWidthClass) * 25 if os2 is not None and os2.usWidthClass else 100
            italic = bool(os2 is not None and os2.fsSelection & 1)
            style = FontStyle.ITALIC if italic else FontStyle.NORMAL
            weight_range = None
            width_range = None
            axes: frozenset[str] = frozenset()
            if "fvar" in tt:
                axes = frozenset(axis.axisTag for axis in tt["fvar"].axes)
                for axis in tt["fvar"].axes:
                    if axis.axisTag == "wght":
                        weight_range = FontRange(int(round(axis.minValue)), int(round(axis.maxValue)))
                    elif axis.axisTag == "wdth":
                        width_range = FontRange(int(round(axis.minValue * 100)), int(round(axis.maxValue * 100)))
            faces.append(
                SystemFace(
                    path=path,
                    index=index if index_count > 1 else 0,
                    family_names=family_names,
                    post_script_name=ps_name,
                    weight=max(1, min(1000, weight)),
                    width=max(1, min(1000, width)),
                    style=style,
                    weight_range=weight_range,
                    width_range=width_range,
                    axes=axes,
                )
            )
        except Exception:
            continue
    return faces


def _axis_distance(value: float, default: int, face_range: Optional[FontRange]) -> int:
    """``axis_distance``: distance from ``value`` to the face's range, or to its default."""
    if face_range is not None:
        if value < face_range.minimum:
            distance = face_range.minimum - value
        elif value > face_range.maximum:
            distance = value - face_range.maximum
        else:
            distance = 0.0
    else:
        distance = abs(default - value)
    return int(round(distance))


def _normalize(value: str) -> str:
    return value.strip().lower()


class FontSystem:
    """``FontSystem``: installed-font discovery, matching, fallback, and source caching."""

    def __init__(self) -> None:
        self._faces: Optional[list[SystemFace]] = None
        self._file_bytes: dict[Path, bytes] = {}
        self._resolved: dict[tuple, list[Font]] = {}
        self._font_cache: dict[tuple, Font] = {}

    # discovery
    def system_faces(self) -> list[SystemFace]:
        if self._faces is None:
            faces: list[SystemFace] = []
            for directory in WINDOWS_FONT_DIRECTORIES:
                if not directory or not directory.is_dir():
                    continue
                for path in sorted(directory.iterdir()):
                    if path.suffix.lower() in _FONT_EXTENSIONS:
                        faces.extend(_scan_font_file(path))
            self._faces = faces
        return self._faces

    def family_names(self) -> list[str]:
        names: dict[str, str] = {}
        for face in self.system_faces():
            for family in face.family_names:
                names.setdefault(_normalize(family), family)
        return list(names.values())

    def family_exists(self, family: str) -> bool:
        key = _normalize(family)
        return any(_normalize(name) == key for face in self.system_faces() for name in face.family_names)

    def _faces_of(self, family: str) -> list[SystemFace]:
        key = _normalize(family)
        return [face for face in self.system_faces() if any(_normalize(name) == key for name in face.family_names)]

    def _best_face(self, faces: Sequence[SystemFace], attributes: Attributes) -> Optional[SystemFace]:
        best = None
        best_score = None
        for face in faces:
            style = 0 if face.style == attributes.style else 2_000
            weight = _axis_distance(attributes.weight, face.weight, face.weight_range)
            stretch = _axis_distance(attributes.width, face.width, face.width_range)
            score = style + weight + stretch
            if best_score is None or score < best_score:
                best, best_score = face, score
        return best

    # resolution
    def resolve(
        self,
        families: Sequence[str],
        attributes: Attributes,
        scripts: Sequence[str],
        language: Optional[str] = None,
    ) -> list["Font"]:
        return self.resolve_with_fallback_families(families, [], attributes, scripts, language)

    def resolve_with_fallback_families(
        self,
        families: Sequence[str],
        fallback_families: Sequence[str],
        attributes: Attributes,
        scripts: Sequence[str],
        language: Optional[str] = None,
    ) -> list["Font"]:
        scripts = list(scripts) if scripts else ["Latn"]
        key = (
            tuple(families),
            tuple(fallback_families),
            attributes,
            tuple(scripts),
            language,
        )
        if key in self._resolved:
            return list(self._resolved[key])

        seen: set[tuple[Path, int]] = set()
        resolved: list[Font] = []
        for script in scripts:
            candidates: list[SystemFace] = []
            named = [family for family in families]
            if not named:
                named = ["Arial"]
            for family in named:
                face = self._best_face(self._faces_of(family), attributes)
                if face is not None:
                    candidates.append(face)
            for family in SCRIPT_FALLBACK_FAMILIES.get(script, DEFAULT_SCRIPT_FALLBACK):
                face = self._best_face(self._faces_of(family), attributes)
                if face is not None:
                    candidates.append(face)
            extra = [f for f in fallback_families if not any(_normalize(f) == _normalize(x) for x in families)]
            for family in extra:
                face = self._best_face(self._faces_of(family), attributes)
                if face is not None:
                    candidates.append(face)

            for face in candidates:
                identity = (face.path, face.index)
                if identity in seen:
                    continue
                seen.add(identity)
                resolved.append(self._load(face, attributes))

        if not resolved and families:
            resolved = self.resolve_with_fallback_families([], fallback_families, attributes, scripts, language)
        if not resolved:
            raise RuntimeError("no usable fonts are installed")
        if len(self._resolved) >= 256:
            self._resolved.clear()
        self._resolved[key] = list(resolved)
        return list(resolved)

    def query_family(self, family: str) -> "Font":
        """``query_family``: the first resolved face whose family name matches ``family``."""
        for font in self.resolve([family], Attributes(), ["Latn"]):
            if _normalize(font.family_name) == _normalize(family):
                return font
        raise LookupError(f"no font found for family {family!r}")

    def first_font(self) -> "Font":
        return self.resolve([], Attributes(), ["Latn"])[0]

    def _load(self, face: SystemFace, attributes: Attributes) -> "Font":
        variations: dict[str, float] = {}
        if face.weight_range is not None:
            variations["wght"] = min(max(attributes.weight, face.weight_range.minimum), face.weight_range.maximum)
        if face.width_range is not None:
            variations["wdth"] = min(max(attributes.width, face.width_range.minimum), face.width_range.maximum)
        # fontique ``FontInfo::synthesis``: a wght axis replaces emboldening, which a static face only
        # gets for a request more than 200 heavier; italic and oblique prefer ital/slnt axes to skew.
        synthetic_bold = face.weight_range is None and attributes.weight > face.weight + 200
        synthetic_skew = None
        if attributes.style != face.style and face.style == FontStyle.NORMAL:
            preferred = ("ital", "slnt") if attributes.style == FontStyle.ITALIC else ("slnt", "ital")
            axis = next((tag for tag in preferred if tag in face.axes), None)
            if axis == "ital":
                variations["ital"] = 1.0
            elif axis == "slnt":
                variations["slnt"] = 14.0
            else:
                synthetic_skew = 14.0
        cache_key = (face.path, face.index, tuple(sorted(variations.items())), synthetic_bold, synthetic_skew)
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]
        data = self._file_bytes.get(face.path)
        if data is None:
            data = face.path.read_bytes()
            self._file_bytes[face.path] = data
        font = Font(
            data=data,
            index=face.index,
            family_name=face.family_name,
            font_name=face.family_name,
            post_script_name=face.post_script_name,
            variations=variations,
            synthetic_bold=synthetic_bold,
            synthetic_skew=synthetic_skew,
        )
        self._font_cache[cache_key] = font
        return font


# ---- bundled catalog ---------------------------------------------------------------------------


@dataclass
class BundledFace:
    family_name: str
    file_name: str
    post_script_name: str
    weight: int
    weight_range: Optional[FontRange]
    stretch: int
    stretch_range: Optional[FontRange]
    style: FontStyle
    preview_file: str

    def info(self) -> FontFace:
        return FontFace(
            post_script_name=self.post_script_name,
            weight=self.weight,
            weight_range=self.weight_range,
            style=self.style,
        )


@dataclass
class BundledFamily:
    family_name: str
    metadata: FontMetadata
    faces: list[BundledFace]

    def info(self) -> FontFamily:
        return FontFamily(
            name=self.family_name,
            metadata=copy.deepcopy(self.metadata),
            sources=[FontSource.BUNDLED],
            faces=[face.info() for face in self.faces],
        )


class FontCatalog:
    """``FontCatalog``: the bundled ``index.json`` (schema 3) and face selection."""

    def __init__(self, families: list[BundledFamily], locate: Callable[[str], Path]) -> None:
        self.families = families
        # A face's file name to its local path (downloading it if needed).
        self.locate = locate
        self.family_names = {_normalize(family.family_name): family for family in families}

    @classmethod
    def from_index(cls, index: dict, locate: Callable[[str], Path]) -> "FontCatalog":
        families: list[BundledFamily] = []
        for family in index["families"]:
            faces: list[BundledFace] = []
            for weight in family["weights"]:
                for font in weight["fonts"]:
                    faces.append(
                        BundledFace(
                            family_name=family["family_name"],
                            file_name=font["file_name"],
                            post_script_name=font["postscript_name"],
                            weight=int(weight["weight"]),
                            weight_range=_catalog_weight_range(font.get("weight_range")),
                            stretch=int(round(float(font["width"]) * 100.0)),
                            stretch_range=_catalog_width_range(font.get("width_range")),
                            style=FontStyle(font["style"]),
                            preview_file=font["preview"]["file"],
                        )
                    )
            families.append(
                BundledFamily(
                    family_name=family["family_name"],
                    metadata=FontMetadata(
                        primary_script=family.get("primary_script"),
                        scripts=list(family.get("scripts", [])),
                        languages=list(family.get("languages", [])),
                        category=family.get("category"),
                        classifications=list(family.get("classifications", [])),
                        use_cases=[value["name"] for value in family.get("use_cases", [])],
                    ),
                    faces=faces,
                )
            )
        return cls(families, locate)

    def select(self, name: str, attributes: Attributes) -> Optional[BundledFace]:
        family = self.family_names.get(_normalize(name))
        if family is None or not family.faces:
            return None
        best = None
        best_score = None
        for face in family.faces:
            style = 0 if face.style == attributes.style else 2_000
            weight = _axis_distance(attributes.weight, face.weight, face.weight_range)
            stretch = _axis_distance(attributes.width, face.stretch, face.stretch_range)
            score = style + weight + stretch
            if best_score is None or score < best_score:
                best, best_score = face, score
        return best


def _catalog_weight_range(value: Optional[dict]) -> Optional[FontRange]:
    if not value:
        return None
    return FontRange(
        minimum=int(min(max(round(value["minimum"]), 1), 1000)),
        maximum=int(min(max(round(value["maximum"]), 1), 1000)),
    )


def _catalog_width_range(value: Optional[dict]) -> Optional[FontRange]:
    if not value:
        return None
    return FontRange(
        minimum=int(min(max(round(value["minimum"] * 100.0), 1), 1000)),
        maximum=int(min(max(round(value["maximum"] * 100.0), 1), 1000)),
    )


def _read_index(path: Path) -> dict:
    index = json.loads(path.read_text(encoding="utf-8"))
    if index.get("schema_version") != FONT_INDEX_SCHEMA:
        raise ValueError(f"unsupported bundled font index schema {index.get('schema_version')}")
    return index


def load_catalog(root: Optional[Path]) -> Optional[FontCatalog]:
    """Reads ``index.json`` from a local bundled font directory, or None when it is absent."""
    if root is None:
        return None
    index_path = root / FONT_INDEX
    if not index_path.is_file():
        return None
    return FontCatalog.from_index(_read_index(index_path), lambda file_name: root / file_name)


def catalog_file(file_name: str) -> HuggingFaceFile:
    return HuggingFaceFile(FONT_REPOSITORY, file_name, FONT_REVISION, repo_type="dataset")


def download_catalog(store: Optional[ModelStore] = None) -> FontCatalog:
    """The pinned ``index.json``; faces download when ``locate`` asks for them."""
    store = store or ModelStore()
    index = _read_index(catalog_file(FONT_INDEX).resolve(store))
    return FontCatalog.from_index(index, lambda file_name: catalog_file(file_name).resolve(store))


def default_catalog_root() -> Optional[Path]:
    value = os.environ.get("URAKATSU_FONT_CATALOG")
    return Path(value) if value else None


# ---- combined font resources -------------------------------------------------------------------


class Fonts:
    """``Fonts``: system and bundled font resolution with the Rust preference order."""

    def __init__(self, catalog: Optional[FontCatalog] = None) -> None:
        self._system: Optional[FontSystem] = None
        self._catalog = catalog
        self._catalog_retry_at = 0.0
        self._catalog_lock = threading.Lock()
        self._bundled: dict[str, Font] = {}
        self._bundled_lock = threading.Lock()

    @property
    def system(self) -> FontSystem:
        if self._system is None:
            self._system = FontSystem()
        return self._system

    @property
    def catalog(self) -> Optional[FontCatalog]:
        """The bundled catalog, fetched once; None (system fonts only) while it cannot be."""
        with self._catalog_lock:
            if self._catalog is None and time.monotonic() >= self._catalog_retry_at:
                try:
                    root = default_catalog_root()
                    self._catalog = load_catalog(root) if root is not None else download_catalog()
                except Exception as error:
                    log.warning("bundled fonts are unavailable, using system fonts only: %s", error)
                if self._catalog is None:
                    self._catalog_retry_at = time.monotonic() + CATALOG_RETRY_SECONDS
            return self._catalog

    def families(self) -> list[FontFamily]:
        """``Fonts::families``: system families merged with bundled ones, sorted like Rust."""
        families: dict[str, FontFamily] = {}
        for face in self.system.system_faces():
            key = _normalize(face.family_name)
            family = families.get(key)
            if family is None:
                family = FontFamily(
                    name=face.family_name,
                    metadata=FontMetadata(),
                    sources=[FontSource.SYSTEM],
                    faces=[],
                )
                families[key] = family
            family.faces.append(
                FontFace(
                    post_script_name=face.post_script_name,
                    weight=face.weight,
                    weight_range=None,
                    style=face.style,
                )
            )
        catalog = self.catalog
        if catalog is not None:
            for bundled in catalog.families:
                key = _normalize(bundled.family_name)
                family = families.get(key)
                if family is None:
                    family = bundled.info()
                    families[key] = family
                elif family.metadata.primary_script is None:
                    family.metadata = copy.deepcopy(bundled.metadata)
                if FontSource.BUNDLED not in family.sources:
                    family.sources.append(FontSource.BUNDLED)
                known = {_normalize(face.post_script_name) for face in family.faces}
                for face in bundled.faces:
                    if _normalize(face.post_script_name) not in known:
                        known.add(_normalize(face.post_script_name))
                        family.faces.append(face.info())
        output = list(families.values())
        for family in output:
            family.faces.sort(key=lambda f: (f.style.value, f.weight, f.post_script_name))
            family.sources.sort(key=lambda s: s.value)
        return output

    def resolve(
        self,
        preferred_font: Optional[str],
        font_weight: Optional[int],
        font_style: Optional[FontStyle],
        theme_families: Sequence[str],
        text: str,
        language: Optional[str] = None,
    ) -> list[Font]:
        """``Fonts::resolve``: preferred family, theme families, Arial; bundled face first if any."""
        families: list[str] = []
        if preferred_font and preferred_font.strip():
            families.append(preferred_font)
        for family in theme_families:
            if not any(value.lower() == family.lower() for value in families):
                families.append(family)
        if not any(family.lower() == "arial" for family in families):
            families.append("Arial")

        attributes = Attributes()
        if font_weight is not None:
            attributes = Attributes(weight=float(font_weight), style=attributes.style, width=attributes.width)
        if font_style is not None:
            attributes = Attributes(weight=attributes.weight, style=font_style, width=attributes.width)

        bundled: Optional[Font] = None
        catalog = self.catalog
        if families and not self.system.family_exists(families[0]) and catalog is not None:
            face = catalog.select(families[0], attributes)
            if face is not None:
                try:
                    bundled = self._load_bundled(catalog, face)
                except Exception as error:
                    log.warning("bundled font %s is unavailable: %s", face.file_name, error)

        resolved = self.system.resolve_with_fallback_families(
            families,
            SYMBOL_FALLBACK_FAMILIES,
            attributes,
            fontique_scripts(text),
            language,
        )
        if bundled is not None:
            resolved = [font for font in resolved if not (font.bytes == bundled.bytes and font.index == bundled.index)]
            resolved.insert(0, bundled)
        return resolved

    def _load_bundled(self, catalog: FontCatalog, face: BundledFace) -> Font:
        with self._bundled_lock:
            cached = self._bundled.get(face.file_name)
            if cached is not None:
                return cached
            font = self._read_bundled(catalog, face)
            self._bundled[face.file_name] = font
            return font

    def _read_bundled(self, catalog: FontCatalog, face: BundledFace) -> Font:
        data = catalog.locate(face.file_name).read_bytes()
        font = Font(
            data=data,
            index=0,
            family_name=face.family_name,
            font_name=face.family_name,
            post_script_name=face.post_script_name,
            variations={},
            synthetic_bold=False,
            synthetic_skew=None,
        )
        # Catalog faces are registered under the catalog family name; the post script name
        # selects the face within a collection.
        font.post_script_name = face.post_script_name
        return font
