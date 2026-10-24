"""Shared public value types for renderer entry points.

Port of ``types.rs``. ``WritingMode`` lives in ``layout.rs``
in Rust and is re-exported from ``layout.py``; it is defined here so that ``script`` and
``layout`` can both import it without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FontSource(str, Enum):
    SYSTEM = "system"
    BUNDLED = "bundled"


class FontStyle(str, Enum):
    NORMAL = "normal"
    ITALIC = "italic"
    OBLIQUE = "oblique"


class TextAlign(str, Enum):
    """Inline-axis alignment within a text layout box (``TextAlign`` in types.rs)."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    JUSTIFY = "justify"


class WritingMode(str, Enum):
    """``WritingMode`` from layout.rs. Vertical is always right-to-left columns (vertical-rl)."""

    HORIZONTAL = "horizontal"
    VERTICAL_RL = "vertical-rl"

    def is_vertical(self) -> bool:
        return self is WritingMode.VERTICAL_RL


@dataclass
class FontMetadata:
    primary_script: Optional[str] = None
    scripts: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    category: Optional[str] = None
    classifications: list[str] = field(default_factory=list)
    use_cases: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FontRange:
    minimum: int
    maximum: int


@dataclass(frozen=True)
class FontFace:
    post_script_name: str
    weight: int
    weight_range: Optional[FontRange]
    style: FontStyle


@dataclass
class FontFamily:
    name: str
    metadata: FontMetadata
    sources: list[FontSource]
    faces: list[FontFace]
