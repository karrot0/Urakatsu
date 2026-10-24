"""Retained rendering of one semantic page.

Modules mirror the Rust sources: ``script`` (script.rs), ``shape`` (shape.rs), ``segment``
(segment.rs), ``fonts`` (fonts.rs), ``layout`` (layout.rs), ``bubble`` (bubble.rs),
``text_renderer`` (text_renderer.rs), ``frame`` (frame.rs), ``renderer`` (renderer.rs), and
``types`` (types.rs). ``images`` is folded into ``renderer`` (decode and cache).
"""

from .fonts import Fonts
from .frame import Frame, ImageKind, ImageMetadata, Layer, Presentation, RenderBounds, RenderDiagnostic, TextMetadata
from .renderer import Renderer, RendererError
from .types import FontFace, FontFamily, FontMetadata, FontRange, FontSource, FontStyle, TextAlign, WritingMode

__all__ = [
    "Fonts",
    "Frame",
    "FontFace",
    "FontFamily",
    "FontMetadata",
    "FontRange",
    "FontSource",
    "FontStyle",
    "ImageKind",
    "ImageMetadata",
    "Layer",
    "Presentation",
    "RenderBounds",
    "RenderDiagnostic",
    "Renderer",
    "RendererError",
    "TextAlign",
    "TextMetadata",
    "WritingMode",
]
