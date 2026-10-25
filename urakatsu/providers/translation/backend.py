"""Translation requests and page-image preparation.

the longest side is capped at 2048 with Lanczos
resampling, and hosted providers receive a JPEG (quality 88) data URL. Local models
receive the prepared pixels losslessly (see ``local.py``).
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from .language import Language

MAX_IMAGE_DIMENSION = 2048
JPEG_QUALITY = 88


@dataclass
class TranslationContext:
    source: str
    translation: str


@dataclass
class TranslationRequest:
    segments: list[str]
    target_language: Language
    source_language: Optional[Language] = None
    instructions: Optional[str] = None
    context: list[TranslationContext] = field(default_factory=list)
    image: Optional[Image.Image] = None
    # Indexes of segments that are sound effects drawn into the art (not in Koharu).
    sound_effects: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        self.segments = [str(segment) for segment in self.segments]
        self.target_language = Language.coerce(self.target_language)
        if self.source_language is not None:
            self.source_language = Language.coerce(self.source_language)

    def prepare_image(self) -> None:
        """Caps the longest side at ``MAX_IMAGE_DIMENSION``."""
        if self.image is None:
            return
        width, height = self.image.size
        longest = max(width, height)
        if longest <= MAX_IMAGE_DIMENSION:
            return
        resized_width = max(width * MAX_IMAGE_DIMENSION // longest, 1)
        resized_height = max(height * MAX_IMAGE_DIMENSION // longest, 1)
        self.image = self.image.convert("RGB").resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    def remove_image(self) -> None:
        self.image = None


def encode_jpeg_data_url(image: Image.Image) -> str:
    """``data:image/jpeg;base64,...`` at quality 88."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def encode_png_data_url(image: Image.Image) -> str:
    """Lossless ``data:image/png;base64,...`` of the same pixels (used where raw pixels are passed)."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
