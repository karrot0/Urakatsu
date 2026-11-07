"""Translation requests and page-image preparation.

The longest side is capped with Lanczos resampling, and hosted providers receive a JPEG
(quality 88) data URL. Local models receive the prepared pixels losslessly (see ``local.py``).
Upstream caps at 2048; here the cap follows ``TranslationConfig.prompt_size`` (2048 at "full"),
since the image is often the largest part of a paid request.
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
class TranslationTerm:
    source: str
    translation: str = ""
    note: str = ""


@dataclass
class TranslationRequest:
    segments: list[str]
    target_language: Language
    source_language: Optional[Language] = None
    instructions: Optional[str] = None
    context: list[TranslationContext] = field(default_factory=list)
    # From the project's knowledge (not in Koharu). Providers without a model prompt ignore these.
    glossary: list[TranslationTerm] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Pinned notes: the same for every page, so they go in the (cacheable) system prompt.
    background: list[str] = field(default_factory=list)
    # Story so far: summaries of earlier pages; ``summarize`` asks for this page's in the reply.
    story: list[str] = field(default_factory=list)
    summarize: bool = False
    image: Optional[Image.Image] = None
    # The longest side the page image is sent at (``TranslationConfig.prompt_size``).
    image_dimension: int = MAX_IMAGE_DIMENSION
    # Indexes of segments that are sound effects drawn into the art (not in Koharu).
    sound_effects: frozenset[int] = frozenset()

    def trim(self) -> bool:
        """Drops the least useful piece of knowledge, for a prompt that does not fit a local model's
        window: the oldest context line, then ranked notes, the story, glossary notes, and glossary
        terms last. False when nothing is left to drop."""
        if self.context:
            self.context.pop(0)
        elif self.notes:
            self.notes.pop()
        elif self.story:
            self.story.pop(0)
        elif self.background:
            self.background.pop()
        elif any(term.note for term in self.glossary):
            for term in reversed(self.glossary):
                if term.note:
                    term.note = ""
                    break
        elif self.glossary:
            self.glossary.pop()
        else:
            return False
        return True

    def __post_init__(self) -> None:
        self.segments = [str(segment) for segment in self.segments]
        self.target_language = Language.coerce(self.target_language)
        if self.source_language is not None:
            self.source_language = Language.coerce(self.source_language)

    def prepare_image(self) -> None:
        """Caps the longest side at ``image_dimension`` (``MAX_IMAGE_DIMENSION`` upstream)."""
        if self.image is None:
            return
        width, height = self.image.size
        longest = max(width, height)
        limit = min(self.image_dimension, MAX_IMAGE_DIMENSION)
        if longest <= limit:
            return
        resized_width = max(width * limit // longest, 1)
        resized_height = max(height * limit // longest, 1)
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
