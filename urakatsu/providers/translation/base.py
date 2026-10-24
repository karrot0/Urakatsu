"""Translator provider base.

Every translation provider receives the pipeline's ``TranslationConfig`` in its
settings under ``translation`` and implements ``_translate``. The public
``translate`` applies the rules that sit above every provider upstream:

- an empty request returns immediately;
- generation settings are narrowed to what the selected model supports;
- the page image is kept (and resized) only when the model can see images;
- the reply must contain exactly one translation per input segment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Optional

from pydantic import Field

from urakatsu.core.config import GenerationConfig, ModelSelection, TranslationConfig
from urakatsu.providers.base import ProviderSettings, TranslationProvider

from .backend import TranslationRequest
from .errors import SegmentCountError, TranslationError
from .model import Model

if TYPE_CHECKING:  # pragma: no cover
    import httpx


class TranslationSettings(ProviderSettings):
    """Settings every translation provider accepts. Service fields are added by subclasses."""

    translation: TranslationConfig = Field(default_factory=TranslationConfig)


class TranslatorProvider(TranslationProvider):
    Settings: ClassVar[type[TranslationSettings]] = TranslationSettings
    # Optional httpx transport override (tests inject ``httpx.MockTransport`` here).
    transport: ClassVar[Optional["httpx.BaseTransport"]] = None

    @property
    def selection(self) -> ModelSelection:
        return self.settings.translation.model

    def model_id(self) -> str:
        """The selected model id; hosted providers and the local runtime need one."""
        if not self.selection.model:
            raise TranslationError(f"{self.id} requires a selected model")
        return self.selection.model

    def generation(self) -> GenerationConfig:
        """``GenerationConfig::for_model``: capabilities the selected model lacks are dropped."""
        return self.settings.translation.generation.for_model(self.selection)

    def supports_vision_model(self, selection: ModelSelection) -> bool:
        """Whether the selected model can take an image. Overridden where the catalog knows."""
        return True

    def vision_enabled(self) -> bool:
        """``Translator::supports_vision``: the setting is on and the model can see images."""
        return bool(self.generation().vision) and self.supports_vision_model(self.selection)

    async def translate(self, request: TranslationRequest) -> list[str]:
        if not request.segments:
            return []
        generation = self.generation()
        if self.vision_enabled():
            request.prepare_image()
        else:
            request.remove_image()
        expected = len(request.segments)
        translated = await self._dispatch(request, generation)
        if len(translated) != expected:
            raise SegmentCountError(self.id, expected, len(translated))
        return translated

    async def _dispatch(self, request: TranslationRequest, generation: GenerationConfig) -> list[str]:
        """Runs ``_translate``; hosted providers override it to rotate API keys."""
        return await self._translate(request, generation)

    async def _translate(self, request: TranslationRequest, generation: GenerationConfig) -> list[str]:
        raise NotImplementedError  # pragma: no cover - interface

    @classmethod
    def list_models(cls, settings: dict, transport: "httpx.BaseTransport | None" = None) -> list[Model]:
        """Models this provider can currently use; empty when unconfigured."""
        return []
