"""Translator provider base.

Every translation provider receives the pipeline's ``TranslationConfig`` in its
settings under ``translation`` and implements ``_translate``. The public
``translate`` applies the rules that sit above every provider upstream:

- an empty request returns immediately;
- generation settings are narrowed to what the selected model supports;
- the page image is kept (and resized) only when the model can see images;
- the reply must contain exactly one translation per input segment.

Not in Koharu: providers that are language models also answer a free prompt with a reply in a
given JSON schema (``complete``), which QA uses to proofread. It needs no tool calling, so a small
local GGUF model serves it as well as a hosted one; translation services leave it out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional

from pydantic import Field

from urakatsu.core.config import GenerationConfig, ModelSelection, TranslationConfig
from urakatsu.providers.base import ProviderSettings, TranslationProvider

from .backend import TranslationRequest
from .errors import CompletionUnsupported, SegmentCountError, TranslationError
from .model import Model

if TYPE_CHECKING:  # pragma: no cover
    import httpx


@dataclass
class Completion:
    """A model's reply to ``TranslatorProvider.complete``, and its token usage when the backend
    reports it (``None`` when it does not)."""

    text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class TranslationSettings(ProviderSettings):
    """Settings every translation provider accepts. Service fields are added by subclasses."""

    translation: TranslationConfig = Field(default_factory=TranslationConfig)


class TranslatorProvider(TranslationProvider):
    Settings: ClassVar[type[TranslationSettings]] = TranslationSettings
    # Optional httpx transport override (tests inject ``httpx.MockTransport`` here).
    transport: ClassVar[Optional["httpx.BaseTransport"]] = None
    # Whether QA's proofreading runs its meaning check by default with this provider's models
    # (``QaConfig.meaning_check`` unset; ``pipeline/qa/proofread.py``): the small on-device models
    # that need it. Hosted models proofread well without its extra calls.
    meaning_check: ClassVar[bool] = False

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

    # -- structured completion (not in Koharu) ---------------------------------------------------

    def can_complete(self) -> bool:
        """Whether the provider is a language model that answers a free prompt (``complete``).
        Translation services (DeepL, Google Cloud, Caiyun) are not."""
        return type(self)._complete is not TranslatorProvider._complete

    async def complete(self, system: str, user: str, schema: dict, *, name: str = "reply", max_tokens: Optional[int] = None) -> Completion:
        """One reply to ``system`` + ``user``, constrained to the JSON ``schema``: no image, no tool
        calls, so any model with structured output can answer. Used by QA (``pipeline/stages/qa.py``).
        Generation settings are the translation's, with ``max_tokens`` in place of its own."""
        if not self.can_complete() or type(self)._complete is TranslatorProvider._complete:
            raise CompletionUnsupported(self.id)
        generation = self.generation()
        if max_tokens is not None:
            generation = generation.model_copy(update={"max_tokens": max_tokens})
        return await self._dispatch_completion(system, user, schema, name, generation)

    async def _dispatch_completion(self, system: str, user: str, schema: dict, name: str, generation: GenerationConfig) -> Completion:
        return await self._complete(system, user, schema, name, generation)

    async def _complete(self, system: str, user: str, schema: dict, name: str, generation: GenerationConfig) -> Completion:
        raise NotImplementedError  # pragma: no cover - interface

    @classmethod
    def list_models(cls, settings: dict, transport: "httpx.BaseTransport | None" = None) -> list[Model]:
        """Models this provider can currently use; empty when unconfigured."""
        return []
