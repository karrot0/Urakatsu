"""Translation providers: local GGUF (llama.cpp) and hosted services.

Every provider returns one string per input segment through ``translate``. The provider
list mirrors Koharu's ``Provider`` enum except ``apple`` (iOS and macOS only).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from urakatsu.core.registry import registry
from urakatsu.core.stage import Stage

from . import local  # noqa: F401  (registers the local provider)
from .backend import TranslationContext, TranslationRequest
from .base import TranslationSettings, TranslatorProvider
from .errors import (
    ApiError,
    QuotaExceeded,
    SegmentCountError,
    TranslationError,
    UnsupportedLanguage,
    UnsupportedSourceLanguage,
)
from .language import Language, languages
from .model import Model, Quantization
from .remote import (  # noqa: F401  (each import registers one hosted provider)
    caiyun,
    claude,
    deepl,
    deepseek,
    gemini,
    google_cloud,
    grok,
    lm_studio,
    minimax,
    openai,
    openai_compatible,
    openrouter,
)

if TYPE_CHECKING:  # pragma: no cover
    from urakatsu.core.config import PipelineConfig

log = logging.getLogger(__name__)


def models(config: "PipelineConfig") -> list[Model]:
    """``Translator::models``: every provider's models, sorted by provider id then model.

    A provider whose listing fails is logged and skipped.
    Hosted listings run concurrently.
    """
    providers = registry.list(Stage.TRANSLATION)
    if not providers:
        return []

    def listing(cls):
        return cls.list_models(config.providers.get(cls.id, {}))

    with ThreadPoolExecutor(max_workers=len(providers)) as pool:
        futures = [(cls, pool.submit(listing, cls)) for cls in providers]
        found: list[Model] = []
        for cls, future in futures:
            try:
                found.extend(future.result())
            except Exception as error:  # a listing failure must not hide other providers
                log.warning("failed to list %s translation models: %s", cls.id, error)
    found.sort(key=lambda model: (model.provider, model.model or ""))
    unique: list[Model] = []
    for model in found:
        if not any(model.provider == kept.provider and model.model == kept.model for kept in unique):
            unique.append(model)
    return unique


__all__ = [
    "ApiError",
    "Language",
    "Model",
    "Quantization",
    "QuotaExceeded",
    "SegmentCountError",
    "TranslationContext",
    "TranslationError",
    "TranslationRequest",
    "TranslationSettings",
    "TranslatorProvider",
    "UnsupportedLanguage",
    "UnsupportedSourceLanguage",
    "languages",
    "models",
]
