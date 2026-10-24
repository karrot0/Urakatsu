"""Gemini generateContent API.

``POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key=...``.
Structured output: ``generationConfig.responseMimeType`` = ``application/json`` with
``responseJsonSchema``. Reasoning maps to ``thinkingConfig`` (``thinkingBudget`` on
Gemini 2.5, ``thinkingLevel`` on other models). No retry. Models come from
``GET /v1beta/models``. API key: settings ``api_key`` or ``GEMINI_API_KEY``.
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional

import httpx

from urakatsu.core.config import GenerationConfig
from urakatsu.core.registry import register
from urakatsu.core.stage import Stage

from .. import prompt
from ..backend import TranslationRequest, encode_jpeg_data_url
from ..base import TranslationSettings
from ..errors import TranslationError
from ..model import Model
from .base import RemoteTranslator, credential, field, send_json_sync

ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
ENV_KEY = "GEMINI_API_KEY"
_EXCLUDED_MARKERS = (
    "antigravity",
    "computer-use",
    "deep-research",
    "embedding",
    "image",
    "imagen",
    "live",
    "lyria",
    "native-audio",
    "omni",
    "robotics",
    "tts",
    "veo",
)


class GeminiSettings(TranslationSettings):
    api_key: Optional[str] = None


def supports_translation(model_id: str) -> bool:
    """Excludes specialised ``generateContent`` models."""
    return not any(marker in model_id for marker in _EXCLUDED_MARKERS)


def _thinking_config(model: str, enabled: bool) -> dict[str, Any]:
    if model.startswith("gemini-2.5"):
        if enabled:
            budget = -1
        elif model.startswith("gemini-2.5-pro"):
            # Gemini 2.5 Pro cannot disable thinking; 128 is its minimum budget.
            budget = 128
        else:
            budget = 0
        return {"thinkingBudget": budget}
    if enabled:
        level = "high"
    elif model.startswith("gemma-4"):
        level = "minimal"
    else:
        level = "low"
    return {"thinkingLevel": level}


@register
class GeminiTranslator(RemoteTranslator):
    id: ClassVar[str] = "gemini"
    name: ClassVar[str] = "Gemini"
    stage: ClassVar[Stage] = Stage.TRANSLATION
    description: ClassVar[str] = "Google Gemini models with a JSON-schema structured reply"
    Settings: ClassVar[type[TranslationSettings]] = GeminiSettings
    env_key: ClassVar[Optional[str]] = ENV_KEY

    @classmethod
    def list_models(cls, settings: dict, transport: Optional[httpx.BaseTransport] = None) -> list[Model]:
        key = credential(settings, "api_key", ENV_KEY)
        if key is None:
            return []
        response = send_json_sync(
            cls.id, "GET", ROOT, transport=transport, params={"key": key, "pageSize": "1000"}
        )
        models = []
        for entry in field(cls.id, response, "models"):
            name = field(cls.id, entry, "name")
            display = field(cls.id, entry, "displayName")
            methods = field(cls.id, entry, "supportedGenerationMethods")
            if "generateContent" not in methods or not supports_translation(name):
                continue
            if not name.startswith("models/"):
                continue
            models.append(
                Model(
                    provider=cls.id,
                    provider_name=cls.name,
                    model=name[len("models/") :],
                    name=display,
                    vision=True,
                    reasoning=bool(entry.get("thinking", False)),
                )
            )
        return models

    async def _translate(self, request: TranslationRequest, generation: GenerationConfig) -> list[str]:
        api_key = self.api_key()
        model = self.model_id()
        system, user = prompt.prompts(request)
        schema = prompt.output_schema(len(request.segments))
        parts: list[dict[str, Any]] = [{"text": user}]
        if request.image is not None:
            parts.append(
                {
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": encode_jpeg_data_url(request.image).split(",", 1)[1],
                    }
                }
            )
        config: dict[str, Any] = {}
        # serde ``skip_serializing_if = "Option::is_none"``: unset options are omitted, not null.
        if generation.temperature is not None:
            config["temperature"] = generation.temperature
        if generation.max_tokens is not None:
            config["maxOutputTokens"] = generation.max_tokens
        if generation.reasoning is not None:
            config["thinkingConfig"] = _thinking_config(model, generation.reasoning)
        config["responseMimeType"] = "application/json"
        config["responseJsonSchema"] = schema
        # The request struct is not camel-cased in Koharu: only generation_config's members are.
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"parts": parts}],
            "generation_config": config,
        }
        response = await self.post(f"{ROOT}/{model}:generateContent", params={"key": api_key}, json_body=body)
        candidates = field(self.id, response, "candidates")
        # Every candidate and part must decode.
        decoded = [
            [field(self.id, part, "text") for part in field(self.id, field(self.id, candidate, "content"), "parts")]
            for candidate in candidates
        ]
        if not decoded or not decoded[0]:
            raise TranslationError("Gemini returned no candidate content")
        return prompt.translations(self.id, decoded[0][0], request.segments)
