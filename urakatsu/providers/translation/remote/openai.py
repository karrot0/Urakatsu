"""OpenAI chat completions.

Structured output: ``response_format`` = ``json_schema`` (strict) with the shared
translation schema. Reasoning maps to ``reasoning_effort`` ("medium" / "none").
API key: settings ``api_key`` or ``OPENAI_API_KEY``.
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
from ..model import Model, display_name
from .base import RemoteTranslator, credential, field, send_json_sync

URL = "https://api.openai.com/v1/chat/completions"
MODELS_URL = "https://api.openai.com/v1/models"


class OpenAiSettings(TranslationSettings):
    api_key: Optional[str] = None


def supports_translation(model_id: str) -> bool:
    """Chat models that accept the translation contract."""
    base = model_id
    if base.startswith("ft:"):
        base = base[len("ft:") :].split(":", 1)[0]
    base = base.lower()
    supported_family = (
        base.startswith("gpt-5")
        or base.startswith("gpt-4.1")
        or base.startswith("gpt-4o")
        or base == "o3"
        or base.startswith("o3-")
    )
    incompatible = any(
        marker in base
        for marker in (
            "-audio",
            "-codex",
            "-deep-research",
            "-image",
            "-pro",
            "-realtime",
            "-search",
            "-transcribe",
            "-tts",
        )
    )
    return supported_family and not incompatible


@register
class OpenAiTranslator(RemoteTranslator):
    id: ClassVar[str] = "openai"
    name: ClassVar[str] = "OpenAI"
    stage: ClassVar[Stage] = Stage.TRANSLATION
    description: ClassVar[str] = "OpenAI chat models with a strict JSON-schema reply"
    Settings: ClassVar[type[TranslationSettings]] = OpenAiSettings
    env_key: ClassVar[Optional[str]] = "OPENAI_API_KEY"

    @classmethod
    def list_models(cls, settings: dict, transport: Optional[httpx.BaseTransport] = None) -> list[Model]:
        key = credential(settings, "api_key", "OPENAI_API_KEY")
        if key is None:
            return []
        response = send_json_sync(
            cls.id, "GET", MODELS_URL, transport=transport, headers={"Authorization": f"Bearer {key}"}
        )
        entries = field(cls.id, response, "data")
        return [
            Model(
                provider=cls.id,
                provider_name=cls.name,
                model=model_id,
                name=display_name(model_id),
                vision=True,
                reasoning=True,
            )
            for model_id in (field(cls.id, entry, "id") for entry in entries)
            if supports_translation(model_id)
        ]

    async def _translate(self, request: TranslationRequest, generation: GenerationConfig) -> list[str]:
        api_key = self.api_key()
        system, user = prompt.prompts(request)
        if request.image is not None:
            user_content: Any = [
                {"type": "text", "text": user},
                {"type": "image_url", "image_url": {"url": encode_jpeg_data_url(request.image)}},
            ]
        else:
            user_content = user
        body: dict[str, Any] = {
            "model": self.model_id(),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        }
        _set(body, "temperature", generation.temperature)
        _set(body, "top_p", generation.top_p)
        _set(body, "max_completion_tokens", generation.max_tokens)
        _set(body, "frequency_penalty", generation.frequency_penalty)
        _set(body, "presence_penalty", generation.presence_penalty)
        if generation.reasoning is not None:
            body["reasoning_effort"] = "medium" if generation.reasoning else "none"
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "manga_translation",
                "strict": True,
                "schema": prompt.output_schema(len(request.segments)),
            },
        }
        response = await self.post(
            URL, headers={"Authorization": f"Bearer {api_key}"}, json_body=body
        )
        choices = field(self.id, response, "choices")
        if not choices:
            raise TranslationError("OpenAI returned no choices")
        # serde ``Option<String>``: an absent or null content is "no content", not a decode error.
        content = field(self.id, choices[0], "message").get("content")
        if content is None:
            raise TranslationError("OpenAI returned no message content")
        return prompt.translations(self.id, content, request.segments)


def _set(body: dict[str, Any], key: str, value: Any) -> None:
    # serde ``skip_serializing_if = "Option::is_none"``: unset options are omitted, not null.
    if value is not None:
        body[key] = value
