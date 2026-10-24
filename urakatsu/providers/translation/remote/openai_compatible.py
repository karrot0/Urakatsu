"""OpenAI-compatible chat completions.

Structured output: ``response_format`` = ``json_schema`` (strict) with the shared
translation schema. Reasoning maps to ``reasoning_effort`` ("medium" / "none").
``base_url`` defaults to a local Ollama (``http://localhost:11434/v1``). The API key is
optional: settings ``api_key`` or ``OPENAI_COMPATIBLE_API_KEY``; without one no
``Authorization`` header is sent.
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

DEFAULT_BASE_URL = "http://localhost:11434/v1"
ENV_KEY = "OPENAI_COMPATIBLE_API_KEY"


class OpenAiCompatibleSettings(TranslationSettings):
    api_key: Optional[str] = None
    base_url: Optional[str] = None


def endpoint(base_url: Optional[str], suffix: str) -> str:
    root = (base_url or DEFAULT_BASE_URL).rstrip("/")
    return root + "/" + suffix.lstrip("/")


def _auth_headers(key: Optional[str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"} if key is not None else {}


@register
class OpenAiCompatibleTranslator(RemoteTranslator):
    id: ClassVar[str] = "openai-compatible"
    name: ClassVar[str] = "OpenAI-compatible"
    stage: ClassVar[Stage] = Stage.TRANSLATION
    description: ClassVar[str] = "Any OpenAI-compatible chat completions server with a strict JSON-schema reply"
    Settings: ClassVar[type[TranslationSettings]] = OpenAiCompatibleSettings
    env_key: ClassVar[Optional[str]] = ENV_KEY

    @classmethod
    def list_models(cls, settings: dict, transport: Optional[httpx.BaseTransport] = None) -> list[Model]:
        response = send_json_sync(
            cls.id,
            "GET",
            endpoint(settings.get("base_url"), "models"),
            transport=transport,
            headers=_auth_headers(credential(settings, "api_key", ENV_KEY)),
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
        ]

    async def _translate(self, request: TranslationRequest, generation: GenerationConfig) -> list[str]:
        key = self.optional_api_key()
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
        _set(body, "max_tokens", generation.max_tokens)
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
            endpoint(self.settings.base_url, "chat/completions"),
            headers=_auth_headers(key),
            json_body=body,
        )
        choices = field(self.id, response, "choices")
        if not choices:
            raise TranslationError("OpenAI-compatible backend returned no choices")
        content = field(self.id, field(self.id, choices[0], "message"), "content")
        if content is None:
            raise TranslationError("OpenAI-compatible backend returned no message content")
        return prompt.translations(self.id, content, request.segments)


def _set(body: dict[str, Any], key: str, value: Any) -> None:
    # serde ``skip_serializing_if = "Option::is_none"``: unset options are omitted, not null.
    if value is not None:
        body[key] = value
