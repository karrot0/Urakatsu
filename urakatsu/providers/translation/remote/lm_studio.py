"""LM Studio.

Chat goes to the OpenAI-compatible ``/v1/chat/completions`` with a strict ``json_schema``
``response_format``; reasoning maps to ``reasoning_effort`` ("medium" / "none"). Models
come from LM Studio's native ``/api/v1/models``. ``base_url`` defaults to
``http://localhost:1234``. The API key is optional: settings ``api_key`` or
``LM_STUDIO_API_KEY``; without one no ``Authorization`` header is sent.
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

DEFAULT_BASE_URL = "http://localhost:1234"
ENV_KEY = "LM_STUDIO_API_KEY"


class LmStudioSettings(TranslationSettings):
    api_key: Optional[str] = None
    base_url: Optional[str] = None


def endpoint(base_url: Optional[str], suffix: str) -> str:
    root = (base_url or DEFAULT_BASE_URL).rstrip("/")
    return root + "/" + suffix.lstrip("/")


def _auth_headers(key: Optional[str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"} if key is not None else {}


@register
class LmStudioTranslator(RemoteTranslator):
    id: ClassVar[str] = "lm-studio"
    name: ClassVar[str] = "LM Studio"
    stage: ClassVar[Stage] = Stage.TRANSLATION
    description: ClassVar[str] = "Local LM Studio server with a strict JSON-schema reply"
    Settings: ClassVar[type[TranslationSettings]] = LmStudioSettings
    env_key: ClassVar[Optional[str]] = ENV_KEY

    @classmethod
    def list_models(cls, settings: dict, transport: Optional[httpx.BaseTransport] = None) -> list[Model]:
        response = send_json_sync(
            cls.id,
            "GET",
            endpoint(settings.get("base_url"), "api/v1/models"),
            transport=transport,
            headers=_auth_headers(credential(settings, "api_key", ENV_KEY)),
        )
        # Every entry is decoded first (serde reads the whole list), then filtered.
        listed = [
            (field(cls.id, entry, "type"), field(cls.id, entry, "key"), entry.get("capabilities"))
            for entry in field(cls.id, response, "models")
        ]
        models = []
        for kind, key, capabilities in listed:
            if capabilities is not None:
                vision = bool(field(cls.id, capabilities, "vision"))
                reasoning = capabilities.get("reasoning") is not None
            else:
                vision = reasoning = False
            if kind not in ("llm", "vlm"):
                continue
            models.append(
                Model(
                    provider=cls.id,
                    provider_name=cls.name,
                    model=key,
                    name=display_name(key),
                    vision=vision,
                    reasoning=reasoning,
                )
            )
        return models

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
            endpoint(self.settings.base_url, "v1/chat/completions"),
            headers=_auth_headers(key),
            json_body=body,
        )
        choices = field(self.id, response, "choices")
        if not choices:
            raise TranslationError("LM Studio returned no choices")
        content = field(self.id, field(self.id, choices[0], "message"), "content")
        if content is None:
            raise TranslationError("LM Studio returned no message content")
        return prompt.translations(self.id, content, request.segments)


def _set(body: dict[str, Any], key: str, value: Any) -> None:
    # serde ``skip_serializing_if = "Option::is_none"``: unset options are omitted, not null.
    if value is not None:
        body[key] = value
