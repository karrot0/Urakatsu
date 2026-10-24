"""Claude messages API.

``POST https://api.anthropic.com/v1/messages`` with ``x-api-key`` and
``anthropic-version: 2023-06-01``. Structured output: ``output_config.format`` =
``json_schema`` with the shared translation schema. Reasoning maps to ``thinking``
(``adaptive`` / ``disabled``). No retry. Models come from ``GET /v1/models``.
API key: settings ``api_key`` or ``ANTHROPIC_API_KEY``.
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

URL = "https://api.anthropic.com/v1/messages"
MODELS_URL = "https://api.anthropic.com/v1/models?limit=1000"
ENV_KEY = "ANTHROPIC_API_KEY"
API_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 8192


class ClaudeSettings(TranslationSettings):
    api_key: Optional[str] = None


def _headers(api_key: str) -> dict[str, str]:
    return {"x-api-key": api_key, "anthropic-version": API_VERSION}


@register
class ClaudeTranslator(RemoteTranslator):
    id: ClassVar[str] = "claude"
    name: ClassVar[str] = "Claude"
    stage: ClassVar[Stage] = Stage.TRANSLATION
    description: ClassVar[str] = "Anthropic Claude models with a JSON-schema structured reply"
    Settings: ClassVar[type[TranslationSettings]] = ClaudeSettings
    env_key: ClassVar[Optional[str]] = ENV_KEY

    @classmethod
    def list_models(cls, settings: dict, transport: Optional[httpx.BaseTransport] = None) -> list[Model]:
        key = credential(settings, "api_key", ENV_KEY)
        if key is None:
            return []
        response = send_json_sync(cls.id, "GET", MODELS_URL, transport=transport, headers=_headers(key))
        models = []
        for entry in field(cls.id, response, "data"):
            capabilities = field(cls.id, entry, "capabilities")
            vision = field(cls.id, field(cls.id, capabilities, "image_input"), "supported")
            thinking = field(cls.id, field(cls.id, capabilities, "thinking"), "types")
            reasoning = field(cls.id, field(cls.id, thinking, "adaptive"), "supported")
            models.append(
                Model(
                    provider=cls.id,
                    provider_name=cls.name,
                    model=field(cls.id, entry, "id"),
                    name=field(cls.id, entry, "display_name"),
                    vision=bool(vision),
                    reasoning=bool(reasoning),
                )
            )
        return models

    async def _translate(self, request: TranslationRequest, generation: GenerationConfig) -> list[str]:
        api_key = self.api_key()
        system, user = prompt.prompts(request)
        content: list[dict[str, Any]] = [{"type": "text", "text": user}]
        if request.image is not None:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": encode_jpeg_data_url(request.image).split(",", 1)[1],
                    },
                }
            )
        body: dict[str, Any] = {
            "model": self.model_id(),
            "max_tokens": generation.max_tokens if generation.max_tokens is not None else DEFAULT_MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": content}],
        }
        # serde ``skip_serializing_if = "Option::is_none"``: unset options are omitted, not null.
        if generation.temperature is not None:
            body["temperature"] = generation.temperature
        if generation.reasoning is not None:
            body["thinking"] = {"type": "adaptive" if generation.reasoning else "disabled"}
        body["output_config"] = {
            "format": {"type": "json_schema", "schema": prompt.output_schema(len(request.segments))}
        }
        response = await self.post(URL, headers=_headers(api_key), json_body=body)
        text = None
        for block in field(self.id, response, "content"):
            kind = field(self.id, block, "type")
            if text is None and kind == "text":
                text = block.get("text")
        if text is None:
            raise TranslationError("Claude returned no text content")
        return prompt.translations(self.id, text, request.segments)
