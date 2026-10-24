"""DeepSeek chat completions.

Structured output: ``response_format`` = ``json_object`` (DeepSeek has no JSON-schema
mode; the schema is enforced by the prompt and the tolerant parser). Reasoning maps to
``thinking`` = ``{"type": "enabled" | "disabled"}``. Temperature defaults to 1.3.
API key: settings ``api_key`` or ``DEEPSEEK_API_KEY``.
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

URL = "https://api.deepseek.com/chat/completions"
MODELS_URL = "https://api.deepseek.com/models"
ENV_KEY = "DEEPSEEK_API_KEY"
DEFAULT_TEMPERATURE = 1.3


class DeepSeekSettings(TranslationSettings):
    api_key: Optional[str] = None


@register
class DeepSeekTranslator(RemoteTranslator):
    id: ClassVar[str] = "deepseek"
    name: ClassVar[str] = "DeepSeek"
    stage: ClassVar[Stage] = Stage.TRANSLATION
    description: ClassVar[str] = "DeepSeek chat models with JSON-object output and a thinking toggle"
    Settings: ClassVar[type[TranslationSettings]] = DeepSeekSettings
    env_key: ClassVar[Optional[str]] = ENV_KEY

    @classmethod
    def list_models(cls, settings: dict, transport: Optional[httpx.BaseTransport] = None) -> list[Model]:
        key = credential(settings, "api_key", ENV_KEY)
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
                vision=False,
                reasoning=True,
            )
            for model_id in (field(cls.id, entry, "id") for entry in entries)
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
        if generation.reasoning is not None:
            body["thinking"] = {"type": "enabled" if generation.reasoning else "disabled"}
        _set(body, "max_tokens", generation.max_tokens)
        body["response_format"] = {"type": "json_object"}
        temperature = generation.temperature if generation.temperature is not None else DEFAULT_TEMPERATURE
        _set(body, "temperature", temperature)
        _set(body, "top_p", generation.top_p)
        response = await self.post(URL, headers={"Authorization": f"Bearer {api_key}"}, json_body=body)
        choices = field(self.id, response, "choices")
        if not choices:
            raise TranslationError("DeepSeek returned no choices")
        message = field(self.id, choices[0], "message")
        if not isinstance(message, dict):
            raise TranslationError(f"failed to decode {self.id} response: invalid type for `message`")
        # ``content`` is an ``Option<String>``: absent and null both mean no content.
        content = message.get("content")
        if content is None:
            raise TranslationError("DeepSeek returned no message content")
        return prompt.translations(self.id, content, request.segments)


def _set(body: dict[str, Any], key: str, value: Any) -> None:
    # serde ``skip_serializing_if = "Option::is_none"``: unset options are omitted, not null.
    if value is not None:
        body[key] = value
