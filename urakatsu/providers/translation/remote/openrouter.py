"""OpenRouter chat completions.

Structured output: ``response_format`` = ``json_schema`` (strict) with the shared
translation schema, plus ``provider.require_parameters`` so routing only picks
backends that honour it. Reasoning maps to ``reasoning: {enabled}``. API key:
settings ``api_key`` or ``OPENROUTER_API_KEY``.
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

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"
ENV_KEY = "OPENROUTER_API_KEY"


class OpenRouterSettings(TranslationSettings):
    api_key: Optional[str] = None


@register
class OpenRouterTranslator(RemoteTranslator):
    id: ClassVar[str] = "openrouter"
    name: ClassVar[str] = "OpenRouter"
    stage: ClassVar[Stage] = Stage.TRANSLATION
    description: ClassVar[str] = "OpenRouter-routed chat models with a strict JSON-schema reply"
    Settings: ClassVar[type[TranslationSettings]] = OpenRouterSettings
    env_key: ClassVar[Optional[str]] = ENV_KEY

    @classmethod
    def list_models(cls, settings: dict, transport: Optional[httpx.BaseTransport] = None) -> list[Model]:
        key = credential(settings, "api_key", ENV_KEY)
        if key is None:
            return []
        # Koharu logs a failed listing and returns no models rather than an error.
        try:
            response = send_json_sync(
                cls.id, "GET", MODELS_URL, transport=transport, headers={"Authorization": f"Bearer {key}"}
            )
            return [cls._listed(entry) for entry in field(cls.id, response, "data")]
        except TranslationError:
            return []

    @classmethod
    def _listed(cls, entry: Any) -> Model:
        model_id = field(cls.id, entry, "id")
        name = field(cls.id, entry, "name")
        modalities = field(cls.id, field(cls.id, entry, "architecture"), "input_modalities")
        parameters = field(cls.id, entry, "supported_parameters")
        return Model(
            provider=cls.id,
            provider_name=cls.name,
            model=model_id,
            name=name,
            vision="image" in modalities,
            reasoning=entry.get("reasoning") is not None or "reasoning" in parameters,
        )

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
        _set(body, "max_tokens", generation.max_tokens)
        _set(body, "frequency_penalty", generation.frequency_penalty)
        _set(body, "presence_penalty", generation.presence_penalty)
        if generation.reasoning is not None:
            body["reasoning"] = {"enabled": generation.reasoning}
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "manga_translation",
                "strict": True,
                "schema": prompt.output_schema(len(request.segments)),
            },
        }
        body["provider"] = {"require_parameters": True}
        response = await self.post(CHAT_URL, headers={"Authorization": f"Bearer {api_key}"}, json_body=body)
        choices = field(self.id, response, "choices")
        if not choices:
            raise TranslationError("OpenRouter returned no choices")
        content = field(self.id, field(self.id, choices[0], "message"), "content")
        if content is None:
            raise TranslationError("OpenRouter returned no message content")
        return prompt.translations(self.id, content, request.segments)


def _set(body: dict[str, Any], key: str, value: Any) -> None:
    # serde ``skip_serializing_if = "Option::is_none"``: unset options are omitted, not null.
    if value is not None:
        body[key] = value
