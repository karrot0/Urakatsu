"""MiniMax native chat completion v2.

Uses MiniMax's native route, not the compatibility APIs. Structured output:
``response_format`` = ``json_schema`` (strict), sent only for ``MiniMax-Text-01``; other
models omit it. The service reports failures in ``base_resp`` on an HTTP 200, so a
non-zero ``status_code`` is an error. API key: settings ``api_key`` or ``MINIMAX_API_KEY``.
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional

import httpx

from urakatsu.core.config import GenerationConfig
from urakatsu.core.registry import register
from urakatsu.core.stage import Stage

from .. import prompt
from ..backend import TranslationRequest
from ..base import TranslationSettings
from ..errors import TranslationError
from ..model import Model, display_name
from .base import RemoteTranslator, credential, field, send_json_sync

URL = "https://api.minimax.io/v1/text/chatcompletion_v2"
MODELS_URL = "https://api.minimax.io/v1/models"
ENV_KEY = "MINIMAX_API_KEY"
SCHEMA_MODEL = "MiniMax-Text-01"


class MiniMaxSettings(TranslationSettings):
    api_key: Optional[str] = None


@register
class MiniMaxTranslator(RemoteTranslator):
    id: ClassVar[str] = "minimax"
    name: ClassVar[str] = "MiniMax"
    stage: ClassVar[Stage] = Stage.TRANSLATION
    description: ClassVar[str] = "MiniMax chat models through the native completion route"
    Settings: ClassVar[type[TranslationSettings]] = MiniMaxSettings
    env_key: ClassVar[Optional[str]] = ENV_KEY

    @classmethod
    def list_models(cls, settings: dict, transport: Optional[httpx.BaseTransport] = None) -> list[Model]:
        key = credential(settings, "api_key", ENV_KEY)
        if key is None:
            return []
        response = send_json_sync(
            cls.id, "GET", MODELS_URL, transport=transport, headers={"Authorization": f"Bearer {key}"}
        )
        return [
            Model(
                provider=cls.id,
                provider_name=cls.name,
                model=model_id,
                name=display_name(model_id),
                vision=False,
                reasoning=True,
            )
            for model_id in (field(cls.id, entry, "id") for entry in field(cls.id, response, "data"))
        ]

    async def _translate(self, request: TranslationRequest, generation: GenerationConfig) -> list[str]:
        api_key = self.api_key()
        system, user = prompt.prompts(request)
        model = self.model_id()
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        _set(body, "temperature", generation.temperature)
        _set(body, "top_p", generation.top_p)
        _set(body, "max_completion_tokens", generation.max_tokens)
        if model == SCHEMA_MODEL:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "manga_translation",
                    "strict": True,
                    "schema": prompt.output_schema(len(request.segments)),
                },
            }
        response = await self.post(URL, headers={"Authorization": f"Bearer {api_key}"}, json_body=body)
        if not isinstance(response, dict):
            raise TranslationError(f"failed to decode {self.id} response: invalid type, expected struct Response")
        # ``base_resp`` may be absent (treated as status 0); inside it ``status_code`` is required.
        if "base_resp" in response:
            base = response["base_resp"]
            status = field(self.id, base, "status_code")
            if status != 0:
                raise TranslationError(f"MiniMax returned status {status}: {base.get('status_msg', '')}")
        choices = response.get("choices", [])
        if not choices:
            raise TranslationError("MiniMax returned no choices")
        content = field(self.id, field(self.id, choices[0], "message"), "content")
        if not isinstance(content, str):
            raise TranslationError(f"failed to decode {self.id} response: invalid type for `content`, expected a string")
        return prompt.translations(self.id, content, request.segments)


def _set(body: dict[str, Any], key: str, value: Any) -> None:
    # serde ``skip_serializing_if = "Option::is_none"``: unset options are omitted, not null.
    if value is not None:
        body[key] = value
