"""xAI Grok Responses API.

Structured output: ``text.format`` = ``json_schema`` (strict) with the shared translation
schema. Reasoning on maps to ``reasoning.effort`` = ``high``; off or unset omits it.
``store`` is always false. API key: settings ``api_key`` or ``XAI_API_KEY``.
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

RESPONSES_URL = "https://api.x.ai/v1/responses"
MODELS_URL = "https://api.x.ai/v1/language-models"
ENV_KEY = "XAI_API_KEY"


class GrokSettings(TranslationSettings):
    api_key: Optional[str] = None


@register
class GrokTranslator(RemoteTranslator):
    id: ClassVar[str] = "grok"
    name: ClassVar[str] = "Grok"
    stage: ClassVar[Stage] = Stage.TRANSLATION
    description: ClassVar[str] = "xAI Grok models through the Responses API with a strict JSON-schema reply"
    Settings: ClassVar[type[TranslationSettings]] = GrokSettings
    env_key: ClassVar[Optional[str]] = ENV_KEY

    @classmethod
    def list_models(cls, settings: dict, transport: Optional[httpx.BaseTransport] = None) -> list[Model]:
        key = credential(settings, "api_key", ENV_KEY)
        if key is None:
            return []
        response = send_json_sync(
            cls.id, "GET", MODELS_URL, transport=transport, headers={"Authorization": f"Bearer {key}"}
        )
        models = []
        for entry in field(cls.id, response, "models"):
            model_id = field(cls.id, entry, "id")
            modalities = field(cls.id, entry, "input_modalities")
            models.append(
                Model(
                    provider=cls.id,
                    provider_name=cls.name,
                    model=model_id,
                    name=display_name(model_id),
                    vision=any(modality == "image" for modality in modalities),
                    reasoning=True,
                )
            )
        return models

    async def _translate(self, request: TranslationRequest, generation: GenerationConfig) -> list[str]:
        api_key = self.api_key()
        system, user = prompt.prompts(request)
        if request.image is not None:
            user_content: Any = [
                {"type": "input_text", "text": user},
                {"type": "input_image", "image_url": encode_jpeg_data_url(request.image)},
            ]
        else:
            user_content = user
        body: dict[str, Any] = {
            "model": self.model_id(),
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "store": False,
        }
        _set(body, "temperature", generation.temperature)
        _set(body, "top_p", generation.top_p)
        _set(body, "max_output_tokens", generation.max_tokens)
        if generation.reasoning:
            body["reasoning"] = {"effort": "high"}
        body["text"] = {
            "format": {
                "type": "json_schema",
                "name": "manga_translation",
                "strict": True,
                "schema": prompt.output_schema(len(request.segments)),
            }
        }
        response = await self.post(RESPONSES_URL, headers={"Authorization": f"Bearer {api_key}"}, json_body=body)
        # Every output item is decoded before the search, as serde does.
        contents = []
        for item in field(self.id, response, "output"):
            if not isinstance(item, dict):
                raise TranslationError(f"failed to decode {self.id} response: invalid type for output item")
            for content in item.get("content") or []:
                field(self.id, content, "type")
                contents.append(content)
        text = next(
            (
                content["text"]
                for content in contents
                if content["type"] == "output_text" and content.get("text") is not None
            ),
            None,
        )
        if text is None:
            raise TranslationError("Grok returned no output text")
        return prompt.translations(self.id, text, request.segments)


def _set(body: dict[str, Any], key: str, value: Any) -> None:
    # serde ``skip_serializing_if = "Option::is_none"``: unset options are omitted, not null.
    if value is not None:
        body[key] = value
