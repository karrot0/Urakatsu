"""OpenAI-compatible chat completions.

Structured output: ``response_format`` = ``json_schema`` (strict) with the shared
translation schema. Reasoning maps to ``reasoning_effort`` ("medium" / "none").
``base_url`` defaults to a local Ollama (``http://localhost:11434/v1``). The API key is
optional: settings ``api_key`` or ``OPENAI_COMPATIBLE_API_KEY``; without one no
``Authorization`` header is sent.

Presets (not in Koharu, which has one OpenAI-compatible endpoint): every
``providers."openai-compatible:<slug>"`` table in the settings is a provider of its own,
with a ``name``, ``base_url`` and ``api_key``, so several servers (Ollama, vLLM, Groq, a
proxy) can be set up side by side and picked like any other provider. A preset has no
environment-variable key fallback, and its key rotation is its own.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping, Optional

import httpx

from urakatsu.core.config import GenerationConfig
from urakatsu.core.registry import Registry, register, registry
from urakatsu.core.stage import Stage

from .. import prompt
from ..backend import TranslationRequest, encode_jpeg_data_url
from ..base import TranslationSettings
from ..errors import TranslationError
from ..model import Model, display_name
from .base import RemoteTranslator, credential, field, send_json_sync

DEFAULT_BASE_URL = "http://localhost:11434/v1"
ENV_KEY = "OPENAI_COMPATIBLE_API_KEY"
PRESET_PREFIX = "openai-compatible:"


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
            headers=_auth_headers(credential(settings, "api_key", cls.env_key or "")),
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

    def base_url(self) -> Optional[str]:
        return self.settings.base_url

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
            endpoint(self.base_url(), "chat/completions"),
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


def preset(provider_id: str, name: str) -> type[OpenAiCompatibleTranslator]:
    """The provider class for one preset: the OpenAI-compatible provider under its own id and name."""
    return type(
        "OpenAiCompatiblePreset",
        (OpenAiCompatibleTranslator,),
        {
            "id": provider_id,
            "name": name,
            "description": "OpenAI-compatible preset",
            "env_key": None,
            "__module__": __name__,
        },
    )


def preset_name(provider_id: str, settings: Any) -> str:
    name = settings.get("name") if isinstance(settings, Mapping) else None
    return name.strip() if isinstance(name, str) and name.strip() else provider_id[len(PRESET_PREFIX) :]


@registry.syncer
def sync_presets(target: Registry, providers: Mapping[str, Any]) -> None:
    """Registers a provider per preset table and removes presets the settings dropped."""
    wanted = {
        provider_id: preset_name(provider_id, settings)
        for provider_id, settings in providers.items()
        if provider_id.startswith(PRESET_PREFIX) and len(provider_id) > len(PRESET_PREFIX)
    }
    for cls in target.list(Stage.TRANSLATION):
        if cls.id.startswith(PRESET_PREFIX) and wanted.get(cls.id) != cls.name:
            target.remove(Stage.TRANSLATION, cls.id)
    registered = {cls.id for cls in target.list(Stage.TRANSLATION)}
    for provider_id, name in wanted.items():
        if provider_id not in registered:
            target.add(preset(provider_id, name))


def _set(body: dict[str, Any], key: str, value: Any) -> None:
    # serde ``skip_serializing_if = "Option::is_none"``: unset options are omitted, not null.
    if value is not None:
        body[key] = value
