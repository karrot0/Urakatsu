"""Urakatsu Cloud: translation on urakatsu.com, metered against the account's plan.

The cloud serves OpenAI-compatible chat completions with a strict JSON-schema reply, so this is
the OpenAI-compatible provider pointed at the cloud. Its models are listed without a key, so the
picker shows them before the account is set up. The key (``api_key`` or
``URAKATSU_CLOUD_API_KEY``) is the same one the ``cloud`` runtime uses for vision models
(``urakatsu.remote.cloud``). Each request is one translated page; a reply of 402 means the
plan's quota and the credits are used up, and is raised as a quota error.
"""

from __future__ import annotations

from typing import ClassVar, Optional

import httpx
from pydantic import Field

from urakatsu.core.registry import register
from urakatsu.core.stage import Stage
from urakatsu.remote import cloud

from ..base import TranslationSettings
from ..errors import TranslationError
from ..model import Model
from .base import field, send_json_sync
from .openai_compatible import OpenAiCompatibleTranslator, endpoint


class UrakatsuCloudSettings(TranslationSettings):
    api_key: Optional[str] = None
    server_url: Optional[str] = Field(default=None, description="Leave empty for urakatsu.com. Set it to use a staging or local Urakatsu Cloud.")


@register
class UrakatsuCloudTranslator(OpenAiCompatibleTranslator):
    id: ClassVar[str] = cloud.PROVIDER
    name: ClassVar[str] = "Urakatsu Cloud"
    stage: ClassVar[Stage] = Stage.TRANSLATION
    description: ClassVar[str] = "Translation on urakatsu.com. The key also runs vision models on the cloud runtime."
    Settings: ClassVar[type[TranslationSettings]] = UrakatsuCloudSettings
    env_key: ClassVar[Optional[str]] = cloud.KEY_ENV

    @classmethod
    def list_models(cls, settings: dict, transport: Optional[httpx.BaseTransport] = None) -> list[Model]:
        try:
            response = send_json_sync(cls.id, "GET", endpoint(f"{cloud.url(settings)}/v1", "models"), transport=transport)
            entries = field(cls.id, response, "data")
            return [
                Model(
                    provider=cls.id,
                    provider_name=cls.name,
                    model=field(cls.id, entry, "id"),
                    name=entry.get("name") or field(cls.id, entry, "id"),
                    vision=bool(entry.get("vision", False)),
                    reasoning=bool(entry.get("reasoning", False)),
                )
                for entry in entries
            ]
        except TranslationError:
            return []

    def base_url(self) -> Optional[str]:
        return f"{cloud.url(self.settings.model_dump())}/v1"
