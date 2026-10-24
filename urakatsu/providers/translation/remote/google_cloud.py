"""Google Cloud Translation v2.

``POST https://translation.googleapis.com/language/translate/v2?key=...`` with a JSON
body ``{q, target, source?, format: "text"}``. Plain text only: no image, no reply
schema. API key: settings ``api_key`` or ``GOOGLE_CLOUD_TRANSLATION_API_KEY``.
"""

from __future__ import annotations

from typing import ClassVar, Optional

import httpx

from urakatsu.core.config import GenerationConfig
from urakatsu.core.registry import register
from urakatsu.core.stage import Stage

from ..backend import TranslationRequest
from ..base import TranslationSettings
from ..model import Model, service_model
from .base import RemoteTranslator, credential, field

URL = "https://translation.googleapis.com/language/translate/v2"
ENV_KEY = "GOOGLE_CLOUD_TRANSLATION_API_KEY"


class GoogleCloudSettings(TranslationSettings):
    api_key: Optional[str] = None


@register
class GoogleCloudTranslator(RemoteTranslator):
    id: ClassVar[str] = "google-cloud-translation"
    name: ClassVar[str] = "Google Cloud Translation"
    stage: ClassVar[Stage] = Stage.TRANSLATION
    description: ClassVar[str] = "Google Cloud Translation v2 (plain text, no image)"
    Settings: ClassVar[type[TranslationSettings]] = GoogleCloudSettings
    env_key: ClassVar[Optional[str]] = ENV_KEY

    @classmethod
    def list_models(cls, settings: dict, transport: Optional[httpx.BaseTransport] = None) -> list[Model]:
        if credential(settings, "api_key", ENV_KEY) is None:
            return []
        return [service_model(cls.id, cls.name, cls.name)]

    async def _translate(self, request: TranslationRequest, generation: GenerationConfig) -> list[str]:
        api_key = self.api_key()
        body: dict[str, object] = {
            "q": request.segments,
            "target": request.target_language.tag,
            "format": "text",
        }
        if request.source_language is not None:
            body["source"] = request.source_language.tag
        response = await self.post(URL, params={"key": api_key}, json_body=body)
        data = field(self.id, response, "data")
        translations = field(self.id, data, "translations")
        return [field(self.id, translation, "translatedText") for translation in translations]
