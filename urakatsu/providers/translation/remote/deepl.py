"""DeepL translation API.

Form-encoded ``POST {root}/v2/translate``. Free-tier keys (ending in ``:fx``) use
``api-free.deepl.com``; others use ``api.deepl.com``; ``base_url`` overrides both.
Source-language context is sent as DeepL ``context``: it informs the translation
and is not itself translated. API key: settings ``api_key`` or ``DEEPL_API_KEY``.
"""

from __future__ import annotations

from typing import ClassVar, Optional

import httpx

from urakatsu.core.config import GenerationConfig
from urakatsu.core.registry import register
from urakatsu.core.stage import Stage

from ..backend import TranslationRequest
from ..base import TranslationSettings
from ..errors import UnsupportedLanguage, UnsupportedSourceLanguage
from ..language import Language
from ..model import Model, service_model
from .base import RemoteTranslator, credential, field

_TARGETS: dict[Language, str] = {
    Language.CHINESE_SIMPLIFIED: "ZH-HANS",
    Language.CHINESE_TRADITIONAL: "ZH-HANT",
    Language.ENGLISH: "EN-US",
    Language.FRENCH: "FR",
    Language.PORTUGUESE: "PT-PT",
    Language.BRAZILIAN_PORTUGUESE: "PT-BR",
    Language.SPANISH: "ES",
    Language.JAPANESE: "JA",
    Language.TURKISH: "TR",
    Language.RUSSIAN: "RU",
    Language.ARABIC: "AR",
    Language.KOREAN: "KO",
    Language.THAI: "TH",
    Language.ITALIAN: "IT",
    Language.GERMAN: "DE",
    Language.VIETNAMESE: "VI",
    Language.MALAY: "MS",
    Language.INDONESIAN: "ID",
    Language.HINDI: "HI",
    Language.POLISH: "PL",
    Language.CZECH: "CS",
    Language.DUTCH: "NL",
    Language.GUJARATI: "GU",
    Language.URDU: "UR",
    Language.TELUGU: "TE",
    Language.MARATHI: "MR",
    Language.HEBREW: "HE",
    Language.BENGALI: "BN",
    Language.BULGARIAN: "BG",
    Language.TAMIL: "TA",
    Language.UKRAINIAN: "UK",
    Language.KAZAKH: "KK",
    Language.BELARUSIAN: "BE",
    Language.HUNGARIAN: "HU",
}


def target_code(language: Language) -> Optional[str]:
    return _TARGETS.get(language)


def source_code(language: Language) -> Optional[str]:
    if language in (Language.CHINESE_SIMPLIFIED, Language.CHINESE_TRADITIONAL):
        return "ZH"
    if language is Language.ENGLISH:
        return "EN"
    if language in (Language.PORTUGUESE, Language.BRAZILIAN_PORTUGUESE):
        return "PT"
    code = target_code(language)
    return code.split("-", 1)[0] if code else None


class DeepLSettings(TranslationSettings):
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@register
class DeepLTranslator(RemoteTranslator):
    id: ClassVar[str] = "deepl"
    name: ClassVar[str] = "DeepL"
    stage: ClassVar[Stage] = Stage.TRANSLATION
    description: ClassVar[str] = "DeepL text translation (no image or structured reply)"
    Settings: ClassVar[type[TranslationSettings]] = DeepLSettings
    env_key: ClassVar[Optional[str]] = "DEEPL_API_KEY"

    @classmethod
    def list_models(cls, settings: dict, transport: Optional[httpx.BaseTransport] = None) -> list[Model]:
        if credential(settings, "api_key", "DEEPL_API_KEY") is None:
            return []
        return [service_model(cls.id, cls.name, "DeepL")]

    async def _translate(self, request: TranslationRequest, generation: GenerationConfig) -> list[str]:
        api_key = self.api_key()
        target = target_code(request.target_language)
        if target is None:
            raise UnsupportedLanguage(self.id, request.target_language.display_name)
        source = None
        if request.source_language is not None:
            source = source_code(request.source_language)
            if source is None:
                raise UnsupportedSourceLanguage(self.id, request.source_language.display_name)
        if self.settings.base_url:
            root = self.settings.base_url.rstrip("/")
        elif api_key.rstrip().endswith(":fx"):
            root = "https://api-free.deepl.com"
        else:
            root = "https://api.deepl.com"
        # DeepL accepts source-language context but does not translate it.
        context = "\n".join(entry.source for entry in request.context)
        form: list[tuple[str, str]] = [("text", text) for text in request.segments]
        form.append(("target_lang", target))
        if source is not None:
            form.append(("source_lang", source))
        if context:
            form.append(("context", context))
        response = await self.post(
            f"{root}/v2/translate",
            headers={"Authorization": f"DeepL-Auth-Key {api_key}"},
            form=form,
        )
        translations = field(self.id, response, "translations")
        return [field(self.id, translation, "text") for translation in translations]
