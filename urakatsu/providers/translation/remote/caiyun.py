"""Caiyun interpreter API.

``POST https://api.interpreter.caiyunai.com/v1/translator`` with
``X-Authorization: token <key>``. Plain text only: no image, no reply schema, no retry;
a non-zero ``rc`` fails immediately. API key: settings ``api_key`` or ``CAIYUN_API_KEY``.
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional

import httpx

from urakatsu.core.config import GenerationConfig
from urakatsu.core.registry import register
from urakatsu.core.stage import Stage

from ..backend import TranslationRequest
from ..base import TranslationSettings
from ..errors import TranslationError, UnsupportedLanguage
from ..language import Language
from ..model import Model, service_model
from .base import RemoteTranslator, credential

URL = "https://api.interpreter.caiyunai.com/v1/translator"
ENV_KEY = "CAIYUN_API_KEY"

_TARGETS: dict[Language, str] = {
    Language.CHINESE_SIMPLIFIED: "zh",
    Language.ENGLISH: "en",
    Language.FRENCH: "fr",
    Language.PORTUGUESE: "pt",
    Language.SPANISH: "es",
    Language.JAPANESE: "ja",
    Language.TURKISH: "tr",
    Language.RUSSIAN: "ru",
    Language.ARABIC: "ar",
    Language.KOREAN: "ko",
    Language.THAI: "th",
    Language.ITALIAN: "it",
    Language.GERMAN: "de",
    Language.VIETNAMESE: "vi",
    Language.INDONESIAN: "id",
    Language.CHINESE_TRADITIONAL: "zh-Hant",
    Language.POLISH: "pl",
}


class CaiyunSettings(TranslationSettings):
    api_key: Optional[str] = None


def target_code(language: Language) -> Optional[str]:
    """Unsupported languages are not substituted."""
    return _TARGETS.get(language)


def _decode_error(message: str) -> TranslationError:
    return TranslationError(f"failed to decode caiyun response: {message}")


def _optional_string(response: dict[str, Any], key: str) -> Optional[str]:
    value = response.get(key)
    if value is not None and not isinstance(value, str):
        raise _decode_error(f"invalid type for field `{key}`, expected a string")
    return value


@register
class CaiyunTranslator(RemoteTranslator):
    id: ClassVar[str] = "caiyun"
    name: ClassVar[str] = "Caiyun"
    stage: ClassVar[Stage] = Stage.TRANSLATION
    description: ClassVar[str] = "Caiyun interpreter text translation (no image or structured reply)"
    Settings: ClassVar[type[TranslationSettings]] = CaiyunSettings
    env_key: ClassVar[Optional[str]] = ENV_KEY

    @classmethod
    def list_models(cls, settings: dict, transport: Optional[httpx.BaseTransport] = None) -> list[Model]:
        if credential(settings, "api_key", ENV_KEY) is None:
            return []
        return [service_model(cls.id, cls.name, "Caiyun")]

    async def _translate(self, request: TranslationRequest, generation: GenerationConfig) -> list[str]:
        api_key = self.api_key()
        target = target_code(request.target_language)
        if target is None:
            raise UnsupportedLanguage(self.id, request.target_language.display_name)
        response = await self.post(
            URL,
            headers={"X-Authorization": f"token {api_key}"},
            json_body={
                "source": request.segments,
                "trans_type": f"auto2{target}",
                "request_id": "koharu-translator",
                "detect": True,
                "media": "text",
            },
        )
        if not isinstance(response, dict):
            raise _decode_error("invalid type: expected struct Response")
        rc = response.get("rc", 0)
        if isinstance(rc, bool) or not isinstance(rc, int):
            raise _decode_error("invalid type for field `rc`, expected i64")
        target_value = response.get("target")
        msg = _optional_string(response, "msg")
        message = _optional_string(response, "message")
        error = _optional_string(response, "error")
        if rc != 0:
            detail = next((value for value in (msg, message, error) if value is not None), "unknown error")
            raise TranslationError(f"Caiyun returned rc={rc}: {detail}")
        if target_value is None:
            raise TranslationError("Caiyun returned no target")
        if isinstance(target_value, str):
            return [target_value]
        if isinstance(target_value, list) and all(isinstance(text, str) for text in target_value):
            return list(target_value)
        raise _decode_error("data did not match any variant of untagged enum Target")
