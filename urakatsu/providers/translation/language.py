"""Translation languages.

Each language has a display name (``str(language)``), a
canonical BCP-47-like tag used on the wire (``tag()`` and serde form),
and the aliases ``parse`` accepts. Parsing is case-insensitive over names, tags
and aliases, exactly like strum's ``ascii_case_insensitive`` EnumString.
"""

from __future__ import annotations

from enum import Enum


class Language(Enum):
    # value: (display name, canonical tag, extra parse aliases); declaration order is the display order.
    CHINESE_SIMPLIFIED = ("Simplified Chinese", "zh-CN", ("zh", "zh-Hans"))
    ENGLISH = ("English", "en-US", ("en",))
    FRENCH = ("French", "fr-FR", ("fr",))
    PORTUGUESE = ("Portuguese", "pt-PT", ("pt",))
    BRAZILIAN_PORTUGUESE = ("Brazilian Portuguese", "pt-BR", ())
    SPANISH = ("Spanish", "es-ES", ("es",))
    JAPANESE = ("Japanese", "ja-JP", ("ja",))
    TURKISH = ("Turkish", "tr-TR", ("tr",))
    RUSSIAN = ("Russian", "ru-RU", ("ru",))
    ARABIC = ("Arabic", "ar-SA", ("ar",))
    KOREAN = ("Korean", "ko-KR", ("ko",))
    THAI = ("Thai", "th-TH", ("th",))
    ITALIAN = ("Italian", "it-IT", ("it",))
    GERMAN = ("German", "de-DE", ("de",))
    VIETNAMESE = ("Vietnamese", "vi-VN", ("vi",))
    MALAY = ("Malay", "ms-MY", ("ms",))
    INDONESIAN = ("Indonesian", "id-ID", ("id",))
    FILIPINO = ("Filipino", "fil-PH", ("fil", "tl"))
    HINDI = ("Hindi", "hi-IN", ("hi",))
    CHINESE_TRADITIONAL = ("Traditional Chinese", "zh-TW", ("zh-Hant",))
    POLISH = ("Polish", "pl-PL", ("pl",))
    CZECH = ("Czech", "cs-CZ", ("cs",))
    DUTCH = ("Dutch", "nl-NL", ("nl",))
    KHMER = ("Khmer", "km-KH", ("km",))
    BURMESE = ("Burmese", "my-MM", ("my",))
    PERSIAN = ("Persian", "fa-IR", ("fa",))
    GUJARATI = ("Gujarati", "gu-IN", ("gu",))
    URDU = ("Urdu", "ur-PK", ("ur",))
    TELUGU = ("Telugu", "te-IN", ("te",))
    MARATHI = ("Marathi", "mr-IN", ("mr",))
    HEBREW = ("Hebrew", "he-IL", ("he",))
    BENGALI = ("Bengali", "bn-BD", ("bn",))
    BULGARIAN = ("Bulgarian", "bg-BG", ("bg",))
    TAMIL = ("Tamil", "ta-IN", ("ta",))
    UKRAINIAN = ("Ukrainian", "uk-UA", ("uk",))
    TIBETAN = ("Tibetan", "bo-CN", ("bo",))
    KAZAKH = ("Kazakh", "kk-KZ", ("kk",))
    MONGOLIAN = ("Mongolian", "mn-MN", ("mn",))
    UYGHUR = ("Uyghur", "ug-CN", ("ug",))
    CANTONESE = ("Cantonese", "yue-HK", ("yue",))
    BELARUSIAN = ("Belarusian", "be-BY", ("be",))
    HUNGARIAN = ("Hungarian", "hu-HU", ("hu",))

    @property
    def display_name(self) -> str:
        return self.value[0]

    @property
    def tag(self) -> str:
        return self.value[1]

    @property
    def aliases(self) -> tuple[str, ...]:
        return self.value[2]

    def __str__(self) -> str:
        return self.display_name

    @classmethod
    def parse(cls, text: str) -> "Language":
        """``str::parse::<Language>``: case-insensitive over display names, tags and aliases."""
        wanted = text.lower()
        for language in cls:
            names = (language.display_name, language.tag, *language.aliases)
            if any(name.lower() == wanted for name in names):
                return language
        raise ValueError(f"not-a-language: {text!r}")

    @classmethod
    def coerce(cls, value: "Language | str") -> "Language":
        return value if isinstance(value, Language) else cls.parse(value)


def languages() -> list[tuple[str, str]]:
    """``(tag, name)`` for every language, in declaration order."""
    return [(language.tag, language.display_name) for language in Language]
