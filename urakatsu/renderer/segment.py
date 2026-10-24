"""Unicode line breaking with language-aware segmentation and optional hyphenation.

Port of ``segment.rs``.

Divergences (no exact Python twin for these ICU/hypher pieces):
- ICU ``LineSegmenter::new_auto`` is replaced by ``uniseg`` (UAX #14 line break
  boundaries). ICU's dictionary-based segmentation for Thai/Lao/Khmer/Burmese is not
  reproduced.
- ICU ``LineBreakWordOption::KeepAll`` for Korean is approximated: UAX #14 breaks
  between two Hangul syllables are suppressed; breaks at spaces and punctuation remain.
- hypher is replaced by ``pyphen`` (Hunspell/LibreOffice hyphenation dictionaries). Break
  positions can differ for some words. Bounds per language are copied from hypher's
  ``Lang::bounds``. Languages that pyphen does not ship are not hyphenated.
- Offsets are code-point indices into the ``str``, not UTF-8 byte offsets.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import jieba
import pyphen
import regex
from uniseg.linebreak import line_break_boundaries

from .script import _script_of

# ---- tables -----------------------------------------------------------------------------------

# hypher ``Lang::from_iso`` (two-letter codes) and the languages used in the port's tests.
# Value: (hypher language name, pyphen dictionary name or None).
_HYPHER_ISO: dict[str, str] = {
    "af": "Afrikaans", "sq": "Albanian", "as": "Assamese", "be": "Belarusian", "bn": "Bengali",
    "bg": "Bulgarian", "ca": "Catalan", "hr": "Croatian", "cs": "Czech", "da": "Danish",
    "nl": "Dutch", "en": "English", "et": "Estonian", "fi": "Finnish", "fr": "French",
    "gl": "Galician", "ka": "Georgian", "de": "German", "el": "Greek", "gu": "Gujarati",
    "hi": "Hindi", "hu": "Hungarian", "is": "Icelandic", "it": "Italian", "kn": "Kannada",
    "ku": "Kurmanji", "la": "Latin", "lt": "Lithuanian", "ml": "Malayalam", "mr": "Marathi",
    "mn": "Mongolian", "no": "Norwegian", "nb": "Norwegian", "nn": "Norwegian", "or": "Oriya",
    "pa": "Panjabi", "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "sa": "Sanskrit",
    "sr": "Serbian", "sk": "Slovak", "sl": "Slovenian", "es": "Spanish", "sv": "Swedish",
    "ta": "Tamil", "te": "Telugu", "tr": "Turkish", "tk": "Turkmen", "uk": "Ukrainian",
}

# hypher ``Lang::bounds`` (minimum prefix, minimum suffix); default (2, 2).
_HYPHER_BOUNDS: dict[str, tuple[int, int]] = {
    "Afrikaans": (1, 2), "Georgian": (1, 2), "Greek": (1, 1), "English": (2, 3),
    "Estonian": (2, 3), "Portuguese": (2, 3), "Slovak": (2, 3),
}

# hypher language name -> pyphen dictionary. Languages without a pyphen dictionary are absent.
_PYPHEN_NAME: dict[str, str] = {
    "Afrikaans": "af_ZA", "Albanian": "sq_AL", "Assamese": "as_IN", "Belarusian": "be_BY",
    "Bulgarian": "bg_BG", "Catalan": "ca", "Croatian": "hr_HR", "Czech": "cs_CZ",
    "Danish": "da_DK", "Dutch": "nl_NL", "English": "en_US", "Estonian": "et_EE",
    "French": "fr", "Galician": "gl", "German": "de_DE", "Greek": "el_GR",
    "Hungarian": "hu_HU", "Icelandic": "is", "Italian": "it_IT", "Kannada": "kn_IN",
    "Lithuanian": "lt", "Marathi": "mr_IN", "Mongolian": "mn_MN", "Norwegian": "nb_NO",
    "Oriya": "or_IN", "Panjabi": "pa_IN", "Polish": "pl_PL", "Portuguese": "pt_BR",
    "Russian": "ru_RU", "Sanskrit": "sa_IN", "Serbian": "sr_Latn", "Slovak": "sk_SK",
    "Slovenian": "sl_SI", "Spanish": "es", "Swedish": "sv", "Telugu": "te_IN",
    "Ukrainian": "uk_UA",
}

# Characters trimmed from a segment that ends in a mandatory break (segment.rs
# ``trim_mandatory_break_suffix``).
_TRIM_BREAK_CHARS = frozenset("\n\r\u0085  ")
# UAX #14 classes BK, CR, LF, NL: the characters that force a break (``is_mandatory``).
_MANDATORY_BREAK_CHARS = frozenset("\n\r\u000b\u000c\u0085  ")


@dataclass(frozen=True)
class LineBreakOpportunity:
    offset: int
    is_mandatory: bool


class LineBreakSuffix(Enum):
    HYPHEN = "-"

    def as_str(self) -> str:
        return self.value


@dataclass
class LineSegment:
    """A trimmed line segment ready for shaping (range excludes trailing mandatory breaks)."""

    range: tuple[int, int]
    next_offset: int
    is_mandatory: bool
    break_suffix: Optional[LineBreakSuffix] = None

    def text(self, source: str) -> str:
        return source[self.range[0] : self.range[1]]


@dataclass(frozen=True)
class HyphenationOptions:
    language: str  # hypher language name, e.g. "English"
    minimum_word_length: int
    minimum_prefix_length: int
    minimum_suffix_length: int

    @classmethod
    def new(cls, language: str, minimum_word_length: int) -> "HyphenationOptions":
        prefix, suffix = _HYPHER_BOUNDS.get(language, (2, 2))
        return cls(language, minimum_word_length, prefix, suffix)

    def with_fragment_bounds(self, minimum_prefix_length: int, minimum_suffix_length: int) -> "HyphenationOptions":
        return HyphenationOptions(
            self.language, self.minimum_word_length, minimum_prefix_length, minimum_suffix_length
        )


def hyphenation_bounds(language: str) -> tuple[int, int]:
    """``Lang::bounds``: (minimum prefix, minimum suffix) for a hypher language name."""
    return _HYPHER_BOUNDS.get(language, (2, 2))


def hyphenation_lang_from_tag(value: str) -> Optional[str]:
    """``hyphenation_lang_from_tag``: primary two-letter BCP 47 subtag to a hypher language."""
    lower = value.strip().lower()
    primary = lower.replace("_", "-").split("-")[0]
    if len(primary) != 2:
        return None
    return _HYPHER_ISO.get(primary)


def _contains_script(text: str, scripts: frozenset[str]) -> bool:
    return any(_script_of(ch) in scripts for ch in text)


def contains_korean(text: str) -> bool:
    return _contains_script(text, frozenset({"Hang"}))


def contains_chinese(text: str) -> bool:
    """``contains_chinese``: Han or Bopomofo present, and no kana or Hangul anywhere.

    The Rust fold returns ``None`` on the first kana or Hangul character, which the
    ``unwrap_or(false)`` turns into ``false`` for the whole string.
    """
    has_chinese = False
    for ch in text:
        script = _script_of(ch)
        if script in ("Hira", "Kana", "Hang"):
            return False
        if script in ("Hani", "Bopo"):
            has_chinese = True
    return has_chinese


def _chinese_word_ranges(text: str) -> list[tuple[int, int]]:
    """``chinese_word_ranges``: jieba words (HMM on) that contain Chinese characters."""
    ranges: list[tuple[int, int]] = []
    offset = 0
    for word in jieba.cut(text, HMM=True):
        start = offset
        end = start + len(word)
        offset = end
        if contains_chinese(word):
            ranges.append((start, end))
    return ranges


def _apply_protected_ranges(
    opportunities: list[LineBreakOpportunity], protected: list[tuple[int, int]]
) -> list[LineBreakOpportunity]:
    if not protected:
        return opportunities
    return [
        op
        for op in opportunities
        if op.is_mandatory or not any(start < op.offset < end for start, end in protected)
    ]


def _trim_mandatory_break_suffix(text: str, start: int, end: int) -> int:
    trimmed_end = end
    while trimmed_end > start:
        ch = text[trimmed_end - 1]
        if ch not in _TRIM_BREAK_CHARS:
            break
        trimmed_end -= 1
    return trimmed_end


_ALPHABETIC = regex.compile(r"\p{Alphabetic}")


def _is_alphabetic(ch: str) -> bool:
    """``char::is_alphabetic``: the Unicode Alphabetic property (``str.isalpha`` misses Mc/Mn/Nl)."""
    return _ALPHABETIC.match(ch) is not None


def _hyphenatable_word_bounds(text: str) -> Optional[tuple[int, int]]:
    start = next((i for i, ch in enumerate(text) if _is_alphabetic(ch)), None)
    if start is None:
        return None
    end = next((i + 1 for i in range(len(text) - 1, -1, -1) if _is_alphabetic(text[i])), None)
    if end is None or start >= end:
        return None
    core = text[start:end]
    if not all(_is_alphabetic(ch) for ch in core):
        return None
    return start, end


def _syllables(word: str, config: HyphenationOptions) -> list[str]:
    """``hyphenate_bounded``: pyphen positions with hypher-equivalent fragment bounds."""
    dictionary = _PYPHEN_NAME.get(config.language)
    if dictionary is None or dictionary not in pyphen.LANGUAGES:
        return [word]
    hyphenator = pyphen.Pyphen(
        lang=dictionary,
        left=config.minimum_prefix_length,
        right=config.minimum_suffix_length,
    )
    positions = hyphenator.positions(word)
    pieces: list[str] = []
    previous = 0
    for position in positions:
        pieces.append(word[previous:position])
        previous = position
    pieces.append(word[previous:])
    return [piece for piece in pieces if piece]


class LineBreaker:
    """``LineBreaker``: UAX #14 opportunities, Chinese word protection, optional hyphenation."""

    def __init__(self) -> None:
        self.hyphenation: Optional[HyphenationOptions] = None

    def with_hyphenation(self, options: HyphenationOptions) -> "LineBreaker":
        self.hyphenation = options
        return self

    def line_break_opportunities(self, text: str) -> list[LineBreakOpportunity]:
        # ICU's segment_str yields the leading 0 boundary; uniseg yields only the breaks.
        boundaries = [0, *line_break_boundaries(text)]
        if contains_korean(text):
            boundaries = _keep_all_boundaries(text, boundaries)
        opportunities = []
        for offset in boundaries:
            is_mandatory = offset > 0 and text[offset - 1] in _MANDATORY_BREAK_CHARS
            opportunities.append(LineBreakOpportunity(offset, is_mandatory))
        if contains_chinese(text):
            return _apply_protected_ranges(opportunities, _chinese_word_ranges(text))
        return opportunities

    def line_segments(self, text: str) -> list[LineSegment]:
        opportunities = self.line_break_opportunities(text)
        segments: list[LineSegment] = []
        for start_op, end_op in zip(opportunities, opportunities[1:]):
            start = start_op.offset
            end = end_op.offset
            is_mandatory = end_op.is_mandatory
            segment_end = _trim_mandatory_break_suffix(text, start, end) if is_mandatory else end
            segment = LineSegment(
                range=(start, segment_end),
                next_offset=end,
                is_mandatory=is_mandatory,
                break_suffix=None,
            )
            segments.extend(self._hyphenated_segments(text, segment))
        return segments

    def _hyphenated_segments(self, text: str, segment: LineSegment) -> list[LineSegment]:
        config = self.hyphenation
        if config is None:
            return [segment]
        if segment.is_mandatory or segment.range[0] >= segment.range[1]:
            return [segment]

        segment_text = segment.text(text)
        bounds = _hyphenatable_word_bounds(segment_text)
        if bounds is None:
            return [segment]
        core_start, core_end = bounds
        core = segment_text[core_start:core_end]
        if len(core) < config.minimum_word_length:
            return [segment]

        syllables = _syllables(core, config)
        if len(syllables) <= 1:
            return [segment]

        result: list[LineSegment] = []
        word_offset = 0
        for index, syllable in enumerate(syllables):
            is_last = index + 1 == len(syllables)
            start = segment.range[0] if index == 0 else segment.range[0] + core_start + word_offset
            word_offset += len(syllable)
            end = segment.range[1] if is_last else segment.range[0] + core_start + word_offset
            result.append(
                LineSegment(
                    range=(start, end),
                    next_offset=segment.next_offset if is_last else end,
                    is_mandatory=False,
                    break_suffix=None if is_last else LineBreakSuffix.HYPHEN,
                )
            )
        return result


def _keep_all_boundaries(text: str, boundaries: list[int]) -> list[int]:
    """Approximates ICU ``KeepAll``: no break between two word characters when a Hangul one is involved.

    Word characters are letters and numbers (Unicode categories L* and N*); breaks at spaces and
    punctuation are kept.
    """
    def is_word(ch: str) -> bool:
        return unicodedata.category(ch)[0] in ("L", "N")

    kept: list[int] = []
    for offset in boundaries:
        if 0 < offset < len(text):
            before, after = text[offset - 1], text[offset]
            if is_word(before) and is_word(after) and (
                _script_of(before) == "Hang" or _script_of(after) == "Hang"
            ):
                continue
        kept.append(offset)
    return kept
