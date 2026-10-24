"""Unicode script detection and shaping direction.

Port of ``script.rs``.

Divergences:
- ICU ``CodePointMapData<Script>`` is replaced by ``fontTools.unicodedata.script``,
  which reads the same Unicode Scripts property and returns ISO 15924 short names
  (``Hani``, ``Latn``, ``Zyyy`` for Common, ``Zinh`` for Inherited).
- ``unicode_bidi::BidiInfo`` paragraph level is replaced by ``bidi.algorithm.get_base_level``
  (P2/P3 first-strong rule, default LTR), which is the same rule upstream relies on.
"""

from __future__ import annotations

from typing import Optional

from bidi.algorithm import get_base_level
from fontTools.unicodedata import script as _unicode_script

from .types import WritingMode

# ICU short names used by the Rust code.
COMMON = "Zyyy"
INHERITED = "Zinh"
HAN = "Hani"
HIRAGANA = "Hira"
KATAKANA = "Kana"
BOPOMOFO = "Bopo"


def _script_of(character: str) -> str:
    return _unicode_script(character)


def shaping_direction_for_text(text: str, writing_mode: WritingMode) -> tuple[str, Optional[str]]:
    """``shaping_direction_for_text``: returns (harfbuzz direction, harfbuzz script tag or None)."""
    if writing_mode.is_vertical():
        return "ttb", dominant_script(text)
    direction = "rtl" if get_base_level(text) == 1 else "ltr"
    return direction, dominant_script(text)


def fontique_scripts(text: str) -> list[str]:
    """``fontique_scripts``: distinct ISO 15924 scripts used for fallback selection.

    Keeps the Rust rule that Hiragana/Katakana/Bopomofo select Han fonts, and that
    emoji (supplementary-plane Common/Inherited characters) add ``Zsye``.
    """
    scripts: list[str] = []
    has_emoji = False
    for character in text:
        script = _script_of(character)
        if script in (COMMON, INHERITED):
            has_emoji |= character >= "\U0001F000"
            continue
        script = fallback_script(script)
        if len(script) != 4:
            continue
        if script not in scripts:
            scripts.append(script)
    if has_emoji and "Zsye" not in scripts:
        scripts.append("Zsye")
    if not scripts:
        scripts.append("Latn")
    return scripts


def harfbuzz_script(script: str) -> Optional[str]:
    """``harfrust_script``: the HarfBuzz script tag for an ICU short name (after fallback)."""
    fallback = fallback_script(script)
    return fallback if len(fallback) == 4 else None


def dominant_script(text: str) -> Optional[str]:
    for character in text:
        script = _script_of(character)
        if script not in (COMMON, INHERITED):
            return harfbuzz_script(fallback_script(script))
    return None


def fallback_script(script: str) -> str:
    if script in (HIRAGANA, KATAKANA, BOPOMOFO):
        return HAN
    return script


def is_chinese_or_japanese_text(text: str) -> bool:
    return any(_script_of(ch) in (HAN, HIRAGANA, KATAKANA, BOPOMOFO) for ch in text)
