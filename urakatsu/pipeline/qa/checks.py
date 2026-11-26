"""QA's model-free wording checks, and the guards every model correction passes.

Some problems need no model, and a small one misjudges them both ways (a Japanese bubble passed as
translated; a correct glossary name flagged and replaced), so the program decides them:

- ``untranslated`` (critical): the translation still holds characters of the source's script
  (kana, CJK ideographs, Hangul) while the target language does not write them.
- ``glossary`` (major): a glossary entry the source line names, whose required rendering is not in
  the translation. The entry's own source text calls for the whole rendering (every word of
  "Leno Ichikawa"); an alias calls for any word of it (市川 alone is "Ichikawa"). Matching ignores
  case, hyphens a letterer breaks a word with, and endings ("Anya's"). When a word of the
  translation is a near spelling of the missing word (ANIA, KAFUKA), the suggestion swaps it.
  A rendering word counts when it is a whole word of the translation (or the word with a plural,
  possessive or honorific ending): "KENJI" is not "Ken". An OCR slip in the source's name
  (カフ力 for カフカ, 霊媒帥 for 霊媒師, a small kana read large) still names it.
- ``glossary`` also for a name the source line does not write: a word of the translation that is a
  near miss of a character's or place's rendering (ANIYA, ANNIE for Anya; KENJI for Ken) and is
  not itself a rendering (``name_slips``). The suggestion puts the rendering in its place.
- ``number`` (major): a number the source writes (digits, or kanji numerals with a counter: 十八歳,
  六人) that the translation changed: the source's number is missing and the translation has one
  the source does not. Either half alone is a free translation ("a couple of", "the best"), and
  so is a unit converted (二週間 as "14 days", 15時 as "3 PM"): ``converted``. The program writes
  the correction itself only for a slip (``number_fix``: 18 for 28, 3,000 for 30,000); any other
  change is flagged without one, since it may be a conversion the program does not know (180センチ
  as "six feet").
- ``full_width`` (minor): full-width Latin letters, digits or punctuation in a Latin translation.

These run on every pass, like the layout checks: they are recomputed, never kept from a review.

``guard_fix`` is what a model's correction must pass before it is offered or written: it must
change more than case and punctuation, keep the source script out, keep every required glossary
rendering the line had, keep the source's numbers (``numbers_agree``), and follow the page's
lettering (an all-caps page gets an all-caps fix, full-width forms are folded). Small models write
fluent corrections that drop a required name or copy another bubble's line; those are worse than none.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional

from urakatsu.scene.components import KnowledgeEntry, QaIssue

from .. import knowledge

KINDS = frozenset({"untranslated", "glossary", "number", "full_width"})

_SOURCE_SCRIPT = re.compile(r"[぀-ヿㇰ-ㇿ㐀-䶿一-鿿豈-﫿가-힯ᄀ-ᇿｦ-ﾟ]")
_FULL_WIDTH = re.compile(r"[！-～　]")
_LETTERS = re.compile(r"[^\W\d_]+")
CJK_LANGUAGES = frozenset({"ja", "zh", "ko"})
STOPWORDS = frozenset({"the", "a", "an", "of", "and", "to", "in", "on", "for", "mr", "ms", "mrs", "miss"})
NEAR_SPELLING = 0.6  # difflib ratio at which a word is taken for a misspelling of the required one
CAPS_SHARE = 0.85  # letters in upper case for a page to count as lettered in capitals
SAME_LINE = 0.85  # difflib ratio at which a correction is taken for another line's text
FIRST_PERSON = frozenset({"i", "me", "my", "mine", "myself"})


def _primary(language: Optional[str]) -> str:
    return (language or "").replace("_", "-").split("-")[0].lower()


def writes_source_script(language: Optional[str]) -> bool:
    return _primary(language) in CJK_LANGUAGES


def has_source_script(text: str) -> bool:
    return _SOURCE_SCRIPT.search(text) is not None


def _letters(text: str) -> str:
    """Lower-case letters only: hyphens, spaces and punctuation gone ("PA-PA" is "papa")."""
    return "".join(_LETTERS.findall(knowledge.normalize(text)))


def _words(text: str) -> list[str]:
    return [word for word in _LETTERS.findall(knowledge.normalize(text)) if word.isascii()]


def _content_words(rendering: str) -> list[str]:
    words = [word for word in _words(rendering) if word not in STOPWORDS]
    return words or _words(rendering)


# Endings a rendering word may carry and still be that word ("Anya's" splits at the apostrophe).
ENDINGS = ("", "s", "es", "san", "kun", "chan", "sama", "senpai", "sensei", "dono")


def _tokens(text: str) -> set[str]:
    """The translation's words, lower case, and each hyphenated run also joined ("PARA-NORMAL" is
    "para", "normal" and "paranormal"; "ICHIKAWA-SAN" is also "ichikawasan")."""
    normal = knowledge.normalize(text)
    found = set(_LETTERS.findall(normal))
    for run in re.findall(r"[^\W\d_]+(?:-\s*[^\W\d_]+)+", normal):
        found.add("".join(_LETTERS.findall(run)))
    return found


def _has_word(tokens: set[str], word: str) -> bool:
    return any(word + ending in tokens for ending in ENDINGS)


def renders(translation: str, rendering: str) -> bool:
    """Whether ``translation`` writes every word of the glossary ``rendering``."""
    tokens = _tokens(translation)
    return all(_has_word(tokens, word) for word in _content_words(rendering))


# OCR slips that turn one character of a name into another: a kanji for the katakana it looks like,
# a small kana read large, a classic kanji pair. Folded on both sides before a source is matched.
_OCR_FOLD = str.maketrans(
    {"力": "カ", "二": "ニ", "口": "ロ", "工": "エ", "夕": "タ", "卜": "ト", "八": "ハ", "一": "ー", "へ": "ヘ", "ぺ": "ペ", "べ": "ベ", "帥": "師",
     "ャ": "ヤ", "ュ": "ユ", "ョ": "ヨ", "ッ": "ツ", "ァ": "ア", "ィ": "イ", "ゥ": "ウ", "ェ": "エ", "ォ": "オ",
     "ゃ": "や", "ゅ": "ゆ", "ょ": "よ", "っ": "つ", "ぁ": "あ", "ぃ": "い", "ぅ": "う", "ぇ": "え", "ぉ": "お"}
)


def _names(name: str, text: str, compact: str) -> bool:
    if not name.strip():
        return False
    if knowledge.mentioned(name, text, compact, fuzzy=True):
        return True
    if name.isascii():
        return False
    return knowledge.mentioned(name.translate(_OCR_FOLD), text.translate(_OCR_FOLD), compact.translate(_OCR_FOLD), fuzzy=True)


@dataclass
class Requirement:
    """A glossary rendering a line's source calls for."""

    source: str  # the name as the source line writes it (the entry's source or alias)
    rendering: str
    words: list[str]  # what must be in the translation
    any_word: bool  # an alias: one of ``words`` will do
    character: bool = False

    def met(self, translation: str) -> bool:
        tokens = _tokens(translation)
        found = [_has_word(tokens, word) for word in self.words]
        if any(found) if self.any_word else all(found):
            return True
        # A speaker naming themself (common in Japanese, アーニャ for "I") is "I" in English; a
        # misspelt name next to the pronoun is still a miss.
        return self.character and bool(FIRST_PERSON & set(_words(translation))) and respell(translation, [self]) == translation

    def missing(self, translation: str) -> list[str]:
        tokens = _tokens(translation)
        missing = [word for word in self.words if not _has_word(tokens, word)]
        return [] if self.any_word and len(missing) < len(self.words) else missing


def requirements(source: str, entries: Iterable[KnowledgeEntry], language: str) -> list[Requirement]:
    """The glossary renderings ``source`` calls for in ``language``."""
    text = knowledge.normalize(source)
    compact = "".join(text.split())
    found = []
    for entry in entries:
        if entry.kind == "note" or not entry.translation.strip() or entry.language not in (None, language):
            continue
        if not _words(entry.translation):
            continue
        character = entry.kind == "character" and _primary(language) == "en"
        if _names(entry.source, text, compact):
            found.append(Requirement(entry.source, entry.translation, _content_words(entry.translation), any_word=False, character=character))
            continue
        alias = next((name for name in entry.aliases if _names(name, text, compact)), None)
        if alias is not None:
            found.append(Requirement(alias, entry.translation, _content_words(entry.translation), any_word=True, character=character))
    return found


def check_line(source: str, translation: str, entries: Iterable[KnowledgeEntry], language: str) -> list[QaIssue]:
    """The model-free findings on one line."""
    issues: list[QaIssue] = []
    if not source.strip() or not translation.strip():
        return issues
    latin_target = not writes_source_script(language)
    if latin_target and has_source_script(translation):
        left = len(_SOURCE_SCRIPT.findall(translation))
        letters = sum(1 for character in translation if character.isalpha())
        whole = left * 2 >= letters
        issues.append(
            QaIssue(
                kind="untranslated",
                severity="error",
                message="The translation is still in the source language." if whole else "Part of the translation is still in the source language.",
            )
        )
    for requirement in requirements(source, entries, language):
        if requirement.met(translation):
            continue
        fixed = respell(translation, [requirement])
        if fixed == translation or not requirement.met(fixed):
            # A near miss too far for ``respell`` (ANNIE for Anya) that the name check knows.
            slip = name_slips(translation, entries, language, only=requirement.rendering, evidence=False)
            fixed = slip[0].suggestion if slip and slip[0].suggestion and requirement.met(slip[0].suggestion) else fixed
        issues.append(
            QaIssue(
                kind="glossary",
                severity="warning",
                message=f"The glossary renders {requirement.source} as \"{requirement.rendering}\", which the translation does not use.",
                suggestion=fixed if fixed != translation and requirement.met(fixed) else None,
            )
        )
    if not any(issue.kind in ("untranslated", "glossary") for issue in issues):
        issues += name_slips(translation, entries, language)
    if latin_target and not any(issue.kind == "untranslated" for issue in issues):
        changed = number_change(source, translation)
        if changed:
            said, wrote = changed
            fixed = number_fix(source, translation)
            issues.append(QaIssue(kind="number", severity="warning", message=f"The source says {said}; the translation says {wrote}.", suggestion=fixed if fixed and fixed != translation else None))
    untranslated = any(issue.kind == "untranslated" for issue in issues)
    # Source text left in brings its full-width punctuation; that line needs translating, not folding.
    if latin_target and not untranslated and _FULL_WIDTH.search(translation):
        issues.append(QaIssue(kind="full_width", severity="info", message="Full-width characters in a Latin translation.", suggestion=fold_width(translation)))
    return issues


def fold_width(text: str) -> str:
    """Full-width Latin forms and the ideographic space as their ASCII forms."""
    return _FULL_WIDTH.sub(lambda match: unicodedata.normalize("NFKC", match.group(0)), text)


# -- numbers ---------------------------------------------------------------------------------------

_KANJI_DIGITS = {"〇": 0, "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_KANJI_SMALL = {"十": 10, "百": 100, "千": 1000}
_KANJI_LARGE = {"万": 10_000, "億": 100_000_000}
_KANJI_RUN = re.compile("[〇零一二两三四五六七八九十百千万億]+")
# Counters: a lone kanji numeral counts only before one (一緒, 三池 are words, 三人 is three people).
COUNTERS = frozenset("歳才人匹頭羽本枚個つ回度年月日時分秒円点番階位冊台件杯発週名話巻ヶヵか箇等号歩倍組軒丁着足周")
_GROUPED = r"\d{1,3}(?:,\d{3})+|\d+"  # a comma only groups thousands: "2,3日" is two numbers
_DIGITS = re.compile(rf"({_GROUPED})(万|千|億)?")
_EN_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_EN_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_EN_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000}
# Digits with a scale after them are one number: "100 MILLION", "30 thousand", "1.5 million", "30K".
_SCALED = re.compile(rf"({_GROUPED})(\.\d+)?(?:\s*(hundred|thousand|million|billion)\b|(k)\b)?", re.IGNORECASE)
# 1 and 0 are left out of the comparison: "a", "one", "this one", "no one" say them without a number.
_UNCOUNTED = frozenset({0, 1})


def _kanji_value(run: str) -> Optional[int]:
    total = section = digit = 0
    seen = False
    for character in run:
        if character in _KANJI_DIGITS:
            digit = digit * 10 + _KANJI_DIGITS[character] if seen and digit else _KANJI_DIGITS[character]
            seen = True
        elif character in _KANJI_SMALL:
            section += (digit or 1) * _KANJI_SMALL[character]
            digit, seen = 0, False
        else:
            total += (section + digit or 1) * _KANJI_LARGE[character]
            section = digit = 0
            seen = False
    return total + section + digit


def source_numbers(text: str) -> set[int]:
    """The numbers a source line writes: digits (full width too, "3万" as 30000), and runs of kanji
    numerals of two or more, or one before a counter."""
    normal = unicodedata.normalize("NFKC", text)
    found = set()
    for match in _DIGITS.finditer(normal):
        digits = match.group(1).replace(",", "")
        if digits:
            found.add(int(digits) * {"万": 10_000, "千": 1000, "億": 100_000_000}.get(match.group(2) or "", 1))
    for match in _KANJI_RUN.finditer(normal):
        run = match.group(0)
        after = normal[match.end() : match.end() + 1]
        if match.start() > 0 and normal[match.start() - 1].isdigit():
            continue  # "3万": read with the digits
        if len(run) >= 2 or after in COUNTERS:
            value = _kanji_value(run)
            if value is not None:
                found.add(value)
    hiragana = "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in normal)
    for match in _KANA_COUNTS.finditer(hiragana):
        found.add(_KANA_NUMBERS[match.group(1)])
    return found


# Things counted in kana (むっつ is 六つ, ふたり 二人); katakana is folded to hiragana first. Left
# out where the kana are also everyday words: いつか (someday), なのか (is it?), ようか (shall we),
# とお (through); やっつ only when not やっつける (to beat).
_KANA_NUMBERS = {
    "ひとつ": 1, "ふたつ": 2, "みっつ": 3, "よっつ": 4, "いつつ": 5, "むっつ": 6, "ななつ": 7, "やっつ": 8, "ここのつ": 9,
    "ふたり": 2, "ここのか": 9, "とおか": 10,
}
_KANA_COUNTS = re.compile("(" + "|".join(sorted(_KANA_NUMBERS, key=len, reverse=True)) + ")(?!け)")


def _number_spans(text: str) -> list[tuple[int, int, int, bool]]:
    """``(start, end, value, in words)`` of each number ``text`` writes: digits, and runs of number
    words ("twenty-eight", "a hundred and five"). Offsets are into ``text``."""
    spans: list[tuple[int, int, int, bool]] = []
    for match in _SCALED.finditer(text):
        scale = _EN_SCALES.get((match.group(3) or "").lower(), 1000 if match.group(4) else 1)
        if scale == 1:  # plain digits; a decimal's fraction is not a number of its own
            spans.append((match.start(), match.end(1), int(match.group(1).replace(",", "")), False))
        else:
            spans.append((match.start(), match.end(), round(float(match.group(1).replace(",", "") + (match.group(2) or "")) * scale), False))
    taken = [(start, end) for start, end, _, _ in spans]
    total = current = 0
    start = end = reach = -1
    last = ""  # the kind of the previous number word: "unit", "tens", "scale"
    for match in [m for m in re.finditer(r"[A-Za-z]+", text) if not any(start <= m.start() < end for start, end in taken)] + [None]:
        word = match.group(0).lower() if match else ""
        joined = match is not None and reach >= 0 and not text[reach : match.start()].strip(" -")
        kind = "scale" if word in _EN_SCALES else "tens" if word in _EN_TENS else "unit" if word in _EN_UNITS else ""
        # A number goes on only as English writes one: a unit after tens ("twenty-eight"), anything
        # after a scale ("a hundred and five"); "four and two" is two numbers.
        goes_on = joined and (last == "scale" or kind == "scale" or (last == "tens" and kind == "unit") or (last == "and" and kind in ("unit", "tens")))
        if kind or (word == "and" and start >= 0 and joined and last == "scale"):
            if start >= 0 and not goes_on and word != "and":
                spans.append((start, end, total + current, True))
                total = current = 0
                start = -1
            if word in _EN_SCALES:
                scale = _EN_SCALES[word]
                if scale >= 1000:
                    total += (current or 1) * scale
                    current = 0
                else:
                    current = (current or 1) * scale
            elif word != "and":
                current += _EN_UNITS.get(word, 0) + _EN_TENS.get(word, 0)
            if start < 0:
                start = match.start()
            if word != "and":
                end = match.end()
            reach = match.end()
            last = kind or "and"
            continue
        if start >= 0:
            spans.append((start, end, total + current, True))
        total = current = 0
        start = end = reach = -1
        last = ""
    return spans


def translation_numbers(text: str) -> set[int]:
    """The numbers a translation writes: digits, and number words ("twenty-eight", "a hundred")."""
    return {value for _, _, value, _ in _number_spans(unicodedata.normalize("NFKC", text))}


_WORD_UNITS = {value: word for word, value in _EN_UNITS.items()}
_WORD_TENS = {value: word for word, value in _EN_TENS.items()}


def number_words(value: int) -> Optional[str]:
    """``value`` in English words, up to 999 ("twenty-eight"); None beyond."""
    if value < 20:
        return _WORD_UNITS.get(value)
    if value < 100:
        tens, unit = divmod(value, 10)
        return _WORD_TENS[tens * 10] + (f"-{_WORD_UNITS[unit]}" if unit else "")
    if value < 1000:
        hundreds, rest = divmod(value, 100)
        return f"{_WORD_UNITS[hundreds]} hundred" + (f" {number_words(rest)}" if rest else "")
    return None


def number_fix(source: str, translation: str) -> Optional[str]:
    """``translation`` with its one changed number put back as the source's, written the way the
    translation wrote it (digits or words, its case): only when exactly one number of the source is
    missing and exactly one of the translation is extra, so the swap is certain."""
    said = source_numbers(source) - _UNCOUNTED
    spans = [span for span in _number_spans(translation) if span[2] not in _UNCOUNTED]
    wrote = {span[2] for span in spans}
    missing, extra = said - wrote, wrote - said
    if len(missing) != 1 or len(extra) != 1:
        return None
    (right,), (wrong,) = missing, extra
    if not _slip(right, wrong):
        return None
    places = [span for span in spans if span[2] == wrong]
    fixed = translation
    for start, end, _, in_words in reversed(places):
        old = translation[start:end]
        new = number_words(right) if in_words else str(right)
        if new is None:
            new = str(right)
        letters = [c for c in old if c.isalpha()]
        if letters and all(c.isupper() for c in letters):
            new = new.upper()
        elif letters and letters[0].isupper():
            new = new[:1].upper() + new[1:]
        fixed = fixed[:start] + new + fixed[end:]
    return fixed


def number_change(source: str, translation: str) -> Optional[tuple[str, str]]:
    """``(what the source says, what the translation says)`` when the translation changed a number."""
    said = source_numbers(source) - _UNCOUNTED
    wrote = translation_numbers(translation) - _UNCOUNTED
    if not said or not wrote or said & wrote == said:
        return None
    extra = wrote - said
    if not extra:
        return None
    missing = said - wrote
    if len(missing) == 1 and len(extra) == 1 and converted(next(iter(missing)), next(iter(extra))):
        return None
    return ", ".join(str(n) for n in sorted(said - wrote)), ", ".join(str(n) for n in sorted(extra))


# What one unit is of the next: days in a week, months in a year, hours in a day, days in a month,
# minutes or seconds in theirs, days in a year, seconds in an hour.
CONVERSIONS = frozenset({7, 12, 24, 30, 60, 365, 3600})


def converted(said: int, wrote: int) -> bool:
    """Whether ``wrote`` is ``said`` in another unit (二週間 as "14 days", 2時間 as "120 minutes")
    or on the twelve-hour clock (15時 as "3 PM"): a translator's choice, not a changed number."""
    low, high = sorted((said, wrote))
    if low > 0 and high % low == 0 and high // low in CONVERSIONS:
        return True
    return 13 <= said <= 24 and wrote == said - 12


def _slip(right: int, wrong: int) -> bool:
    """Whether ``wrong`` for ``right`` looks like a slip the program may put back: the same number
    of digits (28 for 18), or a power of ten off (3,000 for 30,000). Anything else may be a unit
    the program does not know, and is left to a reader."""
    if len(str(right)) == len(str(wrong)):
        return True
    low, high = sorted((right, wrong))
    return low > 0 and high % low == 0 and str(high // low).strip("0") == "1"


def numbers_agree(source: str, fix: str, translation: str = "") -> bool:
    """Whether a correction of ``translation`` keeps the source's numbers: none changed, and none
    added that neither the source nor the translation had."""
    said = source_numbers(source) - _UNCOUNTED
    wrote = translation_numbers(fix) - _UNCOUNTED
    if said:
        return wrote <= said
    return wrote <= translation_numbers(translation) - _UNCOUNTED


# -- names the source line does not write -----------------------------------------------------------

NAME_KINDS = frozenset({"character", "place"})
# English words a name's near miss could be; never taken for a misspelt name.
COMMON = frozenset(
    "any anyway anyone anybody anew every everyone many only lady baby buddy daddy mommy mummy honey sorry kitty "
    "money monkey kenya yore your year yeah keen kent kind king kill dance dandy danny eventually even ever event "
    "maid main many mama papa nana nanny grandpa granny auntie uncle sonny sunny".split()
)
NAME_ENDINGS = ("i", "y", "ie", "ji", "a", "ya", "o", "ko", "ki", "ri")
_VOWELS = re.compile("[aeiouy]")


def _skeleton(word: str) -> str:
    collapsed = re.sub(r"(.)\1+", r"\1", word)
    return collapsed[0] + _VOWELS.sub("", collapsed[1:])


def _near_name(word: str, name: str) -> bool:
    """Whether ``word`` is a near miss of the rendering word ``name`` (both lower case)."""
    if word == name or word in COMMON or len(word) < 3:
        return False
    ratio = difflib.SequenceMatcher(None, word, name).ratio()
    if len(name) >= 4 and len(word) >= 4 and abs(len(word) - len(name)) <= 1 and ratio >= 0.75 and (word[0] == name[0] or ratio >= 0.8):
        return True  # ANIYA, KAFUKA, ITCHIKAWA, CAFCA
    if len(name) >= 3 and word.startswith(name) and word[len(name) :] in NAME_ENDINGS:
        return True  # KENJI for Ken
    # ANNIE for Anya: the same start and consonants, an English diminutive's ending.
    return len(name) >= 4 and word[:2] == name[:2] and len(word) in (len(name), len(name) + 1) and word.endswith(("ie", "y", "i")) and _skeleton(word) == _skeleton(name)


# Words before a name in a bubble ("HEY KAFUKA", "I'M ANIA", "MISS AYANE").
NAME_CUES = frozenset(
    "hey oi yo oh ah miss mister mr mrs ms sir madam dear little big brother sister uncle aunt im i'm it's its "
    "named called call tell ask meet and with thanks thank sorry".split()
)


def _named_here(translation: str, match: re.Match, known: set[str]) -> bool:
    """Whether the word at ``match`` stands where a name does: after a cue word or another name, as
    the one addressed (", ANIYA!", "CAFCA, YOU'LL..."), or before a possessive. In mixed case it must
    also be capitalised, and capitalised mid-sentence counts on its own."""
    text = match.group(0)
    before = translation[: match.start()]
    after = translation[match.end() :]
    mixed = any(character.islower() for character in translation)
    if mixed:
        if not text[:1].isupper():
            return False
        if before.strip() and not re.search(r"[.!?…]\s*[\"'“‘]?$", before):
            return True
    previous = re.findall(r"[A-Za-z']+", before[-24:])
    if previous and (previous[-1].lower().replace("’", "'") in NAME_CUES or previous[-1].lower() in known):
        return True
    if re.match(r"['’]s\b", after, re.IGNORECASE):
        return True
    addressed_after = re.search(r",\s*$", before) and re.match(r"\s*(?:[!?.…~]+|$)", after)
    addressed_first = not re.search(r"[A-Za-z]", re.split(r"[.!?…]", before)[-1]) and re.match(r"\s*[,!]", after)
    return bool(addressed_after or addressed_first)


def name_slips(translation: str, entries: Iterable[KnowledgeEntry], language: str, only: str = "", evidence: bool = True) -> list[QaIssue]:
    """Words of ``translation`` that are near misses of a character's or place's rendering, whether
    or not the source names it (``_near_name``), where the word stands as a name (``_named_here``):
    a near miss with nothing marking it as a name is an English word (a verb taken for a place name
    on the round-four test pages) and is left alone. ``only``: that rendering alone; ``evidence``
    False when the source line itself names the entry."""
    if writes_source_script(language):
        return []
    entries = [e for e in entries if e.kind != "note" and e.translation.strip() and e.language in (None, language)]
    known = {word for entry in entries for word in _words(entry.translation)}
    candidates = [e for e in entries if (e.translation == only if only else e.kind in NAME_KINDS)]
    slips: list[tuple[int, int, str, str, str]] = []  # start, end, replacement, the word, the rendering
    for match in re.finditer(r"[^\W\d_]+", translation):
        text = match.group(0)
        word = knowledge.normalize(text)
        if not word.isascii() or word in known or word in STOPWORDS:
            continue
        if evidence and not _named_here(translation, match, known):
            continue
        hit = next(((name, entry.translation) for entry in candidates for name in _content_words(entry.translation) if _near_name(word, name)), None)
        if hit is not None:
            slips.append((match.start(), match.end(), _cased(hit[0], text), text, hit[1]))
    if not slips:
        return []
    fixed = translation
    for start, end, replacement, _, _ in reversed(slips):
        fixed = fixed[:start] + replacement + fixed[end:]
    named = "; ".join(f"\"{text}\" looks like a misspelling of the glossary's \"{rendering}\"" for _, _, _, text, rendering in slips)
    return [QaIssue(kind="glossary", severity="warning", message=named + ".", suggestion=fixed)]


def respell(text: str, requirements: Iterable[Requirement]) -> str:
    """``text`` with each word that is a near spelling of a missing required word replaced by it,
    in the word's case (ANIA -> ANYA, Kafuka -> Kafka). Words with no near spelling stay missing."""
    for requirement in requirements:
        for word in requirement.missing(text):
            best, best_ratio, plural = None, NEAR_SPELLING, False
            for match in re.finditer(r"[^\W\d_]+(?:-[^\W\d_]+)*", text):
                plain = _letters(match.group(0))
                if not plain.isascii() or plain in STOPWORDS or plain in requirement.words:
                    continue
                for stem, is_plural in ((plain, False), (plain[:-1], True) if plain.endswith("s") and len(plain) > 3 else (plain, False)):
                    ratio = difflib.SequenceMatcher(None, word, stem).ratio()
                    # A different first letter is an OCR or hearing slip only when the rest is close.
                    if ratio > best_ratio and (stem[:1] == word[:1] or ratio >= 0.75):
                        best, best_ratio, plural = match, ratio, is_plural
            if best is None:
                continue
            replacement = _cased(word + ("s" if plural else ""), best.group(0))
            text = text[: best.start()] + replacement + text[best.end() :]
    return text


def _cased(word: str, like: str) -> str:
    letters = [character for character in like if character.isalpha()]
    if letters and all(character.isupper() for character in letters):
        return word.upper()
    if letters and letters[0].isupper():
        return word[:1].upper() + word[1:]
    return word


def capitals(translations: Iterable[str]) -> bool:
    """Whether a page is lettered in capitals."""
    letters = [character for text in translations for character in text if character.isalpha() and character.isascii()]
    return len(letters) >= 8 and sum(character.isupper() for character in letters) >= CAPS_SHARE * len(letters)


def _plain(text: str) -> str:
    return "".join(re.findall(r"[^\W_]+", knowledge.normalize(text)))


def same_text(a: str, b: str) -> bool:
    """Equal but for case, spacing, punctuation and hyphenation (digits count: 18 is not 28)."""
    return _plain(a) == _plain(b)


def alike(a: str, b: str) -> bool:
    a, b = _plain(a), _plain(b)
    return bool(a) and bool(b) and difflib.SequenceMatcher(None, a, b).ratio() >= SAME_LINE


@dataclass
class PageStyle:
    """What a correction must follow on its page."""

    language: str
    capitals: bool = False


def guard_fix(fix: str, translation: str, required: list[Requirement], style: PageStyle, source: str = "") -> tuple[Optional[str], str]:
    """``(fix, "")`` with the page's lettering applied, or ``(None, why)`` when the correction must
    not be offered. ``translation`` is the line's current text; ``required`` its glossary renderings;
    ``source`` its source line, whose numbers the correction must keep."""
    fix = " ".join(fix.split())
    if not fix:
        return None, "no correction"
    fix = fold_width(fix) if not writes_source_script(style.language) else fix
    if not writes_source_script(style.language) and has_source_script(fix):
        return None, "the correction is not in the target language"
    if source and not writes_source_script(style.language) and not numbers_agree(source, fix, translation):
        return None, "the correction changes a number of the source"
    fix = respell(fix, [r for r in required if not r.met(fix)])
    for requirement in required:
        if requirement.met(translation) and not requirement.met(fix):
            return None, f"the correction drops the glossary's \"{requirement.rendering}\""
    if style.capitals:
        fix = fix.upper()
    if same_text(fix, translation):
        return None, "the correction only changes case or punctuation"
    return fix, ""
