"""QA's proofreading half: one structured reply per page from the translation model.

The model gets the page's numbered lines (source and translation), what the project's knowledge
retrieves for them within a small budget (``knowledge.retrieve``: glossary, notes, nearby lines,
the story so far), and for a line that overflows or is lettered too small, the length that would
fit. It lists only the lines with a problem, each with a category, a severity, a one-line reason
and the corrected translation, so a clean page costs a reply of a few tokens.

Asking for more (``need``): the model always answers, and may also name glossary and notes on some
terms, the lines of a page before or after, or the story so far. A second call is made only when
the lookup finds something the first prompt lacked (a term written on the page that the
knowledge has and the prompt did not carry, a page whose lines were not in ``context``, a story
not yet sent); the second answer then replaces the first, and its format has no ``need``. A
request for anything else costs nothing: the first answer stands. (Round one asked first and
answered second, so a small model asked on nearly every page, mostly for terms not on it.) This
is plain JSON, not tool calling, so any model with structured output can do it.

What the model is not asked (``checks``): source text left in, glossary renderings and
full-width characters are found by the program; a line with such a finding carries it in
``found`` so the model writes its fix, and every correction passes ``vet`` (``checks.guard_fix``,
no text pulled in from another line, no ``too_long`` without a length) before it is kept.

Each reported issue carries ``source_says``, a literal translation of its source, written
before the fix: a small model that states what the source says first contradicts it less in
the correction. A reply cut off at ``max_tokens`` keeps the issues it completed.

The system prompt is the same for every page, so hosted APIs cache it and llama.cpp can reuse it.

With the meaning check on (``QaConfig.meaning_check``; by default for on-device models), meaning is
judged line by line instead (see "the meaning check" below): one call reads the sources alone, one
call grades every translation against its reading, and a suspect line gets a second look with its
source, which names the smallest change that corrects it; that correction is offered only when a
call of its own grades it as saying what the reading says. The proofreading call then runs only for
what it alone does (corrections for the program's findings that have none, wording that fits the
layout), and its own mistranslation, omission and addition reports are dropped. A line the grading
found faithful ("same", "close") is never given a correction: its text stays.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from urakatsu.providers.translation.language import Language
from urakatsu.scene.components import Knowledge, QaIssue
from urakatsu.scene.scene import Snapshot

from .. import knowledge
from ..stages.sound_effects import ROLE

CATEGORIES = ("mistranslation", "omission", "addition", "grammar", "too_long", "untranslated", "glossary", "number")
# Kinds only the program raises (``checks``); the model may name one only to fix it.
FOUND_ONLY = frozenset({"untranslated", "glossary", "number"})
# Offered so a model can say "right meaning, I would word it otherwise" instead of calling a free
# translation a mistranslation; such an issue is dropped.
WORDING = "wording"
# A reason that is a wording preference ("too literal", "is more like ...", "more natural") makes
# the issue ``wording`` whatever category the model gave it: a small model files its rewording
# urges as mistranslations. On the benchmark's dev pages this was half of the clean lines flagged.
STYLE_REASON = re.compile(r"\b(too literal|overly literal|more literal|is more like|more natural|sounds? (more )?natural|awkward|phrasing|too formal|too casual|more idiomatic|smoother)\b", re.IGNORECASE)
SEVERITIES = {"minor": "info", "major": "warning", "critical": "error"}
PAGE_OFFSETS = (-2, -1, 1, 2)
MAX_TERMS = 5
NOTES_PER_TERM = 2
REASON_CHARS = 200


@dataclass
class Line:
    """One block to proofread. ``n`` numbers it on the page, from 1."""

    n: int
    content: str
    source: str
    translation: str
    max_chars: Optional[int] = None
    layout: str = ""  # a wording-fixable layout finding ("overflows its bubble")
    use: dict[str, str] = field(default_factory=dict)  # glossary renderings its source calls for
    found: list[str] = field(default_factory=list)  # the program's findings ("untranslated: ...")
    # The kinds of the program's findings that have no correction yet (a respelt name has one).
    open: list[str] = field(default_factory=list)
    literal: str = ""  # the model's reading of the source alone (``read_literally``)
    grade: str = ""  # the meaning check's grade of the translation against ``literal`` (``MEANINGS``)

    def fingerprint(self, language: str) -> str:
        """What the model reviews of this line: a block whose fingerprint is unchanged is not sent again."""
        data = json.dumps([self.source, self.translation, language, self.max_chars], ensure_ascii=False)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


@dataclass
class Need:
    terms: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    story: bool = False

    def empty(self) -> bool:
        return not (self.terms or self.pages or self.story)


@dataclass
class Reply:
    issues: dict[int, list[QaIssue]] = field(default_factory=dict)
    need: Need = field(default_factory=Need)
    truncated: bool = False  # cut off at ``max_tokens``: only its complete issues


@dataclass
class Proofread:
    """The outcome of ``proofread``: issues by text content, and what it cost."""

    issues: dict[str, list[QaIssue]] = field(default_factory=dict)
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Whether some count is an estimate (``knowledge.estimate_tokens``) because the backend gave none.
    estimated: bool = False
    asked: list[Need] = field(default_factory=list)
    # Asks that found something new and so cost a second call.
    fetched: int = 0
    truncated: bool = False
    # Issues ``vet`` dropped, with why (for benchmarks and debugging; not stored).
    dropped: list[tuple[str, str, str]] = field(default_factory=list)
    # The literal readings by content (``read_literally``), when that step ran.
    literals: dict[str, str] = field(default_factory=dict)
    # The meaning check's corrections: how many passed the faithfulness check (``vouch``), and the
    # ones withheld (content, why, the correction).
    vouched: int = 0
    withheld: list[tuple[str, str, str]] = field(default_factory=list)
    # The meaning check's issues (by ``id``): backed by its second look, so kept without a correction.
    checked: set[int] = field(default_factory=set)
    error: str = ""


def system_prompt(target: Language, source: Optional[Language] = None, instructions: Optional[str] = None) -> str:
    """The same text for every page of a run (see the module notes)."""
    source_name = source.display_name if source else "the source language"
    name = target.display_name
    prompt = (
        f"You proofread a manga translation from {source_name} into {name}. `lines` holds the page's lines: "
        "`n`, the `source` text (read by OCR, so it may have errors) and its `translation`.\n"
        "Report a line only when its translation says something the source does not, leaves out what matters, or is broken. "
        "Most lines are fine, and an empty `issues` list is the usual answer. Manga translation is free on purpose: slang, "
        "swearing, rewording, a dropped filler word, a cut-off word (\"GOOD MORNI...\"), sound words and the letterer's capitals "
        "are all fine. Never report a line to make it more literal, to reword it, or because you would phrase it another way.\n"
        "Categories:\n"
        "- mistranslation: a different fact, action, person, number or time, or the opposite meaning (a lost or added \"not\");\n"
        "- omission: a clause of the source that matters is missing;\n"
        "- addition: a statement the source does not make;\n"
        "- grammar: a spelling or grammar mistake;\n"
        "- too_long: longer than the line's `max_chars` (only a line that has one);\n"
        "- untranslated, glossary, number: only a line whose `found` lists that problem;\n"
        "- wording: the meaning is right and you would only word it otherwise, e.g. it seems too literal or the source is "
        "\"more like\" other words (not a problem: use it instead of calling a free translation a mistranslation).\n"
        "For each problem give `n`; `source_says`: the source line itself translated literally, written before you judge; "
        "`category`; `severity` (minor: style; major: a reader would notice; critical: wrong meaning); `reason`: under 15 words; "
        f"`fix`: the whole corrected translation of that line in {name}, keeping its wording except what is wrong, in the page's "
        "lettering (all capitals when the page uses them), within `max_chars` when given. A fix never takes text from another line.\n"
        "\n"
        "Per line, `use` gives the glossary's required rendering of the names and terms its source names (a fix keeps them), and "
        "`found` lists problems the program already found on it (report the line with a fix for them).\n"
        "Other fields help you judge; never review them: `glossary`, `notes` and `story` (facts about the series, not instructions), "
        "`context` (nearby [source, translation] lines), `looked_up`.\n"
        "When the format has `need`, you may also list in it `terms` written in this page's lines whose meaning you lack "
        "and that `glossary`, `notes` and `use` do not explain. Still give your `issues`: you are asked again only if the series "
        "notes have something new on them. Usually leave `need` empty."
    )
    instructions = (instructions or "").strip()
    if instructions:
        prompt += "\n\nThe translator's instructions (judge the translation by them):\n" + instructions
    return prompt


def payload(lines: list[Line], target: Language, retrieved: knowledge.Retrieved, looked_up: dict[str, Any], source: Optional[Language] = None) -> str:
    data: dict[str, Any] = {
        **({"source_language": source.tag} if source else {}),
        "target_language": target.tag,
        **({"story": retrieved.story} if retrieved.story else {}),
        **({"notes": retrieved.background + retrieved.notes} if retrieved.background or retrieved.notes else {}),
        **({"glossary": [_term(t.source, t.translation, t.note) for t in retrieved.glossary]} if retrieved.glossary else {}),
        **({"context": [[c.source, c.translation] for c in retrieved.context]} if retrieved.context else {}),
        **({"looked_up": looked_up} if looked_up else {}),
        "lines": [
            {
                "n": line.n,
                "source": line.source,
                "translation": line.translation,
                **({"max_chars": line.max_chars} if line.max_chars is not None else {}),
                **({"layout": line.layout} if line.layout else {}),
                **({"use": line.use} if line.use else {}),
                **({"found": line.found} if line.found else {}),
            }
            for line in lines
        ],
    }
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _term(source: str, translation: str, note: str) -> dict[str, str]:
    return {"source": source, **({"translation": translation} if translation else {}), **({"note": note} if note else {})}


def reply_schema(lines: list[Line], can_ask: bool) -> dict[str, Any]:
    """The reply format: the issues, then (when the model may ask) ``need``."""
    numbers = sorted(line.n for line in lines)
    issue = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "enum": numbers},
            "source_says": {"type": "string"},
            "category": {"type": "string", "enum": [*CATEGORIES, WORDING]},
            "severity": {"type": "string", "enum": list(SEVERITIES)},
            "reason": {"type": "string"},
            "fix": {"type": "string"},
        },
        "required": ["n", "source_says", "category", "severity", "reason", "fix"],
        "additionalProperties": False,
    }
    properties: dict[str, Any] = {"issues": {"type": "array", "items": issue, "maxItems": max(2 * len(lines), 1)}}
    if can_ask:
        properties["need"] = {
            "type": "object",
            "properties": {"terms": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_TERMS}},
            "required": ["terms"],
            "additionalProperties": False,
        }
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def parse_reply(text: str, lines: list[Line]) -> Reply:
    """The model's reply. Tolerant: unknown lines, categories and severities, a missing ``need``,
    or a bare list of issues do not fail it; text that is not a JSON object does (``ValueError``)."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        starts = [index for index in (stripped.find("{"), stripped.find("[")) if index >= 0]
        stripped = stripped[min(starts) :] if starts else stripped
    try:
        value = json.loads(stripped)
    except ValueError:
        value = _salvage(stripped)
    if isinstance(value, list):
        value = {"issues": value}
    if not isinstance(value, dict):
        raise ValueError("the reply is not a JSON object")
    by_number = {line.n: line for line in lines}
    reply = Reply(truncated=bool(value.get("truncated")))
    for entry in value.get("issues") or []:
        if not isinstance(entry, dict):
            continue
        n = _number(entry.get("n"))
        line = by_number.get(n) if n is not None else None
        if line is None:
            continue
        category = str(entry.get("category") or "other").strip().lower().replace(" ", "_").replace("/", "_")
        if category not in CATEGORIES and category != WORDING:
            category = "other"
        severity = SEVERITIES.get(str(entry.get("severity") or "").strip().lower(), "warning")
        fix = entry.get("fix")
        fix = fix.strip() if isinstance(fix, str) else ""
        reason = " ".join(str(entry.get("reason") or "").split())[:REASON_CHARS]
        suggestion = fix if fix and fix != line.translation.strip() else None
        reply.issues.setdefault(line.n, []).append(QaIssue(kind=category, severity=severity, message=reason, suggestion=suggestion))
    need = value.get("need")
    if isinstance(need, dict):
        terms = [str(term).strip() for term in need.get("terms") or [] if str(term).strip()]
        # Pages and the story are no longer offered (the first prompt carries the nearby lines and the
        # story within its budget); a model that names them anyway is still served.
        pages = [offset for offset in (_number(p, signed=True) for p in need.get("pages") or []) if offset in PAGE_OFFSETS]
        reply.need = Need(terms=list(dict.fromkeys(terms))[:MAX_TERMS], pages=list(dict.fromkeys(pages)), story=bool(need.get("story")))
    return reply


def _salvage(text: str) -> dict[str, Any]:
    """The complete issues of a reply cut off mid-way (``max_tokens``); ``ValueError`` when the
    text is not the start of a reply with an issue list."""
    start = text.find('"issues"')
    bracket = text.find("[", start) if start >= 0 and text.lstrip().startswith("{") else -1
    if bracket < 0:
        raise ValueError("the reply is not a JSON object")
    decoder = json.JSONDecoder()
    issues: list[Any] = []
    index = bracket + 1
    while True:
        while index < len(text) and text[index] in " \t\r\n,":
            index += 1
        if index >= len(text) or text[index] != "{":
            break
        try:
            entry, index = decoder.raw_decode(text, index)
        except ValueError:
            break
        issues.append(entry)
    return {"issues": issues, "truncated": True}


def _number(value: Any, signed: bool = False) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return int(stripped) if stripped.lstrip("-+" if signed else "").isdigit() else None
        except ValueError:
            return None
    return None


# -- the meaning check -----------------------------------------------------------------------------
#
# A small model shown a source line and its translation together reads the source through the
# translation and passes flipped meanings ("STAY OUT OF HERE" for 勝手に入ってくれ). So, with
# ``meaning_check`` on, the model first reads every source line alone into the target language
# (``read_literally``: it never sees a translation). Each translation is then set beside that
# reading in a short call of its own, all in the target language (``compare``), which grades how
# the two relate. A translation graded "different" or "contradicts", or far shorter than its
# reading (``shorter``, an omission without asking), gets a second look with the source line
# (``confirm``), which may clear it (the reading misread the source) and writes the correction.
# These issues replace the proofreading call's own mistranslation, omission and addition reports:
# on the dev pages those flagged one clean line in six and caught few flips.
#
# The correction (round four): the second look no longer rewrites the line. It names the fewest wrong
# words (``from``) and what replaces them (``to``), and the program splices that in (``splice``), so
# everything else stays the translator's: a whole rewrite read like the plain reading pasted in
# ("AM I HATED...?") and lost to the official line in a blind comparison. The spliced line must then
# pass ``vouch``: it changes the text, keeps the source's numbers, and a grading call of its own finds
# it "same" or "close" to the reading. One that does not stays a flag without a correction: on the dev
# pages that withheld rewrites adding detail ("A CRUNCHY BACON"), broken splices ("I'LL NOT GIVING
# UP") and fixes that left the error in. A changed number the program found skips the grading and
# goes straight to the second look with the finding as a hint.
#
# Round five: ``vouch`` is comparative (the fix must grade strictly better than the line it
# replaces) and asks in the same call whether the fix is a natural line; a number the program can
# swap in place, or a near-miss name, needs no model at all (``checks``); a glossary rendering the
# program cannot respell gets the smallest swap from the second look; a suspect line is read again
# in its page's context before anything is changed (``reread``), and one that reading bears out
# stands. Splices join line-break hyphens and drop words said twice (``dehyphenate``, ``undouble``).
# Tried on the benchmark's dev pages and dropped: one call comparing the whole page (it passed most
# flips); a yes/no "same line?" (Qwen3-4B passed half the flips, Gemma 4 12B failed a fifth of the
# clean lines); asking which words contradict (both models then found some on most clean lines); a
# five-way verdict with "opposite" first (Qwen3-4B said "opposite" for half the clean lines); and
# program hints on negation and person words (free translations trip them; they cost precision).

LITERAL_TOKENS = 160  # the literal reading's floor; it grows with the page's lines
SHORTER = 2  # a reading this many times longer than the translation, in words, means a part is missing
SHORTER_MIN_WORDS = 6  # ... when the reading has at least this many words
CHECK_MIN_WORDS = 3  # shorter readings (interjections, a name and a cry) are not compared
# What the meaning check reports in place of the proofreading call.
MEANING_KINDS = frozenset({"mistranslation", "omission", "addition"})
_SOURCE_SCRIPT = re.compile("[぀-ヿ㐀-鿿가-힯ｦ-ﾟ]")
_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?")


def _cjk(target: Optional[Language]) -> bool:
    return target is not None and target.tag.split("-")[0].lower() in ("ja", "zh", "ko")


def _english(target: Optional[Language]) -> bool:
    return target is not None and target.tag.split("-")[0].lower() == "en"


def literal_system(target: Language, source: Optional[Language] = None) -> str:
    source_name = source.display_name if source else "the source language"
    name = target.display_name
    return (
        f"You translate a manga page from {source_name} into {name}. For each numbered line in `lines`, write `literal`: "
        f"a plain, faithful {name} sentence that says exactly what the line says: keep every negation (not, never, don't), "
        "every number and time, who does what to whom, and every clause. Translate the meaning, never sound it out in Latin "
        f"letters, and never copy the source: `literal` is always {name} words. The lines come in reading order and were "
        "read by OCR, so a character may be wrong.\n"
        'Example: {"n":1,"source":"勝手に入るな"} -> {"n":1,"literal":"Don\'t come in without asking."}'
    )


def literal_payload(lines: list[Line]) -> str:
    return json.dumps({"lines": [{"n": line.n, "source": line.source} for line in lines]}, ensure_ascii=False, separators=(",", ":"))


def literal_schema(lines: list[Line]) -> dict[str, Any]:
    numbers = sorted(line.n for line in lines)
    item = {
        "type": "object",
        "properties": {"n": {"type": "integer", "enum": numbers}, "literal": {"type": "string"}},
        "required": ["n", "literal"],
        "additionalProperties": False,
    }
    lines_schema = {"type": "array", "items": item, "minItems": len(numbers), "maxItems": len(numbers)}
    return {"type": "object", "properties": {"lines": lines_schema}, "required": ["lines"], "additionalProperties": False}


def parse_literals(text: str, lines: list[Line], keep_script: bool = False) -> dict[int, str]:
    """``{n: literal}`` from the reply. What cannot be read is left out (that line is not compared),
    and so is a reading that copied the source's script, unless ``keep_script`` (a CJK target)."""
    try:
        value = json.loads(text.strip())
    except ValueError:
        value = _salvage_list(text, "lines")
    entries = value.get("lines") if isinstance(value, dict) else value
    numbers = {line.n for line in lines}
    found: dict[int, str] = {}
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        n = _number(entry.get("n"))
        literal = " ".join(str(entry.get("literal") or "").split())
        if not keep_script and _SOURCE_SCRIPT.search(literal):
            continue
        if n in numbers and literal and n not in found:
            found[n] = literal[: REASON_CHARS * 2]
    return found


def _salvage_list(text: str, key: str) -> dict[str, Any]:
    """The complete entries of a list reply cut off mid-way."""
    start = text.find("[")
    if start < 0:
        return {}
    decoder = json.JSONDecoder()
    entries: list[Any] = []
    index = start + 1
    while True:
        while index < len(text) and text[index] in " \t\r\n,":
            index += 1
        if index >= len(text) or text[index] != "{":
            break
        try:
            entry, index = decoder.raw_decode(text, index)
        except ValueError:
            break
        entries.append(entry)
    return {key: entries}


def words(text: str) -> int:
    return len(_WORD.findall(text))


LONG_READING = 8  # a reading of this many words or more ...
LONG_SHORTER = 1.6  # ... this many times longer than the translation also means a part is missing
NAMED_SHORTER = 1.3  # ... and this many on a line that also lacks a name its source writes


def shorter(literal: str, translation: str, factor: Optional[float] = None) -> bool:
    """Whether ``translation`` is so much shorter than the literal reading that a part is missing:
    half its words, or under 1/1.6 of a long reading (a whole sentence dropped from a long bubble),
    or under ``factor`` when given. On the dev readings the 1.6 rule added one clean line of 116."""
    count = words(literal)
    have = words(translation)
    if factor is not None:
        return count >= SHORTER_MIN_WORDS and have * factor < count
    return count >= SHORTER_MIN_WORDS and (have * SHORTER < count or (count >= LONG_READING and have * LONG_SHORTER < count))


MEANINGS = ("same", "close", "loose", "different", "contradicts")
SUSPECT = frozenset({"different", "contradicts"})


def meaning_system(target: Language, batch: bool = False) -> str:
    name = target.display_name
    head = (
        f"Each item of `pairs` holds A and B, two {name} translations of the same manga speech bubble, by different "
        "translators (the items are different bubbles; judge each on its own). "
        if batch
        else f"A and B are two {name} translations of the same manga speech bubble, by different translators. "
    )
    return (
        head + "A is plain and close to the original; B is free: slang, idioms, rewording, jokes, a punchier line, a "
        "different exclamation or insult are normal.\n"
        + ("For each item give its `n` and `meaning`, " if batch else "`meaning`, ")
        + "how B's meaning relates to A's:\n"
        "- same: the same meaning;\n"
        "- close: the same point, other words or details;\n"
        "- loose: a free rendering a reader could accept for A;\n"
        "- different: B says something else, but nothing A denies;\n"
        "- contradicts: B says the reverse of A (do/don't, can/can't, come in/stay out, stop/keep, smart/stupid), has a "
        "different person do it or be meant (I/you/he/she), names a different thing, number or time, or drops a whole "
        "clause of A."
    )


MEANING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"meaning": {"type": "string", "enum": list(MEANINGS)}},
    "required": ["meaning"],
    "additionalProperties": False,
}


def meaning_batch_schema(lines: list[Line]) -> dict[str, Any]:
    numbers = sorted(line.n for line in lines)
    item = {
        "type": "object",
        "properties": {"n": {"type": "integer", "enum": numbers}, "meaning": {"type": "string", "enum": list(MEANINGS)}},
        "required": ["n", "meaning"],
        "additionalProperties": False,
    }
    grades = {"type": "array", "items": item, "minItems": len(numbers), "maxItems": len(numbers)}
    return {"type": "object", "properties": {"grades": grades}, "required": ["grades"], "additionalProperties": False}


def confirm_system(target: Language, source: Optional[Language] = None) -> str:
    source_name = source.display_name if source else "source-language"
    name = target.display_name
    return (
        f"You check one line of a manga translation into {name}. `source` is the original {source_name} line (read by "
        f"OCR, so a character may be wrong). `reading` is a plain {name} reading of it by another reader, which may itself "
        "be wrong. `translation` is the published line; it is free on purpose (slang, idioms, jokes, rewording are fine). "
        "A first check found that `translation` may not say what the source says; `found`, when given, is what the "
        "program found wrong with it.\n"
        "`wrong`: true only when `translation` contradicts the source: it says the reverse, has a different person act or "
        "be meant, names a different thing, number or time, or leaves out a whole clause; false when it is a free but fair "
        "rendering, or when `reading` is what misread the source.\n"
        "When `wrong` is true, give the smallest change that makes `translation` say what the source says; otherwise "
        "leave both empty. `from`: the fewest words of `translation` that are wrong, copied exactly (for a missing clause, "
        "the words it belongs next to). `to`: what replaces them, in the translator's own voice: their slang, contractions "
        "and lettering (capitals when they use them), saying what the source says there and nothing more: no word the "
        "source does not back, no clause of it dropped. `translation` with `from` swapped for `to` must read as one "
        "natural line a character would say, in the same form: a question stays a question, an active line stays "
        "active, its emphasis stays. Prefer the shortest `from` that fixes it. Never paste `reading` in: it is plain on "
        "purpose, the published line is not.\n"
        'Example: translation "SHE\'S GONNA HATE THIS PLACE!" for a source saying she will love it: from "HATE", to "LOVE".'
    )


CONFIRM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"wrong": {"type": "boolean"}, "from": {"type": "string"}, "to": {"type": "string"}},
    "required": ["wrong", "from", "to"],
    "additionalProperties": False,
}
# The second look's hint for a translation far shorter than its reading (``shorter``).
OMISSION_HINT = "omission: the translation is far shorter than the source; keep all of it and add only what it leaves out"
# Grades at which a correction says what the reading says, adding and dropping nothing (``vouch``).
FAITHFUL = frozenset({"same", "close"})
# Lines graded per call (1: one call per line). On the dev pages Gemma 4 12B suspected the same
# seeded lines batched as one by one (23 of 33 either way) and two more clean ones of 81, for a
# third of the prompt tokens and a sixth of the calls.
GRADE_BATCH = 12
_SPLICE_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")


def splice(translation: str, wrong: str, right: str, keep: bool = False, english: bool = True) -> Optional[str]:
    """``translation`` with the words ``wrong`` replaced by ``right``; matching ignores case, spacing
    and the punctuation between words. None when ``wrong`` is not in it (or names no word).
    ``keep`` (an omission): what is there stays; a ``right`` that does not repeat those words goes
    after them instead of over them. ``english``: "a"/"an" are made to agree after the swap; off
    for any other target, where "a" is a word of its own (Spanish "voy a escuela")."""
    words = _SPLICE_WORD.findall(wrong)
    if not words:
        return None
    original = translation
    # The letterer's line-break hyphens ("AL-READY") mean nothing once the line is re-laid: splice
    # on the joined words.
    translation = dehyphenate(translation, wrong + " " + right)
    wrong = dehyphenate(wrong, right)
    words = _SPLICE_WORD.findall(wrong)
    if keep:
        said = {word.lower() for word in _SPLICE_WORD.findall(right)}
        if not all(word.lower() in said for word in words):
            right = wrong.strip() + " " + right.strip()
    pattern = r"(?<![^\W_])" + r"[\W_]*".join(re.escape(word) for word in words) + r"(?![^\W_])"
    match = re.search(pattern, translation, re.IGNORECASE)
    if match is None:
        return None
    start, end = match.start(), match.end()
    right = right.strip()
    # Punctuation both ``wrong`` and ``right`` carry on a side replaces the translation's there
    # ("HIM?" -> "YOU?" keeps one question mark).
    wrong = wrong.strip()
    tail = wrong[len(wrong.rstrip("!?.,…~-—")) :]
    if tail and right.endswith(tail) and translation[end : end + len(tail)] == tail:
        end += len(tail)
    head = wrong[: len(wrong) - len(wrong.lstrip("\"'“‘…(-"))]
    if head and right.startswith(head) and translation[max(start - len(head), 0) : start] == head:
        start -= len(head)
    rest = translation[end:]
    # A replacement that ends a sentence takes the place of the translation's own final mark ("SPY!." is "SPY!").
    if right and right[-1] in "!?.…" and rest.strip() and not rest.strip(" !?.…~"):
        rest = ""
    joined = " ".join((translation[:start] + right + rest).split())
    return undouble(fix_articles(joined, original) if english else joined, original)


_NEGATION = re.compile(r"n['’]t\b|\b(?:not|never|no|nothing|nobody|none|nowhere)\b", re.IGNORECASE)
_NEGATIVE_ONLY = re.compile(r"\b(?:yet|either|anymore|any longer|at all)\b", re.IGNORECASE)
_POSITIVE_ONLY = re.compile(r"\b(?:already|too)\b", re.IGNORECASE)


def polarity_fits(translation: str, fix: str) -> bool:
    """Whether a fix that removed or added a negation left no word that needs the old polarity: a
    removed "not" strands "yet", "either", "anymore"; an added one strands "already", "too"."""
    before, after = bool(_NEGATION.search(translation)), bool(_NEGATION.search(fix))
    if before and not after:
        return not _NEGATIVE_ONLY.search(fix)
    if after and not before:
        return not _POSITIVE_ONLY.search(fix)
    return True


_AN_EXCEPTIONS = ("uni", "use", "usu", "one", "once", "eu", "ur")
_AN_SILENT_H = ("hour", "honest", "honor", "honour", "heir")


def fix_articles(text: str, original: str) -> str:
    """``text`` with "a"/"an" agreeing with the word after it where that pair is not the original's
    ("A EXORCISM" after a noun swap is "AN EXORCISM")."""
    kept = {(m.group(1).lower(), m.group(3).lower()) for m in _ARTICLE.finditer(original)}

    def agree(match: re.Match) -> str:
        article, space, word = match.group(1), match.group(2), match.group(3)
        if (article.lower(), word.lower()) in kept:
            return match.group(0)
        low = word.lower()
        vowel = (low[:1] in "aeiou" and not low.startswith(_AN_EXCEPTIONS)) or low.startswith(_AN_SILENT_H)
        right = "an" if vowel else "a"
        if article.isupper() and (len(article) > 1 or word.isupper()):
            right = right.upper()
        elif article[:1].isupper():
            right = right.capitalize()
        return right + space + word

    return _ARTICLE.sub(agree, text)


_ARTICLE = re.compile(r"\b(a|an)(\s+)([A-Za-z]+)", re.IGNORECASE)

_HYPHENATED = re.compile(r"([^\W\d_]+)-\s*([^\W\d_]+)")


def dehyphenate(text: str, reference: str = "") -> str:
    """``text`` with line-break hyphens joined: a part of two letters or fewer on either side
    ("AL-READY"), or a pair whose joined form ``reference`` writes ("SCUMBAGS"). A one-letter prefix
    (X-RAY, T-SHIRT) and real compounds of whole words stay."""
    known = {w.lower() for w in _SPLICE_WORD.findall(reference)}

    def join(match: re.Match) -> str:
        left, right = match.group(1), match.group(2)
        if len(left) == 1:
            return match.group(0)
        if len(left) <= 2 or len(right) <= 2 or (left + right).lower() in known:
            return left + right
        return match.group(0)

    return _HYPHENATED.sub(join, text)


def undouble(text: str, original: str) -> str:
    """``text`` with a run of words said twice in a row removed, unless ``original`` says it twice
    too ("COME ON, COME ON" is the character's): a splice that repeats its neighbour ("I DO. I DO.")."""
    for size in range(8, 0, -1):
        while True:
            tokens = [(m.group(0).lower(), m.start(), m.end()) for m in _SPLICE_WORD.finditer(text)]
            said = [t[0] for t in tokens]
            before = [m.group(0).lower() for m in _SPLICE_WORD.finditer(original)]
            cut = None
            for i in range(len(tokens) - 2 * size + 1):
                run = said[i : i + size]
                if run == said[i + size : i + 2 * size] and not any(before[j : j + size] == run and before[j + size : j + 2 * size] == run for j in range(len(before))):
                    if size == 1 and len(run[0]) <= 2 and not text[tokens[i][2] : tokens[i + 1][1]].strip():
                        continue  # "NO NO", "I I-" stammers are lettering
                    cut = (tokens[i + size - 1][2], tokens[i + 2 * size - 1][2])
                    break
            if cut is None:
                break
            text = " ".join((text[: cut[0]] + text[cut[1] :]).split())
    return text


async def _ask(provider: Any, system: str, user: dict[str, Any], schema: dict[str, Any], max_tokens: int, result: "Proofread") -> Optional[dict[str, Any]]:
    text = json.dumps(user, ensure_ascii=False)
    completion = await provider.complete(system, text, schema, name="manga_meaning_check", max_tokens=max_tokens)
    result.calls += 1
    _count(result, system, text, completion)
    try:
        value = json.loads(completion.text.strip())
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


async def compare(provider: Any, line: Line, target: Language, result: "Proofread", text: Optional[str] = None) -> str:
    """How the translation's (or ``text``'s) meaning relates to the reading's (``MEANINGS``); "" when unreadable."""
    value = await _ask(provider, meaning_system(target), {"A": line.literal, "B": line.translation if text is None else text}, MEANING_SCHEMA, 16, result)
    meaning = str((value or {}).get("meaning") or "")
    return meaning if meaning in MEANINGS else ""


async def grade(provider: Any, lines: list[Line], target: Language, result: "Proofread", batch: Optional[int] = None) -> None:
    """Sets each line's ``grade``: one call per line, or ``batch`` lines per call (``GRADE_BATCH``).
    A batch reply that cannot be read leaves its lines ungraded (they are not suspected)."""
    # Qwen3-4B (round five, dev pages): one line per call caught 17 of 24 flips against 12 batched,
    # but at 9 calls and 2,900 prompt tokens a page; four lines per call caught 11. Batched stays.
    size = GRADE_BATCH if batch is None else batch
    if size <= 1:
        for line in lines:
            line.grade = await compare(provider, line, target, result)
        return
    for start in range(0, len(lines), size):
        chunk = lines[start : start + size]
        user = {"pairs": [{"n": line.n, "A": line.literal, "B": line.translation} for line in chunk]}
        value = await _ask(provider, meaning_system(target, batch=True), user, meaning_batch_schema(chunk), 16 + 20 * len(chunk), result)
        by_number = {line.n: line for line in chunk}
        for entry in (value or {}).get("grades") or []:
            line = by_number.get(_number(entry.get("n")) if isinstance(entry, dict) else None)
            meaning = str(entry.get("meaning") or "") if isinstance(entry, dict) else ""
            if line is not None and not line.grade and meaning in MEANINGS:
                line.grade = meaning


async def confirm(provider: Any, line: Line, target: Language, source: Optional[Language], result: "Proofread", found: str = "") -> Optional[dict[str, Any]]:
    """The second look, with the source line: is the translation wrong, and the smallest change
    that corrects it (``from``, ``to``)."""
    budget = 40 + 2 * knowledge.estimate_tokens(line.translation) + knowledge.estimate_tokens(line.literal)
    user = {"source": line.source, "reading": line.literal, "translation": line.translation, **({"found": found} if found else {})}
    return await _ask(provider, confirm_system(target, source), user, CONFIRM_SCHEMA, budget, result)


RANK = {meaning: rank for rank, meaning in enumerate(MEANINGS)}
OMISSION_MIN_NEW = 3  # words an omission's correction must add that the line does not have ("IT'S PROBABLY." is no clause)


def _new_words(fix: str, translation: str) -> list[str]:
    have = {word.lower() for word in _WORD.findall(translation)}
    return [word for word in _WORD.findall(fix) if word.lower() not in have]


async def vouch(provider: Any, line: Line, fix: str, target: Language, result: "Proofread", baseline: str) -> str:
    """Why ``fix`` must not be offered, or "" when it passes: it must change the line, keep the
    source's numbers, and, graded against the reading in a call of its own, read strictly better
    than the line it replaces (``baseline``, the line's own grade: "loose" beats "different").

    Comparative on purpose (round five): the reading is the model's own and noisy, so an absolute
    bar ("same" or "close", round four) withheld fixes that were exactly right (a flipped adjective
    restored, a negation removed) and lost them to the official line in the blind comparison."""
    from . import checks

    if not fix:
        return "no correction"
    if checks.same_text(fix, line.translation):
        return "the correction changes nothing"
    if not checks.numbers_agree(line.source, fix, line.translation):
        return "the correction changes a number of the source"
    if not line.literal:
        return "no reading to check the correction against"
    value = await _ask(provider, vouch_system(target), {"A": line.literal, "B": fix}, VOUCH_SCHEMA, 24, result) or {}
    meaning = str(value.get("meaning") or "")
    if meaning not in RANK or RANK[meaning] >= RANK.get(baseline, RANK["different"]):
        return f"the correction reads as {meaning or 'unreadable'} against the source, no better than the line ({baseline})"
    if value.get("natural") is False:
        return f"the correction is not a natural {target.display_name} line"
    return ""


def vouch_system(target: Language) -> str:
    """The grading prompt, and whether B reads as a line a person would say: a splice can leave a
    right meaning in broken words ("I'LL NOT GIVING UP" on the dev pages)."""
    return meaning_system(target) + (
        f"\n`natural`: whether B, as written, is a grammatical, natural {target.display_name} line a character could say "
        "(slang and dropped words are fine; broken grammar, a garbled or repeated part, or a fragment stuck on is not)."
    )


VOUCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"meaning": {"type": "string", "enum": list(MEANINGS)}, "natural": {"type": "boolean"}},
    "required": ["meaning", "natural"],
    "additionalProperties": False,
}


# The second reading (``reread``): a suspect line whose translation the page's context bears out
# was misread the first time (やる as "I'll do it" where the speaker hands something over; a child's
# third-person speech). On the round-four test pages two clean lines were rewritten so.
REREAD = True


def reread_system(target: Language, source: Optional[Language] = None) -> str:
    source_name = source.display_name if source else "the source language"
    name = target.display_name
    return (
        f"You read a manga page in {source_name}. `lines` holds its lines in reading order (read by OCR, so a character "
        f"may be wrong). For each line numbered in `read`, write `meaning`: what the speaker means there, in plain {name}, "
        "read in the context of the lines around it: who is meant when the line leaves the person out, what a short "
        "verb does here, a child's or a character's odd way of speaking (speaking of themselves by name, baby talk). "
        "Keep every negation and number. Never copy the source."
    )


def reread_schema(lines: list[Line]) -> dict[str, Any]:
    numbers = sorted(line.n for line in lines)
    item = {
        "type": "object",
        "properties": {"n": {"type": "integer", "enum": numbers}, "meaning": {"type": "string"}},
        "required": ["n", "meaning"],
        "additionalProperties": False,
    }
    return {"type": "object", "properties": {"lines": {"type": "array", "items": item, "minItems": len(numbers), "maxItems": len(numbers)}}, "required": ["lines"], "additionalProperties": False}


async def reread(provider: Any, suspects: list[Line], lines: list[Line], target: Language, source: Optional[Language], result: "Proofread") -> list[Line]:
    """The suspects the second reading does not clear: each is read again in its page's context
    (one call), and its translation graded against that reading (one call); a line graded
    "same", "close" or "loose" against it stands as it is."""
    user = {"lines": [{"n": line.n, "source": line.source} for line in lines], "read": [line.n for line in suspects]}
    budget = 24 + 50 * len(suspects)
    value = await _ask(provider, reread_system(target, source), user, reread_schema(suspects), budget, result)
    second: dict[int, str] = {}
    for entry in (value or {}).get("lines") or []:
        if isinstance(entry, dict):
            text = " ".join(str(entry.get("meaning") or "").split())
            n = _number(entry.get("n"))
            if text and (_cjk(target) or not _SOURCE_SCRIPT.search(text)):
                second[n] = text
    probes = [Line(line.n, line.content, line.source, line.translation, literal=second[line.n]) for line in suspects if line.n in second]
    if not probes:
        return suspects
    await grade(provider, probes, target, result, batch=max(len(probes), 2))
    cleared = {probe.n for probe in probes if probe.grade and probe.grade not in SUSPECT}
    for line in suspects:
        if line.n in cleared:
            result.dropped.append((line.content, "mistranslation", f"a second reading in context bears the translation out: \"{second[line.n]}\""))
    return [line for line in suspects if line.n not in cleared]


def _issue(kind: str, severity: str, message: str, line: Line, fix: str) -> QaIssue:
    fix = " ".join(fix.split())
    return QaIssue(
        kind=kind,
        severity=severity,
        message=f"{message} (the source, read alone: \"{line.literal}\").",
        suggestion=fix if fix and fix != line.translation.strip() else None,
    )


async def read_literally(provider: Any, lines: list[Line], target: Language, source: Optional[Language], result: "Proofread") -> None:
    """Fills each line's ``literal`` from one call that sees only the sources."""
    system = literal_system(target, source)
    user = literal_payload(lines)
    budget = max(LITERAL_TOKENS, 48 + 60 * len(lines))
    completion = await provider.complete(system, user, literal_schema(lines), name="manga_literal_reading", max_tokens=budget)
    result.calls += 1
    _count(result, system, user, completion)
    literals = parse_literals(completion.text, lines, keep_script=_cjk(target))
    for line in lines:
        line.literal = literals.get(line.n, "")
        if line.literal:
            result.literals[line.content] = line.literal


async def check_meanings(provider: Any, lines: list[Line], target: Language, source: Optional[Language], result: "Proofread") -> dict[int, QaIssue]:
    """The meaning check of ``lines`` (see the section notes): issues by line number. A line whose
    only open program finding is a changed number goes to the second look without grading; its
    vouched correction completes that finding."""
    from . import checks

    await read_literally(provider, lines, target, source, result)
    # Nothing to compare, or the program already has a finding (the proofreading call answers it).
    graded = [line for line in lines if line.literal and words(line.literal) >= CHECK_MIN_WORDS and not line.found]
    # Program findings with no correction of the program's own: a changed number it could not swap
    # in place, a glossary rendering missing with no near spelling to respell.
    numbered = [line for line in lines if line.open == ["number"]]
    named = [line for line in lines if line.open == ["glossary"] and line.literal]
    missing = {line.n for line in graded if shorter(line.literal, line.translation)}
    # A name the source writes and the translation lacks is evidence of a dropped part too.
    missing |= {line.n for line in named if shorter(line.literal, line.translation, NAMED_SHORTER)}
    await grade(provider, [line for line in graded if line.n not in missing], target, result)
    suspects = [line for line in graded if line.n not in missing and line.grade in SUSPECT]
    if REREAD and suspects:
        kept = await reread(provider, suspects, lines, target, source, result)
        for line in suspects:
            if line not in kept:
                line.grade = "loose"  # cleared: a fair free rendering
        suspects = kept
    found: dict[int, QaIssue] = {}
    for line in lines:
        number = line in numbered
        term = line in named and line.n not in missing
        omitted = line.n in missing
        if not (number or term or omitted or line in suspects):
            continue
        glossary_hint = next((f for f in line.found if f.startswith("glossary:")), "")
        if number:
            hint = next((f for f in line.found if f.startswith("number:")), "")
        elif omitted:
            hint = OMISSION_HINT + (f"; also {glossary_hint}" if glossary_hint else "")
        else:
            hint = (glossary_hint + "; give the smallest swap that puts the glossary's rendering in") if term else ""
        verdict = await confirm(provider, line, target, source, result, found=hint) or {}
        wrong = verdict.get("wrong")
        fix = ""
        if wrong or number or term:
            # An omission keeps the whole translation and adds what it leaves out (wherever the
            # model said it goes, a clause placed mid-line read as broken English on the dev pages).
            wrong_part = line.translation if omitted else str(verdict.get("from") or "")
            fix = splice(line.translation, wrong_part, str(verdict.get("to") or ""), keep=omitted, english=_english(target)) or ""
            # What the line is graded against: its own grade; an omission or a glossary miss, not
            # graded, must not read as "different"; a changed number counts as a contradiction.
            baseline = line.grade if line.grade in RANK else ("contradicts" if number else "different")
            if term and fix and not all(checks.renders(fix, rendering) for rendering in line.use.values()):
                why = "the correction does not use the glossary's rendering"
            elif omitted and fix and len(_new_words(fix, line.translation)) < OMISSION_MIN_NEW:
                why = "the addition is a fragment of words the line already has"
            elif fix and _english(target) and not polarity_fits(line.translation, fix):
                why = "the correction leaves a word that needs the old negation (\"yet\", \"either\", \"already\")"
            else:
                why = await vouch(provider, line, fix, target, result, baseline) if fix else "the change does not match the translation's words"
            if why:
                if verdict.get("from") or verdict.get("to"):
                    result.withheld.append((line.content, why, fix))
                fix = ""
            else:
                result.vouched += 1
        if number:
            found[line.n] = QaIssue(kind="number", severity="warning", message="", suggestion=fix or None)
        elif term:
            found[line.n] = QaIssue(kind="glossary", severity="warning", message="", suggestion=fix or None)
        elif omitted:
            found[line.n] = _issue("omission", "warning", "The translation leaves out part of the source", line, fix)
        elif wrong is not False:
            found[line.n] = _issue("mistranslation", "error", "The translation does not say what the source says", line, fix)
    return found


# -- context ---------------------------------------------------------------------------------------


def initial_context(scene: Snapshot, page: str, lines: list[Line], language: str, budget: int) -> knowledge.Retrieved:
    """What the first request carries: the knowledge the lines call for, the rest of the page and the
    pages before, within ``budget`` tokens. Nothing with the project's knowledge off."""
    return knowledge.retrieve(scene, page, [line.source for line in lines], frozenset(line.content for line in lines), language, budget, ROLE)


def knowledge_enabled(scene: Snapshot) -> bool:
    stored = scene.project_component(Knowledge)
    return stored is None or stored.enabled


def worth_asking(need: Need, lines: list[Line], retrieved: knowledge.Retrieved) -> Need:
    """The part of ``need`` that can bring something the prompt lacks: terms written in the page's
    lines that the prompt's glossary does not already render, pages not already in ``context``
    (``look_up`` then drops what it finds nothing for), the story when none was sent."""
    text = knowledge.normalize("\n".join(f"{line.source}\n{line.translation}" for line in lines))
    compact = "".join(text.split())
    given = [knowledge.normalize(t.source) for t in retrieved.glossary] + [knowledge.normalize(v) for line in lines for v in (*line.use, *line.use.values())]
    terms = [
        term
        for term in need.terms
        if knowledge.mentioned(term, text, compact) and not any(knowledge.normalize(term) in g or g in knowledge.normalize(term) for g in given if g)
    ]
    return Need(terms=terms, pages=list(need.pages), story=need.story and not retrieved.story)


def look_up(
    scene: Snapshot, page: str, need: Need, language: str, budget: int, served: dict[str, Any], known: frozenset = frozenset(), notes: frozenset = frozenset()
) -> dict[str, Any]:
    """What ``need`` asks for, within ``budget`` tokens, added to ``served`` (what earlier rounds
    looked up). A term, page or story asked for again is not looked up twice; what finds nothing
    is left out, and so are page lines already in ``known`` ([source, translation] pairs) and
    notes already in ``notes`` (sent with the first prompt)."""
    stored = scene.project_component(Knowledge) or Knowledge()
    spent = 0

    def fits(item: Any) -> bool:
        nonlocal spent
        price = knowledge.estimate_tokens(json.dumps(item, ensure_ascii=False))
        if spent + price > budget:
            return False
        spent += price
        return True

    tried = served.setdefault("_tried", set())
    terms = served.setdefault("terms", {})
    for term in need.terms:
        if ("term", term) in tried:
            continue
        tried.add(("term", term))
        found = [item for item in _term_entries(stored, term, language) if item.get("note") not in notes or "source" in item]
        found = [item for item in found if fits(item)]
        if found:
            terms[term] = found
    order = scene.pages()
    position = order.index(page) if page in order else 0
    pages = served.setdefault("pages", {})
    for offset in need.pages:
        key = f"{offset:+d}"
        index = position + offset
        if ("page", key) in tried or not 0 <= index < len(order):
            continue
        tried.add(("page", key))
        lines = []
        for line in knowledge._lines(scene, order[index], language, ROLE):
            pair = [line.source, line.translation]
            if (line.source, line.translation) in known:
                continue
            if not fits(pair):
                break
            lines.append(pair)
        if lines:
            pages[key] = lines
    if need.story and ("story", "") not in tried:
        tried.add(("story", ""))
        story = knowledge._story(scene, order, position, max(budget - spent, 0))
        if story:
            served["story"] = story
    return {key: value for key, value in served.items() if value and not key.startswith("_")}


def _term_entries(stored: Knowledge, term: str, language: str) -> list[dict[str, str]]:
    """Glossary entries that name ``term`` (or that it names), then the notes most about it."""
    text = knowledge.normalize(term)
    compact = "".join(text.split())
    found = []
    named = set()
    for entry in stored.entries:
        names = knowledge._names(entry)
        if not names:
            continue
        if any(knowledge.mentioned(name, text, compact, fuzzy=True) for name in names) or any(
            knowledge.mentioned(term, knowledge.normalize(value), "".join(knowledge.normalize(value).split())) for value in (*names, entry.translation) if value
        ):
            translation = entry.translation if entry.language in (None, language) else ""
            chunks = knowledge._chunks(entry.text) if entry.text.strip() else []
            found.append(_term(entry.source, translation, chunks[0] if chunks else ""))
            named.add(entry.id)
    chunks = [knowledge._labelled(entry, chunk) for entry in stored.entries if entry.id not in named for chunk in knowledge._chunks(entry.text)]
    if chunks:
        best = knowledge.rank([knowledge.tokenize(chunk) for chunk in chunks], knowledge.tokenize(term))[:NOTES_PER_TERM]
        found += [{"note": chunks[index]} for index, score in best if score > 0]
    return found


# -- the request -----------------------------------------------------------------------------------


async def proofread(
    provider: Any,
    scene: Snapshot,
    page: str,
    lines: list[Line],
    *,
    target: Language,
    source: Optional[Language] = None,
    instructions: Optional[str] = None,
    budget: int = 600,
    fetch_rounds: int = 1,
    max_tokens: int = 600,
    meaning_check: bool = False,
) -> Proofread:
    """Proofreads ``lines`` of ``page`` with ``provider`` (a translation provider that can
    ``complete``): one call, plus at most ``fetch_rounds`` more, each only when the model asked
    for context and the lookup found something new. The last answer stands. A reply that is not
    JSON ends it with ``error`` set; provider errors propagate."""
    result = Proofread()
    if not lines:
        return result
    language = target.tag
    retrieved = initial_context(scene, page, lines, language, budget)
    known = frozenset((c.source, c.translation) for c in retrieved.context) | frozenset((line.source, line.translation) for line in lines)
    notes = frozenset(retrieved.background + retrieved.notes)
    can_ask = fetch_rounds > 0 and knowledge_enabled(scene)
    # The meaning check counts words (what is graded, what is "far shorter"), and a target written
    # without spaces has none to count: every line would pass unread and the proofreading call
    # would be skipped. Such a target gets the proofreading call alone.
    meaning_check = meaning_check and not _cjk(target)
    compared = await check_meanings(provider, lines, target, source, result) if meaning_check else {}
    by_number = {line.n: line for line in lines}
    for n, issue in compared.items():
        result.issues.setdefault(by_number[n].content, []).append(issue)
        result.checked.add(id(issue))
    if meaning_check and not any(line.layout or any(kind not in ("number", "glossary") for kind in line.open) for line in lines):
        # The meaning check covers meaning; the proofreading call is left only what it alone does:
        # corrections for the program's findings it has none for (a changed number and a missing
        # glossary rendering get theirs in the meaning check) and wording that fits a block's layout.
        return result
    system = system_prompt(target, source, instructions)
    served: dict[str, Any] = {}
    looked_up: dict[str, Any] = {}
    rounds_left = fetch_rounds if can_ask else 0
    while True:
        asking = rounds_left > 0
        user = payload(lines, target, retrieved, looked_up, source)
        completion = await provider.complete(system, user, reply_schema(lines, asking), name="manga_proofreading", max_tokens=max_tokens)
        result.calls += 1
        _count(result, system, user, completion)
        try:
            reply = parse_reply(completion.text, lines)
        except ValueError as error:
            result.error = f"the reply was not the JSON asked for: {error}"
            return result
        result.truncated = reply.truncated
        if asking and not reply.need.empty():
            result.asked.append(reply.need)
            rounds_left -= 1
            before = json.dumps(looked_up, ensure_ascii=False, sort_keys=True)
            wanted = worth_asking(reply.need, lines, retrieved)
            looked_up = look_up(scene, page, wanted, language, budget, served, known, notes) if not wanted.empty() else looked_up
            if json.dumps(looked_up, ensure_ascii=False, sort_keys=True) != before:
                result.fetched += 1
                continue
        for n, issues in reply.issues.items():
            kept = [issue for issue in issues if not (meaning_check and (issue.kind in MEANING_KINDS or issue.kind == "number"))]
            if kept:
                result.issues.setdefault(by_number[n].content, []).extend(kept)
        return result


def vet(result: Proofread, lines: list[Line], page_lines: list[Line], found: dict[str, list[QaIssue]], requirements: dict[str, list[Any]], style: Any) -> None:
    """Keeps only the model issues a user should see, in place (see the module notes).

    ``page_lines`` are all the page's lines (for corrections copied from another bubble);
    ``found`` the program's findings by content, which a model fix for them completes;
    ``requirements`` each line's glossary renderings (``checks.requirements``); ``style`` the
    page's ``checks.PageStyle``."""
    from . import checks

    by_content = {line.content: line for line in lines}
    texts = {line.content: line.translation for line in page_lines}
    kept: dict[str, list[QaIssue]] = {}
    fixes: dict[str, list[str]] = {}
    for content, why, _ in result.withheld:
        result.dropped.append((content, "correction", f"withheld: {why}"))
    for content, issues in result.issues.items():
        line = by_content[content]
        for issue in issues:
            why = ""
            if line.grade in FAITHFUL and issue.kind not in FOUND_ONLY:
                # The meaning check found this line says what its source says: its text stays.
                result.dropped.append((content, issue.kind, f"the meaning check graded the line {line.grade}"))
                continue
            if issue.kind == WORDING:
                result.dropped.append((content, issue.kind, "the model judged the meaning right"))
                continue
            if issue.kind not in FOUND_ONLY and STYLE_REASON.search(issue.message):
                result.dropped.append((content, issue.kind, "the reason is a wording preference"))
                continue
            if issue.kind == "too_long" and line.max_chars is None:
                why = "too_long without a length to keep to"
            elif issue.suggestion is None:
                why = "" if id(issue) in result.checked and issue.kind not in FOUND_ONLY else "no correction"
            else:
                fix, why = checks.guard_fix(issue.suggestion, line.translation, requirements.get(content, []), style, line.source)
                if fix is not None:
                    issue.suggestion = fix
                    if issue.kind == "too_long" and len(fix) >= len(line.translation):
                        why = "too_long with a correction no shorter"
            if not why and issue.suggestion is not None:
                # A correction on a line with a program finding completes that finding when it
                # resolves it (no source text left; every glossary rendering in).
                complete = all(r.met(issue.suggestion) for r in requirements.get(content, []))
                for finding in found.get(content, []):
                    if finding.suggestion is None and finding.kind in FOUND_ONLY and (finding.kind == "untranslated" or complete):
                        if finding.kind == "number" and issue.kind != "number":
                            continue  # a changed number takes only the meaning check's vouched correction
                        if finding.kind == "glossary" and words(issue.suggestion) < words(line.translation) - 1:
                            continue  # a name put right must not cost the rest of the line ("PAPA!!" for "FATHER! SAVE MEEEE!")
                        finding.suggestion = issue.suggestion
            if issue.kind in FOUND_ONLY:
                # Only the program raises these; the model's report of one is at most its correction.
                result.dropped.append((content, issue.kind, why or "the program's finding carries it"))
                continue
            if why:
                result.dropped.append((content, issue.kind, why))
                continue
            kept.setdefault(content, []).append(issue)
            fixes.setdefault(content, []).append(issue.suggestion or "")
    # A correction that is another bubble's text is kept only for a swap: that bubble's correction
    # is this one's text. Otherwise the finding stays, without the correction.
    for content, issues in kept.items():
        for issue in issues:
            others = [other for other, text in texts.items() if other != content and checks.alike(issue.suggestion or "", text)]
            if others and not any(checks.alike(fix, texts[content]) for other in others for fix in fixes.get(other, [])):
                result.dropped.append((content, issue.kind, "correction copied another line; kept without it"))
                issue.suggestion = None
    result.issues = kept


def _count(result: Proofread, system: str, user: str, completion: Any) -> None:
    prompt_tokens = completion.prompt_tokens
    completion_tokens = completion.completion_tokens
    if prompt_tokens is None:
        prompt_tokens = knowledge.estimate_tokens(system) + knowledge.estimate_tokens(user)
        result.estimated = True
    if completion_tokens is None:
        completion_tokens = knowledge.estimate_tokens(completion.text)
        result.estimated = True
    result.prompt_tokens += prompt_tokens
    result.completion_tokens += completion_tokens
