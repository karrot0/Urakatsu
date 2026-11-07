"""Project knowledge: what the translator should know about a series, retrieved page by page.

An Urakatsu addition: Koharu translates each page on its own. The knowledge base is the
``Knowledge`` project component, so it travels with the project and every change is an
undoable commit. It holds entries of two uses:

* a glossary: an entry with a ``source`` name and aliases, sent whenever the page mentions one;
* notes: free information (a synopsis, a character sheet, a pasted wiki page) split into
  chunks. ``pinned`` entries go with every page; the others are ranked against the page.

The project's own pages are the third source and need no storage: lines around the page and
earlier lines that resemble it become ``TranslationRequest.context``, and a line the user
edited outranks a generated one, so a correction carries forward. A line that names a glossary
entry but does not use its translation is left out: it would teach the model a rendering the
user has since replaced. Lines with no words in them ("…", "!?") and repeats are left out too.

Retrieval is lexical (BM25 over words and CJK character bigrams): no model to load, and it
finds proper nouns exactly. A CJK name of four or more characters also matches with one
character wrong, missing or extra, since OCR misreads one now and then. A note in another
language than the page is found through the glossary, whose matched names and translations
join the query, and through the translations of the pages just before it.

Pinned notes are the ``background``: the same for every page, so providers put them in the
system prompt, where prompt caching (hosted APIs) and KV reuse (llama.cpp) can keep them.

Story so far (``Knowledge.story``, off by default): the translator also writes a sentence or
two on what happens on the page (``PageSummary``), and later pages get the latest of those,
grouped by chapter. It costs a few output tokens per page instead of a separate summary call,
and needs nothing from a provider beyond the shared output schema.

Everything is limited to the provider's ``context_budget``, spent glossary first.
"""

from __future__ import annotations

import math
import re
import threading
import unicodedata
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Iterable, Literal, Optional

from pydantic import BaseModel, Field

from urakatsu.providers.translation.backend import TranslationContext, TranslationTerm
from urakatsu.scene import Generation, Relations, Session, Snapshot, SourceText, TextGroup, TextRole, Translation, generated, is_user, user
from urakatsu.scene.components import Chapters, Knowledge, KnowledgeEntry, PageSummary

PRODUCER = "dev.urakatsu.knowledge"
SERIES_ENTRY = "series"  # the note written from the linked source's series facts
CHUNK_CHARS = 500
NOTE_RESULTS = 4
RECENT_PAGES = 6
MEMORY_RESULTS = 12
MEMORY_MATCH = 0.5  # how alike a line must be (Dice over token weights) to count as a match
USER_BOOST = 0.15  # a line the user edited wins over a generated line that is about as alike
# A token in more lines than this share says little, and walking its lines is the slow part.
MEMORY_COMMON = 0.05
MEMORY_COMMON_MINIMUM = 50
# Shares of the budget; what one leaves unused passes to the next.
GLOSSARY_SHARE = 0.4
NOTES_SHARE = 0.5
STORY_SHARE = 0.3
MEMORY_SHARE = 1 / 3
# OCR misreads a character now and then; a CJK name this long still matches with one edit.
FUZZY_MINIMUM = 4
# A target-language word in more than this share of note chunks says nothing about which to pick.
BRIDGE_COMMON = 0.5
BRIDGE_WEIGHT = 0.5
BRIDGE_PAGES = 2
SUMMARY_CHARS = 300
BM25_K1 = 1.2
BM25_B = 0.75

_WORD = re.compile(r"[^\W_]+")
# Knowledge changes are read-modify-write of one component; concurrent ones must not lose each other.
_lock = threading.Lock()


# -- reading text ----------------------------------------------------------------------------------


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


@lru_cache(maxsize=65536)  # every page of a run reads the same earlier lines
def tokenize(text: str) -> tuple[str, ...]:
    """Latin words whole, everything else as character bigrams: CJK has no spaces to split on."""
    tokens: list[str] = []
    for run in _WORD.findall(normalize(text)):
        if run.isascii():
            tokens.append(run)
        elif len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tuple(tokens)


def estimate_tokens(text: str) -> int:
    """A rough count, high for CJK on purpose: an overrun fails on a local model, spare room does not."""
    wide = sum(1 for character in text if ord(character) > 0x2E7F)
    return wide * 3 // 2 + (len(text) - wide) // 4 + 1


def mentioned(name: str, text: str, compact: str, fuzzy: bool = False) -> bool:
    """Whether normalized ``text`` names ``name``. OCR puts stray spaces in CJK, so that compares
    without spaces (``compact``); a Latin name must stand alone, or "Al" would match "also".
    ``fuzzy`` lets a CJK name of FUZZY_MINIMUM characters or more be one edit off."""
    name = normalize(name).strip()
    if not name:
        return False
    if name.isascii():
        return re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", text) is not None
    name = "".join(name.split())
    if name in compact:
        return True
    return fuzzy and len(name) >= FUZZY_MINIMUM and _near(name, compact)


def _one_edit(a: str, b: str) -> bool:
    """Whether ``a`` and ``b`` differ by at most one substitution, insertion or deletion."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return a[index:] == b[index + 1 :]
    return True


def _near(name: str, compact: str) -> bool:
    """Whether ``compact`` holds ``name`` with one edit. One edit leaves a half of the name intact,
    so only the windows around each place a half occurs are compared."""
    half = len(name) // 2
    left, right = name[:half], name[half:]
    sizes = (len(name) - 1, len(name), len(name) + 1)
    for part, anchored_left in ((left, True), (right, False)):
        start = compact.find(part)
        while start != -1:
            for size in sizes:
                begin = start if anchored_left else start + len(part) - size
                if begin >= 0 and begin + size <= len(compact) and _one_edit(name, compact[begin : begin + size]):
                    return True
            start = compact.find(part, start + 1)
    return False


def _chunks(text: str) -> list[str]:
    """``text`` as chunks of about CHUNK_CHARS, broken between lines where it can be."""
    chunks: list[str] = []
    current = ""
    for line in (line.strip() for line in text.splitlines()):
        if not line:
            continue
        while len(line) > CHUNK_CHARS:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:CHUNK_CHARS])
            line = line[CHUNK_CHARS:]
        if len(current) + len(line) + 1 > CHUNK_CHARS and current:
            chunks.append(current)
            current = ""
        current = f"{current}\n{line}" if current else line
    if current:
        chunks.append(current)
    return chunks


def rank(documents: list[tuple[str, ...]], query: Iterable[str], weights: Optional[dict[str, float]] = None) -> list[tuple[int, float]]:
    """BM25: ``(document index, score)`` of the documents sharing a query token, best first.
    ``weights`` scales a token's part of the score (1 when absent)."""
    if not documents:
        return []
    frequency: Counter[str] = Counter()
    for tokens in documents:
        frequency.update(set(tokens))
    average = sum(len(tokens) for tokens in documents) / len(documents)
    scored = []
    terms = set(query)
    weights = weights or {}
    for index, tokens in enumerate(documents):
        counts = Counter(tokens)
        score = 0.0
        for term in terms & counts.keys():
            idf = math.log(1 + (len(documents) - frequency[term] + 0.5) / (frequency[term] + 0.5))
            denominator = counts[term] + BM25_K1 * (1 - BM25_B + BM25_B * len(tokens) / max(average, 1))
            score += weights.get(term, 1.0) * idf * counts[term] * (BM25_K1 + 1) / denominator
        if score > 0:
            scored.append((index, score))
    return sorted(scored, key=lambda item: item[1], reverse=True)


# -- retrieval -------------------------------------------------------------------------------------


@dataclass
class Retrieved:
    glossary: list[TranslationTerm] = field(default_factory=list)
    # Pinned notes: the same for every page.
    background: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # What happened on earlier pages, oldest first, one string per chapter (Story so far).
    story: list[str] = field(default_factory=list)
    context: list[TranslationContext] = field(default_factory=list)
    # Whether the translator should write this page's summary.
    summarize: bool = False


@dataclass
class _Line:
    source: str
    translation: str
    edited: bool


def _lines(scene: Snapshot, page: str, language: str, ignore_role: Optional[str], skip: frozenset[str] = frozenset()) -> list[_Line]:
    """The page's translated lines in this language, apart from ``skip`` and the ``ignore_role``.

    This walks every page of a project per translated page, so it reads the page's state: the
    Snapshot accessors search the pages for each entity and copy each component."""
    state = scene.state.pages[page]
    entities = state.entities
    group = next((child for child in entities[page].children if TextGroup.KIND in entities[child].components), None)
    if group is None:
        return []
    presents = {relation.source: relation.target for relation in state.relations.values() if relation.kind == Relations.PRESENTS}
    lines = []
    for layer in entities[group].children:
        content = presents.get(layer)
        if content is None or content in skip:
            continue
        components = entities[content].components
        role = components.get(TextRole.KIND)
        source = components.get(SourceText.KIND)
        translation = components.get(Translation.KIND)
        if (role is not None and role.role == ignore_role) or source is None or translation is None or translation.language != language:
            continue
        if source.text.value.strip() and translation.text.value.strip():
            lines.append(_Line(source.text.value, translation.text.value, is_user(translation.text.origin)))
    return lines


def _fits(items: list, cost: Callable, budget: int, ordered: bool = False) -> tuple[list, int]:
    """The ``items`` that fit ``budget`` in turn, and what they cost. An item too large is passed
    over so the smaller ones after it still go, unless the items are ``ordered`` (lines in reading
    order, where a gap would mislead): those stop at the first that does not fit."""
    kept = []
    spent = 0
    for item in items:
        price = cost(item)
        if spent + price > budget:
            if ordered:
                break
            continue
        kept.append(item)
        spent += price
    return kept, spent


def _names(entry: KnowledgeEntry) -> list[str]:
    return [name for name in (entry.source, *entry.aliases) if name.strip()]


def _labelled(entry: KnowledgeEntry, text: str) -> str:
    return f"{entry.source}: {text}" if entry.source.strip() else text


def _glossary(entries: list[KnowledgeEntry], texts: list[str], language: str) -> tuple[list[TranslationTerm], list[KnowledgeEntry]]:
    """Entries the page names, as terms: a translation for another language is left out, its note kept."""
    text = normalize("\n".join(texts))
    compact = "".join(text.split())
    matched = [entry for entry in entries if any(mentioned(name, text, compact, fuzzy=True) for name in _names(entry))]
    # The user's entries first, so they survive the budget.
    matched.sort(key=lambda entry: not is_user(entry.origin))
    terms = []
    for entry in matched:
        translation = entry.translation if entry.language in (None, language) else ""
        if translation.strip() or entry.text.strip():
            terms.append(TranslationTerm(source=entry.source, translation=translation, note=_chunks(entry.text)[0] if entry.text.strip() else ""))
    return terms, matched


def _notes(entries: list[KnowledgeEntry], matched: list[KnowledgeEntry], texts: list[str], bridge: list[str]) -> tuple[list[str], list[str]]:
    """``(pinned, ranked)``: pinned notes, and the chunks of other entries most like the page. An
    entry the page names already sent its first chunk with its glossary term.

    ``bridge`` is target-language text from just before the page: its words find a note written
    in that language when the page names nothing in the glossary. They weigh half, and a word in
    most chunks ("the", "and") is not used."""
    named = {entry.id for entry in matched}

    def rest(entry: KnowledgeEntry) -> list[str]:
        return [_labelled(entry, chunk) for chunk in _chunks(entry.text)[entry.id in named :]]

    pinned = [chunk for entry in entries if entry.pinned for chunk in rest(entry)]
    chunks = [chunk for entry in entries if not entry.pinned for chunk in rest(entry)]
    if not chunks:
        return pinned, []
    documents = [tokenize(chunk) for chunk in chunks]
    query = [*tokenize("\n".join(texts)), *(token for entry in matched for token in tokenize(" ".join((*_names(entry), entry.translation))))]
    weights: dict[str, float] = {}
    if bridge:
        frequency: Counter[str] = Counter()
        for tokens in documents:
            frequency.update(set(tokens))
        own = set(query)
        for token in set(tokenize("\n".join(bridge))) - own:
            if 0 < frequency[token] <= len(documents) * BRIDGE_COMMON:
                weights[token] = BRIDGE_WEIGHT
        query += list(weights)
    best = rank(documents, query, weights)[:NOTE_RESULTS]
    return pinned, [chunks[index] for index, _ in best]


class _Renderings:
    """The glossary's translations in one language, to tell which earlier lines still follow it."""

    def __init__(self, entries: list[KnowledgeEntry], language: str):
        self.entries = []
        for entry in entries:
            if entry.translation.strip() and entry.language in (None, language):
                self.entries.append((_names(entry), _stem(normalize(entry.translation).strip())))

    def follows(self, line: "_Line") -> bool:
        """False when the line names an entry and its translation does not use the entry's."""
        if not self.entries:
            return True
        source = normalize(line.source)
        compact = "".join(source.split())
        translation = normalize(line.translation)
        for names, stem in self.entries:
            if stem not in translation and any(mentioned(name, source, compact) for name in names):
                return False
        return True


def _stem(translation: str) -> str:
    """A translation without its last letters, so an inflected form still counts (Роза, Розы)."""
    if len(translation) < 4:
        return translation
    return translation[: -1 if len(translation) < 7 else -2]


def _useful(lines: Iterable["_Line"], renderings: _Renderings, seen: set[str]) -> list["_Line"]:
    """``lines`` without wordless ones, repeats of a line in ``seen`` (which they join), and ones
    that contradict the glossary."""
    kept = []
    for line in lines:
        key = normalize(line.source).strip()
        if key in seen or not tokenize(line.source) or not renderings.follows(line):
            continue
        seen.add(key)
        kept.append(line)
    return kept


def summary(scene: Snapshot, page: str) -> Optional[PageSummary]:
    state = scene.state.pages.get(page)
    if state is None:
        return None
    return state.entities[page].components.get(PageSummary.KIND)


def _story(scene: Snapshot, order: list[str], position: int, budget: int) -> list[str]:
    """The latest page summaries before ``position`` that fit ``budget``, oldest first, one string per
    chapter: "Chapter 3: summary summary ..."."""
    chapters = scene.project_component(Chapters)
    names = {each: chapter.name for chapter in (chapters.chapters if chapters else []) for each in chapter.pages}
    picked: list[tuple[str, str]] = []
    spent = 0
    for each in reversed(order[:position]):
        found = summary(scene, each)
        if found is None or not found.text.strip():
            continue
        price = estimate_tokens(found.text) + 1
        if spent + price > budget:
            break
        spent += price
        picked.append((names.get(each, ""), found.text.strip()))
    story: list[tuple[str, list[str]]] = []
    for chapter, text in reversed(picked):
        if story and story[-1][0] == chapter:
            story[-1][1].append(text)
        else:
            story.append((chapter, [text]))
    return [f"{chapter}: {' '.join(texts)}" if chapter else " ".join(texts) for chapter, texts in story]


def _memory(segments: list[str], lines: list[_Line]) -> list[_Line]:
    """Earlier lines most like the page's segments: repeated phrases, catchphrases, recurring terms."""
    if not lines:
        return []
    documents = [Counter(tokenize(line.source)) for line in lines]
    postings: dict[str, list[int]] = defaultdict(list)
    for index, tokens in enumerate(documents):
        for token in tokens:
            postings[token].append(index)

    weights = {token: math.log(1 + len(lines) / (1 + len(indexes))) for token, indexes in postings.items()}
    common = max(MEMORY_COMMON_MINIMUM, int(len(lines) * MEMORY_COMMON))
    unseen = math.log(1 + len(lines))

    sizes = [sum(weights[token] for token in tokens) for tokens in documents]
    found: dict[int, float] = {}
    for segment in segments:
        tokens = set(tokenize(segment))
        size = sum(weights.get(token, unseen) for token in tokens)
        shared: Counter[int] = Counter()
        for token in tokens:
            matching = postings.get(token, ())
            if len(matching) <= common:
                for index in matching:
                    shared[index] += weights[token]
        for index, overlap in shared.items():
            likeness = 2 * overlap / (size + sizes[index])
            if likeness >= MEMORY_MATCH:
                found[index] = max(found.get(index, 0.0), likeness + (USER_BOOST if lines[index].edited else 0.0))
    best = sorted(found, key=found.__getitem__, reverse=True)[:MEMORY_RESULTS]
    return [lines[index] for index in best]


def retrieve(
    scene: Snapshot,
    page: str,
    segments: list[str],
    targets: frozenset[str],
    language: str,
    budget: int,
    ignore_role: Optional[str] = None,
) -> Retrieved:
    """What the translator gets for ``segments`` of ``page``.

    ``targets`` are the contents being translated, ``language`` the target tag, ``ignore_role`` the
    text role that is no use as context (sound effects). A project with knowledge turned off gets nothing.
    """
    stored = scene.project_component(Knowledge) or Knowledge()
    result = Retrieved()
    if not stored.enabled or not segments:
        return result
    result.summarize = stored.story

    def cost(line: _Line) -> int:
        return estimate_tokens(line.source + line.translation) + 2

    terms, matched = _glossary(stored.entries, segments, language)
    result.glossary, spent = _fits(terms, lambda term: estimate_tokens(f"{term.source}{term.translation}{term.note}"), int(budget * GLOSSARY_SHARE))
    budget -= spent

    renderings = _Renderings(stored.entries, language)
    seen: set[str] = set()
    order = scene.pages()
    position = order.index(page) if page in order else 0
    earlier = order[max(0, position - RECENT_PAGES) : position]
    # The rest of this page, when only part of it is being translated, reads closest of all.
    neighbours = _useful(_lines(scene, page, language, ignore_role, targets), renderings, seen)
    # Newest first, so a line said again keeps its latest place.
    recent = _useful((line for each in reversed(earlier) for line in reversed(_lines(scene, each, language, ignore_role))), renderings, seen)

    bridge = [line.translation for line in neighbours]
    bridge += [line.translation for each in earlier[-BRIDGE_PAGES:] for line in _lines(scene, each, language, ignore_role)]
    pinned, ranked = _notes(stored.entries, matched, segments, bridge)
    share = int(budget * NOTES_SHARE)
    result.background, spent = _fits(pinned, estimate_tokens, share)
    result.notes, more = _fits(ranked, estimate_tokens, share - spent)
    budget -= spent + more
    if stored.story:
        result.story =_story(scene, order, position, int(budget * STORY_SHARE))
        budget -= sum(estimate_tokens(text) for text in result.story)

    neighbours, spent = _fits(neighbours, cost, budget, ordered=True)
    budget -= spent
    others = [
        line
        for each in order
        if each != page and each not in earlier
        for line in _lines(scene, each, language, ignore_role)
        if normalize(line.source).strip() not in seen
    ]
    memory, spent = _fits(_useful(_memory(segments, others), renderings, seen), cost, int(budget * MEMORY_SHARE))
    budget -= spent
    # The lines nearest the page, kept in reading order.
    kept, _ = _fits(recent, cost, budget, ordered=True)
    lines = memory + list(reversed(kept)) + neighbours
    result.context = [TranslationContext(source=line.source, translation=line.translation) for line in lines]
    return result


# -- editing ---------------------------------------------------------------------------------------


class EntryInput(BaseModel):
    kind: Literal["character", "place", "term", "note"] = "term"
    source: str = ""
    aliases: list[str] = Field(default_factory=list)
    translation: str = ""
    text: str = ""
    pinned: bool = False


def read(snapshot: Snapshot) -> Knowledge:
    return snapshot.project_component(Knowledge) or Knowledge()


def _write(session: Session, change: Callable[[Knowledge], None]) -> Knowledge:
    with _lock:
        snapshot = session.snapshot()
        knowledge = read(snapshot)
        change(knowledge)
        edit = snapshot.edit()
        edit.set_project(knowledge)
        session.commit(edit.finish())
        return knowledge


def _clean(names: list[str]) -> list[str]:
    return [name.strip() for name in names if name.strip()]


def _find(knowledge: Knowledge, entry_id: str) -> KnowledgeEntry:
    for entry in knowledge.entries:
        if entry.id == entry_id:
            return entry
    raise KeyError(f"no knowledge entry {entry_id}")


def add(session: Session, inputs: list[EntryInput], language: Optional[str]) -> Knowledge:
    """Adds entries. One whose source name exists already updates that entry, with what it gives."""

    def change(knowledge: Knowledge) -> None:
        for value in inputs:
            source = value.source.strip()
            if not source and not value.text.strip():
                raise ValueError("an entry needs a name or some text")
            key = normalize(source)
            existing = next((entry for entry in knowledge.entries if key and normalize(entry.source) == key), None)
            if existing is None:
                existing = KnowledgeEntry(id=uuid.uuid4().hex[:12], kind=value.kind, source=source, language=language)
                knowledge.entries.append(existing)
            existing.kind = value.kind
            existing.aliases = _clean([*existing.aliases, *(a for a in value.aliases if a not in existing.aliases)])
            existing.pinned = existing.pinned or value.pinned
            existing.origin = user()
            if value.translation.strip():
                existing.translation = value.translation.strip()
                existing.language = language
            if value.text.strip():
                existing.text = value.text.strip()

    return _write(session, change)


def update(session: Session, entry_id: str, fields: dict, language: Optional[str]) -> Knowledge:
    """Replaces the given fields of an entry; it is the user's from then on."""

    def change(knowledge: Knowledge) -> None:
        entry = _find(knowledge, entry_id)
        for name, value in fields.items():
            setattr(entry, name, _clean(value) if name == "aliases" else value.strip() if isinstance(value, str) else value)
        if not entry.source and not entry.text:
            raise ValueError("an entry needs a name or some text")
        if "translation" in fields:
            entry.language = language
        entry.origin = user()

    return _write(session, change)


def remove(session: Session, entry_ids: list[str]) -> Knowledge:
    def change(knowledge: Knowledge) -> None:
        for entry_id in entry_ids:
            _find(knowledge, entry_id)
        knowledge.entries = [entry for entry in knowledge.entries if entry.id not in entry_ids]

    return _write(session, change)


def set_enabled(session: Session, enabled: bool) -> Knowledge:
    def change(knowledge: Knowledge) -> None:
        knowledge.enabled = enabled

    return _write(session, change)


def set_story(session: Session, enabled: bool) -> Knowledge:
    """Turns Story so far on or off. Summaries already written stay with their pages."""

    def change(knowledge: Knowledge) -> None:
        knowledge.story = enabled

    return _write(session, change)


def clean_summary(text: str) -> str:
    """A summary as stored: one line, at most SUMMARY_CHARS, cut at a sentence end where it can be."""
    text = " ".join(text.split())
    if len(text) <= SUMMARY_CHARS:
        return text
    cut = text[:SUMMARY_CHARS]
    end = max(cut.rfind(mark) for mark in (". ", "! ", "? ", "。"))
    return cut[: end + 1] if end > SUMMARY_CHARS // 2 else cut.rstrip() + "…"


def _series_text(series) -> str:
    facts = []
    if series.authors:
        facts.append("By " + ", ".join(series.authors) + ".")
    if series.tags:
        facts.append("Tags: " + ", ".join(series.tags) + ".")
    summary = series.summary.strip()
    return "\n".join(part for part in (" ".join(facts), summary) if part)


def seed_series(session: Session, series) -> None:
    """Keeps the note written from the source's series facts current, until the user edits it."""
    text = _series_text(series)
    with _lock:
        snapshot = session.snapshot()
        knowledge = read(snapshot)
        existing = next((entry for entry in knowledge.entries if entry.id == SERIES_ENTRY), None)
        if existing is not None and is_user(existing.origin):
            return
        if (existing.text, existing.source) == (text, series.title) if existing is not None else not text:
            return
        if existing is None:
            existing = KnowledgeEntry(id=SERIES_ENTRY, kind="note", pinned=True, origin=generated(Generation(producer=PRODUCER)))
            knowledge.entries.insert(0, existing)
        existing.source = series.title
        existing.text = text
        edit = snapshot.edit()
        edit.set_project(knowledge)
        session.commit(edit.finish())
