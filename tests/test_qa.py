"""QA stage (not in Koharu): stage selection, layout checks, proofreading, fix mode, fingerprints,
and the provider QA shares with translation. Fakes only: no model, no server, no GPU."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

import httpx
import pytest

import urakatsu.pipeline.pipeline as pipeline_module
from urakatsu.core import Operation, Stage
from urakatsu.core.stage import PREREQUISITES
from urakatsu.core.config import ModelSelection, PipelineConfig, QaConfig, TypesettingConfig
from urakatsu.pipeline import Pipeline, PipelineError, ProjectStore, Request, RunStatus, knowledge
from urakatsu.pipeline.knowledge import EntryInput
from urakatsu.pipeline.qa import checks, review_page
from urakatsu.pipeline.qa.layout import check_layout
from urakatsu.pipeline.qa.proofread import Line, parse_reply, reply_schema
from urakatsu.pipeline.stages.qa import QaStage
from urakatsu.pipeline.stages.translation import PRODUCER as TRANSLATION_PRODUCER
from urakatsu.pipeline.stages.translation import TranslationStage
from urakatsu.providers.translation import Completion, TranslatorProvider
from urakatsu.providers.translation.remote.openai_compatible import OpenAiCompatibleTranslator
from urakatsu.renderer import Renderer
from urakatsu.runtimes import RuntimeManager
from urakatsu.scene import (
    At,
    Authored,
    Generation,
    Geometry,
    Page,
    QaPage,
    QaReview,
    RegionKinds,
    Relations,
    Session,
    SourceText,
    TextLayout,
    TextLayoutKind,
    Translation,
    Typography,
)
from urakatsu.scene import components as c
from urakatsu.scene.components import KnowledgeEntry

from fakes import FakeDetectionStage, FakeOcrStage, FakeTranslator, fake_config, fake_processors, fake_registry, page_png

GENERATED = Generation(producer=TRANSLATION_PRODUCER, model="test")
CLEAN = '{"need":{"terms":[],"pages":[],"story":false},"issues":[]}'


class FakeLlm(TranslatorProvider):
    """A translation provider that is a language model: it answers ``complete`` from a script."""

    id = "fake-llm"
    name = "Fake LLM"
    loads = 0

    def __init__(self, settings=None, runtimes=None, runtime=None):
        super().__init__(settings or {"translation": {"model": {"provider": "fake-llm", "model": "m", "vision": False}}}, runtimes, runtime)
        self.replies: list[str] = []
        self.calls: list[dict[str, Any]] = []

    def _load(self) -> None:
        type(self).loads += 1

    async def _translate(self, request, generation) -> list[str]:
        return [f"EN:{segment}" for segment in request.segments]

    async def _complete(self, system, user, schema, name, generation) -> Completion:
        self.calls.append({"system": system, "user": json.loads(user), "schema": schema, "max_tokens": generation.max_tokens})
        text = self.replies.pop(0) if self.replies else CLEAN
        return Completion(text, 100, len(text) // 4)


def issue(n: int, category: str = "mistranslation", severity: str = "critical", fix: str = "Fixed", reason: str = "Wrong.") -> dict:
    return {"n": n, "category": category, "severity": severity, "reason": reason, "fix": fix}


def reply(*issues: dict, need: Optional[dict] = None) -> str:
    return json.dumps({"need": need or {"terms": [], "pages": [], "story": False}, "issues": list(issues)})


# -- stage selection -------------------------------------------------------------------------------


def test_qa_is_the_last_stage_and_waits_for_translation():
    assert Stage.all()[-1] is Stage.QA
    assert Operation.through(Stage.QA).resolve() == [Stage.DETECTION, Stage.OCR, Stage.TRANSLATION, Stage.QA]
    assert Stage.QA not in Operation.through(Stage.SOUND_EFFECTS).resolve()
    assert PREREQUISITES[Stage.QA] == (Stage.TRANSLATION,)


@pytest.mark.parametrize("mode, kept", [("off", False), ("flag", True), ("fix", True)])
def test_a_run_has_qa_only_while_it_is_on(mode, kept):
    stages = pipeline_module._qa_stages(Stage.all(), mode)
    assert (Stage.QA in stages) is kept
    if mode == "off":
        with pytest.raises(ValueError, match="QA is off"):
            pipeline_module._qa_stages([Stage.QA], mode)


@pytest.fixture
def project(isolated_home):
    store = ProjectStore(isolated_home / "projects")
    project_id = store.create("test")
    pages = [store.import_page(project_id, f"p{i}.png", page_png()) for i in range(2)]
    return store, project_id, pages


def make_pipeline(config, translator_stage=None) -> Pipeline:
    registry = fake_registry()
    registry.add(FakeLlm)
    processors = fake_processors()
    if translator_stage is not None:
        processors[Stage.TRANSLATION] = translator_stage
    processors[Stage.QA] = QaStage()
    return Pipeline(config, RuntimeManager("cpu"), registry, processors)


async def test_only_qa_while_it_is_off_is_refused(project):
    store, project_id, _ = project
    pipeline = make_pipeline(fake_config())
    with pytest.raises(PipelineError, match="QA is off"):
        await pipeline.execute(store.session(project_id), Request(operation=Operation.only(Stage.QA)))


async def test_a_full_run_with_a_translation_service_checks_layout_only(project):
    store, project_id, pages = project
    config = fake_config()  # fake-translator: translates, cannot proofread
    config.qa.mode = "flag"
    pipeline = make_pipeline(config)
    report = await pipeline.execute(store.session(project_id), Request(operation=Operation.full()))
    assert report.status is RunStatus.COMPLETED
    scene = store.session(project_id).snapshot()
    for page in pages:
        summary = scene.component(page, QaPage)
        assert summary is not None and not summary.proofread and summary.calls == 0
        assert "only translates" in summary.reason


# -- layout checks ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def renderer() -> Renderer:
    return Renderer(TypesettingConfig())


def lettered(tmp_path, blocks: list[dict], width: float = 600, height: float = 400):
    """A page of balloons, each with a translated text layer flowing in it."""
    session = Session(tmp_path)
    edit = session.snapshot().edit()
    page = edit.add_page(Page(label="p", width=width, height=height))
    contents = []
    for block in blocks:
        balloon = edit.add_analysis_region(page, At.END, RegionKinds.BUBBLE, Geometry.rectangle(*block["balloon"]))
        content = edit.add_text_content(page)
        edit.set(content, SourceText(text=Authored.user(block.get("source", "source"))))
        edit.set(content, Translation(text=Authored.user(block["text"]), language="en-US"))
        layer = edit.add_text_layer(page, At.END, content, TextLayout(kind=TextLayoutKind.PARAGRAPH))
        edit.relate(Relations.FLOWS_IN, layer, balloon)
        if "geometry" in block:
            edit.set(layer, Geometry.rectangle(*block["geometry"]))
        if "typography" in block:
            edit.set(layer, block["typography"])
        contents.append(content)
    session.commit(edit.finish())
    return session.snapshot(), page, contents


def kinds(blocks, content) -> set[str]:
    return {issue.kind for issue in blocks[content].issues}


def test_a_clean_page_has_no_findings(tmp_path, renderer):
    scene, page, contents = lettered(tmp_path, [{"balloon": (20, 20, 220, 140), "text": "Hello there!"}, {"balloon": (300, 40, 220, 140), "text": "Who are you?"}])
    blocks = check_layout(scene, page, renderer)
    assert set(blocks) == set(contents)
    assert all(not block.issues for block in blocks.values()), {k: [i.kind for i in b.issues] for k, b in blocks.items()}
    assert all(block.max_chars is None for block in blocks.values())


def test_overflow_and_tiny_text_are_found_with_a_length_that_fits(tmp_path, renderer):
    long = "This line is far too long to fit inside such a small speech balloon at this size"
    scene, page, (overflowing, tiny) = lettered(
        tmp_path,
        [
            {"balloon": (20, 20, 120, 60), "text": long, "typography": Typography(size=30, auto_fit=False)},
            {"balloon": (300, 40, 220, 140), "text": "Whisper", "typography": Typography(size=5, auto_fit=False)},
        ],
    )
    blocks = check_layout(scene, page, renderer)
    assert "overflow" in kinds(blocks, overflowing)
    assert 0 < blocks[overflowing].max_chars < len(long)
    assert blocks[overflowing].wording_hint() == "overflows its bubble"
    assert "small_font" in kinds(blocks, tiny)


def test_a_font_size_far_from_the_pages_others_is_an_outlier(tmp_path, renderer):
    balloons = [(20, 20, 160, 100), (200, 20, 160, 100), (380, 20, 160, 100), (20, 200, 160, 100)]
    clean, page, contents = lettered(tmp_path / "a", [{"balloon": b, "text": "Hello there"} for b in balloons])
    sizes = [block.font_size for block in check_layout(clean, page, renderer).values()]
    small = min(sizes) / 3
    blocks_spec = [{"balloon": b, "text": "Hello there"} for b in balloons]
    blocks_spec[3]["typography"] = Typography(size=max(small, 9.5), auto_fit=False)
    scene, page, contents = lettered(tmp_path / "b", blocks_spec)
    blocks = check_layout(scene, page, renderer)
    assert "font_outlier" in kinds(blocks, contents[3])
    assert all("font_outlier" not in kinds(blocks, content) for content in contents[:3])


def _laid_out(lines: list[tuple[str, float, float]], balloon=(0, 0, 300, 200), size: float = 10.0, advance: float = 11.0):
    """A fake laid-out block: ``(text, baseline y, left x)`` per line, glyphs ``advance`` apart."""
    from types import SimpleNamespace

    from urakatsu.pipeline.qa.layout import BlockLayout
    from urakatsu.renderer.frame import RenderBounds
    from urakatsu.renderer.text_renderer import GlyphRun

    glyphs, starts, text = [], [], ""
    for piece, baseline, left in lines:
        starts.append(len(text))
        text += piece
        glyphs += [(0, left + i * advance, baseline) for i, ch in enumerate(piece.rstrip()) if not ch.isspace()]
    run = GlyphRun(font=None, font_size=size, glyphs=glyphs, transform=(1.0, 0.0, 0.0, 1.0, 0.0, 0.0), glyph_skew=None, embolden=0.0, color=(0, 0, 0, 255))
    layer = SimpleNamespace(scene=SimpleNamespace(runs=[run]), placement=(1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    xs = [g[1] for g in glyphs]
    ys = [g[2] for g in glyphs]
    x0, y0, x1, y1 = balloon
    block = BlockLayout(
        layer="l", content="c", text=text, font_size=size,
        ink=RenderBounds(min(xs), min(ys) - 0.7 * size, max(xs) + 0.6 * size - min(xs), max(ys) - min(ys) + 0.7 * size),
        frame=RenderBounds(x0, y0, x1 - x0, y1 - y0),
        balloon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
    )
    return block, layer, tuple(starts)


def test_what_a_reader_sees_is_checked_from_the_laid_out_lines():
    from urakatsu.pipeline.qa.layout import _as_read

    def read(lines, **kw):
        block, layer, starts = _laid_out(lines, **kw)
        _as_read(block, layer, starts)
        return {issue.kind for issue in block.issues}

    # "A" alone on its line, and the next line has room for it.
    assert "stranded_line" in read([("YOU WANNA ", 60, 100), ("FOSTER ", 80, 110), ("A ", 100, 140), ("KID?", 120, 125)])
    # A broken word that fits whole on its line at this size: the next line is as wide as both halves.
    assert "hyphenation" in read([("THE PARA-", 80, 90), ("NORMAL IS REAL STUFF", 100, 40)])
    # Many short lines of small letters in a big balloon.
    assert "underfilled" in read([("WELL ", 70, 130), ("I ", 85, 140), ("TRIED ", 100, 125), ("MY ", 115, 138), ("BEST", 130, 128)], size=8, advance=8)
    # A tall narrow column in a wide balloon.
    assert "narrow_column" in read([("ONE ", 40, 140), ("TWO ", 70, 140), ("SIX ", 100, 140), ("TEN", 130, 140)], balloon=(0, 0, 300, 160), size=20, advance=14)
    # Ink within a third of its size of the bottom outline (but not touching it).
    assert "crowds_outline" in read([("HELLO THERE", 198, 90)], size=10)
    # Evenly set lines in a balloon of their size: nothing.
    assert read([("HELLO THERE,", 90, 85), ("NICE TO MEET YOU.", 120, 60)], size=14, advance=10) == set()


def test_a_misplaced_block_is_found(tmp_path, renderer):
    scene, page, (moved, off_page, first, second) = lettered(
        tmp_path,
        [
            # Its frame hangs out of the balloon's right side.
            {"balloon": (20, 20, 160, 100), "text": "Over here", "geometry": (120, 30, 160, 80)},
            # Its frame leaves the page.
            {"balloon": (420, 300, 170, 90), "text": "Falling off", "geometry": (520, 330, 160, 60)},
            # Two blocks lettered on the same spot.
            {"balloon": (20, 200, 200, 120), "text": "Same place"},
            {"balloon": (20, 200, 200, 120), "text": "Same place"},
        ],
    )
    blocks = check_layout(scene, page, renderer)
    assert {"crosses_contour", "off_centre"} <= kinds(blocks, moved)
    assert "off_page" in kinds(blocks, off_page)
    assert "overlap" in kinds(blocks, first) and "overlap" in kinds(blocks, second)


# -- proofreading ----------------------------------------------------------------------------------


@pytest.fixture
def book(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    return store, store.create("book")


def page_of(store, project_id, lines):
    """A page of ``(source, translation)`` lines; a third item ``"user"`` makes the translation the user's."""
    page_id = store.import_page(project_id, "page.png", page_png())
    session = store.session(project_id)
    edit = session.snapshot().edit_as(GENERATED)
    contents = []
    for line in lines:
        content = edit.add_text_content(page_id)
        edit.add_text_layer(page_id, At.END, content, TextLayout(kind=TextLayoutKind.PARAGRAPH))
        edit.set(content, SourceText(text=Authored.generated(line[0], GENERATED)))
        edit.set(content, Translation(text=Authored.generated(line[1], GENERATED), language="en-US"))
        contents.append(content)
    session.commit(edit.finish())
    users = [content for content, line in zip(contents, lines) if len(line) > 2 and line[2] == "user"]
    if users:
        edit = session.snapshot().edit()
        for content in users:
            edit.set(content, Translation(text=Authored.user(edit.component(content, Translation).text.value), language="en-US"))
            edit.set(content, Typography(size=20, auto_fit=False))
        session.commit(edit.finish())
    return page_id, contents


def review(store, project_id, page_id, provider, mode="flag", force=False, **config):
    session = store.session(project_id)
    result = asyncio.run(review_page(session.snapshot(), page_id, provider, QaConfig(mode=mode, **config), force=force))
    if not result.patch.is_empty():
        session.commit(result.patch)
    return result


def test_the_prompt_numbers_the_lines_and_carries_what_knowledge_retrieves(book):
    store, project_id = book
    knowledge.add(store.session(project_id), [EntryInput(source="アリス", translation="Alice", text="A girl of twelve.")], "en-US")
    page_id, contents = page_of(store, project_id, [("アリスだ", "It's Arisu"), ("待って", "Wait")])
    provider = FakeLlm()
    provider.replies = [reply(issue(1, "glossary", "major", "It's Alice"), {"n": 9, "category": "grammar", "severity": "minor", "reason": "", "fix": ""})]
    result = review(store, project_id, page_id, provider)
    (call,) = provider.calls
    assert [line["n"] for line in call["user"]["lines"]] == [1, 2]
    # The line carries the glossary rendering its source calls for, and the program's finding.
    assert call["user"]["lines"][0] == {
        "n": 1,
        "source": "アリスだ",
        "translation": "It's Arisu",
        "use": {"アリス": "Alice"},
        "found": ['glossary: The glossary renders アリス as "Alice", which the translation does not use.'],
    }
    assert call["user"]["lines"][1] == {"n": 2, "source": "待って", "translation": "Wait"}
    assert call["user"]["glossary"] == [{"source": "アリス", "translation": "Alice", "note": "A girl of twelve."}]
    assert call["schema"]["properties"]["issues"]["items"]["properties"]["n"]["enum"] == [1, 2]
    assert list(call["schema"]["properties"]) == ["issues", "need"] and call["max_tokens"] == 600
    assert "image" not in json.dumps(call["user"])
    # Line 9 does not exist: dropped. The glossary finding is the program's; the model's report of
    # it only brings the correction. Nothing on line 2.
    first = result.reviews[contents[0]]
    assert [(i.kind, i.severity, i.suggestion, i.applied) for i in first.issues] == [("glossary", "warning", "It's Alice", False)]
    assert result.reviews[contents[1]].issues == []
    assert result.summary.calls == 1 and result.summary.prompt_tokens == 100 and result.summary.sent == 2
    scene = store.session(project_id).snapshot()
    assert scene.component(contents[0], Translation).text.value == "It's Arisu"  # flag mode only records
    assert scene.component(page_id, QaPage).total_prompt_tokens == 100


def test_the_reply_parser_is_tolerant():
    lines = [Line(1, "a", "src", "Hello"), Line(2, "b", "src", "World")]
    parsed = parse_reply('```json\n[{"n": "2", "category": "Too Long", "severity": "CRITICAL", "reason": " x ", "fix": "World"}]\n```', lines)
    (found,) = parsed.issues[2]
    assert (found.kind, found.severity, found.suggestion) == ("too_long", "error", None)  # a fix equal to the text is none
    assert parse_reply('{"issues": [{"n": 1, "category": "made-up", "severity": "?", "fix": "Hi"}]}', lines).issues[1][0].kind == "other"
    with pytest.raises(ValueError):
        parse_reply("not json", lines)
    assert "need" not in reply_schema(lines, can_ask=False)["properties"]
    # A reply cut off at max_tokens keeps the issues it completed.
    cut = '{"issues":[{"n":1,"source_says":"x","category":"grammar","severity":"major","reason":"r","fix":"Hullo"},{"n":2,"source_says":"y","categ'
    parsed = parse_reply(cut, lines)
    assert parsed.truncated and [i.suggestion for i in parsed.issues[1]] == ["Hullo"] and 2 not in parsed.issues


def test_a_clean_page_costs_a_near_empty_reply(book):
    store, project_id = book
    page_id, contents = page_of(store, project_id, [("待って", "Wait!")])
    provider = FakeLlm()
    result = review(store, project_id, page_id, provider)
    assert result.summary.completion_tokens <= 20 and result.summary.issues == 0
    assert store.session(project_id).snapshot().component(contents[0], QaReview).fingerprint


def test_the_model_asks_again_only_when_the_lookup_finds_something_new(book, monkeypatch):
    import sys

    proofread_module = sys.modules["urakatsu.pipeline.qa.proofread"]  # the package's ``proofread`` is the function
    store, project_id = book
    knowledge.add(store.session(project_id), [EntryInput(source="ボブ", translation="Bob", text="Alice's older brother; she calls him 兄さん.")], "en-US")
    before, _ = page_of(store, project_id, [("ボブが来た", "Bob is here")])
    page_id, contents = page_of(store, project_id, [("兄さん", "Big sister")])
    # The first prompt carries no knowledge, so the notes on 兄さん are new to it.
    monkeypatch.setattr(proofread_module, "initial_context", lambda *args: knowledge.Retrieved())
    provider = FakeLlm()
    asking = {"terms": ["兄さん"], "pages": [-1], "story": False}
    provider.replies = [reply(issue(1, "mistranslation", "minor", "Sister"), need=asking), reply(issue(1, "mistranslation", "critical", "Big brother"), need=asking)]
    result = review(store, project_id, page_id, provider, fetch_rounds=1)
    first, second = provider.calls
    assert "looked_up" not in first["user"]
    looked_up = second["user"]["looked_up"]
    assert looked_up["terms"]["兄さん"] == [{"note": "ボブ: Alice's older brother; she calls him 兄さん."}]
    assert looked_up["pages"]["-1"] == [["ボブが来た", "Bob is here"]]  # a model off the schema is still served
    # The last round cannot ask: the format has no ``need``, and its answer replaces the first.
    assert "need" in first["schema"]["properties"] and "need" not in second["schema"]["properties"]
    assert result.summary.calls == 2 and result.proofread.fetched == 1
    assert [i.suggestion for i in result.reviews[contents[0]].issues] == ["Big brother"]
    assert result.summary.prompt_tokens == 200


def test_an_ask_that_finds_nothing_new_costs_no_second_call(book):
    store, project_id = book
    knowledge.add(store.session(project_id), [EntryInput(source="ボブ", translation="Bob", text="Alice's older brother.")], "en-US")
    page_id, contents = page_of(store, project_id, [("ボブの兄さん", "Bob's big sister")])
    provider = FakeLlm()
    # ボブ is already in the prompt's glossary; "Dracula" is not on the page; the story is not kept.
    provider.replies = [reply(issue(1, fix="Bob's big brother"), need={"terms": ["ボブ", "Dracula"], "pages": [], "story": True})]
    result = review(store, project_id, page_id, provider, fetch_rounds=1)
    assert len(provider.calls) == 1 and result.summary.calls == 1
    assert result.proofread.asked and result.proofread.fetched == 0
    assert [i.suggestion for i in result.reviews[contents[0]].issues] == ["Bob's big brother"]


def test_without_fetch_rounds_the_model_cannot_ask(book):
    store, project_id = book
    page_id, _ = page_of(store, project_id, [("兄さん", "Big sister")])
    provider = FakeLlm()
    provider.replies = [reply(need={"terms": ["x"], "pages": [], "story": False})]
    review(store, project_id, page_id, provider, fetch_rounds=0)
    assert len(provider.calls) == 1 and "need" not in provider.calls[0]["schema"]["properties"]


def test_a_reply_that_is_not_json_is_recorded_not_raised(book):
    store, project_id = book
    page_id, contents = page_of(store, project_id, [("兄さん", "Big sister")])
    provider = FakeLlm()
    provider.replies = ["Sure! Here are the issues:"]
    result = review(store, project_id, page_id, provider)
    assert not result.summary.proofread and "not the JSON" in result.summary.reason
    # Not reviewed, so the next pass sends it again.
    assert store.session(project_id).snapshot().component(contents[0], QaReview).fingerprint is None


# -- the meaning check (QaConfig.literal) ------------------------------------------------------------


import importlib  # noqa: E402

_PROOFREAD = importlib.import_module("urakatsu.pipeline.qa.proofread")  # the package's ``proofread`` is the function
DEFAULT_GRADE_BATCH = _PROOFREAD.GRADE_BATCH


@pytest.fixture(autouse=True)
def grade_one_line_per_call(monkeypatch):
    """The meaning check's tests script one grading reply per line, except where they batch."""
    monkeypatch.setattr(_PROOFREAD, "GRADE_BATCH", 1)
    monkeypatch.setattr(_PROOFREAD, "REREAD", False)


def literals(*texts: str) -> str:
    return json.dumps({"lines": [{"n": n, "literal": text} for n, text in enumerate(texts, 1)]})


def test_the_meaning_check_reads_the_sources_alone_then_compares_each_line(book):
    store, project_id = book
    page_id, contents = page_of(
        store,
        project_id,
        [
            ("勝手に入ってくれ", "STAY OUT OF HERE."),  # flipped
            ("くそーあの特集のせいだ…", "DAMMIT..."),  # half the bubble dropped
            ("言い過ぎた", "YOU WENT TOO FAR."),  # graded "different", but the second look clears it
            ("うい", "CLEAR"),  # an interjection: not compared
        ],
    )
    provider = FakeLlm()
    provider.replies = [
        literals("Just come in.", "Damn, it's all that special feature's fault...", "I said way too much.", "Yeah"),
        json.dumps({"meaning": "contradicts"}),
        json.dumps({"meaning": "different"}),
        json.dumps({"wrong": True, "from": "STAY OUT OF HERE", "to": "come on in"}),
        json.dumps({"meaning": "same"}),
        json.dumps({"wrong": True, "from": "DAMMIT...", "to": "DAMMIT... IT'S ALL 'CAUSE OF THAT SPECIAL..."}),
        json.dumps({"meaning": "close"}),
        json.dumps({"wrong": False, "from": "", "to": ""}),
    ]
    result = review(store, project_id, page_id, provider, meaning_check=True)
    read, flipped, graded, sure, vouched, dropped, vouched_too, cleared = provider.calls  # no proofreading call
    # The reading never sees a translation; the grading sees one line, in English only; the second
    # look adds the source; the correction is graded against the reading on its own.
    assert read["user"] == {"lines": [{"n": 1, "source": "勝手に入ってくれ"}, {"n": 2, "source": "くそーあの特集のせいだ…"}, {"n": 3, "source": "言い過ぎた"}, {"n": 4, "source": "うい"}]}
    assert flipped["user"] == {"A": "Just come in.", "B": "STAY OUT OF HERE."}
    assert graded["user"]["A"] == "I said way too much."
    assert sure["user"] == {"source": "勝手に入ってくれ", "reading": "Just come in.", "translation": "STAY OUT OF HERE."}
    assert vouched["user"] == {"A": "Just come in.", "B": "come on in."}  # the smallest change, spliced in
    assert dropped["user"]["translation"] == "DAMMIT..."  # far shorter: an omission without grading
    assert dropped["user"]["found"].startswith("omission:")
    assert vouched_too["user"]["B"] == "DAMMIT... IT'S ALL 'CAUSE OF THAT SPECIAL..."
    assert cleared["user"]["source"] == "言い過ぎた"
    issues = [[(i.kind, i.severity, i.suggestion) for i in result.reviews[c].issues] for c in contents]
    assert issues == [
        [("mistranslation", "error", "COME ON IN.")],  # the page's capitals
        [("omission", "warning", "DAMMIT... IT'S ALL 'CAUSE OF THAT SPECIAL...")],
        [],
        [],
    ]
    assert "Just come in." in result.reviews[contents[0]].issues[0].message
    assert result.summary.calls == 8 and result.summary.prompt_tokens == 800
    assert result.proofread.literals[contents[0]] == "Just come in." and result.proofread.vouched == 2


def test_a_correction_that_does_not_read_as_the_source_is_withheld(book):
    store, project_id = book
    # A glossary term the first line's source names (and its translation keeps): the line has a
    # requirement, so a flag without a correction is checked against it too.
    knowledge.add(store.session(project_id), [EntryInput(source="勝手", translation="Out")], "en-US")
    page_id, contents = page_of(
        store,
        project_id,
        [("勝手に入ってくれ", "STAY OUT OF HERE."), ("いいよどれでも好きなの持ってきな", "JUST DON'T TAKE ANY OF THEM."), ("言い過ぎた", "YOU WENT TOO FAR.")],
    )
    provider = FakeLlm()
    provider.replies = [
        literals("Just come in.", "Fine, bring whichever you like.", "I said too much."),
        json.dumps({"meaning": "different"}),
        json.dumps({"meaning": "contradicts"}),
        json.dumps({"meaning": "contradicts"}),
        # Adds a detail the source lacks: graded "different" against the reading, no better than
        # the line it would replace, so withheld.
        json.dumps({"wrong": True, "from": "STAY OUT OF HERE", "to": "COME IN, THE MISSION STARTS NOW"}),
        json.dumps({"meaning": "different"}),
        # Words the translation does not have: nothing to splice, no call spent checking it.
        json.dumps({"wrong": True, "from": "TAKE NONE", "to": "TAKE ANY ONE"}),
        # Changes nothing.
        json.dumps({"wrong": True, "from": "YOU WENT", "to": "YOU WENT"}),
    ]
    result = review(store, project_id, page_id, provider, meaning_check=True)
    assert len(provider.calls) == 8
    # Still flagged, without a correction.
    assert [[(i.kind, i.suggestion) for i in result.reviews[c].issues] for c in contents] == [[("mistranslation", None)]] * 3
    assert [why for _, why, _ in result.proofread.withheld] == [
        "the correction reads as different against the source, no better than the line (different)",
        "the change does not match the translation's words",
        "the correction changes nothing",
    ]
    assert result.proofread.vouched == 0


def test_the_correction_is_the_translation_with_one_span_replaced():
    from urakatsu.pipeline.qa.proofread import splice

    assert splice("IS THIS OKAY WITH HIM?", "HIM?", "YOU?") == "IS THIS OKAY WITH YOU?"
    assert splice("IS THIS OKAY WITH HIM?", "him", "you") == "IS THIS OKAY WITH you?"
    assert splice("YOU WANNA GIVE UP A KID?", "give  up", "FOSTER") == "YOU WANNA FOSTER A KID?"
    assert splice("YEAH, WHATEVER. JUST DON'T TAKE ANY OF THEM.", "DON'T TAKE ANY OF THEM", "TAKE WHICHEVER ONE YOU WANT") == "YEAH, WHATEVER. JUST TAKE WHICHEVER ONE YOU WANT."
    assert splice("THIMBLE", "HIM", "HER") is None  # whole words only
    # An omission keeps what is there: a replacement that drops those words goes after them.
    assert splice("I DO.", "I DO.", "I HEARD YOU ARRANGE ADOPTIONS HERE.", keep=True) == "I DO. I HEARD YOU ARRANGE ADOPTIONS HERE."
    assert splice("DAMMIT...", "DAMMIT...", "DAMMIT... IT'S THAT SPECIAL'S FAULT...", keep=True) == "DAMMIT... IT'S THAT SPECIAL'S FAULT..."
    assert splice("WAIT!", "...", "x") is None
    assert splice("I MUST BECOME INCONSPICUOUS.", "INCONSPICUOUS", "INCONSPICUOUS. THAT'S THE KEY!") == "I MUST BECOME INCONSPICUOUS. THAT'S THE KEY!"


def test_a_splice_drops_line_break_hyphens_and_words_said_twice():
    from urakatsu.pipeline.qa.proofread import dehyphenate, splice, undouble

    assert splice("I'VE AL-READY TOLD HIM.", "HIM", "HER") == "I'VE ALREADY TOLD HER."
    assert splice("IT'S AL-READY OVER.", "ALREADY OVER", "NOT OVER YET") == "IT'S NOT OVER YET."
    assert dehyphenate("A TEN-PENT COIN, AN X-RAY") == "A TEN-PENT COIN, AN X-RAY"  # compounds stay
    assert dehyphenate("FULL OF SCUM-BAGS", "scumbags") == "FULL OF SCUMBAGS"
    assert splice("I DO.", "DO.", "DO. I DO.") == "I DO."
    assert undouble("COME ON, COME ON! HURRY UP HURRY UP", "COME ON, COME ON! HURRY UP") == "COME ON, COME ON! HURRY UP"
    assert undouble("NO NO NO!", "NO!") == "NO NO NO!"  # a stammer is lettering


def test_a_correction_in_broken_english_is_withheld(book):
    store, project_id = book
    page_id, (line,) = page_of(store, project_id, [("オレは諦めないんで", "I'LL GIVE UP SOON")])
    provider = FakeLlm()
    provider.replies = [
        literals("I'm not going to give up."),
        json.dumps({"meaning": "contradicts"}),
        json.dumps({"wrong": True, "from": "GIVE UP SOON", "to": "NOT GIVING UP"}),
        json.dumps({"meaning": "close", "natural": False}),
    ]
    result = review(store, project_id, page_id, provider, meaning_check=True)
    assert provider.calls[3]["schema"]["required"] == ["meaning", "natural"]
    assert [(i.kind, i.suggestion) for i in result.reviews[line].issues] == [("mistranslation", None)]
    assert [why for _, why, _ in result.proofread.withheld] == ["the correction is not a natural English line"]


def test_a_second_reading_in_context_clears_a_line_the_first_reading_misread(book, monkeypatch):
    monkeypatch.setattr(_PROOFREAD, "REREAD", True)
    store, project_id = book
    page_id, contents = page_of(store, project_id, [("これ食べる？", "WANT THIS?"), ("やる", "YOU CAN HAVE IT."), ("勝手に入ってくれ", "STAY OUT OF HERE.")])
    provider = FakeLlm()
    provider.replies = [
        literals("Will you eat this?", "I'll do it.", "Just come in."),
        json.dumps({"meaning": "close"}),
        json.dumps({"meaning": "contradicts"}),
        json.dumps({"meaning": "contradicts"}),
        # One call reads both suspects again in context; one grades them against it.
        json.dumps({"lines": [{"n": 2, "meaning": "Here, you can have it."}, {"n": 3, "meaning": "Come on in."}]}),
        json.dumps({"grades": [{"n": 2, "meaning": "same"}, {"n": 3, "meaning": "contradicts"}]}),
        json.dumps({"wrong": True, "from": "STAY OUT OF HERE", "to": "COME ON IN"}),
        json.dumps({"meaning": "same"}),
    ]
    result = review(store, project_id, page_id, provider, meaning_check=True)
    reread_call = provider.calls[4]
    assert reread_call["user"]["read"] == [2, 3] and len(reread_call["user"]["lines"]) == 3  # the whole page as context
    assert [[(i.kind, i.suggestion) for i in result.reviews[c].issues] for c in contents] == [[], [], [("mistranslation", "COME ON IN.")]]
    assert len(provider.calls) == 8


def test_a_swap_keeps_the_article_and_no_word_that_needs_the_old_negation(book):
    from urakatsu.pipeline.qa.proofread import fix_articles, polarity_fits, splice

    assert splice("IT'S A PROPHECY.", "PROPHECY", "EXORCISM") == "IT'S AN EXORCISM."
    assert fix_articles("AN UNIFORM AND A HOUR", "") == "A UNIFORM AND AN HOUR"
    assert fix_articles("an apple", "an apple") == "an apple"
    assert not polarity_fits("I HAVEN'T EATEN YET.", "I HAVE EATEN YET.") and polarity_fits("I HAVEN'T EATEN.", "I HAVE EATEN.")
    store, project_id = book
    page_id, (line,) = page_of(store, project_id, [("まだ食べてない", "I HAVEN'T EATEN YET, OKAY?")])
    provider = FakeLlm()
    provider.replies = [
        literals("I've eaten already."),
        json.dumps({"meaning": "contradicts"}),
        json.dumps({"wrong": True, "from": "HAVEN'T", "to": "HAVE"}),  # strands "YET": withheld, no grading call
    ]
    result = review(store, project_id, page_id, provider, meaning_check=True)
    assert len(provider.calls) == 3
    assert [(i.kind, i.suggestion) for i in result.reviews[line].issues] == [("mistranslation", None)]


def test_a_missing_glossary_term_gets_the_smallest_swap(book):
    store, project_id = book
    knowledge.add(store.session(project_id), [EntryInput(source="霊媒師", translation="spirit medium")], "en-US")
    page_id, (line,) = page_of(store, project_id, [("霊媒師なの？", "YOU'RE A PSYCHIC?")])
    provider = FakeLlm()
    provider.replies = [
        literals("Are you a spirit medium?"),
        json.dumps({"wrong": True, "from": "PSYCHIC", "to": "SPIRIT MEDIUM"}),
        json.dumps({"meaning": "same"}),
    ]
    result = review(store, project_id, page_id, provider, meaning_check=True)
    assert len(provider.calls) == 3 and provider.calls[1]["user"]["found"].startswith("glossary:")  # no proofreading call
    assert [(i.kind, i.suggestion) for i in result.reviews[line].issues] == [("glossary", "YOU'RE A SPIRIT MEDIUM?")]


def test_a_name_missing_with_a_shorter_line_counts_as_an_omission(book):
    store, project_id = book
    knowledge.add(store.session(project_id), [EntryInput(source="アーニャ", translation="Anya", kind="character")], "en-US")
    page_id, (line,) = page_of(store, project_id, [("アーニャはピーナッツがすきでもうたべた", "SHE LOVES PEANUTS, YOU KNOW.")])
    provider = FakeLlm()
    provider.replies = [
        literals("Anya likes peanuts and has already eaten them."),
        json.dumps({"wrong": True, "from": "", "to": "AND I ATE 'EM ALREADY."}),
        json.dumps({"meaning": "close"}),
    ]
    result = review(store, project_id, page_id, provider, meaning_check=True)
    assert provider.calls[1]["user"]["found"].startswith("omission:") and "glossary:" in provider.calls[1]["user"]["found"]
    assert ("omission", "SHE LOVES PEANUTS, YOU KNOW. AND I ATE 'EM ALREADY.") in [(i.kind, i.suggestion) for i in result.reviews[line].issues]


def test_a_line_graded_same_or_close_keeps_its_text(book):
    store, project_id = book
    page_id, contents = page_of(store, project_id, [("待ってくれよ", "HEY, WAIT FOR ME!"), ("どうした急に", "どうした急に")])
    provider = FakeLlm()
    provider.replies = [
        literals("Hey, wait for me.", "What happened all of a sudden?"),
        json.dumps({"meaning": "close"}),
        # The proofreading call runs for line 2's open finding and rewrites line 1 too.
        reply(issue(1, "grammar", "major", "HEY, WAIT UP FOR ME!"), issue(2, "untranslated", "critical", "WHAT'S WRONG ALL OF A SUDDEN?")),
    ]
    result = review(store, project_id, page_id, provider, meaning_check=True, mode="fix")
    assert result.reviews[contents[0]].issues == []
    assert (contents[0], "grammar", "the meaning check graded the line close") in result.proofread.dropped
    scene = store.session(project_id).snapshot()
    assert scene.component(contents[0], Translation).text.value == "HEY, WAIT FOR ME!"
    assert scene.component(contents[1], Translation).text.value == "WHAT'S WRONG ALL OF A SUDDEN?"


def test_a_changed_number_is_put_back_by_the_program_or_by_a_vouched_swap(book):
    store, project_id = book
    page_id, contents = page_of(
        store, project_id, [("十八歳です", "I'M 28 YEARS OLD."), ("六人と三人", "FOUR AND TWO OF THEM."), ("六人と三人", "FOUR AND TWO OF US.")]
    )
    provider = FakeLlm()
    provider.replies = [
        literals("I am eighteen years old.", "Six people and three people.", "Six people and three people."),
        # Line 1: one number changed, swapped in place by the program; no model call. Line 2: which
        # is which is the model's, with the finding as a hint; "loose" beats the changed number.
        json.dumps({"wrong": True, "from": "FOUR AND TWO", "to": "SIX AND THREE"}),
        json.dumps({"meaning": "loose"}),
        json.dumps({"wrong": True, "from": "FOUR", "to": "FIVE"}),  # still not the source's number: no call spent
    ]
    result = review(store, project_id, page_id, provider, meaning_check=True)
    assert len(provider.calls) == 4 and provider.calls[1]["user"]["found"] == "number: The source says 3, 6; the translation says 2, 4."
    assert [[(i.kind, i.suggestion) for i in result.reviews[c].issues] for c in contents] == [
        [("number", "I'M 18 YEARS OLD.")],
        [("number", "SIX AND THREE OF THEM.")],
        [("number", None)],
    ]
    assert result.proofread.withheld == [(contents[2], "the correction changes a number of the source", "FIVE AND TWO OF US.")]


def test_the_grading_takes_the_page_in_one_call_by_default(book, monkeypatch):
    store, project_id = book
    page_id, contents = page_of(store, project_id, [("勝手に入ってくれ", "STAY OUT OF HERE."), ("言い過ぎた", "I WENT TOO FAR.")])
    provider = FakeLlm()
    provider.replies = [
        literals("Just come in.", "I said too much."),
        json.dumps({"grades": [{"n": 2, "meaning": "same"}, {"n": 1, "meaning": "contradicts"}]}),
        json.dumps({"wrong": True, "from": "STAY OUT OF HERE", "to": "COME ON IN"}),
        json.dumps({"meaning": "same"}),
    ]
    monkeypatch.setattr(_PROOFREAD, "GRADE_BATCH", DEFAULT_GRADE_BATCH)
    assert DEFAULT_GRADE_BATCH > 2
    result = review(store, project_id, page_id, provider, meaning_check=True)
    assert len(provider.calls) == 4
    assert provider.calls[1]["user"] == {"pairs": [{"n": 1, "A": "Just come in.", "B": "STAY OUT OF HERE."}, {"n": 2, "A": "I said too much.", "B": "I WENT TOO FAR."}]}
    assert provider.calls[1]["schema"]["properties"]["grades"]["minItems"] == 2
    assert [[(i.kind, i.suggestion) for i in result.reviews[c].issues] for c in contents] == [[("mistranslation", "COME ON IN.")], []]


def test_with_the_meaning_check_the_proofreading_call_keeps_to_what_only_it_does(book):
    store, project_id = book
    page_id, contents = page_of(store, project_id, [("どうした急に", "どうした急に"), ("言い過ぎた", "I WENT TOO FAR.")])
    provider = FakeLlm()
    provider.replies = [
        literals("What happened all of a sudden?", "I said too much."),
        json.dumps({"meaning": "close"}),  # line 1 has a program finding: not graded
        reply(issue(1, "untranslated", "critical", "WHAT'S WRONG ALL OF A SUDDEN?"), issue(2, "mistranslation", "critical", "I SAID TOO MUCH.")),
    ]
    result = review(store, project_id, page_id, provider, meaning_check=True)
    assert len(provider.calls) == 3 and "lines" in provider.calls[2]["user"]
    # The finding gets its correction; the call's own meaning report is the meaning check's to make.
    assert [(i.kind, i.suggestion) for i in result.reviews[contents[0]].issues] == [("untranslated", "WHAT'S WRONG ALL OF A SUDDEN?")]
    assert result.reviews[contents[1]].issues == []


def test_a_reading_that_copies_the_source_or_breaks_is_not_compared(book):
    store, project_id = book
    page_id, contents = page_of(store, project_id, [("勝手に入ってくれ", "STAY OUT OF HERE."), ("言い過ぎた", "I WENT TOO FAR.")])
    provider = FakeLlm()
    provider.replies = [literals("勝手に入ってくれ", "I went too far, sorry."), json.dumps({"meaning": "same"})]
    result = review(store, project_id, page_id, provider, meaning_check=True)
    assert len(provider.calls) == 2 and provider.calls[1]["user"]["A"] == "I went too far, sorry."
    assert all(not result.reviews[c].issues for c in contents)
    provider = FakeLlm()
    provider.replies = ["not json"]
    result = review(store, project_id, page_id, provider, meaning_check=True, force=True)
    assert len(provider.calls) == 1 and result.summary.proofread  # the review goes on without the check


def test_the_meaning_check_is_on_by_default_only_for_providers_that_ask_for_it(book):
    from urakatsu.pipeline.qa import meaning_check
    from urakatsu.providers.translation.local import LocalTranslator

    class SmallLlm(FakeLlm):
        meaning_check = True

    assert LocalTranslator.meaning_check and not OpenAiCompatibleTranslator.meaning_check
    assert meaning_check(SmallLlm(), QaConfig()) and not meaning_check(FakeLlm(), QaConfig())
    assert not meaning_check(SmallLlm(), QaConfig(meaning_check=False)) and meaning_check(FakeLlm(), QaConfig(meaning_check=True))
    store, project_id = book
    page_id, _ = page_of(store, project_id, [("言い過ぎた", "I WENT TOO FAR.")])
    provider = SmallLlm()
    provider.replies = [literals("I said way too much."), json.dumps({"meaning": "same"})]
    review(store, project_id, page_id, provider)
    assert [call["schema"]["required"] for call in provider.calls] == [["lines"], ["meaning"]]


def test_only_a_far_shorter_translation_counts_as_an_omission():
    from urakatsu.pipeline.qa.proofread import shorter

    assert shorter("Damn, it's all that special feature's fault...", "DAMMIT...")
    assert not shorter("Damn, it's all that special feature's fault...", "DAMMIT... IT'S THAT SPECIAL'S FAULT")
    assert not shorter("Come in.", "OK")  # too short a reading to tell


# -- the program's own checks and the guards on corrections ----------------------------------------


GLOSSARY = [
    KnowledgeEntry(id="1", kind="character", source="アーニャ", translation="Anya"),
    KnowledgeEntry(id="2", kind="term", source="フォージャー", translation="Forger"),
    KnowledgeEntry(id="3", kind="character", source="市川レノ", aliases=["市川"], translation="Leno Ichikawa"),
    KnowledgeEntry(id="4", kind="term", source="オカルト", translation="the paranormal"),
]


def found(source: str, translation: str) -> list[tuple[str, str, Optional[str]]]:
    return [(i.kind, i.severity, i.suggestion) for i in checks.check_line(source, translation, GLOSSARY, "en-US")]


def test_source_text_left_in_is_found_without_a_model():
    assert found("どうした急に", "どうした急に") == [("untranslated", "error", None)]
    assert found("どうした！？", "どうした！？") == [("untranslated", "error", None)]  # its full-width marks go with it
    assert found("クズしかいねえのか", "IS THE WORLD JUST FULL OF クズ?")[0][:2] == ("untranslated", "error")
    assert found("どうした", "WHAT'S WRONG?") == []
    assert checks.check_line("どうした", "どうした", [], "ja-JP") == []  # a Japanese target may write it


def test_glossary_renderings_are_checked_and_near_spellings_respelt():
    assert found("おいアーニヤ！", "ANIA, GET OVER HERE!") == [("glossary", "warning", "ANYA, GET OVER HERE!")]  # OCR's ヤ for ャ
    assert found("フォージャーです", "WE'RE THE FORGES.") == [("glossary", "warning", "WE'RE THE FORGERS.")]
    assert found("市川レノ", "RENO ICHIKAWA") == [("glossary", "warning", "LENO ICHIKAWA")]  # the full name: every word
    assert found("市川おつかれ", "GOOD JOB, ICHIKAWA.") == []  # an alias: any word of it
    assert found("オカルト好き", "YOU LIKE THE OCCULT?") == [("glossary", "warning", None)]  # nothing near to respell
    assert found("オカルト好き", "YOU LIKE THE PARA-NORMAL?") == []  # the letterer's hyphen
    # A speaker naming themself is "I" in English; a misspelt name next to it is still a miss.
    assert found("アーニャピーナッツすき", "I LOVE PEANUTS!") == []
    assert found("アーニャです", "I'M ANIA.") == [("glossary", "warning", "I'M ANYA.")]


def test_a_near_miss_of_a_name_is_found_even_where_the_source_does_not_write_it():
    glossary = GLOSSARY + [
        KnowledgeEntry(id="5", kind="character", source="高倉健", aliases=["健さん"], translation="Ken Takakura"),
        KnowledgeEntry(id="6", kind="character", source="カフカ", translation="Kafka"),
        KnowledgeEntry(id="7", kind="term", source="霊媒師", translation="spirit medium"),
    ]

    def slips(source, translation):
        return [(i.kind, i.suggestion) for i in checks.check_line(source, translation, glossary, "en-US")]

    assert slips("ほら挨拶せんか", "GREET THE MAN, ANIYA!") == [("glossary", "GREET THE MAN, ANYA!")]
    assert slips("ほら挨拶せんか", "GREET THE MAN, ANNIE!") == [("glossary", "GREET THE MAN, ANYA!")]
    assert slips("ほら", "Then Annie said so.") == [("glossary", "Then Anya said so.")]
    assert slips("ほら", "annie said so.") == []  # mixed case: a name is capitalised
    assert slips("健さん！", "HEY, KENJI!") == [("glossary", "HEY, KEN!")]  # a whole word: KENJI is not KEN
    assert slips("おいカフ力こっち来い", "HEY KAFUKA, C'MERE!") == [("glossary", "HEY KAFKA, C'MERE!")]  # OCR's 力 for カ
    assert slips("霊媒帥なの", "YOU'RE A PSYCHIC?") == [("glossary", None)]  # OCR's 帥 for 師
    assert slips("市川おつかれ", "GOOD JOB, ICHIKAWA-SAN.") == []
    assert slips("なに", "ANYWAY, EVERY MOMMY IS KEEN ON MANY THINGS.") == []  # English words near a name


def test_a_changed_number_is_found_without_a_model():
    # One number changed: put back in place, as the translation wrote it.
    assert found("十八歳です", "I'M 28.") == [("number", "warning", "I'M 18.")]
    assert found("六人いた", "THERE WERE FOUR OF THEM.") == [("number", "warning", "THERE WERE SIX OF THEM.")]
    assert found("むっつあった", "THERE WERE FOUR.") == [("number", "warning", "THERE WERE SIX.")]  # counted in kana
    assert found("二十歳", "I'M TWENTY-EIGHT") == [("number", "warning", "I'M TWENTY")]
    assert found("六人と三人", "FOUR AND TWO OF THEM.") == [("number", "warning", None)]  # which is which: the model's
    assert found("33歳未満に", "UNDER 33 YEARS OLD") == []
    assert found("１Ｄの物は10ｐ硬貨じゃ", "THAT COSTS ONE DALC. NOT WITH A TEN-PENT COIN.") == []
    assert found("3時半", "AT 3:30") == []  # the source's number is there
    assert found("一人で十分", "ONE IS ENOUGH") == []  # 1 is not counted; a lone kanji needs a counter
    assert found("三池班は", "MIIKE'S TEAM, ALL FOUR OF THEM") == []  # 三池 is a name: the source writes no number
    assert checks.numbers_agree("十八歳です", "I'M 18.", "I'M 28.") and not checks.numbers_agree("十八歳です", "I'M 28 OR 18.", "I'M 28.")


def test_full_width_forms_are_folded():
    assert found("なに", "ＷＨＡＴ？") == [("full_width", "info", "WHAT?")]


def test_a_correction_must_keep_the_glossary_and_the_pages_lettering():
    style = checks.PageStyle(language="en-US", capitals=True)
    required = checks.requirements("アーニャ", GLOSSARY, "en-US")
    assert checks.guard_fix("Anya wants it", "ANYA WANT IT", required, style) == ("ANYA WANTS IT", "")
    assert checks.guard_fix("Ania wants it", "ANYA WANT IT", required, style) == ("ANYA WANTS IT", "")  # respelt
    assert checks.guard_fix("She wants it", "ANYA WANT IT", required, style)[0] is None  # drops the name
    assert checks.guard_fix("anya want it!", "ANYA WANT IT", required, style)[0] is None  # case and punctuation only
    assert checks.guard_fix("アーニャ WANTS IT", "ANYA WANT IT", required, style)[0] is None
    assert checks.guard_fix("ＷＨＡＴ？", "WHO?", [], style) == ("WHAT?", "")


def test_the_model_is_kept_to_problems_it_can_back(book):
    store, project_id = book
    page_id, contents = page_of(
        store,
        project_id,
        [("ゴメン", "SORRY."), ("言い過ぎた", "I WENT TOO FAR."), ("はい", "LOOK"), ("待って", "WAIT!"), ("どうした", "どうした")],
    )
    provider = FakeLlm()
    provider.replies = [
        reply(
            issue(1, "wording", "minor", "I APOLOGIZE."),  # the model says the meaning is right
            issue(2, "too_long", "major", "TOO FAR."),  # no length to keep to
            issue(2, "mistranslation", "major", "I SAID TOO MUCH.", reason="The translation is too literal."),  # a preference
            issue(3, "mistranslation", "major", "Yes"),  # lower case on a page lettered in capitals
            issue(4, "mistranslation", "major", "I WENT TOO FAR."),  # another bubble's line
            issue(5, "untranslated", "critical", "what's wrong?"),  # completes the program's finding
            issue(1, "grammar", "minor", "Sorry!"),  # punctuation only
        )
    ]
    result = review(store, project_id, page_id, provider)
    by_line = [[(i.kind, i.suggestion) for i in result.reviews[content].issues] for content in contents]
    assert by_line == [[], [], [("mistranslation", "YES")], [("mistranslation", None)], [("untranslated", "WHAT'S WRONG?")]]
    assert {why for _, _, why in result.proofread.dropped} >= {
        "the model judged the meaning right",
        "too_long without a length to keep to",
        "the reason is a wording preference",
    }


def test_a_names_correction_must_keep_the_rest_of_the_line(book):
    store, project_id = book
    knowledge.add(store.session(project_id), [EntryInput(source="ちち", translation="Papa")], "en-US")
    page_id, (line,) = page_of(store, project_id, [("ちちーっ！！", "FATHER! SAVE MEEEE!")])
    provider = FakeLlm()
    provider.replies = [reply(issue(1, "glossary", "major", "PAPA!!"))]
    result = review(store, project_id, page_id, provider)
    assert [(i.kind, i.suggestion) for i in result.reviews[line].issues] == [("glossary", None)]


def test_fix_mode_writes_the_models_correction_of_a_program_finding(book):
    store, project_id = book
    page_id, (left,) = page_of(store, project_id, [("どうした", "どうした")])
    provider = FakeLlm()
    provider.replies = [reply(issue(1, "untranslated", "critical", "What's wrong?"))]
    review(store, project_id, page_id, provider, mode="fix")
    scene = store.session(project_id).snapshot()
    assert scene.component(left, Translation).text.value == "What's wrong?"
    issues = scene.component(left, QaReview).issues
    assert [(i.kind, i.applied) for i in issues] == [("untranslated", True)]
    # The next pass finds nothing left and keeps no stale finding.
    review(store, project_id, page_id, provider, mode="fix")
    assert [(i.kind, i.applied) for i in store.session(project_id).snapshot().component(left, QaReview).issues] == [("untranslated", True)]


def test_an_overlong_text_shrunk_by_auto_fit_is_found(tmp_path, renderer):
    padded = "I'M SORRY. I'M REALLY, TRULY SORRY ABOUT ALL OF THAT, AND I HOPE YOU CAN FORGIVE ME FOR IT."
    scene, page, (long, short) = lettered(
        tmp_path,
        [{"balloon": (20, 20, 220, 160), "text": padded, "source": "ゴメン"}, {"balloon": (300, 40, 220, 140), "text": "SORRY.", "source": "ゴメン"}],
    )
    blocks = check_layout(scene, page, renderer)
    assert kinds(blocks, long) == {"overlong"} and kinds(blocks, short) == set()
    assert 0 < blocks[long].max_chars < len(padded) and blocks[long].wording_hint().startswith("far longer")


# -- fix mode --------------------------------------------------------------------------------------


def test_fix_mode_rewrites_generated_translations_and_only_flags_the_users(book):
    store, project_id = book
    page_id, (generated, users, minor) = page_of(store, project_id, [("兄さん", "Big sister"), ("姉さん", "Big brother", "user"), ("はい", "Yes")])
    provider = FakeLlm()
    provider.replies = [reply(issue(1, fix="Big brother"), issue(2, fix="Big sister"), issue(3, "tone", "minor", "Yeah"))]
    result = review(store, project_id, page_id, provider, mode="fix")
    scene = store.session(project_id).snapshot()
    fixed = scene.component(generated, Translation)
    assert fixed.text.value == "Big brother"
    # The fix stays the translation stage's, so re-translating can replace it.
    assert fixed.text.origin.generation.producer == TRANSLATION_PRODUCER
    assert scene.component(users, Translation).text.value == "Big brother" and c.is_user(scene.component(users, Translation).text.origin)
    assert scene.component(users, Typography).size == 20 and c.is_user(scene.component(users, Typography).origin)
    assert scene.component(minor, Translation).text.value == "Yes"  # a minor issue is only flagged
    assert scene.component(generated, QaReview).original == "Big sister"
    assert [i.applied for i in scene.component(generated, QaReview).issues] == [True]
    assert [i.applied for i in scene.component(users, QaReview).issues] == [False]
    assert result.summary.fixed == 1
    # The fixed block's fingerprint is of the new text: the next pass does not send it again.
    provider.calls.clear()
    review(store, project_id, page_id, provider, mode="fix")
    assert provider.calls == []
    # Re-translating still replaces the fixed translation.
    edit = store.session(project_id).snapshot().edit_as(GENERATED)
    edit.set(generated, Translation(text=Authored.generated("Brother", GENERATED), language="en-US"))
    store.session(project_id).commit(edit.finish())


def test_fix_mode_writes_what_an_earlier_pass_only_flagged_without_asking_again(book):
    store, project_id = book
    page_id, (generated, users) = page_of(store, project_id, [("兄さん", "Big sister"), ("姉さん", "Big brother", "user")])
    provider = FakeLlm()
    provider.replies = [reply(issue(1, fix="Big brother"), issue(2, fix="Big sister"))]
    review(store, project_id, page_id, provider, mode="flag")
    provider.calls.clear()
    result = review(store, project_id, page_id, provider, mode="fix")
    scene = store.session(project_id).snapshot()
    assert provider.calls == [] and result.summary.fixed == 1
    assert scene.component(generated, Translation).text.value == "Big brother" and scene.component(generated, QaReview).original == "Big sister"
    assert scene.component(users, Translation).text.value == "Big brother"  # the user's line is still only flagged
    # A third pass has nothing left to write.
    again = review(store, project_id, page_id, provider, mode="fix")
    assert provider.calls == [] and again.summary.fixed == 0


def test_a_fix_is_remembered_only_while_it_is_the_blocks_translation(book):
    store, project_id = book
    page_id, (fixed, cleared) = page_of(store, project_id, [("兄さん", "Big sister"), ("姉さん", "Big brother")])
    provider = FakeLlm()
    provider.replies = [reply(issue(1, fix="Big brother"), issue(2, fix="Big sister"))]
    review(store, project_id, page_id, provider, mode="fix")
    session = store.session(project_id)
    before = session.snapshot().component(fixed, QaReview)
    assert (before.original, before.written) == ("Big sister", "Big brother")
    # An unchanged pass keeps what the fix replaced.
    review(store, project_id, page_id, provider, mode="fix")
    assert session.snapshot().component(fixed, QaReview).original == "Big sister"

    # Re-translated: the text QA replaced belongs to a line that is gone. Cleared: nothing is left
    # to review, so the block's findings and corrections go with its text.
    edit = session.snapshot().edit_as(GENERATED)
    edit.set(fixed, Translation(text=Authored.generated("Elder brother", GENERATED), language="en-US"))
    session.commit(edit.finish())
    edit = session.snapshot().edit()
    edit.set(cleared, Translation(text=Authored.user(""), language="en-US"))
    session.commit(edit.finish())
    result = review(store, project_id, page_id, provider, mode="fix")
    scene = session.snapshot()
    again = scene.component(fixed, QaReview)
    assert (again.original, again.written, again.issues) == (None, None, [])
    assert scene.component(cleared, QaReview) is None and result.summary.issues == 0


def test_a_run_on_some_blocks_leaves_and_counts_the_others(book):
    store, project_id = book
    page_id, (first, second) = page_of(store, project_id, [("兄さん", "Big sister"), ("姉さん", "Big brother")])
    provider = FakeLlm()
    provider.replies = [reply(issue(1, fix="Big brother"), issue(2, fix="Big sister"))]
    review(store, project_id, page_id, provider)
    session = store.session(project_id)
    scene = session.snapshot()
    provider.replies = [reply()]
    result = asyncio.run(review_page(scene, page_id, provider, QaConfig(mode="flag"), entities={scene.text_layers(page_id)[0]}, force=True))
    session.commit(result.patch)
    scene = session.snapshot()
    assert scene.component(first, QaReview).issues == [] and len(scene.component(second, QaReview).issues) == 1
    assert scene.component(page_id, QaPage).issues == 1  # the page's count, not the run's


def test_a_target_written_without_spaces_gets_the_proofreading_call(book):
    # The meaning check counts words; Japanese has none to count, so it would pass every line unread.
    store, project_id = book
    page_id, _ = page_of(store, project_id, [("I will never go to school again, sorry", "学校に行く"), ("Yes", "はい")])
    provider = FakeLlm({"translation": {"model": {"provider": "fake-llm", "model": "m", "vision": False}, "target_language": "ja-JP"}})
    review(store, project_id, page_id, provider, meaning_check=True)
    assert [sorted(call["user"])[0] for call in provider.calls] == ["lines"] and "translation" in provider.calls[0]["user"]["lines"][0]


def test_an_article_is_repaired_only_in_english():
    from urakatsu.pipeline.qa.proofread import splice

    assert splice("IT'S A SPIRIT", "SPIRIT", "EXORCISM") == "IT'S AN EXORCISM"
    assert splice("Voy al parque", "al parque", "a escuela", english=False) == "Voy a escuela"


@pytest.mark.parametrize(
    "source, translation",
    [
        ("二週間待って", "WAIT 14 DAYS"),  # a unit converted
        ("2時間", "120 MINUTES"),
        ("15時に来い", "COME AT 3 PM"),
        ("1億円だ", "IT'S 100 MILLION YEN"),  # digits with a scale are one number
        ("3万円", "30K YEN"),
        ("3万円", "30 THOUSAND YEN"),
        ("500万", "1.5 MILLION? NO, 5 MILLION"),
        ("2,3日", "2 OR 3 DAYS"),  # a comma only groups thousands
    ],
)
def test_a_converted_unit_or_a_scaled_number_is_not_a_changed_number(source, translation):
    assert checks.number_change(source, translation) is None


def test_the_program_puts_back_only_a_slip():
    assert checks.number_fix("十八歳", "I'M 28") == "I'M 18"
    assert checks.number_fix("3万円", "3,000 YEN") == "30000 YEN"
    # Flagged, but what the right number is in the translation's unit is not the program's to say.
    assert checks.number_change("170センチ", "67 INCHES TALL") and checks.number_fix("170センチ", "67 INCHES TALL") is None


# -- fingerprints ----------------------------------------------------------------------------------


def test_an_unchanged_page_is_not_sent_again_and_its_model_is_not_loaded(book):
    store, project_id = book
    page_id, (first, second) = page_of(store, project_id, [("兄さん", "Big sister"), ("はい", "Yes")])
    provider = FakeLlm()
    provider.replies = [reply(issue(1, fix="Big brother"))]
    review(store, project_id, page_id, provider)
    assert len(provider.calls) == 1 and provider.loaded
    provider.unload()

    again = review(store, project_id, page_id, provider)
    assert len(provider.calls) == 1 and not provider.loaded
    assert again.summary.unchanged == 2 and again.summary.calls == 0
    # The earlier finding is kept for the unchanged block.
    assert [i.suggestion for i in again.reviews[first].issues] == ["Big brother"]

    session = store.session(project_id)
    edit = session.snapshot().edit()
    edit.set(second, Translation(text=Authored.user("Yeah"), language="en-US"))
    session.commit(edit.finish())
    review(store, project_id, page_id, provider)
    assert [line["translation"] for line in provider.calls[-1]["user"]["lines"]] == ["Yeah"]
    # The unchanged line goes as context, not to be checked.
    assert provider.calls[-1]["user"]["context"] == [["兄さん", "Big sister"]]

    review(store, project_id, page_id, provider, force=True)
    assert len(provider.calls[-1]["user"]["lines"]) == 2


# -- the provider QA shares with translation -------------------------------------------------------


def llm_config() -> PipelineConfig:
    config = fake_config("fake-llm")
    config.translation.model = ModelSelection(provider="fake-llm", model="m", vision=False)
    config.qa.mode = "flag"
    return config


def test_qa_with_the_translation_model_is_the_translation_provider():
    config = llm_config()
    pipeline = make_pipeline(config)
    shared = pipeline.provider(Stage.TRANSLATION)
    assert pipeline.provider(Stage.QA) is shared

    other = config.model_copy(deep=True)
    other.qa.model = ModelSelection(provider="fake-llm", model="other", vision=False)
    pipeline.configure(other)
    separate = pipeline.provider(Stage.QA)
    assert separate is not pipeline.provider(Stage.TRANSLATION)
    assert separate.settings.translation.model.model == "other"

    back = other.model_copy(deep=True)
    back.qa.model = None
    pipeline.configure(back)
    assert pipeline.provider(Stage.QA) is pipeline.provider(Stage.TRANSLATION)
    assert Stage.QA not in pipeline._providers


async def test_a_run_loads_the_shared_model_once(project):
    store, project_id, pages = project
    FakeLlm.loads = 0
    pipeline = make_pipeline(llm_config(), TranslationStage())
    report = await pipeline.execute(store.session(project_id), Request(operation=Operation.full()))
    assert report.status is RunStatus.COMPLETED
    assert FakeLlm.loads == 1
    loaded = pipeline.loaded()
    assert loaded[Stage.TRANSLATION] == "fake-llm" and Stage.QA not in loaded
    scene = store.session(project_id).snapshot()
    for page in pages:
        summary = scene.component(page, QaPage)
        assert summary.proofread and summary.calls == 1 and summary.sent == 1


# -- providers -------------------------------------------------------------------------------------


async def test_an_openai_compatible_server_answers_with_its_usage(monkeypatch):
    seen: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": CLEAN}}], "usage": {"prompt_tokens": 321, "completion_tokens": 12}})

    monkeypatch.setattr(OpenAiCompatibleTranslator, "transport", httpx.MockTransport(handle))
    provider = OpenAiCompatibleTranslator({"translation": {"model": {"provider": "openai-compatible", "model": "qwen3-4b"}}})
    schema = reply_schema([Line(1, "a", "s", "t")], can_ask=True)
    completion = await provider.complete("system", "user", schema, name="manga_proofreading", max_tokens=300)
    assert (completion.text, completion.prompt_tokens, completion.completion_tokens) == (CLEAN, 321, 12)
    body = seen[0]
    assert body["response_format"]["json_schema"] == {"name": "manga_proofreading", "strict": True, "schema": schema}
    assert body["max_tokens"] == 300 and body["messages"][0] == {"role": "system", "content": "system"}


async def test_the_local_model_completes_on_the_server_it_has_loaded():
    from urakatsu.providers.translation.local import LocalTranslator

    class Server:
        def __init__(self):
            self.prompts: list[tuple[str, dict]] = []

        def strip_bos(self, prompt: str) -> str:
            return prompt

        def complete_with_usage(self, prompt, images=(), **params):
            self.prompts.append((prompt, params))
            return CLEAN, 77, 9

    provider = LocalTranslator({})
    provider._server = Server()
    provider._template = "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}{% if enable_thinking %}<think>{% endif %}"
    completion = await provider.complete("system", "user", {"type": "object"}, max_tokens=256)
    assert (completion.text, completion.prompt_tokens, completion.completion_tokens) == (CLEAN, 77, 9)
    prompt, params = provider._server.prompts[0]
    assert prompt == "<system>system<user>user" and params["json_schema"] == {"type": "object"} and params["n_predict"] == 256


def test_translation_services_cannot_proofread():
    from urakatsu.providers.translation.remote.deepl import DeepLTranslator

    assert not DeepLTranslator({}).can_complete()
    assert FakeLlm().can_complete()
