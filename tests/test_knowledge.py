"""Project knowledge: storage, retrieval, and what reaches the translator."""

import asyncio
import json
from types import SimpleNamespace
from typing import Optional

import pytest

from urakatsu.core.config import TranslationConfig
from urakatsu.pipeline import knowledge
from urakatsu.pipeline.images import ImageCache
from urakatsu.pipeline.knowledge import EntryInput
from urakatsu.pipeline.projects import ProjectStore
from urakatsu.pipeline.stages.base import StageInput
from urakatsu.pipeline.stages.sound_effects import ROLE
from urakatsu.pipeline.stages.translation import TranslationStage, retrieve
from urakatsu.providers.translation.backend import TranslationRequest
from urakatsu.providers.translation.prompt import prompts
from urakatsu.scene import At, Authored, Generation, Knowledge, SourceText, TextLayout, TextLayoutKind, TextRole, Translation
from urakatsu.sources.base import Series

from fakes import page_png

GENERATED = Generation(producer="test")


class Capture:
    id = "capture"
    context_budget = 6000

    def __init__(self):
        self.settings = SimpleNamespace(translation=TranslationConfig(target_language="en-US"))
        self.request: Optional[TranslationRequest] = None

    def vision_enabled(self) -> bool:
        return False

    async def translate(self, request: TranslationRequest) -> list[str]:
        self.request = request
        return [f"EN:{segment}" for segment in request.segments]


@pytest.fixture
def project(tmp_path):
    store = ProjectStore(tmp_path)
    return store, store.create("p")


def page(store, project_id, lines):
    """A page of ``(source, translation)`` lines; a third item is ``"user"`` or a language tag."""
    page_id = store.import_page(project_id, "page.png", page_png())
    session = store.session(project_id)
    edit = session.snapshot().edit_as(GENERATED)
    contents = []
    edited = []
    for line in lines:
        content = edit.add_text_content(page_id)
        edit.add_text_layer(page_id, At.END, content, TextLayout(kind=TextLayoutKind.PARAGRAPH))
        edit.set(content, SourceText(text=Authored.generated(line[0], GENERATED)))
        if line[1] is not None:
            extra = line[2] if len(line) > 2 else ""
            edit.set(content, Translation(text=Authored.generated(line[1], GENERATED), language=extra if "-" in extra else "en-US"))
            if extra == "user":
                edited.append((content, line[1]))
        contents.append(content)
    session.commit(edit.finish())
    if edited:
        edit = session.snapshot().edit()
        for content, text in edited:
            edit.set(content, Translation(text=Authored.user(text), language="en-US"))
        session.commit(edit.finish())
    return page_id, contents


def retrieved(store, project_id, page_id, budget=6000):
    scene = store.session(project_id).snapshot()
    targets = []
    for layer in scene.text_layers(page_id):
        content = scene.layer_content(layer)
        source = scene.component(content, SourceText)
        translation = scene.component(content, Translation)
        if translation is None:
            targets.append((content, source.text.value))
    return retrieve(scene, page_id, targets, "en-US", budget)


# -- reading text ----------------------------------------------------------------------------------


def test_cjk_is_read_as_bigrams_and_latin_as_words():
    assert knowledge.tokenize("アリス様 Hello") == ("アリ", "リス", "ス様", "hello")


def test_a_latin_name_must_stand_alone_and_ocr_spaces_do_not_hide_a_cjk_name():
    def named(name, text):
        normalized = knowledge.normalize(text)
        return knowledge.mentioned(name, normalized, "".join(normalized.split()))

    assert named("Al", "Al said so")
    assert not named("Al", "also")
    assert named("アリス", "ア リ ス 様")
    assert named("ＡＬＩＣＥ", "alice!")


def test_long_notes_become_bounded_chunks():
    chunks = knowledge._chunks("first\n\n" + "x" * 1200 + "\nlast")
    assert all(len(chunk) <= knowledge.CHUNK_CHARS for chunk in chunks)
    assert chunks[0] == "first" and chunks[-1].endswith("last")


# -- the glossary and notes ------------------------------------------------------------------------


def test_only_terms_the_page_names_are_sent(project):
    store, project_id = project
    session = store.session(project_id)
    knowledge.add(
        session,
        [
            EntryInput(kind="character", source="アリス", aliases=["アリスちゃん"], translation="Alice", text="A knight. Speaks formally."),
            EntryInput(source="魔王", translation="Demon King"),
            EntryInput(source="勇者", aliases=["ゆうしゃ"], translation="Hero"),
        ],
        "en-US",
    )
    page_id, _ = page(store, project_id, [("アリスちゃん、ゆうしゃが来た", None)])
    found = retrieved(store, project_id, page_id)
    assert [(t.source, t.translation) for t in found.glossary] == [("アリス", "Alice"), ("勇者", "Hero")]
    assert found.glossary[0].note == "A knight. Speaks formally."


def test_a_translation_for_another_language_is_left_out_but_its_note_stays(project):
    store, project_id = project
    knowledge.add(store.session(project_id), [EntryInput(source="アリス", translation="Alicia", text="A knight.")], "es-ES")
    page_id, _ = page(store, project_id, [("アリスが来た", None)])
    term = retrieved(store, project_id, page_id).glossary[0]
    assert term.translation == "" and term.note == "A knight."


def test_pinned_notes_always_go_and_other_notes_are_ranked(project):
    store, project_id = project
    knowledge.add(
        store.session(project_id),
        [
            EntryInput(kind="note", text="A fantasy comedy.", pinned=True),
            EntryInput(kind="note", source="王都", text="王都の騎士団は三つある。"),
            EntryInput(kind="note", text="海辺の村の話。"),
        ],
        "en-US",
    )
    page_id, _ = page(store, project_id, [("騎士団が来たぞ", None)])
    notes = retrieved(store, project_id, page_id).notes
    assert notes == ["A fantasy comedy.", "王都: 王都の騎士団は三つある。"]


def test_a_note_in_the_target_language_is_found_through_the_glossary(project):
    store, project_id = project
    knowledge.add(
        store.session(project_id),
        [EntryInput(source="アリス", translation="Alice"), EntryInput(kind="note", text="Alice lost her sister in the war."), EntryInput(kind="note", text="The harvest festival is in autumn.")],
        "en-US",
    )
    page_id, _ = page(store, project_id, [("アリスが来た", None)])
    assert retrieved(store, project_id, page_id).notes == ["Alice lost her sister in the war."]


def test_a_named_entry_sends_its_note_once_and_the_rest_of_a_long_one_can_still_be_found(project):
    store, project_id = project
    first, second = "a" * (knowledge.CHUNK_CHARS - 10), "騎士団の団長を務めた過去がある。"
    knowledge.add(
        store.session(project_id),
        [EntryInput(source="東京", text="Where it opens.", pinned=True), EntryInput(source="アリス", text=f"{first}\n{second}")],
        "en-US",
    )
    page_id, _ = page(store, project_id, [("東京のアリスは騎士団の団長だ", None)])
    found = retrieved(store, project_id, page_id)
    assert {t.source: t.note for t in found.glossary} == {"東京": "Where it opens.", "アリス": first}
    assert found.notes == [f"アリス: {second}"]


# -- the project's own lines -----------------------------------------------------------------------


def test_recent_pages_come_in_reading_order_without_other_languages_or_sound_effects(project):
    store, project_id = project
    page(store, project_id, [("一", "One"), ("二", "Dos", "es-ES")])
    _, contents = page(store, project_id, [("三", "Three"), ("ドン", "BAM")])
    session = store.session(project_id)
    edit = session.snapshot().edit()
    edit.set(contents[1], TextRole(role=ROLE))
    session.commit(edit.finish())
    page_id, _ = page(store, project_id, [("四", None)])
    assert [(c.source, c.translation) for c in retrieved(store, project_id, page_id).context] == [("一", "One"), ("三", "Three")]


def test_an_earlier_line_like_the_page_is_found_far_back_and_the_users_wins(project):
    store, project_id = project
    page(store, project_id, [("お前はもう死んでいる", "You are already dead."), ("今日はいい天気だ", "Nice weather today.")])
    page(store, project_id, [("お前はもう死んでいる!", "You're already dead!", "user")])
    for number in range(knowledge.RECENT_PAGES):
        page(store, project_id, [(f"ページ{number}", f"Page {number}")])
    page_id, _ = page(store, project_id, [("お前はもう死んでいる", None)])
    context = retrieved(store, project_id, page_id).context
    translations = [c.translation for c in context]
    assert translations[:2] == ["You're already dead!", "You are already dead."]
    assert "Nice weather today." not in translations


def test_budget_limits_what_is_sent_and_glossary_comes_first(project):
    store, project_id = project
    knowledge.add(store.session(project_id), [EntryInput(source=f"用語{i}", translation=f"Term {i}", text="長い説明" * 20) for i in range(40)], "en-US")
    page(store, project_id, [("前の頁の台詞" * 10, "A line from the page before. " * 10) for _ in range(30)])
    page_id, _ = page(store, project_id, [(" ".join(f"用語{i}" for i in range(40)), None)])
    found = retrieved(store, project_id, page_id, budget=2000)
    spent = sum(knowledge.estimate_tokens(f"{t.source}{t.translation}{t.note}") for t in found.glossary)
    spent += sum(knowledge.estimate_tokens(note) for note in found.notes)
    spent += sum(knowledge.estimate_tokens(c.source + c.translation) for c in found.context)
    assert found.glossary and found.context
    assert spent <= 2000


def test_turning_knowledge_off_sends_nothing(project):
    store, project_id = project
    session = store.session(project_id)
    knowledge.add(session, [EntryInput(source="アリス", translation="Alice")], "en-US")
    page(store, project_id, [("一", "One")])
    page_id, _ = page(store, project_id, [("アリス", None)])
    knowledge.set_enabled(session, False)
    found = retrieved(store, project_id, page_id)
    assert not found.glossary and not found.notes and not found.context


# -- the stage and the prompt ----------------------------------------------------------------------


def run_stage(store, project_id, page_id, entities=None):
    provider = Capture()
    scene = store.session(project_id).snapshot()
    patch = asyncio.run(TranslationStage().process(StageInput(scene, page_id, entities, None, ImageCache()), provider))
    return provider, patch


def test_the_stage_gives_the_translator_what_it_retrieved(project):
    store, project_id = project
    knowledge.add(store.session(project_id), [EntryInput(source="アリス", translation="Alice"), EntryInput(kind="note", text="Set in 1920.", pinned=True)], "en-US")
    page(store, project_id, [("こんにちは", "Hello")])
    page_id, _ = page(store, project_id, [("アリスだ", None)])
    provider, _ = run_stage(store, project_id, page_id)
    payload = json.loads(prompts(provider.request)[1])
    assert payload["glossary"] == [{"source": "アリス", "translation": "Alice"}]
    assert payload["notes"] == ["Set in 1920."]
    assert payload["context"] == [{"source": "こんにちは", "translation": "Hello"}]
    system = prompts(provider.request)[0]
    assert "Glossary requirements" in system and "Background requirements" in system


def test_retranslating_one_block_reads_the_rest_of_its_page(project):
    store, project_id = project
    page_id, contents = page(store, project_id, [("誰だ", "Who's there?"), ("俺だ", None)])
    scene = store.session(project_id).snapshot()
    layer = next(each for each in scene.text_layers(page_id) if scene.layer_content(each) == contents[1])
    provider, _ = run_stage(store, project_id, page_id, {layer})
    assert provider.request.segments == ["俺だ"]
    assert [(c.source, c.translation) for c in provider.request.context] == [("誰だ", "Who's there?")]


def test_a_stale_neighbour_does_not_reject_the_page(project):
    store, project_id = project
    _, contents = page(store, project_id, [("一", "One")])
    page_id, _ = page(store, project_id, [("二", None)])
    _, patch = run_stage(store, project_id, page_id)
    session = store.session(project_id)
    edit = session.snapshot().edit()
    edit.set(contents[0], Translation(text=Authored.user("Uno"), language="en-US"))
    session.commit(edit.finish())
    session.commit(patch)  # the rebase rule only looks at what the stage observed: its own page


def test_a_project_without_knowledge_or_other_pages_keeps_the_plain_payload(project):
    store, project_id = project
    page_id, _ = page(store, project_id, [("一", None)])
    provider, _ = run_stage(store, project_id, page_id)
    assert set(json.loads(prompts(provider.request)[1])) == {"source_language", "target_language", "context", "segments"}


# -- storage ---------------------------------------------------------------------------------------


def test_adding_a_known_name_updates_that_entry(project):
    store, project_id = project
    session = store.session(project_id)
    knowledge.add(session, [EntryInput(source="アリス", aliases=["アリスちゃん"], text="A knight.")], "en-US")
    stored = knowledge.add(session, [EntryInput(source="アリス", translation="Alice", aliases=["ありす"])], "en-US")
    assert len(stored.entries) == 1
    entry = stored.entries[0]
    assert (entry.translation, entry.text, entry.aliases) == ("Alice", "A knight.", ["アリスちゃん", "ありす"])


def test_an_entry_needs_a_name_or_text(project):
    store, project_id = project
    with pytest.raises(ValueError):
        knowledge.add(store.session(project_id), [EntryInput(kind="note")], "en-US")


def test_entries_are_saved_with_the_project_and_can_be_undone(project, tmp_path):
    store, project_id = project
    session = store.session(project_id)
    stored = knowledge.add(session, [EntryInput(source="アリス", translation="Alice")], "en-US")
    entry_id = stored.entries[0].id
    knowledge.update(session, entry_id, {"translation": "Alicia", "pinned": True}, "es-ES")
    reopened = ProjectStore(tmp_path).session(project_id).snapshot().project_component(Knowledge)
    assert (reopened.entries[0].translation, reopened.entries[0].language, reopened.entries[0].pinned) == ("Alicia", "es-ES", True)
    session.undo()
    assert knowledge.read(session.snapshot()).entries[0].translation == "Alice"
    knowledge.remove(session, [entry_id])
    assert knowledge.read(session.snapshot()).entries == []
    with pytest.raises(KeyError):
        knowledge.remove(session, [entry_id])


def test_series_facts_become_a_pinned_note_until_the_user_edits_it(project):
    store, project_id = project
    session = store.session(project_id)
    series = Series(id="1", title="Wataten", summary="A girl falls for her friend.", authors=["Mukuro"], tags=["Comedy"])
    knowledge.seed_series(session, series)
    entry = knowledge.read(session.snapshot()).entries[0]
    assert entry.pinned and entry.text == "By Mukuro. Tags: Comedy.\nA girl falls for her friend."

    revision = session.snapshot().revision
    knowledge.seed_series(session, series)
    assert session.snapshot().revision == revision  # nothing new, no undo step

    knowledge.seed_series(session, series.model_copy(update={"summary": "New summary."}))
    assert knowledge.read(session.snapshot()).entries[0].text.endswith("New summary.")

    knowledge.update(session, knowledge.SERIES_ENTRY, {"text": "My own words."}, "en-US")
    knowledge.seed_series(session, series)
    assert knowledge.read(session.snapshot()).entries[0].text == "My own words."


def test_a_series_with_nothing_to_say_writes_nothing(project):
    store, project_id = project
    session = store.session(project_id)
    knowledge.seed_series(session, Series(id="1", title="Untold"))
    assert session.snapshot().project_component(Knowledge) is None


# -- the API and the assistant ---------------------------------------------------------------------


def test_the_routes_edit_entries_and_show_what_a_page_gets(isolated_home):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from urakatsu.service.knowledge_api import knowledge_router
    from urakatsu.service.workspace import Workspace

    workspace = Workspace(device="cpu")
    app = FastAPI()
    app.include_router(knowledge_router(workspace), prefix="/api/v1")
    client = TestClient(app)
    project_id = workspace.create_project("p").id
    root = f"/api/v1/projects/{project_id}/knowledge"

    assert client.get(root).json() == {"enabled": True, "entries": []}
    assert client.get("/api/v1/projects/missing/knowledge").status_code == 404
    assert client.post(f"{root}/entries", json={"entries": [{"kind": "note"}]}).status_code == 400
    added = client.post(f"{root}/entries", json={"entries": [{"kind": "character", "source": "アリス", "translation": "Alice"}]}).json()
    entry = added["entries"][0]
    assert entry["language"] == workspace.config.translation.target_language

    assert client.patch(f"{root}/entries/{entry['id']}", json={"text": "A knight."}).json()["entries"][0]["text"] == "A knight."
    assert client.patch(f"{root}/entries/missing", json={"text": "x"}).status_code == 404

    page_id, _ = page(workspace.store, project_id, [("アリスだ", None)])
    picked = client.get(f"{root}/pages/{page_id}").json()
    assert picked["glossary"] == [{"source": "アリス", "translation": "Alice", "note": "A knight."}]
    assert client.get(f"{root}/pages/missing").status_code == 404

    assert client.put(f"{root}/enabled", json={"enabled": False}).json()["enabled"] is False
    assert client.get(f"{root}/pages/{page_id}").json()["glossary"] == []
    assert client.delete(f"{root}/entries/{entry['id']}").status_code == 204
    assert client.get(root).json()["entries"] == []


def test_the_assistant_reads_and_saves_knowledge(isolated_home):
    from urakatsu.agents.app_tools import delete_knowledge, get_knowledge, save_knowledge
    from urakatsu.agents.tools import ToolContext
    from urakatsu.service.workspace import Workspace

    workspace = Workspace(device="cpu")
    context = ToolContext(workspace, workspace.create_project("p").id)
    entries = [{"kind": "character", "source": "アリス", "translation": "Alice"}, {"kind": "note", "text": "Set in 1920.", "pinned": True}]
    assert save_knowledge.handler(context, {"entries": entries}) == {"saved": 2, "total": 2}
    found = get_knowledge.handler(context, {"query": "alice"})["entries"]
    assert [entry["source"] for entry in found] == ["アリス"]
    assert delete_knowledge.handler(context, {"ids": [found[0]["id"]]}) == {"total": 1}
