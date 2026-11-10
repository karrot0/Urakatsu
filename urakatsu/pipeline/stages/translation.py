"""Translation stage.

Targets are the non-blank source texts of the page's text layers, in layer order,
restricted to the run's scope. The provider translates them as one request. Results are
written as generated ``Translation`` values, except where the user already authored one.
An ellipsis source keeps the ellipsis.

Not in Koharu: the request carries the project's knowledge (``pipeline/knowledge.py``), sized by
``TranslationConfig.prompt_size``; with Story so far on, a whole-page run also writes the page's
``PageSummary`` from the reply, unless the user wrote that page's summary. Source text is sent
through ``ocr.trim_runaway``, so a runaway OCR repeat stored before that existed (or typed in)
cannot make the model loop until it returns no JSON; the stored text is left as it is.
"""

from __future__ import annotations

import asyncio

from urakatsu.pipeline import knowledge
from urakatsu.providers.translation.backend import TranslationRequest
from urakatsu.providers.translation.language import Language
from urakatsu.scene import Authored, PageSummary, Patch, Snapshot, SourceText, Translation
from urakatsu.scene.components import is_user

from .base import StageInput, StageProcessor, finish, generation
from .ocr import trim_runaway
from .sound_effects import ROLE, is_sound_effect
from .watermarks import page_image

PRODUCER = "dev.koharu.pipeline.translation"
ELLIPSIS = "…"


def retrieve(scene: Snapshot, page: str, targets: list[tuple[str, str]], language: str, budget: int) -> knowledge.Retrieved:
    """The page's knowledge for ``targets``: sound effects are neither searched for nor quoted."""
    speech = [text for content, text in targets if not is_sound_effect(scene, content)]
    return knowledge.retrieve(scene, page, speech, frozenset(content for content, _ in targets), language, budget, ROLE)


class TranslationStage(StageProcessor):
    async def process(self, input: StageInput, provider) -> Patch:
        scene = input.scene
        targets: list[tuple[str, str]] = []
        for layer in scene.text_layers(input.page):
            if not input.contains_entity(layer):
                continue
            content = scene.layer_content(layer)
            source = scene.component(content, SourceText)
            if source is None:
                continue
            if source.text.value.strip():
                targets.append((content, source.text.value))

        config = provider.settings.translation
        target_language = Language.coerce(config.target_language)
        budget, image_dimension = config.prompt_limits(provider.context_budget)
        # Other pages' text is read, not observed: an edit there must not reject this page's result.
        # Off the event loop: a long series means thousands of lines to read per page.
        retrieved = await asyncio.to_thread(retrieve, scene, input.page, targets, target_language.tag, budget)
        # A summary covers the page: re-translating part of it keeps the one it has.
        whole_page = len(targets) == len(_sources(scene, input.page))
        summarize = retrieved.summarize and whole_page and not _user_summary(scene, input.page)
        request = TranslationRequest(
            segments=[trim_runaway(text) for _, text in targets],
            target_language=target_language,
            instructions=config.instructions,
            context=retrieved.context,
            glossary=retrieved.glossary,
            notes=retrieved.notes,
            background=retrieved.background,
            story=retrieved.story,
            summarize=summarize,
            image_dimension=image_dimension,
            sound_effects=frozenset(i for i, (content, _) in enumerate(targets) if is_sound_effect(scene, content)),
        )
        if provider.vision_enabled():
            image = page_image(input)
            if image is not None:
                request.image = image

        translated = await provider.translate(request)
        generated = generation(PRODUCER, provider.id)
        edit = scene.edit_as(generated)
        for entity, _ in targets:
            edit.observe(entity, SourceText)
            edit.observe(entity, Translation)
        for (entity, source_text), text in zip(targets, translated):
            existing = scene.component(entity, Translation)
            if existing is not None and is_user(existing.text.origin):
                continue
            if source_text.strip() == ELLIPSIS:
                text = ELLIPSIS
            edit.set(
                entity,
                Translation(
                    text=Authored.generated(text, generated),
                    language=target_language.tag,
                ),
            )
        written = knowledge.clean_summary(getattr(translated, "summary", "") or "") if summarize else ""
        if written:
            edit.set(input.page, PageSummary(text=written, language=target_language.tag))
        return finish(edit)


def _sources(scene: Snapshot, page: str) -> list[str]:
    """The page's text contents that have source text to translate."""
    found = []
    for layer in scene.text_layers(page):
        content = scene.layer_content(layer)
        source = scene.component(content, SourceText)
        if source is not None and source.text.value.strip():
            found.append(content)
    return found


def _user_summary(scene: Snapshot, page: str) -> bool:
    existing = knowledge.summary(scene, page)
    return existing is not None and is_user(existing.origin)
