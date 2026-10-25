"""Translation stage.

Targets are the non-blank source texts of the page's text layers, in layer order,
restricted to the run's scope. The provider translates them as one request. Results are
written as generated ``Translation`` values, except where the user already authored one.
An ellipsis source keeps the ellipsis.
"""

from __future__ import annotations

from urakatsu.providers.translation.backend import TranslationRequest
from urakatsu.providers.translation.language import Language
from urakatsu.scene import Authored, Patch, SourceText, Translation
from urakatsu.scene.components import is_user

from .base import StageInput, StageProcessor, finish, generation
from .sound_effects import is_sound_effect

PRODUCER = "dev.koharu.pipeline.translation"
ELLIPSIS = "…"


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

        target_language = Language.coerce(provider.settings.translation.target_language)
        request = TranslationRequest(
            segments=[text for _, text in targets],
            target_language=target_language,
            instructions=provider.settings.translation.instructions,
            sound_effects=frozenset(i for i, (content, _) in enumerate(targets) if is_sound_effect(scene, content)),
        )
        if provider.vision_enabled():
            image = input.images.get(scene, input.page, "source")
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
        return finish(edit)
