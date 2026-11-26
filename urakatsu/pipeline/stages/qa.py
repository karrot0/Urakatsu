"""QA stage: proofreading and layout checks after translation. Not in Koharu.

The logic is ``pipeline/qa`` (see its notes for what is written, when the model is asked, and
what fix mode changes). The stage's model is the translation provider (``QaConfig.model``, else
the translation model), and the pipeline hands it the instance already loaded for translation
when the selection is the same (``Pipeline.provider``), so a local model is never started twice.

``loads_provider``: the pipeline leaves loading to the stage. A page whose text is unchanged since
its last review, or a provider that only translates, needs no model, so the layout checks run
without starting one.
"""

from __future__ import annotations

from urakatsu.core.config import QaConfig
from urakatsu.providers.base import Provider
from urakatsu.scene import Patch, QaPage

from ..qa import has_work, review_page
from .base import StageInput, StageProcessor


class QaStage(StageProcessor):
    loads_provider = True

    def skip(self, input: StageInput, provider: Provider) -> bool:
        return not has_work(input.scene, input.page, _layers(input))

    def done(self, input: StageInput) -> bool:
        return input.scene.has(input.page, QaPage)

    async def process(self, input: StageInput, provider: Provider) -> Patch:
        config = input.qa or QaConfig(mode="flag")
        review = await review_page(
            input.scene,
            input.page,
            provider,
            config,
            entities=_layers(input),
            typesetting=input.typesetting,
            force=input.force,
        )
        return review.patch


def _layers(input: StageInput):
    """The run's text layers on this page, or None for the whole page."""
    if input.entities is None and input.region is None:
        return None
    return {layer for layer in input.scene.text_layers(input.page) if input.contains_entity(layer)}
