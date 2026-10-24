"""Stage processor contract.

A stage processor owns the stage logic (what to read from the scene, how
to build the patch). The model it runs is a *provider*, selected by
configuration and bound to a runtime, so any provider of the right kind can be
swapped in without touching stage logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from urakatsu.pipeline.images import ImageCache
from urakatsu.pipeline.scope import Bounds, contains_entity
from urakatsu.providers.base import Provider
from urakatsu.scene import Edit, Generation, Patch, Snapshot


@dataclass
class InpaintingMask:
    page: str
    png: bytes
    # Generative fill (the canvas's selection + prompt): the fill model repaints one crop
    # around the mask, following ``prompt`` (empty removes). Remove-tool masks are not generative.
    prompt: Optional[str] = None
    generative: bool = False


@dataclass
class StageInput:
    scene: Snapshot
    page: str
    entities: Optional[set[str]]
    region: Optional[Bounds]
    images: ImageCache
    inpainting_mask: Optional[InpaintingMask] = None

    def contains_entity(self, entity: str) -> bool:
        return contains_entity(self.scene, self.page, self.entities, self.region, entity)


class StageProcessor:
    """One stage. ``skip`` is cheap and runs before the provider is loaded."""

    def skip(self, input: StageInput, provider: Provider) -> bool:
        return False

    async def process(self, input: StageInput, provider: Provider) -> Patch:
        raise NotImplementedError


def generation(producer: str, model: str) -> Generation:
    return Generation(producer=producer, model=model)


def finish(edit: Edit) -> Patch:
    return edit.finish()
