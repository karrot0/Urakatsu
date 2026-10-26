"""OCR stage.

For each text region on the page that a ``RecognizedFrom`` relation points at a content
entity, crop the region from the page source image, run the OCR provider on the crop, and
commit the result as generated ``SourceText`` plus ``OcrAnalysis`` on the region. Source
text that the user wrote is never overwritten.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from urakatsu.providers.base import OcrProvider, Provider
from urakatsu.providers.ocr._common import is_whitespace
from urakatsu.pipeline.scope import geometry_extents
from urakatsu.scene import components as c
from urakatsu.scene import Relations

from .base import StageInput, StageProcessor, finish, generation
from .watermarks import page_image

PRODUCER = "dev.koharu.pipeline.ocr"
DEFAULT_LANGUAGE = "ja-JP"
# ocr.rs: a region taller than 1.15x its width is vertical text.
VERTICAL_ASPECT = 1.15
# ocr.rs ``normalize_ocr_text``: replacement glyphs that Manga OCR emits for an isolated ellipsis.
PLACEHOLDER_GLYPHS = frozenset("☐□▢▣�")


@dataclass
class _Target:
    content: str
    region: str
    geometry: c.Geometry
    previous: Optional[c.SourceText]


def normalize_ocr_text(text: str) -> str:
    """Only an all-placeholder sequence (two or more glyphs) becomes an ellipsis; other text is kept."""
    visible = [ch for ch in text if not is_whitespace(ch)]
    if len(visible) >= 2 and all(ch in PLACEHOLDER_GLYPHS for ch in visible):
        return "…"
    return text


def crop(source: Image.Image, geometry: c.Geometry) -> Image.Image:
    """``ocr.rs`` ``crop``: floor the minimum, ceil the maximum, clamp to the image."""
    extents = geometry_extents(geometry)
    if extents is None:
        raise ValueError("geometry is empty")
    min_x, min_y, max_x, max_y = extents
    x = max(math.floor(min_x), 0)
    y = max(math.floor(min_y), 0)
    right = min(max(math.ceil(max_x), 0), source.width)
    bottom = min(max(math.ceil(max_y), 0), source.height)
    if right <= x or bottom <= y:
        raise ValueError("geometry does not overlap the image")
    return source.crop((x, y, right, bottom))


def _text_direction(geometry: c.Geometry) -> c.TextDirection:
    extents = geometry_extents(geometry)
    if extents is None:
        raise ValueError("text region has empty geometry")
    min_x, min_y, max_x, max_y = extents
    if max_y - min_y >= (max_x - min_x) * VERTICAL_ASPECT:
        return c.TextDirection.VERTICAL
    return c.TextDirection.HORIZONTAL


class OcrStage(StageProcessor):
    def _targets(self, input: StageInput) -> list[_Target]:
        targets: list[_Target] = []
        for region in input.scene.descendants(input.page):
            if not input.contains_entity(region):
                continue
            value = input.scene.component(region, c.Region)
            if value is None or value.kind != c.RegionKinds.TEXT:
                continue
            geometry = input.scene.component(region, c.Geometry)
            if geometry is None:
                raise ValueError(f"text region {region} has no geometry")
            for relation in input.scene.relations_to(region, Relations.RECOGNIZED_FROM):
                content = relation.source
                previous = input.scene.component(content, c.SourceText)
                if previous is not None and c.is_user(previous.text.origin):
                    continue
                targets.append(_Target(content=content, region=region, geometry=geometry, previous=previous))
        return targets

    def skip(self, input: StageInput, provider: Provider) -> bool:
        return not self._targets(input)

    async def process(self, input: StageInput, provider: Provider):
        if not isinstance(provider, OcrProvider):
            raise TypeError(f"OCR stage cannot run provider {provider.id!r}")
        source = page_image(input)
        if source is None:
            raise ValueError(f"page {input.page} has no source image")

        # ocr.rs crops every text region before it looks at its recognition relations.
        targets = self._targets(input)
        crops: dict[str, Image.Image] = {}
        for target in targets:
            if target.region not in crops:
                crops[target.region] = crop(source, target.geometry)
        texts = await provider.recognize([crops[target.region] for target in targets]) if targets else []

        model = generation(PRODUCER, provider.id)
        edit = input.scene.edit_as(model)
        edit.observe_assets(input.page)
        for target in targets:
            edit.observe(target.region, c.Region)
            edit.observe(target.region, c.Geometry)
            edit.observe(target.content, c.SourceText)
        for target, text in zip(targets, texts):
            language = (target.previous.language if target.previous is not None else None) or DEFAULT_LANGUAGE
            edit.set(
                target.content,
                c.SourceText(text=c.Authored.generated(normalize_ocr_text(text), model), language=language),
            )
            edit.set(
                target.region,
                c.OcrAnalysis(
                    origin=c.generated(model),
                    direction=_text_direction(target.geometry),
                    confidence=None,
                    line_boundaries=[],
                ),
            )
        return finish(edit)
