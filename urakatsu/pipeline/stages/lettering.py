"""Lettering stage: letter each translation in its source's kind of face.

For each text region on the page, the font reader (``providers/lettering``) says how likely each
lettering style is to be the face its text is lettered in, and how heavy the fonts it named are;
the glyph stroke is measured from the page. The region keeps that as ``LetteringAnalysis``. The
text's layer then takes a style, and bold when its source is heavier than the book's dialogue
(``pipeline/lettering.py``), kept as its ``LetteringStyle``, which the renderer sets in that
style's font (``core/styles.py``) wherever the layer's typography sets none of its own.

Koharu 0.100 reads the faces inside detection and picks the style inside translation. As a stage
of its own it runs on pages that are already detected, so a project made before it gets styles
without being detected again (Koharu: "run Detect again, then Translate"), the font reader is a
provider like any other, and it can be turned off (``LetteringConfig``).

Each lettering is read in five crops, its box with 0, 3, 7, 12 and 20% more on each side, and the
readings are averaged, as Koharu does: YuzuMarker trains on 7%, and its reading moves with the
margin (Koharu, on Manga109: moving one crop's margin changed the style of 8% of balloon texts
and 18% of sound effects; the mean of five, 1% and 2%).

A page's styles are placed from the readings the project has when the page runs, its own
included. The typography, detection's or the user's, is not touched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from urakatsu.core.styles import STYLES
from urakatsu.pipeline import lettering
from urakatsu.pipeline.scope import geometry_extents
from urakatsu.providers.base import FontReading, LetteringProvider, Provider
from urakatsu.scene import Relations, Snapshot
from urakatsu.scene import components as c

from .base import StageInput, StageProcessor, finish, generation
from .ocr import VERTICAL_ASPECT
from .sound_effects import ROLE as SOUND_EFFECT_ROLE
from .watermarks import page_image

PRODUCER = "dev.urakatsu.pipeline.lettering"
# The margins a lettering's font is read under, as shares of its box on each side.
MARGINS = (0.0, 0.03, 0.07, 0.12, 0.2)
# The least probability a region keeps of a style: none under it changes a choice of lettering.
MIN_KEPT = 0.01


@dataclass
class _Target:
    region: str
    content: str
    layer: Optional[str]
    box: tuple[float, float, float, float]


def _targets(input: StageInput) -> list[_Target]:
    """The page's text regions in scope, each with the content read from it and that content's layer."""
    targets: list[_Target] = []
    for region in input.scene.descendants(input.page):
        if not input.contains_entity(region):
            continue
        value = input.scene.component(region, c.Region)
        if value is None or value.kind != c.RegionKinds.TEXT:
            continue
        geometry = input.scene.component(region, c.Geometry)
        box = geometry_extents(geometry) if geometry is not None else None
        if box is None:
            continue
        for relation in input.scene.relations_to(region, Relations.RECOGNIZED_FROM):
            presents = input.scene.relations_to(relation.source, Relations.PRESENTS)
            targets.append(_Target(region=region, content=relation.source, layer=presents[0].source if presents else None, box=box))
            break
    return targets


def crops(image: Image.Image, box: tuple[float, float, float, float]) -> list[Image.Image]:
    """``layers::crops``: a lettering's box with each of the ``MARGINS`` more on each side."""
    x0, y0, x1, y1 = box
    result = []
    for margin in MARGINS:
        across, down = (x1 - x0) * margin, (y1 - y0) * margin
        left, top = max(int(x0 - across), 0), max(int(y0 - down), 0)
        right, bottom = min(int(math.ceil(x1 + across)), image.width), min(int(math.ceil(y1 + down)), image.height)
        if right > left and bottom > top:
            result.append(image.crop((left, top, right, bottom)))
    return result


def mean_reading(readings: list[FontReading]) -> FontReading:
    """A lettering's readings, one a crop, averaged."""
    kernels: dict[str, float] = {}
    for item in readings:
        for style, probability in item.kernels.items():
            kernels[style] = kernels.get(style, 0.0) + probability / len(readings)
    weights = [item.weight for item in readings if item.weight is not None]
    return FontReading(kernels=kernels, weight=sum(weights) / len(weights) if weights else None)


def block_kind(scene: Snapshot, region: str, content: str) -> str:
    """What a block is: a sound effect, a text in a box, a text in a speech balloon, or a text outside both."""
    role = scene.component(content, c.TextRole)
    if role is not None and role.role == SOUND_EFFECT_ROLE:
        return lettering.SOUND_EFFECT
    inside = scene.relation_from(region, Relations.INSIDE)
    if inside is None:
        return lettering.FREE
    outline = scene.component(inside.target, c.Geometry)
    points = [(point.x, point.y) for point in outline.points] if outline is not None else []
    return lettering.BOXED if lettering.boxed(points) else lettering.BALLOON


def _source_text(scene: Snapshot, content: str) -> str:
    source = scene.component(content, c.SourceText)
    return source.text.value if source is not None else ""


def project_readings(scene: Snapshot) -> dict[str, lettering.Reading]:
    """Every read text region of the project, by region."""
    scene = scene.borrowed()
    readings: dict[str, lettering.Reading] = {}
    for page in scene.pages():
        for region in scene.descendants(page):
            analysis = scene.component(region, c.LetteringAnalysis)
            if analysis is None:
                continue
            relations = scene.relations_to(region, Relations.RECOGNIZED_FROM)
            if not relations:
                continue
            content = relations[0].source
            readings[region] = lettering.reading(analysis.kernels, _source_text(scene, content), block_kind(scene, region, content), analysis.stroke, analysis.weight)
    return readings


def _vertical(scene: Snapshot, target: _Target) -> bool:
    analysis = scene.component(target.region, c.OcrAnalysis)
    if analysis is not None and analysis.direction is not c.TextDirection.AUTO:
        return analysis.direction is c.TextDirection.VERTICAL
    x0, y0, x1, y1 = target.box
    return (y1 - y0) >= (x1 - x0) * VERTICAL_ASPECT


class LetteringStage(StageProcessor):
    def done(self, input: StageInput) -> bool:
        targets = _targets(input)
        return bool(targets) and all(input.scene.has(target.region, c.LetteringAnalysis) for target in targets)

    def skip(self, input: StageInput, provider: Provider) -> bool:
        # A page that is already read is not read again, unless the run is forced (Redo).
        return not _targets(input) or (not input.force and self.done(input))

    async def process(self, input: StageInput, provider: Provider):
        if not isinstance(provider, LetteringProvider):
            raise TypeError(f"lettering stage cannot run provider {provider.id!r}")
        image = page_image(input)
        if image is None:
            raise ValueError(f"page {input.page} has no source image")
        image = image.convert("RGB")

        targets = _targets(input)
        batches = [crops(image, target.box) for target in targets]
        flat = [crop for batch in batches for crop in batch]
        read = await provider.read_fonts(flat) if flat else []
        if len(read) != len(flat):
            raise ValueError(f"{provider.id} read {len(read)} fonts for {len(flat)} crops")
        pixels = np.asarray(image, dtype=np.uint8)

        gen = generation(PRODUCER, provider.id)
        edit = input.scene.edit_as(gen)
        edit.observe_assets(input.page)
        readings = project_readings(input.scene)
        start = 0
        for target, batch in zip(targets, batches):
            edit.observe(target.region, c.Region)
            edit.observe(target.region, c.Geometry)
            edit.observe(target.content, c.SourceText)
            font = mean_reading(read[start : start + len(batch)]) if batch else FontReading()
            start += len(batch)
            kernels = {style: round(font.kernels[style], 3) for style in STYLES if font.kernels.get(style, 0.0) >= MIN_KEPT}
            x0, y0, x1, y1 = target.box
            window = pixels[max(int(y0), 0) : max(int(math.ceil(y1)), 0), max(int(x0), 0) : max(int(math.ceil(x1)), 0)]
            stroke = lettering.measure_stroke(window, _vertical(input.scene, target)) if window.size else None
            weight = round(font.weight, 1) if font.weight is not None else None
            edit.set(target.region, c.LetteringAnalysis(origin=c.generated(gen), kernels=kernels, weight=weight, stroke=stroke))
            readings[target.region] = lettering.reading(
                kernels, _source_text(input.scene, target.content), block_kind(input.scene, target.region, target.content), stroke, weight
            )

        placement = lettering.Placement(readings)
        for target in targets:
            if target.layer is None:
                continue
            style, weight = placement.of(target.region)
            edit.set(target.layer, c.LetteringStyle(origin=c.generated(gen), style=style, font_weight=weight))
        return finish(edit)
