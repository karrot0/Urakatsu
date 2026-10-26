"""Sound effects: turning detected onomatopoeia into translated lettering. Not in Koharu.

Koharu's layout detector finds onomatopoeia but nothing downstream uses them. With
``SoundEffectsConfig.mode`` set, Urakatsu handles them in two places:

- Detection (``write_sound_effects``) merges onomatopoeia boxes that touch or nearly touch,
  because the detector often cuts one sound effect into pieces (달 | 칵, グ | ビ), and writes each
  group as a text region with a text block whose role is ``ROLE``. The OCR and translation stages
  then read and translate it like any other block; translation asks for a sound effect
  (``TranslationRequest.sound_effects``). In ``typeset`` mode the block is lettered and the
  onomatopoeia mask (``coo-mask``) lets inpainting erase the original. In ``redraw`` mode the
  block starts hidden and the original stays in the art for this stage to redraw; the pieces' own
  boxes are kept as ``sound-effect-mask``, because a merged box can span art between the pieces
  (a trail of ドスススス across a fighter, two 쾅 either side of a door).
- The sound effects stage (``SoundEffectsStage``, redraw mode only) sends each translated sound
  effect, with context around it, to an inpainting provider that follows prompts, with an
  instruction to re-letter it in the target language in the same style. Only the pieces' boxes are
  kept. The results form one generated raster layer per page above the Cleanup layer; like the
  Cleanup layer, it is not redone while it exists (delete it to redraw), so a re-run never paints
  over it.

Readings with no letters (art, speed lines and punctuation the detector boxed) are not redrawn.
"""

from __future__ import annotations

import asyncio
import functools
import io
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from urakatsu.core.results import LayoutDetection
from urakatsu.providers.base import Provider
from urakatsu.providers.translation.language import Language
from urakatsu.scene import At, Edit, Patch
from urakatsu.scene import components as c

from .base import StageInput, StageProcessor, finish, generation
from .inpainting import fill_region
from .watermarks import page_image

PRODUCER = "dev.urakatsu.pipeline.sound-effects"
ROLE = "dev.urakatsu.text.sound-effect"
LABEL = "onomatopoeia"
MASK = "coo-mask"
PIECES_MASK = "sound-effect-mask"
LAYER_NAME = "Sound effects"
# Boxes closer than this share of the smaller box's short side belong to one sound effect.
MERGE_GAP = 0.35
# Pixels around a sound effect's box that the redraw may repaint (outlines reach past the box).
REDRAW_PADDING = 4


# ---- detection ------------------------------------------------------------------------------------


@dataclass
class SoundEffectGroup:
    members: list[LayoutDetection]
    bbox: tuple[float, float, float, float]

    @property
    def score(self) -> float:
        return max(d.score for d in self.members)

    @property
    def largest(self) -> LayoutDetection:
        return max(self.members, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))


def _gap(a, b) -> float:
    """Distance between two boxes (0 when they overlap)."""
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dy = max(b[1] - a[3], a[1] - b[3], 0.0)
    return float(max(dx, dy))


def _short_side(box) -> float:
    return float(min(box[2] - box[0], box[3] - box[1]))


def group_sound_effects(detections: list[LayoutDetection]) -> list[SoundEffectGroup]:
    """Onomatopoeia detections merged into sound effects, in the order of their first piece."""
    pieces = [d for d in detections if d.label == LABEL]
    parents = list(range(len(pieces)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left in range(len(pieces)):
        for right in range(left + 1, len(pieces)):
            a, b = pieces[left].bbox, pieces[right].bbox
            if _gap(a, b) <= MERGE_GAP * min(_short_side(a), _short_side(b)):
                root_left, root_right = find(left), find(right)
                if root_left != root_right:
                    parents[max(root_left, root_right)] = min(root_left, root_right)

    groups: dict[int, list[LayoutDetection]] = {}
    for index, piece in enumerate(pieces):
        groups.setdefault(find(index), []).append(piece)
    result = []
    for root in sorted(groups):
        members = groups[root]
        bbox = (
            min(d.bbox[0] for d in members),
            min(d.bbox[1] for d in members),
            max(d.bbox[2] for d in members),
            max(d.bbox[3] for d in members),
        )
        result.append(SoundEffectGroup(members=members, bbox=bbox))
    return result


def write_sound_effects(edit: Edit, page: str, image: np.ndarray, groups: list[SoundEffectGroup], mode: str, gen: c.Generation) -> list[str]:
    """One text region and block per sound effect; returns the text layers written."""
    from .detection import infer_typography, rectangle_geometry, write_text_role

    layers = []
    for group in groups:
        typography = infer_typography(image, group.largest)
        entity = edit.add_entity(page, At.END)
        edit.set(entity, rectangle_geometry(group.bbox))
        edit.set(entity, c.Region(origin=c.generated(gen), kind=c.RegionKinds.TEXT, label=LABEL))
        edit.set(
            entity,
            c.DetectionAnalysis(origin=c.generated(gen), labels=[c.DetectionLabel(kind=c.RegionKinds.TEXT, confidence=group.score)]),
        )
        content = edit.add_text_content(page, At.END)
        layer = edit.add_text_layer(
            page,
            At.END,
            content,
            c.TextLayout(origin=c.generated(gen), kind=c.TextLayoutKind.PARAGRAPH, angle_degrees=None),
        )
        write_text_role(edit, content, ROLE, gen)
        edit.set(
            layer,
            c.Typography(
                origin=c.generated(gen),
                preferred_font=None,
                font_weight=None,
                font_style=None,
                size=None,
                auto_fit=True,
                color=(*typography.color, 255) if typography is not None else None,
                stroke_color=(*typography.stroke_color, 255) if typography is not None and typography.stroke_color is not None else None,
                stroke_width=typography.stroke_width if typography is not None else None,
                alignment=None,
                writing_mode=None,  # sound effects are lettered horizontally in the target language
                extensions={},
            ),
        )
        if mode == "redraw":
            # The redrawn art carries the translation; the block holds the reading and translation.
            edit.set(layer, c.Visibility(origin=c.generated(gen), visible=False))
        edit.relate(c.Relations.RECOGNIZED_FROM, content, entity)
        edit.relate(c.Relations.FITS_TO, layer, entity)
        layers.append(layer)
    return layers


def write_sound_effect_mask(input: StageInput, edit: Edit, page: str, detections: list, size: tuple[int, int]) -> None:
    """``typeset`` erases the originals through the onomatopoeia mask (``coo-mask``, which inpainting
    reads); ``redraw`` keeps them in the art and records the pieces' boxes for the redraw instead."""
    from .detection import _radius_for, closed_mask_for, preserve_mask_outside_region

    width, height = size
    if input.sound_effects == "typeset":
        role, remove = MASK, PIECES_MASK
        mask = closed_mask_for(detections, LABEL, size, _radius_for(size))
    else:
        role, remove = PIECES_MASK, MASK
        mask = np.zeros((height, width), dtype=np.uint8)
        for piece in detections:
            if piece.label == LABEL:
                left, top, right, bottom = _padded(piece.bbox, width, height)
                mask[top:bottom, left:right] = 255
    edit.remove_asset(page, remove)
    if input.region is not None:
        preserve_mask_outside_region(input, page, role, input.region, mask)
    buffer = io.BytesIO()
    Image.fromarray(mask).save(buffer, format="PNG", optimize=False)
    edit.set_asset(
        page,
        role,
        c.AssetInput(data=buffer.getvalue(), media_type="image/png", metadata=c.AssetMetadata(width=width, height=height, attributes={})),
    )


def _padded(bbox, width: int, height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    return (
        max(int(np.floor(left)) - REDRAW_PADDING, 0),
        max(int(np.floor(top)) - REDRAW_PADDING, 0),
        min(int(np.ceil(right)) + REDRAW_PADDING, width),
        min(int(np.ceil(bottom)) + REDRAW_PADDING, height),
    )


# ---- redraw stage ---------------------------------------------------------------------------------


@dataclass
class _Target:
    content: str
    box: tuple[int, int, int, int]  # left, top, right, bottom (exclusive), padded and clamped
    source: str
    translation: str
    language: str


def readable(text: str) -> bool:
    """A reading worth redrawing: it has at least one letter (kana, hangul, latin...)."""
    return any(ch.isalpha() for ch in text)


def is_sound_effect(scene, content: str) -> bool:
    role = scene.component(content, c.TextRole)
    return role is not None and role.role == ROLE


def redraw_targets(input: StageInput, width: int, height: int) -> list[_Target]:
    scene = input.scene
    targets = []
    for layer in scene.text_layers(input.page):
        if not input.contains_entity(layer):
            continue
        content = scene.layer_content(layer)
        if not is_sound_effect(scene, content):
            continue
        source = scene.component(content, c.SourceText)
        translation = scene.component(content, c.Translation)
        if source is None or translation is None or not readable(source.text.value) or not translation.text.value.strip():
            continue
        region = scene.source_region(content)
        geometry = scene.component(region, c.Geometry) if region is not None else None
        if geometry is None or not geometry.points:
            continue
        box = _padded(geometry.bounds(), width, height)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        targets.append(_Target(content, box, source.text.value.strip(), translation.text.value.strip(), translation.language))
    return targets


def redraw_layer(scene, page: str) -> Optional[str]:
    """The page's generated sound effects layer, if this stage has drawn one."""
    for entity in scene.raster_layers(page):
        layer = scene.component(entity, c.RasterLayer)
        origin = layer.origin if layer is not None else None
        if isinstance(origin, c.GeneratedOrigin) and origin.generation.producer == PRODUCER:
            return entity
    return None


def instruction(target: _Target, instructed: bool) -> str:
    language = Language.coerce(target.language).display_name if target.language else "the target language"
    if instructed:
        return (
            f'Replace the sound effect "{target.source}" in the middle of this image with the {language} sound effect '
            f'"{target.translation}". Hand-letter it in the same style, outline, colour, size and angle as the original, '
            "remove the original lettering completely, and keep all the artwork around it unchanged."
        )
    return f'manga sound effect lettering "{target.translation}", bold hand-drawn comic letters matching the surrounding artwork'


def _instructed(provider: Provider) -> bool:
    """Whether the loaded model takes edit instructions; the generic diffusion provider knows only once loaded."""
    processor = getattr(provider, "_processor", None)
    edits = getattr(processor, "edits", None)
    return bool(edits) if edits is not None else bool(getattr(provider, "instructed", False))


class SoundEffectsStage(StageProcessor):
    def skip(self, input: StageInput, provider: Provider) -> bool:
        if input.sound_effects != "redraw" or redraw_layer(input.scene, input.page) is not None:
            return True
        meta = input.scene.page(input.page)
        return not redraw_targets(input, int(meta.width), int(meta.height))

    async def process(self, input: StageInput, provider: Provider) -> Patch:
        if not getattr(provider, "prompted", False):
            raise ValueError(
                f"{provider.name} only removes lettering, so it cannot redraw sound effects; "
                "pick an image model that follows prompts in Settings, Pipeline, Sound effects"
            )
        source = page_image(input)
        if source is None:
            raise ValueError(f"page {input.page} has no source image")
        source = source.convert("RGB")
        targets = redraw_targets(input, source.width, source.height)
        pieces = input.images.get(input.scene, input.page, PIECES_MASK)
        pieces_mask = np.asarray(pieces.convert("L")) if pieces is not None else None
        if pieces_mask is not None and pieces_mask.shape != (source.height, source.width):
            pieces_mask = None
        instructed = _instructed(provider)
        overlay = np.zeros((source.height, source.width, 4), dtype=np.uint8)
        for target in targets:
            left, top, right, bottom = target.box
            mask = np.zeros((source.height, source.width), dtype=np.uint8)
            # Only the pieces' own boxes: a merged box can span art between them.
            window = pieces_mask[top:bottom, left:right] if pieces_mask is not None else None
            mask[top:bottom, left:right] = window if window is not None and window.any() else 255
            fill = functools.partial(provider._fill, prompt=instruction(target, instructed))
            redrawn = np.asarray((await asyncio.to_thread(fill_region, source, mask, fill)).convert("RGB"))
            keep = mask >= 127
            overlay[keep, :3] = redrawn[keep]
            overlay[keep, 3] = 255
        return commit(input, provider, overlay, [t.content for t in targets])


def commit(input: StageInput, provider: Provider, overlay: np.ndarray, contents: list[str]) -> Patch:
    page = input.page
    gen = generation(PRODUCER, provider.id)
    edit = input.scene.edit_as(gen)
    edit.observe_assets(page)
    for content in contents:
        edit.observe(content, c.SourceText)
        edit.observe(content, c.Translation)
    if not overlay[..., 3].any():
        return finish(edit)
    rasters = input.scene.raster_layers(page)
    entity = edit.add_entity(page, At.after(rasters[-1]) if rasters else At.START)
    edit.set(entity, c.RasterLayer(origin=c.generated(gen), name=LAYER_NAME, kind=c.RasterLayerKind.EDIT))
    buffer = io.BytesIO()
    Image.fromarray(overlay, "RGBA").save(buffer, format="PNG")
    height, width = overlay.shape[:2]
    edit.set_asset(
        entity,
        "source",
        c.AssetInput(data=buffer.getvalue(), media_type="image/png", metadata=c.AssetMetadata(width=width, height=height, attributes={})),
    )
    return finish(edit)
