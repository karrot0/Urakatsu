"""Sound effects: grouping split onomatopoeia, the typeset and redraw modes, and the redraw stage."""

from __future__ import annotations

import asyncio
import io
import json

import numpy as np
import pytest
from PIL import Image

from urakatsu.core.config import PipelineConfig, SoundEffectsConfig
from urakatsu.core.results import LayoutDetection, LayoutDetections, LayoutMask
from urakatsu.core.stage import Operation, Stage
from urakatsu.pipeline.images import ImageCache
from urakatsu.pipeline.stages import sound_effects as sfx
from urakatsu.pipeline.stages.base import StageInput, generation
from urakatsu.pipeline.stages.detection import DetectionStage
from urakatsu.pipeline.stages.sound_effects import SoundEffectsStage
from urakatsu.providers.translation.backend import TranslationRequest
from urakatsu.providers.translation.prompt import prompts
from urakatsu.scene import At, AssetInput, AssetMetadata, Page, Session
from urakatsu.scene import components as c


def _piece(label: str, x: int, y: int, width: int, height: int, score: float = 0.5) -> LayoutDetection:
    return LayoutDetection(
        label_id=0,
        label=label,
        score=score,
        bbox=(float(x), float(y), float(x + width), float(y + height)),
        area=width * height,
        mask=LayoutMask(x=x, y=y, width=width, height=height, pixels=np.full((height, width), 255, dtype=np.uint8)),
    )


class _Detector:
    def __init__(self, detections):
        self.detections = detections
        self.id = "fake-detector"

    async def detect(self, image):
        return LayoutDetections(image.width, image.height, list(self.detections))


class _Fill:
    """A prompted inpainting provider that paints its crop red and records the prompts."""

    id = "fake-fill"
    name = "Fake fill"
    prompted = True
    instructed = True

    def __init__(self):
        self.prompts: list[str] = []

    def _fill(self, image, mask, prompt):
        self.prompts.append(prompt)
        return Image.new("RGB", image.size, (255, 0, 0))


def _page(session: Session, size: int = 200):
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), "white").save(buffer, format="PNG")
    edit = session.snapshot().edit()
    page = edit.add_page(Page(label="page", width=float(size), height=float(size)), At.END)
    edit.set_asset(page, "source", AssetInput(data=buffer.getvalue(), media_type="image/png", metadata=AssetMetadata(width=size, height=size)))
    return session.commit(edit.finish()), page


# Two halves of one sound effect side by side (달 | 칵), one far away, and a line of dialogue.
DETECTIONS = [
    _piece("onomatopoeia", 20, 20, 30, 30),
    _piece("onomatopoeia", 52, 22, 30, 28, score=0.7),
    _piece("onomatopoeia", 140, 140, 30, 30),
    _piece("text", 100, 20, 40, 20),
]


def _detect(session: Session, snapshot, page, mode: str):
    stage_input = StageInput(snapshot, page, None, None, ImageCache(), None, mode)
    return session.commit(asyncio.run(DetectionStage().process(stage_input, _Detector(DETECTIONS))))


def _sound_effect_layers(snapshot, page):
    return [layer for layer in snapshot.text_layers(page) if sfx.is_sound_effect(snapshot, snapshot.layer_content(layer))]


# ---- grouping -------------------------------------------------------------------------------------


def test_touching_pieces_merge_into_one_sound_effect_and_distant_ones_stay_apart():
    groups = sfx.group_sound_effects(DETECTIONS)
    assert [g.bbox for g in groups] == [(20.0, 20.0, 82.0, 50.0), (140.0, 140.0, 170.0, 170.0)]
    assert groups[0].score == 0.7


# ---- detection ------------------------------------------------------------------------------------


def test_off_keeps_onomatopoeia_as_plain_regions(tmp_path):
    session = Session(tmp_path)
    snapshot, page = _page(session)
    snapshot = _detect(session, snapshot, page, "off")
    assert _sound_effect_layers(snapshot, page) == []
    assert len(snapshot.text_layers(page)) == 1
    assert snapshot.asset(page, sfx.MASK) is None


def test_typeset_writes_a_text_block_per_sound_effect_and_the_onomatopoeia_mask(tmp_path):
    session = Session(tmp_path)
    snapshot, page = _page(session)
    snapshot = _detect(session, snapshot, page, "typeset")
    layers = _sound_effect_layers(snapshot, page)
    assert len(layers) == 2 and len(snapshot.text_layers(page)) == 3
    region = snapshot.source_region(snapshot.layer_content(layers[0]))
    assert snapshot.component(region, c.Region).kind == c.RegionKinds.TEXT
    assert snapshot.component(region, c.Geometry).bounds() == (20.0, 20.0, 82.0, 50.0)
    assert (snapshot.component(layers[0], c.Visibility) or c.Visibility()).visible
    mask = np.asarray(Image.open(io.BytesIO(snapshot.asset_bytes(page, sfx.MASK))).convert("L"))
    assert mask[35, 35] == 255 and mask[30, 120] == 0  # the sound effect, not the dialogue
    text_mask = np.asarray(Image.open(io.BytesIO(snapshot.asset_bytes(page, "text-mask"))).convert("L"))
    assert text_mask[35, 35] == 0


def test_redraw_hides_the_blocks_and_keeps_the_originals_in_the_art(tmp_path):
    session = Session(tmp_path)
    snapshot, page = _page(session)
    snapshot = _detect(session, snapshot, page, "redraw")
    layers = _sound_effect_layers(snapshot, page)
    assert len(layers) == 2
    assert all(not snapshot.component(layer, c.Visibility).visible for layer in layers)
    assert snapshot.asset(page, sfx.MASK) is None


def _detect_pair(session: Session, snapshot, page, mode: str, text_score: float, sound_effect_score: float, force: bool = False):
    """One lettering found under both labels."""
    found = [_piece("text", 20, 20, 40, 100, score=text_score), _piece("onomatopoeia", 24, 24, 24, 90, score=sound_effect_score)]
    stage_input = StageInput(snapshot, page, None, None, ImageCache(), None, mode, force)
    return session.commit(asyncio.run(DetectionStage().process(stage_input, _Detector(found))))


def test_lettering_found_under_both_labels_is_one_block_of_the_stronger_kind(tmp_path):
    session = Session(tmp_path)
    snapshot, page = _page(session)
    snapshot = _detect_pair(session, snapshot, page, "typeset", 0.46, 0.22)
    assert len(snapshot.text_layers(page)) == 1 and _sound_effect_layers(snapshot, page) == []

    session = Session(tmp_path / "sound-effect")
    snapshot, page = _page(session)
    snapshot = _detect_pair(session, snapshot, page, "typeset", 0.38, 0.74)
    assert len(snapshot.text_layers(page)) == 1 and len(_sound_effect_layers(snapshot, page)) == 1


def test_off_never_gives_up_text_to_an_onomatopoeia_that_makes_no_block(tmp_path):
    session = Session(tmp_path)
    snapshot, page = _page(session)
    snapshot = _detect_pair(session, snapshot, page, "off", 0.38, 0.74)
    assert len(snapshot.text_layers(page)) == 1 and _sound_effect_layers(snapshot, page) == []


def test_a_sound_effect_is_not_written_over_a_block_the_user_edited(tmp_path):
    session = Session(tmp_path)
    snapshot, page = _page(session)
    snapshot = _detect_pair(session, snapshot, page, "typeset", 0.46, 0.22)
    layer = snapshot.text_layers(page)[0]
    edit = snapshot.edit()
    edit.set(snapshot.layer_content(layer), c.Translation(text=c.Authored.user("Mine"), language="en"))
    snapshot = session.commit(edit.finish())

    snapshot = _detect_pair(session, snapshot, page, "typeset", 0.46, 0.8, force=True)

    assert snapshot.text_layers(page) == [layer] and _sound_effect_layers(snapshot, page) == []


# ---- translation ----------------------------------------------------------------------------------


def test_sound_effect_segments_are_tagged_and_explained():
    request = TranslationRequest(segments=["ドン", "行くぞ"], target_language="en-US", sound_effects=frozenset({0}))
    system, user = prompts(request)
    payload = json.loads(user)
    assert payload["segments"] == [{"id": 0, "text": "ドン", "kind": "sound_effect"}, {"id": 1, "text": "行くぞ"}]
    assert "Sound effect requirements" in system


def test_requests_without_sound_effects_are_unchanged():
    system, user = prompts(TranslationRequest(segments=["行くぞ"], target_language="en-US"))
    assert json.loads(user)["segments"] == [{"id": 0, "text": "行くぞ"}]
    assert "Sound effect" not in system


# ---- redraw stage ---------------------------------------------------------------------------------


def _read_and_translate(session: Session, snapshot, page, readings: list[tuple[str, str]]):
    gen = generation("test", "test")
    edit = snapshot.edit_as(gen)
    for layer, (source, translation) in zip(_sound_effect_layers(snapshot, page), readings):
        content = snapshot.layer_content(layer)
        edit.set(content, c.SourceText(text=c.Authored.generated(source, gen), language="ko-KR"))
        edit.set(content, c.Translation(text=c.Authored.generated(translation, gen), language="en-US"))
    return session.commit(edit.finish())


def test_redraw_paints_only_each_sound_effects_box_on_its_own_layer(tmp_path):
    session = Session(tmp_path)
    snapshot, page = _page(session)
    snapshot = _detect(session, snapshot, page, "redraw")
    # The second "sound effect" read as punctuation: junk the detector boxed, so it is not redrawn.
    snapshot = _read_and_translate(session, snapshot, page, [("달칵", "CLICK"), ("!", "!")])
    stage, provider = SoundEffectsStage(), _Fill()
    stage_input = StageInput(snapshot, page, None, None, ImageCache(), None, "redraw")
    assert not stage.skip(stage_input, provider)

    snapshot = session.commit(asyncio.run(stage.process(stage_input, provider)))
    assert len(provider.prompts) == 1
    assert '"달칵"' in provider.prompts[0] and '"CLICK"' in provider.prompts[0] and "English" in provider.prompts[0]
    layer = sfx.redraw_layer(snapshot, page)
    assert snapshot.component(layer, c.RasterLayer).kind is c.RasterLayerKind.EDIT
    pixels = np.asarray(Image.open(io.BytesIO(snapshot.asset_bytes(layer, "source"))).convert("RGBA"))
    assert tuple(pixels[35, 50]) == (255, 0, 0, 255)  # inside the merged box
    assert pixels[35, 100, 3] == 0 and pixels[150, 150, 3] == 0  # dialogue and the junk box untouched

    # Like the Cleanup layer, it is not redrawn while it exists.
    assert stage.skip(StageInput(snapshot, page, None, None, ImageCache(), None, "redraw"), provider)


def test_redraw_skips_outside_redraw_mode_and_refuses_models_that_only_remove(tmp_path):
    session = Session(tmp_path)
    snapshot, page = _page(session)
    snapshot = _detect(session, snapshot, page, "redraw")
    snapshot = _read_and_translate(session, snapshot, page, [("달칵", "CLICK")])
    stage = SoundEffectsStage()
    assert stage.skip(StageInput(snapshot, page, None, None, ImageCache(), None, "typeset"), _Fill())

    class Remover(_Fill):
        prompted = False

    with pytest.raises(ValueError, match="only removes"):
        asyncio.run(stage.process(StageInput(snapshot, page, None, None, ImageCache(), None, "redraw"), Remover()))


def test_the_stage_joins_a_run_only_while_sound_effects_are_redrawn():
    assert Stage.SOUND_EFFECTS in Operation.full().resolve()
    config = PipelineConfig()
    assert config.redraw_provider() == config.inpainting.model
    config.sound_effects = SoundEffectsConfig(mode="redraw", model="diffusion-model")
    assert config.provider_for(Stage.SOUND_EFFECTS) == "diffusion-model"


def test_redraw_leaves_the_art_between_merged_pieces(tmp_path):
    # Two halves 10 px apart: close enough to be one sound effect, with art between them.
    pieces = [_piece("onomatopoeia", 20, 20, 30, 30), _piece("onomatopoeia", 60, 20, 30, 30)]
    session = Session(tmp_path)
    snapshot, page = _page(session)
    stage_input = StageInput(snapshot, page, None, None, ImageCache(), None, "redraw")
    snapshot = session.commit(asyncio.run(DetectionStage().process(stage_input, _Detector(pieces))))
    assert len(_sound_effect_layers(snapshot, page)) == 1
    snapshot = _read_and_translate(session, snapshot, page, [("쾅쾅", "BAM BAM")])

    snapshot = session.commit(asyncio.run(SoundEffectsStage().process(StageInput(snapshot, page, None, None, ImageCache(), None, "redraw"), _Fill())))
    pixels = np.asarray(Image.open(io.BytesIO(snapshot.asset_bytes(sfx.redraw_layer(snapshot, page), "source"))).convert("RGBA"))
    assert pixels[35, 30, 3] == 255 and pixels[35, 75, 3] == 255  # both pieces redrawn
    assert pixels[35, 55, 3] == 0  # the gap between them keeps the art
