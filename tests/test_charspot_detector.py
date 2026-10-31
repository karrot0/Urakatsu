"""Charspot detector: tiling, character grouping and text blocks, without weights."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from urakatsu.catalog.facts import ModelFacts
from urakatsu.core.errors import ProcessingError
from urakatsu.providers.detection.charspot_detector import (
    CharspotDetectorProvider,
    group_characters,
    text_block,
    tile_spans,
)


@pytest.mark.parametrize("length, tile", [(100, 256), (256, 256), (257, 256), (770, 256), (1080, 256), (5000, 333)])
def test_tiles_own_every_pixel_once(length, tile):
    spans = tile_spans(length, tile)
    assert spans[0][1] == 0 and spans[-1][2] == length
    for (_, _, own_end), (_, next_start, _) in zip(spans, spans[1:]):
        assert own_end == next_start
    for start, own_start, own_end in spans:
        assert 0 <= start <= own_start < own_end <= min(start + tile, length)


def _column(x: float, y: float, count: int, size: float = 20.0, step: float = 24.0) -> np.ndarray:
    return np.array([[x, y + i * step, x + size, y + i * step + size] for i in range(count)], np.float32)


def test_near_characters_form_one_block_and_far_ones_another():
    boxes = np.concatenate([_column(0, 0, 3), _column(26, 0, 3), _column(300, 0, 2)])
    groups = group_characters(boxes, np.ones(len(boxes), bool))
    assert sorted(sorted(g) for g in groups) == [[0, 1, 2, 3, 4, 5], [6, 7]]


def test_a_weak_character_joins_a_block_but_does_not_extend_it():
    boxes = _column(0, 0, 4)
    strong = np.array([True, True, False, False])
    assert sorted(group_characters(boxes, strong)[0]) == [0, 1, 2]


def test_a_much_larger_character_does_not_join():
    boxes = np.array([[0, 0, 20, 20], [22, 0, 102, 80]], np.float32)
    groups = group_characters(boxes, np.ones(2, bool))
    assert sorted(sorted(g) for g in groups) == [[0], [1]]


def test_a_block_masks_its_characters_and_stays_on_the_page():
    boxes = np.array([[-2.5, 3.0, 10.0, 12.5], [90.0, 80.0, 130.0, 95.0]], np.float32)
    block = text_block(boxes, 0.7, 100, 90)
    assert block.label == "text"
    assert block.bbox == (0.0, 3.0, 100.0, 90.0)
    assert block.mask.contains(5, 5)
    assert block.mask.contains(95, 85)
    assert not block.mask.contains(50, 50)
    assert block.area == int(np.count_nonzero(block.mask.pixels))


class _Session:
    """Finds two characters at fixed spots in the first tile only."""

    def __init__(self, scores):
        self.scores = np.array(scores, np.float32)
        self.calls = 0

    def run(self, outputs, inputs):
        assert outputs == ["boxes", "scores"]
        assert inputs["img"].dtype == np.float32
        self.calls += 1
        if self.calls > 1:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        return np.array([[10, 10, 30, 30], [10, 34, 30, 54]], np.float32), self.scores


def _provider(session) -> CharspotDetectorProvider:
    provider = CharspotDetectorProvider()
    provider._session = session
    provider._cell, provider._budget = 16, 256
    return provider


def test_detect_builds_a_block_in_page_pixels():
    session = _Session([0.9, 0.8])
    result = _provider(session)._detect(Image.new("RGB", (770, 1080), "white"))
    assert session.calls == 4 * 6
    assert (result.image_width, result.image_height) == (770, 1080)
    assert [d.bbox for d in result.detections] == [(10.0, 10.0, 30.0, 54.0)]
    assert result.detections[0].score == pytest.approx(0.85)


def test_detect_drops_a_block_without_enough_confident_characters():
    result = _provider(_Session([0.9, 0.4]))._detect(Image.new("RGB", (770, 1080), "white"))
    assert result.detections == []


def test_detect_needs_a_loaded_model():
    with pytest.raises(ProcessingError):
        CharspotDetectorProvider()._detect(Image.new("RGB", (64, 64), "white"))


def test_it_accepts_the_charspot_models():
    files = ["config.json", "vocab.json", "model.onnx"]
    assert CharspotDetectorProvider.accepts(ModelFacts(runtime="onnx", repo="muscgab/manga-ocr-charspot-nano", files=files)) == "onnx"
    assert CharspotDetectorProvider.accepts(ModelFacts(runtime="onnx", repo="someone/other", files=files)) is None
