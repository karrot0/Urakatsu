"""DBNet ResNet34 provider: strip ownership, transposition, settings, and a weight-gated fixture run."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from urakatsu.providers.detection.ctd.processor import LinePolygon
from urakatsu.providers.detection.dbnet_resnet34 import (
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_DETECT_SIZE,
    DBNetResNet34Provider,
    _transposed,
)
from urakatsu.runtimes.store import ModelStore

FIXTURE = Path(os.environ.get("KOHARU_FIXTURE", r"C:\Users\karrot\AppData\Local\Temp\claude\B--Urakatsu\bca97ddd-a10d-4816-bbf1-9861555d82fb\scratchpad\fixtures\1.jpg"))


def _line(x0: float, y0: float, x1: float, y1: float, score: float = 0.9) -> LinePolygon:
    return LinePolygon(quad=np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64), score=score)


def test_out_of_range_options_fall_back_to_defaults():
    provider = DBNetResNet34Provider({"detect_size": 10, "box_threshold": 3.0, "unclip_ratio": -1.0})
    assert provider.settings.detect_size == DEFAULT_DETECT_SIZE
    assert provider.settings.box_threshold == DEFAULT_BOX_THRESHOLD
    assert provider.settings.unclip_ratio == 2.3


def test_transposed_quad_swaps_axes_and_keeps_winding():
    line = _line(10, 20, 30, 60)
    flipped = _transposed(line)
    xs, ys = flipped.quad[:, 0], flipped.quad[:, 1]
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (20, 60, 10, 30)

    def signed_area(quad):
        x, y = quad[:, 0], quad[:, 1]
        return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    assert np.sign(signed_area(flipped.quad)) == np.sign(signed_area(line.quad))


def test_short_pages_take_one_pass():
    provider = DBNetResNet34Provider()
    calls = []
    provider._detect_once = lambda page: calls.append(page.shape) or (np.zeros(page.shape[:2], np.uint8), [])
    provider._detect_tall(np.zeros((3000, 800, 3), np.uint8))  # aspect 3.75 but under 2.5 x 2048
    assert calls == [(3000, 800, 3)]


def test_long_pages_are_cut_into_owned_strips():
    provider = DBNetResNet34Provider()
    height, width = 6000, 400  # aspect 15, 2.9 x 2048
    seen = []

    def fake(page):
        index = len(seen)
        seen.append(page.shape[0])
        mask = np.full(page.shape[:2], index + 1, np.uint8)
        # One line at the top of every strip and one in its middle.
        return mask, [_line(10, 5, 50, 25), _line(10, page.shape[0] / 2 - 10, 50, page.shape[0] / 2 + 10)]

    provider._detect_once = fake
    mask, lines = provider._detect_tall(np.zeros((height, width, 3), np.uint8))
    assert all(length == 2 * width for length in seen)
    # Every row is owned by exactly one strip, in order.
    owners = mask[:, 0]
    assert owners.min() >= 1 and np.all(np.diff(owners.astype(int)) >= 0)
    assert len(np.unique(owners)) == len(seen)
    # Middle lines are all kept; top lines survive only in the first strip (the others fall in an overlap owned by the previous strip).
    centres = sorted(float(line.quad[:, 1].mean()) for line in lines)
    assert len(lines) == len(seen) + 1
    assert centres[0] == 15.0
    assert all(0 <= c < height for c in centres)


def test_wide_pages_are_detected_transposed(monkeypatch):
    provider = DBNetResNet34Provider()
    provider._network = object()
    shapes = []

    def fake_tall(page):
        shapes.append(page.shape)
        mask = np.zeros(page.shape[:2], np.uint8)
        mask[100:140, 20:60] = 255  # in the transposed frame: rows are page x, columns page y
        return mask, [_line(20, 100, 60, 140)]

    monkeypatch.setattr(provider, "_detect_tall", fake_tall)
    image = Image.new("RGB", (900, 300), "white")
    result = provider._detect(image)
    assert shapes == [(900, 300, 3)]
    assert result.image_width == 900 and result.image_height == 300
    for detection in result.detections:
        x0, y0, x1, y1 = detection.bbox
        assert x1 <= 900 and y1 <= 300


# ---- weights ------------------------------------------------------------------------------------------


def _real_store() -> ModelStore:
    root = os.environ.get("URAKATSU_WEIGHTS_HOME")
    return ModelStore(root=Path(root) if root else Path.home() / ".urakatsu" / "models", offline=True)


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


@pytest.mark.skipif(not _real_store().cached(DBNetResNet34Provider.weight_sources()[0]), reason="DBNet weights are not cached")
def test_checkpoint_finds_the_speech_text_on_the_fixture_page():
    if not _cuda_available():
        pytest.skip("requires CUDA")
    if not FIXTURE.exists():
        pytest.skip("Koharu fixture image is not available")
    from urakatsu.runtimes.manager import RuntimeManager

    provider = DBNetResNet34Provider(None, RuntimeManager(store=_real_store()), runtime="torch")
    asyncio.run(provider.load())
    try:
        image = Image.open(FIXTURE).convert("RGB")
        result = asyncio.run(provider.detect(image))
    finally:
        provider.unload()
    assert result.image_width == image.width and result.image_height == image.height
    texts = [d for d in result.detections if d.label == "text"]
    assert len(texts) >= 5
    for detection in texts:
        x0, y0, x1, y1 = detection.bbox
        assert 0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height
        assert detection.area > 0 and detection.mask.pixels.any()
        assert detection.score > DEFAULT_BOX_THRESHOLD
