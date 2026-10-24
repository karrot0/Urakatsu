"""Comic text detector: unit tests for the DBNet post-processing, and a weight-gated provider test.

The unit tests run on synthetic inputs. The provider test loads the cached checkpoints strictly and
runs them on the fixture page; it skips when the weights are not cached or CUDA is absent.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from urakatsu.providers.detection.ctd.processor import (
    LinePolygon,
    LINE_SCORE_THRESHOLD,
    DBNET_BINARY_THRESHOLD,
    convex_hull,
    extract_line_polygons,
    group_text_lines,
    minimum_area_rect,
    polygon_score,
)
from urakatsu.providers.detection.comic_text_detector import ComicTextDetectorProvider
from urakatsu.runtimes.manager import RuntimeManager
from urakatsu.runtimes.store import ModelStore

DEFAULT_FIXTURE = r"C:\Users\karrot\AppData\Local\Temp\claude\B--Urakatsu\bca97ddd-a10d-4816-bbf1-9861555d82fb\scratchpad\fixtures\1.jpg"


def _quad(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


# ---- (a) pure unit tests ----------------------------------------------------------------------------


def test_minimum_area_rect_axis_aligned_rectangle():
    points = [(10, 20), (50, 20), (50, 30), (10, 30)]
    rect = minimum_area_rect(points)
    assert rect is not None
    assert abs(rect.width * rect.height - 40 * 10) < 1e-6
    assert {round(rect.width), round(rect.height)} == {40, 10}


def test_minimum_area_rect_rotated_rectangle_keeps_minimum_area():
    # A 400 x 100 rectangle rotated by 30 degrees (scaled so integer rounding is negligible): the
    # minimum-area box is the rectangle itself, and its sides are the rectangle's sides.
    angle = np.deg2rad(30.0)
    corners = np.array([[0, 0], [400, 0], [400, 100], [0, 100]], dtype=np.float64)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = np.round(corners @ rotation.T + 1000).astype(int)
    rect = minimum_area_rect(rotated.tolist())
    assert rect is not None
    assert abs(rect.width * rect.height - 400 * 100) / (400 * 100) < 0.01
    assert sorted([rect.width, rect.height]) == pytest.approx([100, 400], abs=2.0)


def test_minimum_area_rect_rejects_collinear_points():
    assert minimum_area_rect([(0, 0), (5, 5), (10, 10)]) is None


def test_convex_hull_drops_interior_and_duplicate_points():
    points = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2), (2, 2), (1, 3), (0, 0)]
    hull = convex_hull(points)
    assert sorted(hull) == [(0.0, 0.0), (0.0, 4.0), (4.0, 0.0), (4.0, 4.0)]
    assert len(hull) == 4


def test_convex_hull_small_inputs_pass_through():
    assert convex_hull([(3, 3)]) == [(3.0, 3.0)]
    assert sorted(convex_hull([(1, 1), (2, 2), (1, 1)])) == [(1.0, 1.0), (2.0, 2.0)]


def test_polygon_score_is_mean_inside_the_polygon():
    prob = np.zeros((20, 30), dtype=np.float32)
    prob[5:10, 5:15] = 0.8  # bright rectangle, pixels x 5..14, y 5..9
    inside = [(5, 5), (14, 5), (14, 9), (5, 9)]
    assert abs(polygon_score(prob, 30, 20, inside) - 0.8) < 1e-6
    outside = [(20, 2), (28, 2), (28, 8), (20, 8)]
    assert polygon_score(prob, 30, 20, outside) == 0.0


def test_polygon_score_averages_mixed_values():
    prob = np.zeros((10, 20), dtype=np.float32)
    prob[:, :10] = 1.0  # left half bright, right half dark
    polygon = [(0, 0), (19, 0), (19, 9), (0, 9)]  # 20 x 10 pixels, half of them bright
    assert abs(polygon_score(prob, 20, 10, polygon) - 0.5) < 1e-6


def _bar_map(height: int = 64, width: int = 64, value: float = 0.9) -> np.ndarray:
    prob = np.full((height, width), 0.2, dtype=np.float32)
    prob[20:30, 10:50] = value  # a text-line-like bar, 40 x 10 pixels
    return prob


def test_extract_line_polygons_finds_one_bar_and_scales_to_destination():
    lines = extract_line_polygons(_bar_map(), 64, 64, 128, 128)
    assert len(lines) == 1
    line = lines[0]
    assert isinstance(line, LinePolygon)
    assert line.quad.shape == (4, 2)
    assert line.score > LINE_SCORE_THRESHOLD
    x0, y0 = line.quad.min(axis=0)
    x1, y1 = line.quad.max(axis=0)
    # The bar covers map x 10..49, y 20..29, i.e. destination x 20..98, y 40..58 after the 2x scale.
    # The quad must contain it and grow by the unclip margin (about 5.5 map px, 11 destination px).
    assert x0 <= 20 and x1 >= 98 and y0 <= 40 and y1 >= 58
    assert x0 >= 20 - 2 * 8 and x1 <= 98 + 2 * 8
    assert y0 >= 40 - 2 * 8 and y1 <= 58 + 2 * 8
    assert (line.quad >= 0).all() and (line.quad[:, 0] <= 128).all() and (line.quad[:, 1] <= 128).all()


def test_extract_line_polygons_thresholds_are_parameters():
    prob = _bar_map(value=0.9)
    assert len(extract_line_polygons(prob, 64, 64, 64, 64)) == 1
    # Above the bar's mean probability the candidate is rejected by the score threshold.
    assert extract_line_polygons(prob, 64, 64, 64, 64, score_threshold=0.95) == []
    # Above the bar's value the binary map is empty, so nothing is extracted.
    assert extract_line_polygons(prob, 64, 64, 64, 64, binary_threshold=0.95) == []
    assert DBNET_BINARY_THRESHOLD == pytest.approx(0.3)


def test_extract_line_polygons_drops_thin_candidates():
    prob = np.full((40, 40), 0.2, dtype=np.float32)
    prob[10:11, 5:35] = 0.9  # one pixel tall: min side below 2 px
    assert extract_line_polygons(prob, 40, 40, 40, 40) == []


def _full_mask(height: int = 80, width: int = 160) -> np.ndarray:
    return np.full((height, width), 255, dtype=np.uint8)


def test_group_text_lines_merges_stacked_lines_and_keeps_distant_ones():
    lines = [
        LinePolygon(_quad(10, 10, 50, 20), 0.9),
        LinePolygon(_quad(10, 22, 50, 32), 0.8),  # one line below, same size and x range
        LinePolygon(_quad(10, 60, 50, 70), 0.7),  # far below: its own block
    ]
    blocks = group_text_lines(lines, _full_mask())
    assert len(blocks) == 2
    merged = next(b for b in blocks if b.xyxy[1] == 10)
    assert merged.xyxy == (10, 10, 50, 32)
    assert len(merged.lines) == 2
    assert merged.line_scores == pytest.approx([0.9, 0.8])
    assert merged.vertical is False
    assert next(b for b in blocks if b.xyxy[1] == 60).xyxy == (10, 60, 50, 70)


def test_group_text_lines_drops_lines_without_mask_under_them():
    mask = np.zeros((80, 160), dtype=np.uint8)
    mask[10:20, 10:50] = 255  # only the first line has mask under it
    lines = [LinePolygon(_quad(10, 10, 50, 20), 0.9), LinePolygon(_quad(100, 60, 140, 70), 0.9)]
    blocks = group_text_lines(lines, mask)
    assert [b.xyxy for b in blocks] == [(10, 10, 50, 20)]


def test_group_text_lines_marks_tall_quads_vertical():
    lines = [LinePolygon(_quad(10, 10, 20, 60), 0.9)]
    blocks = group_text_lines(lines, _full_mask())
    assert len(blocks) == 1
    assert blocks[0].vertical is True
    # Vertical blocks measure the angle from the vertical axis: a straight-up line is 0 degrees.
    assert blocks[0].angle == 0


def test_provider_metadata_needs_no_weights_and_no_catalog():
    assert ComicTextDetectorProvider.id == "comic-text-detector"
    assert ComicTextDetectorProvider.runtimes == ("torch",)
    assert ComicTextDetectorProvider.uses_catalog() is False
    sources = ComicTextDetectorProvider.weight_sources()
    assert [s.filename for s in sources] == ["yolo-v5.safetensors", "unet.safetensors", "dbnet.safetensors"]


# ---- (b) weight-gated provider test -----------------------------------------------------------------


def _real_store() -> ModelStore:
    """The user's cached weights (``URAKATSU_HOME`` is redirected to a temp dir by conftest)."""
    root = os.environ.get("URAKATSU_WEIGHTS_HOME")
    return ModelStore(root=Path(root) if root else Path.home() / ".urakatsu" / "models", offline=True)


def _cached() -> bool:
    store = _real_store()
    return all(store.cached(source) for source in ComicTextDetectorProvider.weight_sources())


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


FIXTURE = Path(os.environ.get("KOHARU_FIXTURE", DEFAULT_FIXTURE))


@pytest.mark.skipif(not _cached(), reason="comic-text-detector weights are not cached")
@pytest.mark.skipif(not _cuda_available(), reason="CUDA is not available")
@pytest.mark.skipif(not FIXTURE.exists(), reason=f"fixture page missing: {FIXTURE}")
def test_ctd_loads_strictly_and_detects_text_on_fixture():
    runtimes = RuntimeManager("auto", store=_real_store())
    provider = ComicTextDetectorProvider(None, runtimes)
    asyncio.run(provider.load())  # strict load of all three checkpoints
    try:
        image = Image.open(FIXTURE).convert("RGB")
        result = provider._detect(image)
    finally:
        provider.unload()

    assert result.image_width == image.width and result.image_height == image.height
    detections = result.detections
    assert len(detections) >= 8
    for detection in detections:
        x0, y0, x1, y1 = detection.bbox
        assert 0 <= x0 < x1 <= image.width
        assert 0 <= y0 < y1 <= image.height
        assert detection.label == "text" and detection.label_id == 0
        assert 0.0 < detection.score <= 1.0
        assert detection.mask.pixels.shape == (detection.mask.height, detection.mask.width)
        assert detection.mask.pixels.any()
        assert detection.area == int(np.count_nonzero(detection.mask.pixels)) > 0
    # The speech bubble with "エビ漁船バイト!" sits at about x 361..380, y 281..420 on the fixture.
    assert any(d.bbox[0] <= 370 <= d.bbox[2] and d.bbox[1] <= 350 <= d.bbox[3] for d in detections)
