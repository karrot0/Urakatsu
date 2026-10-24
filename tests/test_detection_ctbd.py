"""Comic Text & Bubble Detector: ports of Koharu's processor.rs on synthetic boxes, plus weight-gated checks.

The weight-gated tests need the pinned checkpoint in the store (``URAKATSU_WEIGHTS_HOME``, else
``~/.urakatsu/models``) and CUDA. ``KOHARU_FIXTURE`` overrides the fixture page.
"""

import asyncio
import os
from pathlib import Path

import pytest
from PIL import Image

from urakatsu.providers.detection.comic_text_bubble_detector import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    ComicTextBubbleDetectorProvider,
    ComicTextBubbleDetectorSettings,
)
from urakatsu.providers.detection.ctbd.processor import (
    ImageSlicer,
    RTDetrImageProcessor,
    ScoredBox,
    SizeDict,
    calculate_iou,
    create_text_blocks,
    does_rectangle_fit,
    filter_and_fix_bboxes,
    merge_overlapping_boxes,
)
from urakatsu.runtimes.manager import RuntimeManager
from urakatsu.runtimes.store import ModelStore

FIXTURE = Path(os.environ.get("KOHARU_FIXTURE", r"C:\Users\karrot\AppData\Local\Temp\claude\B--Urakatsu\bca97ddd-a10d-4816-bbf1-9861555d82fb\scratchpad\fixtures\1.jpg"))


def box(x0, y0, x1, y1, score=0.5):
    return ScoredBox((float(x0), float(y0), float(x1), float(y1)), float(score))


def blank(width, height):
    return Image.new("RGB", (width, height), (255, 255, 255))


# ---- settings and provider surface (no weights) ----------------------------------------------------


def test_default_threshold_is_koharu_default():
    assert DEFAULT_CONFIDENCE_THRESHOLD == 0.3
    assert ComicTextBubbleDetectorSettings().confidence_threshold == 0.3


@pytest.mark.parametrize("value, expected", [(0.5, 0.5), (0.0, 0.0), (1.5, 0.3), (-0.1, 0.3), (float("nan"), 0.3)])
def test_out_of_range_threshold_falls_back_to_the_default(value, expected):
    provider = ComicTextBubbleDetectorProvider({"confidence_threshold": value})
    assert provider.settings.confidence_threshold == expected


def test_provider_declares_pinned_weights_and_no_catalog():
    sources = ComicTextBubbleDetectorProvider.weight_sources("torch")
    assert [source.filename for source in sources] == ["config.json", "preprocessor_config.json", "model.safetensors"]
    assert {source.revision for source in sources} == {"16e8a622f91fabc6b5b65c96d32d1183f8843546"}
    assert {source.repository for source in sources} == {"ogkalu/comic-text-and-bubble-detector"}
    assert not ComicTextBubbleDetectorProvider.uses_catalog()


# ---- merge_overlapping_boxes (slicer, Koharu ImageSlicer::merge_overlapping_boxes) -------------------


def test_contained_box_is_absorbed_and_the_container_keeps_the_max_score():
    slicer = ImageSlicer()
    merged = slicer.merge_overlapping_boxes([box(0, 0, 100, 100, 0.5), box(10, 10, 50, 50, 0.9)], image_height=1000)
    assert merged == [box(0, 0, 100, 100, 0.9)]


def test_containment_keeps_the_larger_box_geometry_when_the_smaller_comes_first():
    slicer = ImageSlicer()
    merged = slicer.merge_overlapping_boxes([box(10, 10, 50, 50, 0.4), box(0, 0, 100, 100, 0.7)], image_height=1000)
    assert merged == [box(0, 0, 100, 100, 0.7)]


def test_duplicate_boxes_collapse_with_the_max_score():
    slicer = ImageSlicer()
    # Containment 8000 / 10000 = 0.8 is below 0.85, but IoU 8000 / 12000 = 0.67 is a duplicate.
    merged = slicer.merge_overlapping_boxes([box(0, 0, 100, 100, 0.3), box(20, 0, 120, 100, 0.8)], image_height=1000)
    assert merged == [box(0, 0, 100, 100, 0.8)]


def test_vertically_adjacent_fragments_of_one_bubble_merge():
    slicer = ImageSlicer()
    fragments = [box(100, 100, 200, 150, 0.8), box(102, 153, 199, 200, 0.6)]
    merged = slicer.merge_overlapping_boxes(fragments, image_height=1000)
    assert merged == [box(100, 100, 200, 200, 0.8)]


def test_distant_boxes_are_not_merged():
    slicer = ImageSlicer()
    boxes = [box(100, 100, 200, 150, 0.8), box(100, 400, 200, 450, 0.6)]
    assert slicer.merge_overlapping_boxes(boxes, image_height=1000) == boxes


# ---- free merge_overlapping_boxes (text boxes, Koharu content.rs) ----------------------------------------


def test_text_merge_grows_boxes_that_mostly_contain_each_other():
    # (0..100) is 2000 px and covers 60% of (70..120), which is 1000 px: the outer box is at least as large
    # and the inner is 30%+ inside it, so they merge into one box keeping the max score.
    merged = merge_overlapping_boxes([box(0, 0, 100, 20, 0.6), box(70, 0, 120, 20, 0.9)])
    assert merged == [box(0, 0, 120, 20, 0.9)]


def test_text_merge_keeps_disjoint_boxes_in_order():
    merged = merge_overlapping_boxes([box(0, 0, 20, 20, 0.4), box(100, 100, 130, 120, 0.7)])
    assert merged == [box(0, 0, 20, 20, 0.4), box(100, 100, 130, 120, 0.7)]


# ---- filter_and_fix_bboxes ------------------------------------------------------------------------------


def test_filter_and_fix_clamps_to_the_image_and_drops_boxes_not_larger_than_five_pixels():
    boxes = [
        box(-10, -10, 50, 50, 0.5),  # clamped to (0, 0, 50, 50)
        box(0, 0, 5, 100, 0.6),  # width 5 is not greater than 5: dropped
        box(900, 0, 1200, 10, 0.7),  # clamped to (900, 0, 1000, 10): kept
        box(10, 10, 16, 16, 0.8),  # 6 by 6: kept
    ]
    fixed = filter_and_fix_bboxes(boxes, image_width=1000, image_height=500)
    assert [item.xyxy for item in fixed] == [(0.0, 0.0, 50.0, 50.0), (900.0, 0.0, 1000.0, 10.0), (10.0, 10.0, 16.0, 16.0)]
    assert [item.score for item in fixed] == [0.5, 0.7, 0.8]


# ---- ImageSlicer slice math ------------------------------------------------------------------------------


def test_should_slice_only_above_height_to_width_3_5():
    slicer = ImageSlicer()
    assert slicer.should_slice(blank(800, 4000))
    assert not slicer.should_slice(blank(800, 2000))


def test_slice_params_match_koharu_for_a_tall_page():
    slicer = ImageSlicer()
    slice_width, slice_height, effective, num_slices = slicer.calculate_slice_params(blank(800, 4000))
    assert (slice_width, slice_height, effective) == (800, 2400, 1920)
    # ceil(4000 / 1920) = 3; the 160 px last slice is under 70% of a slice, so the count drops to 2,
    # but processing loops over div_ceil slices (Koharu's flow ignores this value).
    assert num_slices == 2


def test_slices_cover_the_page_with_koharu_overlap():
    slicer = ImageSlicer()
    image = blank(800, 4000)
    _, slice_height, effective, _ = slicer.calculate_slice_params(image)
    count = -(-image.height // effective)
    spans = []
    for number in range(count):
        crop, start_y, end_y = slicer.get_slice(image, number, effective, slice_height)
        assert crop.size == (800, end_y - start_y)
        spans.append((start_y, end_y))
    assert spans == [(0, 2400), (1920, 4000), (3840, 4000)]


def test_adjust_box_coordinates_shifts_only_y():
    slicer = ImageSlicer()
    shifted = slicer.adjust_box_coordinates([box(5, 6, 7, 8, 0.2)], start_y=100)
    assert shifted == [box(5, 106, 7, 108, 0.2)]


def test_tall_page_goes_through_the_slicer_and_merges_across_slice_seams():
    """A page ratio above 3.5 is sliced; a bubble and a text box repeated in two slices collapse to one each."""
    image = blank(800, 4000)
    starts = [0, 1920, 3840]
    page_bubbles = [(100, 2300, 300, 2400, 0.9)]
    page_texts = [(100, 3900, 300, 3950, 0.7)]
    calls = []

    def detect(slice_image):
        index = len(calls)
        calls.append(slice_image.size)
        start = starts[index]
        top, bottom = start, start + slice_image.height
        bubbles = [
            ScoredBox((x0, y0 - start, x1, y1 - start), score)
            for x0, y0, x1, y1, score in page_bubbles
            if top <= y0 and y1 <= bottom
        ]
        # A second, lower-scoring copy of the same bubble inside the overlap, as a real slice would give.
        bubbles += [
            ScoredBox((x0, y0 - start, x1, y1 - start), score - 0.1)
            for x0, y0, x1, y1, score in page_bubbles
            if top <= y0 and y1 <= bottom and start > 0
        ]
        texts = [
            ScoredBox((x0, y0 - start, x1, y1 - start), score)
            for x0, y0, x1, y1, score in page_texts
            if top <= y0 and y1 <= bottom
        ]
        return bubbles, texts

    bubbles, texts = ImageSlicer().process_slices_for_detection(image, detect)
    assert len(calls) == 3
    assert [item.xyxy for item in bubbles] == [(100.0, 2300.0, 300.0, 2400.0)]
    assert bubbles[0].score == pytest.approx(0.9)
    assert [item.xyxy for item in texts] == [(100.0, 3900.0, 300.0, 3950.0)]
    assert texts[0].score == pytest.approx(0.7)


def test_short_page_is_detected_in_one_pass():
    image = blank(800, 1200)
    calls = []

    def detect(slice_image):
        calls.append(slice_image.size)
        return [box(1, 1, 50, 50, 0.9)], []

    bubbles, texts = ImageSlicer().process_slices_for_detection(image, detect)
    assert calls == [(800, 1200)]
    assert bubbles == [box(1, 1, 50, 50, 0.9)]
    assert texts == []


# ---- create_text_blocks (bubble assignment) ---------------------------------------------------------------


def test_text_inside_a_bubble_is_text_bubble_and_elsewhere_text_free():
    bubbles = [box(100, 100, 300, 300, 0.9)]
    texts = [box(150, 150, 250, 200, 0.8), box(500, 500, 600, 540, 0.7)]
    blocks = create_text_blocks(1000, 1000, texts, bubbles)
    by_xyxy = {block.xyxy: block for block in blocks}
    inside = by_xyxy[(150, 150, 250, 200)]
    outside = by_xyxy[(500, 500, 600, 540)]
    assert inside.text_class == "text_bubble"
    assert inside.bubble_xyxy == (100, 100, 300, 300)
    assert inside.score == pytest.approx(0.8)
    assert outside.text_class == "text_free"
    assert outside.bubble_xyxy is None


def test_text_with_iou_of_at_least_point_two_counts_as_in_the_bubble():
    bubbles = [box(100, 100, 300, 300, 0.9)]
    # Not fitted inside (it sticks out), but IoU = 22500 / 57500 = 0.39.
    texts = [box(150, 150, 350, 350, 0.8)]
    blocks = create_text_blocks(1000, 1000, texts, bubbles)
    assert blocks[0].text_class == "text_bubble"


def test_text_just_under_the_iou_threshold_stays_free():
    bubbles = [box(100, 100, 300, 300, 0.9)]
    texts = [box(200, 200, 400, 400, 0.8)]  # IoU = 10000 / 70000 = 0.14, not contained
    blocks = create_text_blocks(1000, 1000, texts, bubbles)
    assert blocks[0].text_class == "text_free"


def test_fit_helper_requires_full_containment():
    assert does_rectangle_fit((0, 0, 10, 10), (2, 2, 8, 8))
    assert not does_rectangle_fit((0, 0, 10, 10), (2, 2, 12, 8))
    assert calculate_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_tiny_text_and_bubble_boxes_are_dropped_before_pairing():
    blocks = create_text_blocks(1000, 1000, [box(150, 150, 153, 153, 0.9)], [box(100, 100, 300, 300, 0.9)])
    assert blocks == []


# ---- RTDetrImageProcessor (preprocess and post_process_object_detection) ------------------------------------


def test_preprocess_resizes_and_rescales_to_nchw_float():
    torch = pytest.importorskip("torch")
    processor = RTDetrImageProcessor(size=SizeDict(height=8, width=16), do_normalize=False)
    image = Image.new("RGB", (4, 2), (255, 0, 51))
    tensor = processor.preprocess(image, torch.device("cpu"))
    assert tuple(tensor.shape) == (1, 3, 8, 16)
    assert tensor.dtype == torch.float32
    assert float(tensor[0, 0].max()) == pytest.approx(1.0)
    assert float(tensor[0, 2].max()) == pytest.approx(51 / 255, rel=1e-6)


def test_post_process_keeps_scores_above_threshold_and_maps_to_target_size():
    torch = pytest.importorskip("torch")
    processor = RTDetrImageProcessor()
    queries, classes = 300, 3
    logits = torch.full((1, queries, classes), -10.0)
    logits[0, 7, 0] = 5.0  # bubble-class logit for query 7
    logits[0, 9, 1] = 0.0  # sigmoid 0.5: above 0.3
    pred_boxes = torch.full((1, queries, 4), 0.5)
    pred_boxes[0, 7] = torch.tensor([0.5, 0.5, 0.2, 0.2])
    pred_boxes[0, 9] = torch.tensor([0.25, 0.5, 0.1, 0.2])
    results = processor.post_process_object_detection(logits, pred_boxes, 0.3, [(500, 1000)], use_focal_loss=True)
    out = results[0]
    assert out.labels == [0, 1]
    assert out.boxes[0] == pytest.approx((400.0, 200.0, 600.0, 300.0))
    assert out.boxes[1] == pytest.approx((200.0, 200.0, 300.0, 300.0))
    assert out.scores[0] == pytest.approx(float(torch.sigmoid(torch.tensor(5.0))), rel=1e-6)


def test_post_process_treats_a_score_equal_to_the_threshold_as_rejected():
    torch = pytest.importorskip("torch")
    processor = RTDetrImageProcessor()
    logits = torch.full((1, 300, 3), -10.0)
    logits[0, 0, 0] = 0.0  # sigmoid(0) == 0.5 exactly in float32
    pred_boxes = torch.full((1, 300, 4), 0.5)
    out = processor.post_process_object_detection(logits, pred_boxes, 0.5, [(100, 100)], use_focal_loss=True)[0]
    assert out.labels == []


# ---- weight-gated checks ---------------------------------------------------------------------------------------


def _real_store() -> ModelStore:
    root = os.environ.get("URAKATSU_WEIGHTS_HOME")
    return ModelStore(root=Path(root) if root else Path.home() / ".urakatsu" / "models", offline=True)


def _cached() -> bool:
    store = _real_store()
    return all(store.cached(source) for source in ComicTextBubbleDetectorProvider.weight_sources("torch"))


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _loaded_provider():
    runtimes = RuntimeManager("cuda:0", store=_real_store())
    provider = ComicTextBubbleDetectorProvider(None, runtimes)
    asyncio.run(provider.load())
    return provider


@pytest.mark.skipif(not _cached(), reason="comic text/bubble detector weights are not cached")
@pytest.mark.skipif(not _cuda_available(), reason="requires CUDA")
@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture page is not available")
def test_detects_bubbles_and_text_on_the_fixture_page():
    provider = _loaded_provider()
    try:
        result = provider._detect(Image.open(FIXTURE).convert("RGB"))
    finally:
        provider.unload()

    bubbles = [d for d in result.detections if d.label_id == 0]
    texts = [d for d in result.detections if d.label == "text"]
    assert bubbles and texts
    assert all(d.label == "bubble" for d in bubbles)
    assert {d.label_id for d in texts} <= {1, 2}

    # Text inside a bubble is text_bubble; the containment of each in its best-fitting bubble is high.
    text_bubbles = [d for d in texts if d.label_id == 1]
    fitted = 0
    for text in text_bubbles:
        tx0, ty0, tx1, ty1 = text.bbox
        area = (tx1 - tx0) * (ty1 - ty0)
        best = 0.0
        for bubble in bubbles:
            bx0, by0, bx1, by1 = bubble.bbox
            inter = max(min(tx1, bx1) - max(tx0, bx0), 0) * max(min(ty1, by1) - max(ty0, by0), 0)
            best = max(best, inter / area)
        fitted += best >= 0.85
    assert text_bubbles and fitted >= 0.8 * len(text_bubbles)

    for detection in result.detections:
        x0, y0, x1, y1 = detection.bbox
        assert 0 <= x0 < x1 <= result.image_width and 0 <= y0 < y1 <= result.image_height
        assert detection.area == int((x1 - x0) * (y1 - y0))
        assert detection.mask.pixels.shape == (int(y1 - y0), int(x1 - x0))
        assert int(detection.mask.pixels.min()) == 255


@pytest.mark.skipif(not _cached(), reason="comic text/bubble detector weights are not cached")
@pytest.mark.skipif(not _cuda_available(), reason="requires CUDA")
def test_tall_page_runs_through_the_slicer_on_the_real_model():
    if not FIXTURE.exists():
        pytest.skip("fixture page is not available")
    page = Image.open(FIXTURE).convert("RGB")
    tall = Image.new("RGB", (page.width, page.height * 4), (255, 255, 255))
    for index in range(4):
        tall.paste(page, (0, index * page.height))
    assert tall.height / tall.width > 3.5

    provider = _loaded_provider()
    try:
        result = provider._detect(tall)
    finally:
        provider.unload()

    assert result.image_height == tall.height
    assert any(d.label_id == 0 for d in result.detections)
    bands = {int(((d.bbox[1] + d.bbox[3]) / 2) // page.height) for d in result.detections}
    assert bands >= {0, 1, 2, 3}
