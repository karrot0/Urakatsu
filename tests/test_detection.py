"""Detection stage and provider: ports of the Rust unit tests.

Tests that need the checkpoint skip when it is not in the local Hugging Face cache or CUDA is absent.
"""

from __future__ import annotations

import asyncio
import io
import math
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from urakatsu.core.results import LayoutDetection, LayoutDetections, LayoutMask
from urakatsu.pipeline.images import ImageCache
from urakatsu.pipeline.scope import Bounds
from urakatsu.pipeline.stages import detection as det
from urakatsu.pipeline.stages.base import StageInput
from urakatsu.pipeline.stages.detection import DetectionStage
from urakatsu.providers.detection.koharu_layout_rfdetr_seg_2xl import (
    KoharuLayoutRFDetrSeg2XLProvider,
    REPOSITORY,
    REVISION,
)
from urakatsu.scene import At, AssetInput, AssetMetadata, Geometry, Page, Session
from urakatsu.scene import components as c

PRODUCER_GENERATION = det.generation(det.PRODUCER, det.MODEL_ID)


def _mask(x: int, y: int, pixels: np.ndarray) -> LayoutMask:
    height, width = pixels.shape
    return LayoutMask(x=x, y=y, width=width, height=height, pixels=pixels.astype(np.uint8))


def _detection(label: str, score: float, bbox) -> LayoutDetection:
    return LayoutDetection(
        label_id=0,
        label=label,
        score=score,
        bbox=tuple(float(v) for v in bbox),
        area=0,
        mask=_mask(0, 0, np.zeros((1, 1), dtype=np.uint8)),
    )


def _page(session: Session, width: float = 100.0, height: float = 100.0):
    edit = session.snapshot().edit()
    page = edit.add_page(Page(label="page", width=width, height=height), At.END)
    snapshot = session.commit(edit.finish())
    return snapshot, page


# ---- provider configuration -------------------------------------------------------------------------


def test_out_of_range_thresholds_fall_back_to_the_model_defaults():
    provider = KoharuLayoutRFDetrSeg2XLProvider(
        {"text_threshold": 15.0, "bubble_threshold": float("nan"), "panel_threshold": 0.55}
    )
    assert provider.settings.text_threshold is None
    assert provider.settings.bubble_threshold is None
    assert provider.settings.panel_threshold == 0.55


# ---- skip, relations, and scene writes ---------------------------------------------------------------


def test_detection_skips_a_page_with_existing_text(tmp_path):
    session = Session(tmp_path)
    snapshot, page = _page(session)
    edit = snapshot.edit()
    edit.add_analysis_region(page, At.END, c.RegionKinds.TEXT, Geometry.rectangle(10, 10, 20, 20))
    snapshot = session.commit(edit.finish())
    stage_input = StageInput(snapshot, page, None, None, ImageCache(), None)

    assert DetectionStage().skip(stage_input, None)


def test_detection_does_not_skip_a_page_without_text_regions(tmp_path):
    session = Session(tmp_path)
    snapshot, page = _page(session)
    edit = snapshot.edit()
    edit.add_analysis_region(page, At.END, c.RegionKinds.BUBBLE, Geometry.rectangle(10, 10, 20, 20))
    snapshot = session.commit(edit.finish())

    assert not DetectionStage().skip(StageInput(snapshot, page, None, None, ImageCache(), None), None)


def test_joined_text_uses_balloon_flow_semantics(tmp_path):
    session = Session(tmp_path)
    snapshot, page = _page(session)
    bubble_mask = _mask(0, 0, np.full((1, 1), 255, dtype=np.uint8))
    text_mask = _mask(0, 0, np.full((1, 1), 255, dtype=np.uint8))
    edit = snapshot.edit_as(PRODUCER_GENERATION)
    bubble = edit.add_analysis_region(page, At.END, c.RegionKinds.BUBBLE, Geometry.rectangle(10, 10, 80, 80))
    region = edit.add_analysis_region(page, At.END, c.RegionKinds.TEXT, Geometry.rectangle(30, 30, 20, 20))
    content = edit.add_text_content(page, At.END)
    layer = edit.add_text_layer(
        page, At.END, content, c.TextLayout(kind=c.TextLayoutKind.PARAGRAPH, angle_degrees=None)
    )
    det.link_dialogue_regions(
        edit,
        det._PageRegions(
            bubbles=[det._Bubble(entity=bubble, mask=bubble_mask, area=1)],
            texts=[det._Text(entity=region, mask=text_mask, bounds=(0.0, 0.0, 1.0, 1.0), content=content, layer=layer)],
        ),
        PRODUCER_GENERATION,
    )
    snapshot = session.commit(edit.finish())

    assert snapshot.relation_from(layer, c.Relations.FLOWS_IN).target == bubble
    assert snapshot.relation_from(layer, c.Relations.FITS_TO) is None
    assert snapshot.relation_from(region, c.Relations.INSIDE).target == bubble


def test_detected_typography_preserves_the_measured_outline(tmp_path):
    session = Session(tmp_path)
    snapshot, page = _page(session, 96.0, 96.0)
    image, detection = _outlined_text_image(3, (0, 0, 0), (255, 255, 255), (16, 24, 88))
    inferred = det.infer_typography(image, detection)
    edit = snapshot.edit_as(PRODUCER_GENERATION)
    _, text = det.write_region(edit, page, detection, inferred, PRODUCER_GENERATION)
    snapshot = session.commit(edit.finish())
    typography = snapshot.component(text.layer, c.Typography)

    assert typography.color == (0, 0, 0, 255)
    assert typography.stroke_color == (255, 255, 255, 255)
    assert typography.stroke_width == 3.0


# ---- non-maximum suppression, layout order, masks ----------------------------------------------------


def test_nms_removes_lower_scored_overlapping_regions_per_class():
    detections = [
        _detection("text", 0.8, [5.0, 5.0, 105.0, 105.0]),
        _detection("bubble", 0.7, [0.0, 0.0, 100.0, 100.0]),
        _detection("text", 0.9, [0.0, 0.0, 100.0, 100.0]),
        _detection("text", 0.5, [20.0, 20.0, 80.0, 80.0]),
        _detection("text", 0.6, [200.0, 0.0, 250.0, 50.0]),
    ]

    det.non_maximum_suppression(detections, np.float32(0.5))

    text_scores = [d.score for d in detections if d.label == "text"]
    assert text_scores == [0.9, 0.6, 0.5]
    assert any(d.label == "bubble" for d in detections)


def test_text_mask_excludes_onomatopoeia():
    def detection(label: str, x: int, value: int) -> LayoutDetection:
        return LayoutDetection(
            label_id=0,
            label=label,
            score=1.0,
            bbox=(float(x), 0.0, float(x) + 1.0, 1.0),
            area=int(value != 0),
            mask=_mask(x, 0, np.array([[value]], dtype=np.uint8)),
        )

    detections = [
        detection("bubble", 0, 0),
        detection("onomatopoeia", 0, 255),
        detection("text", 1, 255),
        detection("onomatopoeia", 3, 255),
    ]

    mask = det.mask_for(detections, "text", (4, 1))

    assert mask.tolist() == [[0, 255, 0, 0]]


def test_bubble_membership_uses_detected_ink_instead_of_text_box_corners():
    bubble = _mask(
        0,
        0,
        np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
            ]
        ),
    )
    dialogue = _mask(
        0,
        0,
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ),
    )
    crossing = _mask(
        0,
        0,
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1],
            ]
        ),
    )
    bounds = (0.0, 0.0, 5.0, 5.0)

    assert det.mask_containment(bubble, dialogue, bounds) == 1.0
    assert det.mask_containment(bubble, crossing, bounds) < det.DIALOGUE_MASK_CONTAINMENT_THRESHOLD


def test_mask_containment_accepts_independent_local_extents():
    valid = _mask(5, 5, np.ones((2, 2), dtype=np.uint8))
    malformed = LayoutMask(x=5, y=5, width=2, height=2, pixels=np.ones(3, dtype=np.uint8))
    different_size = _mask(5, 5, np.ones((1, 1), dtype=np.uint8))
    bounds = (5.0, 5.0, 7.0, 7.0)

    assert det.mask_containment(valid, malformed, bounds) == 0.0
    assert det.mask_containment(valid, different_size, bounds) == 1.0


def _local_text(x: int, y: int, pixels: np.ndarray) -> LayoutDetection:
    height, width = pixels.shape
    return LayoutDetection(
        label_id=0,
        label="text",
        score=1.0,
        bbox=(float(x), float(y), float(x + width), float(y + height)),
        area=int(np.count_nonzero(pixels)),
        mask=_mask(x, y, pixels),
    )


def test_local_mask_morphology_matches_the_full_page_operation():
    size = (64, 48)
    detections = [
        _local_text(0, 0, np.full((4, 4), 255)),
        _local_text(10, 8, np.full((4, 4), 255)),
        _local_text(19, 10, np.full((5, 3), 255)),
        _local_text(52, 40, np.full((3, 4), 255)),
    ]
    radius = 3
    page = det.mask_for(detections, "text", size)
    expected = det.close_l2(det.dilate_l2(page, radius), radius)

    assert np.array_equal(det.closed_mask_for(detections, "text", size, radius), expected)


def test_local_mask_morphology_matches_varied_sparse_instances():
    size = (64, 48)
    state = 0x7F4A7C15

    def next_value() -> int:
        nonlocal state
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        return state

    for case in range(24):
        detections = []
        for _ in range(8):
            width = next_value() % 8 + 1
            height = next_value() % 8 + 1
            x = next_value() % (size[0] - width + 1)
            y = next_value() % (size[1] - height + 1)
            pixels = np.array(
                [255 if next_value() % 3 else 0 for _ in range(width * height)], dtype=np.uint8
            ).reshape(height, width)
            detections.append(_local_text(x, y, pixels))
        radius = case % 5 + 1
        page = det.mask_for(detections, "text", size)
        expected = det.close_l2(det.dilate_l2(page, radius), radius)

        assert np.array_equal(det.closed_mask_for(detections, "text", size, radius), expected), f"case {case}"


def test_bubble_geometry_follows_the_instance_mask_polygon():
    width = 9
    height = 9
    pixels = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if abs(x - 4) + abs(y - 4) <= 4:
                pixels[y, x] = 255
    geometry = det.mask_geometry(_mask(10, 20, pixels))

    assert geometry is not None
    assert len(geometry.points) >= 4
    assert all(not (p.x == 10.0 and p.y == 20.0) for p in geometry.points)
    assert any(p.x == 14.0 and p.y == 20.0 for p in geometry.points)


def test_layout_order_follows_panels_then_bubbles_then_their_text():
    detections = [
        _detection("text", 0.63, [20.0, 30.0, 70.0, 70.0]),
        _detection("bubble", 0.7, [10.0, 20.0, 80.0, 80.0]),
        _detection("panel", 0.9, [100.0, 0.0, 200.0, 200.0]),
        _detection("text", 0.62, [130.0, 110.0, 180.0, 150.0]),
        _detection("bubble", 0.8, [120.0, 20.0, 190.0, 80.0]),
        _detection("panel", 0.9, [0.0, 0.0, 95.0, 200.0]),
        _detection("text", 0.61, [130.0, 30.0, 180.0, 70.0]),
        _detection("bubble", 0.7, [120.0, 100.0, 190.0, 160.0]),
    ]

    order = det.layout_order(detections)
    text_scores = [detections[i].score for i in order if detections[i].label == "text"]

    assert text_scores == [0.61, 0.62, 0.63]


def test_layout_order_merges_containers_and_text_at_each_spatial_level():
    detections = [
        _detection("panel", 0.9, [0.0, 40.0, 200.0, 240.0]),
        _detection("bubble", 0.8, [100.0, 120.0, 190.0, 210.0]),
        _detection("text", 0.3, [120.0, 140.0, 180.0, 190.0]),
        _detection("text", 0.2, [20.0, 60.0, 180.0, 90.0]),
        _detection("text", 0.1, [20.0, 0.0, 180.0, 20.0]),
    ]

    order = det.layout_order(detections)
    text_scores = [detections[i].score for i in order if detections[i].label == "text"]

    assert text_scores == [0.1, 0.2, 0.3]


# ---- typography from pixels ---------------------------------------------------------------------------


def _masked_text(local_width: float, local_height: float, angle_degrees: float, color):
    size = 96
    center = size * 0.5
    radians = math.radians(angle_degrees)
    sin, cos = math.sin(radians), math.cos(radians)
    grid = np.arange(size, dtype=np.float64) + 0.5 - center
    dx = grid[None, :]
    dy = grid[:, None]
    local_x = dx * cos + dy * sin
    local_y = -dx * sin + dy * cos
    inside = (np.abs(local_x) <= local_width * 0.5) & (np.abs(local_y) <= local_height * 0.5)
    image = np.full((size, size, 3), (200, 180, 160), dtype=np.uint8)
    image[inside] = color
    return image, _detection_from_mask("text", 0, 0, size, size, inside.astype(np.uint8) * 255)


def _detection_from_mask(label: str, x: int, y: int, width: int, height: int, pixels: np.ndarray) -> LayoutDetection:
    return LayoutDetection(
        label_id=0,
        label=label,
        score=1.0,
        bbox=(float(x), float(y), float(x + width), float(y + height)),
        area=int(np.count_nonzero(pixels)),
        mask=_mask(x, y, pixels),
    )


def _outlined_text_image(stroke_width: int, fill, stroke, background):
    size = 96
    left, top, right, bottom = 20, 38, 76, 58
    image = np.full((size, size, 3), background, dtype=np.uint8)
    pixels = np.zeros((size, size), dtype=np.uint8)
    for y in range(top, bottom):
        for x in range(left, right):
            pixels[y, x] = 255
            is_fill = left + stroke_width <= x < right - stroke_width and top + stroke_width <= y < bottom - stroke_width
            image[y, x] = fill if is_fill else stroke
    detection = LayoutDetection(
        label_id=0,
        label="text",
        score=1.0,
        bbox=(float(left), float(top), float(right), float(bottom)),
        area=int(np.count_nonzero(pixels)),
        mask=_mask(0, 0, pixels),
    )
    return image, detection


def _region_masked_text(fill, background):
    size = 96
    left, top, right, bottom = 20, 20, 76, 76
    image = np.full((size, size, 3), background, dtype=np.uint8)
    pixels = np.zeros((size, size), dtype=np.uint8)
    pixels[top:bottom, left:right] = 255
    for x in (27, 35, 43, 51, 59, 67):
        image[28:68, x : x + 3] = fill
    detection = LayoutDetection(
        label_id=0,
        label="text",
        score=1.0,
        bbox=(float(left), float(top), float(right), float(bottom)),
        area=int(np.count_nonzero(pixels)),
        mask=_mask(0, 0, pixels),
    )
    return image, detection


def _outlined_text_on_textured_background():
    size = 96
    left, top, right, bottom = 20, 20, 76, 76
    image = np.full((size, size, 3), (24, 32, 96), dtype=np.uint8)
    pixels = np.zeros((size, size), dtype=np.uint8)
    palette = [(0, 0, 0), (38, 48, 128), (24, 32, 96)]
    for y in range(top, bottom):
        for x in range(left, right):
            pixels[y, x] = 255
            image[y, x] = palette[(x // 7 + y // 5) % 3]
    for glyph_left in (26, 38, 50, 62):
        for y in range(28, 68):
            for x in range(glyph_left, glyph_left + 9):
                is_fill = glyph_left + 3 <= x < glyph_left + 6 and 31 <= y < 65
                image[y, x] = (0, 0, 0) if is_fill else (255, 255, 255)
    detection = LayoutDetection(
        label_id=0,
        label="text",
        score=1.0,
        bbox=(float(left), float(top), float(right), float(bottom)),
        area=int(np.count_nonzero(pixels)),
        mask=_mask(0, 0, pixels),
    )
    return image, detection


def test_typography_comes_from_horizontal_text_mask():
    image, detection = _masked_text(52.0, 12.0, 12.0, (24, 80, 160))

    inferred = det.infer_typography(image, detection)

    assert abs(inferred.angle_degrees - 12.0) < 1.0
    assert inferred.color == (24, 80, 160)
    assert inferred.stroke_color is None
    assert inferred.stroke_width is None
    assert inferred.writing_mode == c.WritingMode.HORIZONTAL


def test_vertical_text_angle_is_relative_to_upright_vertical():
    image, detection = _masked_text(12.0, 52.0, 9.0, (120, 80, 40))

    inferred = det.infer_typography(image, detection)

    assert abs(inferred.angle_degrees - 9.0) < 1.0
    assert inferred.writing_mode == c.WritingMode.VERTICAL


def test_tall_multiline_mask_uses_text_lines_instead_of_block_aspect():
    size = 96
    angle_degrees = 8.0
    radians = math.radians(angle_degrees)
    sin, cos = math.sin(radians), math.cos(radians)
    grid = np.arange(size, dtype=np.float64) + 0.5 - size * 0.5
    dx = grid[None, :]
    dy = grid[:, None]
    local_x = dx * cos + dy * sin
    local_y = -dx * sin + dy * cos
    inside = np.zeros((size, size), dtype=bool)
    for line_y in (-24.0, 0.0, 24.0):
        inside |= (np.abs(local_x) <= 15.0) & (np.abs(local_y - line_y) <= 2.0)
    image = np.full((size, size, 3), (240, 240, 240), dtype=np.uint8)
    image[inside] = (8, 8, 8)
    detection = _detection_from_mask("text", 0, 0, size, size, inside.astype(np.uint8) * 255)

    inferred = det.infer_typography(image, detection)

    assert inferred.writing_mode == c.WritingMode.HORIZONTAL
    assert abs(inferred.angle_degrees - 8.0) < 1.0


def test_antialiased_dark_text_uses_the_high_contrast_core():
    image, detection = _masked_text(52.0, 12.0, 0.0, (96, 94, 92))
    flat = detection.mask.pixels.reshape(-1)
    for index in np.nonzero(flat)[0]:
        if index % 3 == 0:
            image[index // image.shape[1], index % image.shape[1]] = (8, 9, 7)

    inferred = det.infer_typography(image, detection)

    assert inferred.color == (0, 0, 0)


def test_text_colors_snap_only_at_luminance_extremes():
    assert det.normalize_text_color((20, 31, 24)) == (0, 0, 0)
    assert det.normalize_text_color((230, 240, 250)) == (255, 255, 255)
    assert det.normalize_text_color((255, 0, 0)) == (0, 0, 0)
    assert det.normalize_text_color((0, 255, 0)) == (0, 255, 0)
    assert det.normalize_text_color((20, 115, 235)) == (20, 115, 235)
    assert det.normalize_text_color((40, 70, 40)) == (0, 0, 0)


@pytest.mark.parametrize("stroke_width", [2, 3, 5])
def test_outlined_text_uses_the_deep_fill_and_measures_the_border(stroke_width):
    image, detection = _outlined_text_image(stroke_width, (0, 0, 0), (255, 255, 255), (16, 24, 88))

    inferred = det.infer_typography(image, detection)

    assert inferred.color == (0, 0, 0)
    assert inferred.stroke_color == (255, 255, 255)
    assert inferred.stroke_width == float(stroke_width)


def test_outlined_text_roles_do_not_flip_with_inverse_luminance():
    image, detection = _outlined_text_image(3, (255, 255, 255), (0, 0, 0), (16, 24, 88))

    inferred = det.infer_typography(image, detection)

    assert inferred.color == (255, 255, 255)
    assert inferred.stroke_color == (0, 0, 0)
    assert inferred.stroke_width == 3.0


def test_glyph_holes_matching_the_background_do_not_invert_the_paint():
    image, detection = _outlined_text_image(3, (255, 255, 255), (0, 0, 0), (255, 255, 255))

    inferred = det.infer_typography(image, detection)

    assert inferred.color == (0, 0, 0)
    assert inferred.stroke_color is None
    assert inferred.stroke_width is None


def test_outlined_text_is_separated_from_matching_dark_artwork():
    image, detection = _outlined_text_on_textured_background()

    inferred = det.infer_typography(image, detection)

    assert inferred.color == (0, 0, 0)
    assert inferred.stroke_color == (255, 255, 255)
    assert inferred.stroke_width == 3.0


def test_colors_in_the_same_luminance_class_do_not_create_a_border():
    image, detection = _outlined_text_image(3, (0, 0, 0), (48, 48, 48), (16, 24, 88))

    inferred = det.infer_typography(image, detection)

    assert inferred.color == (0, 0, 0)
    assert inferred.stroke_color is None
    assert inferred.stroke_width is None


def test_wide_antialias_bands_do_not_create_borders():
    for fill, antialias, background in [
        ((20, 115, 235), (133, 180, 240), (245, 245, 245)),
        ((0, 0, 0), (96, 96, 96), (245, 245, 245)),
    ]:
        image, detection = _outlined_text_image(4, fill, antialias, background)

        inferred = det.infer_typography(image, detection)

        assert inferred.color == fill
        assert inferred.stroke_color is None
        assert inferred.stroke_width is None


def test_text_region_background_is_not_mistaken_for_the_fill():
    for fill, background, expected in [
        ((24, 40, 80), (225, 130, 175), (0, 0, 0)),
        ((220, 240, 255), (16, 24, 88), (255, 255, 255)),
        ((20, 115, 235), (245, 245, 245), (20, 115, 235)),
    ]:
        image, detection = _region_masked_text(fill, background)

        inferred = det.infer_typography(image, detection)

        assert inferred.color == expected
        assert inferred.stroke_color is None
        assert inferred.stroke_width is None


def test_color_palette_is_bounded_independently_of_crop_area():
    index = np.arange(65_536, dtype=np.int64)
    color = np.stack([index & 0xFF, (index >> 8) & 0xFF, (index * 31) & 0xFF], axis=1)

    colors, counts, _ = det._palette(color)

    assert len(colors) <= 16 * 16 * 16


# ---- imageproc-equivalent primitives -------------------------------------------------------------------


def _brute_force_ceil_distance(sites: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(sites)
    height, width = sites.shape
    out = np.full(sites.shape, 255, dtype=np.uint8)
    if len(ys) == 0:
        return out
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    squared = np.min((grid_y[..., None] - ys) ** 2 + (grid_x[..., None] - xs) ** 2, axis=-1)
    root = np.ceil(np.sqrt(squared.astype(np.float64)))
    return np.minimum(root, 255).astype(np.uint8)


def test_distance_transform_matches_exact_euclidean_ceiling():
    rng = np.random.default_rng(7)
    for _ in range(6):
        sites = rng.random((23, 31)) < 0.04
        sites[0, 0] = True
        assert np.array_equal(det._distance_from_sites(sites), _brute_force_ceil_distance(sites))


def test_suzuki_abe_traces_the_outer_border_of_a_filled_square():
    binary = np.zeros((7, 7), dtype=bool)
    binary[1:6, 1:6] = True

    contours = det.trace_contours(binary)
    outer = [points for points, is_outer in contours if is_outer]

    assert len(outer) == 1
    assert set(outer[0]) == {(x, y) for x in range(1, 6) for y in range(1, 6) if x in (1, 5) or y in (1, 5)}


def test_contour_simplification_matches_imageproc_closed_mode():
    # imageproc simplifies the open curve and then pops its last point when closed, so the final corner
    # of an open Suzuki contour is dropped. mask_geometry inherits this; the port keeps it.
    outline = [(0, 0), (2, 0), (4, 0), (4, 4), (2, 4), (0, 4)]

    simplified = det.approximate_polygon_dp(outline, 0.5, closed=True)

    assert simplified == [(0, 0), (4, 0), (4, 4)]


class _FakeDetector:
    """Stands in for the provider: returns a fixed set of detections."""

    def __init__(self, detections, width: int, height: int):
        self.output = LayoutDetections(image_width=width, image_height=height, detections=detections)

    async def detect(self, image):
        return self.output


def _solid_text(x: int, y: int, width: int, height: int) -> LayoutDetection:
    return LayoutDetection(
        label_id=0,
        label="text",
        score=0.9,
        bbox=(float(x), float(y), float(x + width), float(y + height)),
        area=width * height,
        mask=_mask(x, y, np.full((height, width), 255, dtype=np.uint8)),
    )


def test_region_scoped_redetection_keeps_masks_and_regions_outside_the_region(tmp_path):
    session = Session(tmp_path)
    blank = io.BytesIO()
    Image.new("RGB", (100, 100), "white").save(blank, format="PNG")
    edit = session.snapshot().edit()
    page = edit.add_page(Page(label="page", width=100.0, height=100.0), At.END)
    edit.set_asset(page, "source", AssetInput(data=blank.getvalue(), media_type="image/png",
                                              metadata=AssetMetadata(width=100, height=100)))
    snapshot = session.commit(edit.finish())

    detector = _FakeDetector([_solid_text(10, 10, 20, 10), _solid_text(60, 60, 20, 20)], 100, 100)
    stage_input = StageInput(snapshot, page, None, None, ImageCache(), None)
    snapshot = session.commit(asyncio.run(DetectionStage().process(stage_input, detector)))
    first = np.asarray(Image.open(io.BytesIO(snapshot.asset_bytes(page, "text-mask"))).convert("L"))
    assert first[15, 15] == 255 and first[70, 70] == 255
    assert len([e for e in snapshot.descendants(page) if snapshot.component(e, c.Region) is not None]) == 2

    # Re-detect only inside (0, 0, 50, 50), where the detector now finds nothing.
    stage_input = StageInput(snapshot, page, None, Bounds(0.0, 0.0, 50.0, 50.0), ImageCache(), None)
    snapshot = session.commit(asyncio.run(DetectionStage().process(stage_input, _FakeDetector([], 100, 100))))
    second = np.asarray(Image.open(io.BytesIO(snapshot.asset_bytes(page, "text-mask"))).convert("L"))

    assert second[15, 15] == 0  # inside the scope: rewritten from the (empty) new detections
    assert second[70, 70] == 255  # outside the scope: preserved from the previous mask
    regions = [e for e in snapshot.descendants(page) if snapshot.component(e, c.Region) is not None]
    assert len(regions) == 1
    assert snapshot.component(regions[0], c.Geometry).bounds() == (60.0, 60.0, 80.0, 80.0)


# ---- checkpoint -----------------------------------------------------------------------------------------


def _cached_checkpoint():
    try:
        from huggingface_hub import hf_hub_download

        config = hf_hub_download(REPOSITORY, "inference_config.json", revision=REVISION, local_files_only=True)
        weights = hf_hub_download(REPOSITORY, "model.safetensors", revision=REVISION, local_files_only=True)
    except Exception:
        return None
    return Path(config), Path(weights)


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


KOHARU_ROOT = Path(os.environ.get("KOHARU_ROOT", r"D:\dev\koharu"))
RUST_FIXTURE = KOHARU_ROOT / "crates" / "koharu-ml" / "benches" / "fixtures" / "object_detection" / "1.jpg"


def test_checkpoint_detects_the_fixture_page(tmp_path):
    if not _cuda_available():
        pytest.skip("requires CUDA")
    checkpoint = _cached_checkpoint()
    if checkpoint is None:
        pytest.skip("checkpoint is not in the local Hugging Face cache (offline)")
    if not RUST_FIXTURE.exists():
        pytest.skip("Koharu fixture image is not available")
    config_path, weights_path = checkpoint

    import torch
    from rfdetr.models.postprocess import PostProcess

    from urakatsu.providers.detection.rfdetr.config import InferenceConfig
    from urakatsu.providers.detection.rfdetr.network import load_network, run_network
    from urakatsu.providers.detection.rfdetr.processor import KoharuLayoutRFDetrImageProcessor

    image = Image.open(RUST_FIXTURE).convert("RGB")

    # Provider path at the precision the runtime selects (bfloat16 on Ampere CUDA, as Koharu's set_precision).
    provider = KoharuLayoutRFDetrSeg2XLProvider(runtime="torch")
    provider._build(config_path, weights_path)
    result = provider._detect(image)
    best = result.detections[0]
    assert best.label == "bubble"
    # The Rust reference for this page is bubble 0.9689; the bf16 path lands within its tolerance.
    assert abs(best.score - 0.968856) < 0.01

    # fp32 path against upstream PostProcess on identical network outputs must match exactly.
    config = InferenceConfig.from_file(config_path)
    processor = KoharuLayoutRFDetrImageProcessor(config)
    device = torch.device("cuda", 0)
    network = load_network(weights_path, config, device, torch.float32)
    thresholds = provider._thresholds
    with torch.inference_mode():
        pixels = processor.preprocess(image, device)
        output = run_network(network, pixels, torch.float32)
        ours = processor.postprocess(output, image.width, image.height, thresholds)
        upstream = PostProcess(num_select=config.num_select)(
            {"pred_logits": output.pred_logits, "pred_boxes": output.pred_boxes, "pred_masks": output.pred_masks},
            torch.tensor([[image.height, image.width]], device=device),
        )[0]
    names = config.class_names()
    per_class = {"text": thresholds.text, "onomatopoeia": thresholds.onomatopoeia, "bubble": thresholds.bubble, "panel": thresholds.panel}
    scores = upstream["scores"].float().cpu().tolist()
    labels = upstream["labels"].cpu().tolist()
    boxes = upstream["boxes"].float().cpu().tolist()
    expected = sorted(
        (
            (scores[i], names[labels[i]], boxes[i])
            for i in range(len(scores))
            if labels[i] < len(names) and scores[i] > per_class[names[labels[i]]]
        ),
        key=lambda item: -item[0],
    )
    got = sorted(ours.detections, key=lambda d: -d.score)
    assert len(got) == len(expected)
    for detection, (score, label, box) in zip(got, expected):
        assert detection.label == label
        assert abs(detection.score - score) < 1e-6
        assert max(abs(a - b) for a, b in zip(detection.bbox, box)) < 1e-3
    del network
    torch.cuda.empty_cache()

    session = Session(tmp_path)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    edit = session.snapshot().edit()
    page = edit.add_page(Page(label="1", width=image.width, height=image.height), At.END)
    edit.set_asset(page, "source", AssetInput(data=buffer.getvalue(), media_type="image/png",
                                              metadata=AssetMetadata(width=image.width, height=image.height)))
    snapshot = session.commit(edit.finish())
    stage_input = StageInput(snapshot, page, None, None, ImageCache(), None)
    import asyncio

    patch = asyncio.run(DetectionStage().process(stage_input, provider))
    committed = session.commit(patch)
    labels = {committed.component(e, c.Region).label for e in committed.descendants(page) if committed.component(e, c.Region)}
    assert {"bubble", "text", "panel"} <= labels
    assert committed.asset(page, "text-mask") is not None
    assert committed.asset(page, "bubble-mask") is None
