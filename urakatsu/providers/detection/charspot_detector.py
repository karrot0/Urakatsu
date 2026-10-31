"""Charspot as a text detector: an experiment. Not in Koharu, and not how upstream uses the model.

The model (``providers/ocr/charspot_ocr.py``) is made to read one cropped text box and takes at
most 256 cells of 16 px, far too few for a whole page. It does find single characters, though, so
this provider runs it on overlapping page tiles and builds text blocks from what it finds:

- A tile is a third of the page's short side, and at least what fits the cell budget at full size
  (256 px), so characters stay near the 12-40 px the model trained on. Each tile owns the pixels
  up to the middle of its overlaps, and a character belongs to the tile that owns its centre.
- The model reads characters into art too, because it expects a crop that holds text, and those
  hits score lower. Blocks therefore grow from characters scoring at least ``threshold``: a
  neighbour joins when the gap to it is below ``GROUP_GAP`` of the member's size and it is at most
  ``SIZE_RATIO`` times as large. A weaker neighbour joins too, so a faint character stays in its
  line, but nothing grows from it, so art cannot chain blocks together. A block must hold
  ``min_characters`` characters that reach ``threshold``, which also drops short low-contrast
  labels. Nothing else separates blocks, so text that sits close together can merge.
- A block's mask is the union of its character boxes. There are no bubble, panel or onomatopoeia
  detections, and furigana is dropped by the model, so it stays out of the mask.

Runs on CPU for the reason ``charspot_ocr.py`` gives. The weights are non-commercial, as there.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, ClassVar, Optional

import numpy as np
from PIL import Image
from pydantic import Field

from urakatsu.core.errors import ProcessingError
from urakatsu.core.registry import register
from urakatsu.core.results import LayoutDetection, LayoutDetections, LayoutMask
from urakatsu.providers.base import DetectionProvider, ProviderSettings
from urakatsu.providers.ocr.charspot_ocr import CharspotOcrProvider, preprocess
from urakatsu.runtimes.store import WeightSource

TILES_PER_SHORT_SIDE = 3
GROUP_GAP = 0.6
SIZE_RATIO = 1.6


class CharspotDetectorSettings(ProviderSettings):
    threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Lowest score a character needs to grow a text block")
    min_characters: int = Field(default=2, ge=1, description="Fewest characters at that score a text block needs")


def tile_spans(length: int, tile: int) -> list[tuple[int, int, int]]:
    """``(start, own_start, own_end)`` along one side: tiles overlap by a quarter, and each owns
    the pixels up to the middle of its overlaps."""
    if length <= tile:
        return [(0, 0, length)]
    starts = [*range(0, length - tile, tile * 3 // 4), length - tile]
    spans = []
    for index, start in enumerate(starts):
        own_start = 0 if index == 0 else (start + starts[index - 1] + tile) // 2
        own_end = length if index == len(starts) - 1 else (start + tile + starts[index + 1]) // 2
        spans.append((start, own_start, own_end))
    return spans


def group_characters(boxes: np.ndarray, strong: np.ndarray) -> list[list[int]]:
    """Indices of ``boxes`` (x0, y0, x1, y1) per block, grown from the ``strong`` characters."""
    sizes = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    gap_x = np.maximum(boxes[:, None, 0] - boxes[None, :, 2], boxes[None, :, 0] - boxes[:, None, 2])
    gap_y = np.maximum(boxes[:, None, 1] - boxes[None, :, 3], boxes[None, :, 1] - boxes[:, None, 3])
    reach = (np.maximum(gap_x, gap_y) <= sizes[:, None] * GROUP_GAP) & (sizes[None, :] <= sizes[:, None] * SIZE_RATIO)
    seen = np.zeros(len(boxes), bool)
    groups = []
    for first in np.nonzero(strong)[0].tolist():
        if seen[first]:
            continue
        seen[first] = True
        members, stack = [], [first]
        while stack:
            index = stack.pop()
            members.append(index)
            if strong[index]:
                found = np.nonzero(reach[index] & ~seen)[0]
                seen[found] = True
                stack.extend(found.tolist())
        groups.append(members)
    return groups


def text_block(boxes: np.ndarray, score: float, width: int, height: int) -> Optional[LayoutDetection]:
    """One ``text`` detection whose mask is the union of the characters' boxes on the page."""
    x0, y0 = max(math.floor(boxes[:, 0].min()), 0), max(math.floor(boxes[:, 1].min()), 0)
    x1, y1 = min(math.ceil(boxes[:, 2].max()), width), min(math.ceil(boxes[:, 3].max()), height)
    if x1 <= x0 or y1 <= y0:
        return None
    pixels = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for left, top, right, bottom in boxes:
        pixels[max(math.floor(top), y0) - y0 : math.ceil(bottom) - y0, max(math.floor(left), x0) - x0 : math.ceil(right) - x0] = 255
    return LayoutDetection(
        label_id=0,
        label="text",
        score=score,
        bbox=(float(x0), float(y0), float(x1), float(y1)),
        area=int(np.count_nonzero(pixels)),
        mask=LayoutMask(x=x0, y=y0, width=x1 - x0, height=y1 - y0, pixels=pixels),
    )


@register
class CharspotDetectorProvider(DetectionProvider):
    id = "charspot-detector"
    name = "Charspot (experimental)"
    description = "Experimental text detector: Charspot OCR's character finder on page tiles (text only). Non-commercial weights."
    runtimes = ("onnx",)
    Settings = CharspotDetectorSettings
    catalog_queries: ClassVar[tuple[str, ...]] = CharspotOcrProvider.catalog_queries

    @classmethod
    def accepts(cls, facts) -> Optional[str]:
        """The same models as Charspot OCR."""
        return CharspotOcrProvider.accepts(facts)

    def __init__(self, settings: Any = None, runtimes=None, runtime: Optional[str] = None):
        super().__init__(settings, runtimes, runtime)
        self._session = None

    @classmethod
    def weight_sources(cls, runtime: Optional[str] = None) -> tuple[WeightSource, ...]:
        return CharspotOcrProvider.weight_sources(runtime)

    def _load(self) -> None:
        config_path, _, weights_path = self.resolve_weights()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self._cell = int(config["input"]["cell"])
        self._budget = int(config["input"]["budget_cells"])
        self._session = self.runtime_instance().session(
            weights_path, cpu=True, intra_op_num_threads=min(8, os.cpu_count() or 1), inter_op_num_threads=1
        )

    def _unload(self) -> None:
        self._session = None

    def _detect(self, image: Image.Image) -> LayoutDetections:
        if self._session is None:
            raise ProcessingError(f"{self.id} is not loaded")
        rgb = image.convert("RGB")
        width, height = rgb.size
        if width == 0 or height == 0:
            raise ProcessingError("cannot detect text in an empty image")
        boxes, scores = self._characters(rgb)
        detections = []
        if len(boxes):
            strong = scores >= self.settings.threshold
            for members in group_characters(boxes, strong):
                if int(strong[members].sum()) >= self.settings.min_characters:
                    block = text_block(boxes[members], float(scores[members].mean()), width, height)
                    if block is not None:
                        detections.append(block)
        return LayoutDetections(image_width=width, image_height=height, detections=detections)

    def _characters(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """Every character's box (in page pixels) and score, each from the tile owning its centre."""
        tile = max(math.isqrt(self._budget) * self._cell, min(image.size) // TILES_PER_SHORT_SIDE)
        all_boxes, all_scores = [np.zeros((0, 4), np.float32)], [np.zeros(0, np.float32)]
        for top, own_top, own_bottom in tile_spans(image.height, tile):
            for left, own_left, own_right in tile_spans(image.width, tile):
                box = (left, top, min(left + tile, image.width), min(top + tile, image.height))
                pixels, (scale_x, scale_y) = preprocess(image.crop(box), self._budget, self._cell)
                boxes, scores = self._session.run(["boxes", "scores"], {"img": pixels[None].astype(np.float32)})
                boxes = boxes * np.array([scale_x, scale_y, scale_x, scale_y], np.float32) + np.array([left, top, left, top], np.float32)
                centre_x, centre_y = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
                owned = (own_left <= centre_x) & (centre_x < own_right) & (own_top <= centre_y) & (centre_y < own_bottom)
                all_boxes.append(boxes[owned])
                all_scores.append(scores[owned])
        return np.concatenate(all_boxes), np.concatenate(all_scores)
