"""YOLO watermark detector (ONNX). Not in Koharu.

Default weights: ``ayan4m1/Watermark-Detection-YOLO26-ONNX`` at a pinned revision, a YOLO26 fine-tune
on the visible-watermark-pita dataset with one class (``watermark``), exported end to end
(``output0`` is ``(1, 300, 6)``: x0, y0, x1, y1, score, class). The catalog can swap in other
single-class YOLO watermark exports; a raw YOLO head (``(1, 4 + classes, anchors)``: centre, size,
class scores) is decoded with NMS instead.

The input is a 640 square: the image scaled so its longer side is 640, at the top left, padded
with grey (114) the way Ultralytics letterboxes. Tall strips (webtoons) are cut into overlapping
square tiles first, because a whole strip scaled to 640 shrinks a 150-pixel logo to a few pixels.

The detector only proposes: its boxes are seeds for ``pipeline/watermarks.py``, which keeps a
mark only when it repeats across the chapter, so its false positives (lace, lettering) and its
partial boxes cost time, not art.
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional

import numpy as np
from PIL import Image
from pydantic import Field

from urakatsu.core.errors import ProcessingError
from urakatsu.core.registry import register
from urakatsu.providers.base import ProviderSettings, WatermarkBox, WatermarkProvider
from urakatsu.runtimes.store import HuggingFaceFile

REPOSITORY = "ayan4m1/Watermark-Detection-YOLO26-ONNX"
REVISION = "c189f35eaf50974610647f58df94ecf94b890a90"
WEIGHTS = "onnx/model.onnx"
SIZE = 640
PAD_VALUE = 114
NMS_IOU = 0.4


class YoloWatermarkSettings(ProviderSettings):
    # Low on purpose: a box is only a seed, kept when its mark repeats across the chapter.
    threshold: float = Field(default=0.25, ge=0.0, le=1.0, description="Lowest detector score proposed as a watermark")


@register
class YoloWatermarkProvider(WatermarkProvider):
    id = "yolo26-watermark"
    name = "YOLO26 watermark detector"
    description = "Finds watermark candidates (platform logos, scanlator marks); repeated ones are removed by unblending."
    runtimes = ("onnx",)
    Settings = YoloWatermarkSettings
    catalog_queries: ClassVar[tuple[str, ...]] = ("watermark detection",)

    @classmethod
    def accepts(cls, facts) -> Optional[str]:
        """Single-purpose watermark detectors exported to ONNX."""
        if facts.runtime != "onnx" or not any(f.endswith(".onnx") for f in facts.files):
            return None
        named = "watermark" in facts.repo.lower() or "watermark" in (t.lower() for t in facts.tags)
        return "onnx" if named and facts.pipeline_tag in (None, "object-detection") else None

    def __init__(self, settings: Any = None, runtimes=None, runtime: Optional[str] = None):
        super().__init__(settings, runtimes, runtime)
        self._session = None

    @classmethod
    def weight_sources(cls, runtime: Optional[str] = None):
        return (HuggingFaceFile(REPOSITORY, WEIGHTS, REVISION),)

    def _load(self) -> None:
        (path,) = self.resolve_weights()
        self._session = self.runtime_instance().session(path)

    def _unload(self) -> None:
        self._session = None

    def _detect_watermarks(self, image: Image.Image) -> list[WatermarkBox]:
        if self._session is None:
            raise ProcessingError(f"{self.id} is not loaded")
        rgb = image.convert("RGB")
        width, height = rgb.size
        boxes: list[np.ndarray] = []
        scores: list[np.ndarray] = []
        for top, tile in _tiles(rgb):
            tile_boxes, tile_scores = self._run(tile)
            tile_boxes[:, [1, 3]] += top
            boxes.append(tile_boxes)
            scores.append(tile_scores)
        all_boxes, all_scores = np.concatenate(boxes), np.concatenate(scores)
        keep = all_scores >= self.settings.threshold
        all_boxes, all_scores = all_boxes[keep], all_scores[keep]
        result = []
        for index in _nms(all_boxes, all_scores, NMS_IOU):
            x0, y0, x1, y1 = all_boxes[index]
            box = (int(max(x0, 0)), int(max(y0, 0)), int(min(x1, width)), int(min(y1, height)))
            if box[2] > box[0] and box[3] > box[1]:
                result.append(WatermarkBox(box=box, score=float(all_scores[index])))
        return result

    def _run(self, tile: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """Boxes (in tile pixels) and scores for one tile."""
        width, height = tile.size
        scale = SIZE / max(width, height)
        canvas = Image.new("RGB", (SIZE, SIZE), (PAD_VALUE,) * 3)
        canvas.paste(tile.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.BILINEAR), (0, 0))
        pixels = (np.asarray(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]
        name = self._session.get_inputs()[0].name
        output = self._session.run(None, {name: pixels})[0][0]
        if output.ndim == 2 and output.shape[1] == 6:  # end to end: x0, y0, x1, y1, score, class
            boxes, scores = output[:, :4].astype(np.float32), output[:, 4].astype(np.float32)
        else:  # raw head: (4 + classes, anchors) of centre x, centre y, width, height, class scores
            predictions = output.T
            scores = predictions[:, 4:].max(1).astype(np.float32)
            cx, cy, w, h = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
            boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1).astype(np.float32)
        return boxes / scale, scores


def _tiles(image: Image.Image):
    """The image, or for a strip taller than 1.6 widths, square tiles overlapping by half."""
    width, height = image.size
    if height <= width * 1.6:
        yield 0, image
        return
    tops = list(range(0, height - width, max(width // 2, 1))) + [height - width]
    for top in tops:
        yield top, image.crop((0, top, width, top + width))


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    order = scores.argsort()[::-1]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        ix = np.clip(np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0]), 0, None)
        iy = np.clip(np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1]), 0, None)
        inter = ix * iy
        order = rest[inter / (area[i] + area[rest] - inter + 1e-9) < iou]
    return keep
