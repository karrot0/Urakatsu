"""Text detector: Comic Text Detector (CTD), a port of Koharu's ``comic_text_detector`` module.

Weights: ``mayocream/comic-text-detector`` at a pinned revision (``yolo-v5.safetensors``,
``unet.safetensors``, ``dbnet.safetensors``), loaded strictly. The network and the pre/post-processing
live in ``ctd/`` (see ``ctd/processor.py`` for the port notes). Torch is imported only in ``_load`` and
``_detect``; weights are resolved in ``_load``, so nothing is downloaded unless this detector is used.

Differences from the Koharu layout detector (``koharu_layout_rfdetr_seg_2xl``):

- CTD detects text only. It never produces ``onomatopoeia``, ``bubble`` or ``panel`` detections.
- CTD produces no confidence score. Each detection's score is the mean of its line polygons' scores
  (the mean DBNet shrink-map value inside each line), or 1.0 when the block has none.
- Each detection's mask is the refined text mask (after Koharu's dilation) cropped to the block bbox.
  If that crop is empty, the block's line quads are filled into the crop instead.
- ``area`` is the nonzero pixel count of the final mask.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw

from urakatsu.core.errors import ProcessingError
from urakatsu.core.registry import register
from urakatsu.core.results import LayoutDetection, LayoutDetections, LayoutMask
from urakatsu.providers.base import DetectionProvider, ProviderSettings
from urakatsu.runtimes.store import HuggingFaceFile

log = logging.getLogger(__name__)

REPOSITORY = "mayocream/comic-text-detector"
REVISION = "15ade029f4dabd502bc97af6051c8b9f2bec24d5"
YOLO_WEIGHTS = "yolo-v5.safetensors"
UNET_WEIGHTS = "unet.safetensors"
DBNET_WEIGHTS = "dbnet.safetensors"
TEXT_LABEL_ID = 0
TEXT_LABEL = "text"


@register
class ComicTextDetectorProvider(DetectionProvider):
    id = "comic-text-detector"
    name = "Comic Text Detector"
    description = "Manga text detector (text only): YOLOv5 blocks, U-Net text mask and DBNet text lines."
    runtimes = ("torch",)
    requires = {"torch": ("safetensors",)}
    Settings = ProviderSettings

    def __init__(self, settings: Any = None, runtimes=None, runtime: Optional[str] = None):
        super().__init__(settings, runtimes, runtime)
        self._network = None
        self._device = None

    @classmethod
    def weight_sources(cls, runtime: Optional[str] = None):
        return tuple(HuggingFaceFile(REPOSITORY, name, REVISION) for name in (YOLO_WEIGHTS, UNET_WEIGHTS, DBNET_WEIGHTS))

    def _load(self) -> None:
        yolo_path, unet_path, dbnet_path = self.resolve_weights()
        from .ctd.network import ComicTextDetectorNetwork

        runtime = self.runtime_instance()
        device = runtime.torch_device
        network = ComicTextDetectorNetwork()
        network.load_checkpoints(yolo_path, unet_path, dbnet_path)
        network.eval()
        network.to(device=device, dtype=runtime.dtype())
        self._network = network
        self._device = device

    def _unload(self) -> None:
        self._network = None
        self._device = None

    def _detect(self, image: Image.Image) -> LayoutDetections:
        """``ComicTextDetector::inference`` followed by one ``LayoutDetection`` per text block."""
        import torch

        from .ctd.processor import inference

        if self._network is None:
            raise ProcessingError(f"{self.id} is not loaded")
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with torch.inference_mode():
            mask, blocks = inference(rgb, self._device, self._network)
        return text_detections(rgb.shape[1], rgb.shape[0], mask, blocks)


def text_detections(width: int, height: int, mask: np.ndarray, blocks) -> LayoutDetections:
    """One ``text`` detection per block: the refined mask cropped to its bbox (line quads when empty)."""
    detections = []
    for block in blocks:
        x0, y0, x1, y1 = block.xyxy
        x0 = min(max(x0, 0), width - 1)
        y0 = min(max(y0, 0), height - 1)
        x1 = min(max(x1, x0 + 1), width)
        y1 = min(max(y1, y0 + 1), height)
        pixels = mask[y0:y1, x0:x1].copy()
        if not pixels.any():
            pixels = _fill_lines(block.lines, x0, y0, x1 - x0, y1 - y0)
        if not pixels.any():
            continue
        scores = block.line_scores
        score = float(np.mean(scores)) if scores else 1.0
        detections.append(
            LayoutDetection(
                label_id=TEXT_LABEL_ID,
                label=TEXT_LABEL,
                score=score,
                bbox=(float(x0), float(y0), float(x1), float(y1)),
                area=int(np.count_nonzero(pixels)),
                mask=LayoutMask(x=x0, y=y0, width=x1 - x0, height=y1 - y0, pixels=pixels),
            )
        )
    return LayoutDetections(image_width=width, image_height=height, detections=detections)


def _fill_lines(lines, x0: int, y0: int, width: int, height: int) -> np.ndarray:
    """Rasterise a block's line quads into its crop (used when the refined mask is empty there)."""
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for quad in lines:
        draw.polygon([(float(x - x0), float(y - y0)) for x, y in quad], fill=255)
    return np.asarray(canvas, dtype=np.uint8)
