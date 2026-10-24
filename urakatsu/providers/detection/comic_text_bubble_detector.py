"""Comic Text & Bubble Detector (Koharu ``comic_text_bubble_detector``): RT-DETR-v2 text and bubble boxes.

Weights: ``ogkalu/comic-text-and-bubble-detector`` at a pinned revision (``config.json``,
``preprocessor_config.json`` and ``model.safetensors``). The network is upstream
``transformers.RTDetrV2ForObjectDetection`` (Koharu's ``model.rs`` is a port of it), loaded strictly.
Pre- and postprocessing, the slicer and the text/bubble pairing live in ``ctbd/processor.py``, a port
of Koharu's ``processor.rs``. Torch and transformers are imported only inside ``_build``/``_detect``.

Detector classes: ``bubble`` (label 0) and text, which is reported as ``text`` with ``label_id`` 1 when
it sits in a bubble (``text_bubble``) and 2 when it does not (``text_free``). This is the model's full
label set. Onomatopoeia and panel classes do not exist in this checkpoint, so no detection carries them.

Masks are rectangles (``LayoutMask``), filled with 255 over the whole box. The model gives boxes only,
and Koharu erases the whole text box when inpainting, so a rectangle matches what is removed.
``area`` is the mask pixel count.

Scores: Koharu discards the score when it merges boxes. Here each output box keeps the maximum score of
the raw boxes merged into it. Bubbles keep their own score. This is a deviation from Koharu.

Confidence: ``confidence_threshold`` (default 0.3, Koharu's bin and bench default). Out-of-range values
fall back to that default, as the RF-DETR provider does for its thresholds.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image
from pydantic import Field

from urakatsu.core.errors import ModelLoadError, ProcessingError
from urakatsu.core.registry import register
from urakatsu.core.results import LayoutDetection, LayoutDetections, LayoutMask
from urakatsu.providers.base import DetectionProvider, ProviderSettings
from urakatsu.runtimes.store import HuggingFaceFile

from .ctbd.processor import (
    ImageSlicer,
    RTDetrImageProcessor,
    ScoredBox,
    create_text_blocks,
    filter_and_fix_bboxes,
)

log = logging.getLogger(__name__)

REPOSITORY = "ogkalu/comic-text-and-bubble-detector"
REVISION = "16e8a622f91fabc6b5b65c96d32d1183f8843546"

# Koharu bin and bench: ``--confidence-threshold`` / ``inference(.., 0.3)``.
DEFAULT_CONFIDENCE_THRESHOLD = 0.3

# config.rs / mod.rs: the model's id2label and the layout ids this provider reports.
LABEL_BUBBLE = 0
LABEL_TEXT_BUBBLE = 1
LABEL_TEXT_FREE = 2
TEXT_LABEL_IDS = {"text_bubble": 1, "text_free": 2}


class ComicTextBubbleDetectorSettings(ProviderSettings):
    """``ComicTextBubbleDetectorConfig``: the confidence threshold for boxes."""

    confidence_threshold: float = Field(
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        description="Detection confidence threshold, between 0 and 1 (Koharu default 0.3).",
    )


def _usable_threshold(name: str, value: float) -> float:
    """A stored threshold outside [0, 1] is a preference and falls back to the default (as RF-DETR does)."""
    if math.isfinite(value) and 0.0 <= value <= 1.0:
        return value
    log.warning("confidence threshold is not between zero and one; using the default", extra={"label": name, "threshold": value})
    return DEFAULT_CONFIDENCE_THRESHOLD


@register
class ComicTextBubbleDetectorProvider(DetectionProvider):
    id = "comic-text-bubble-detector"
    name = "Comic Text & Bubble Detector"
    description = "Comic text and speech-bubble detection (RT-DETR-v2): bubble boxes and text boxes with bubble membership."
    runtimes = ("torch",)
    requires = {"torch": ("transformers", "safetensors")}
    Settings = ComicTextBubbleDetectorSettings

    def __init__(self, settings: Any = None, runtimes=None, runtime: Optional[str] = None):
        super().__init__(settings, runtimes, runtime)
        # Normalise once at construction.
        self.settings = self.settings.model_copy(
            update={"confidence_threshold": _usable_threshold("confidence_threshold", self.settings.confidence_threshold)}
        )
        self._processor: Optional[RTDetrImageProcessor] = None
        self._model = None
        self._config = None
        self._slicer = ImageSlicer()
        self._device = None
        self._dtype = None

    @classmethod
    def weight_sources(cls, runtime: Optional[str] = None):
        return (
            HuggingFaceFile(REPOSITORY, "config.json", REVISION),
            HuggingFaceFile(REPOSITORY, "preprocessor_config.json", REVISION),
            HuggingFaceFile(REPOSITORY, "model.safetensors", REVISION),
        )

    def _load(self) -> None:
        config_path, processor_path, weights_path = self.resolve_weights()
        self._build(config_path, processor_path, weights_path)

    def _build(self, config_path: Path, processor_path: Path, weights_path: Path) -> None:
        """``RTDetrV2Detection::load``: parse both configs, fix the anchors, build the net and load it strictly."""
        import torch
        from transformers import RTDetrV2Config, RTDetrV2ForObjectDetection

        if config_path.parent != weights_path.parent:
            raise ModelLoadError("comic text/bubble detector config and weights must come from the same revision folder")
        processor = RTDetrImageProcessor.from_dict(json.loads(processor_path.read_text(encoding="utf-8")))
        config = RTDetrV2Config.from_json_file(str(config_path))
        # processor.size is always the network resolution, so RT-DETR reuses its fixed anchors (mod.rs).
        config.anchor_image_size = [processor.size.height, processor.size.width]

        # from_pretrained applies Transformers' rt_detr checkpoint key conversions (the checkpoint's
        # ``encoder.encoder.N`` is ``encoder.aifi.N``). Tied heads (top-level class_embed/bbox_embed
        # alias model.decoder.*) are not checkpoint keys and are not reported as missing.
        model, info = RTDetrV2ForObjectDetection.from_pretrained(
            str(weights_path.parent), config=config, dtype=torch.float32, output_loading_info=True
        )
        missing = list(info["missing_keys"])
        unexpected = list(info["unexpected_keys"])
        mismatched = list(info["mismatched_keys"])
        errors = list(info["error_msgs"])
        if missing or unexpected or mismatched or errors:
            raise ModelLoadError(
                f"incompatible comic text/bubble detector weights: missing {missing[:8]} ({len(missing)}), "
                f"unexpected {unexpected[:8]} ({len(unexpected)}), mismatched {mismatched[:4]} ({len(mismatched)}), "
                f"errors {errors[:2]}"
            )

        runtime = self.runtime_instance()
        device = runtime.torch_device
        dtype = runtime.dtype()
        # set_precision: the whole var store takes one dtype (bfloat16 on Ampere+ CUDA, else float32).
        model.to(device=device, dtype=dtype)
        model.eval()

        self._processor = processor
        self._model = model
        self._config = config
        self._slicer = ImageSlicer()
        self._device = device
        self._dtype = dtype

    def _unload(self) -> None:
        self._processor = None
        self._model = None
        self._config = None
        self._device = None
        self._dtype = None

    def _detect(self, image: Image.Image) -> LayoutDetections:
        """``RTDetrV2Detection::inference``: no-grad slicing, per-slice detection, then text-block pairing."""
        import torch

        if self._model is None or self._processor is None or self._config is None:
            raise ProcessingError(f"{self.id} is not loaded")
        image = image.convert("RGB")
        if image.width <= 0 or image.height <= 0:
            raise ProcessingError("cannot detect text in an empty image")

        with torch.inference_mode():
            bubble_boxes, text_boxes = self._slicer.process_slices_for_detection(image, self._detect_single_image)
        return self._layout(image, bubble_boxes, text_boxes)

    def _detect_single_image(self, image: Image.Image) -> tuple[list[ScoredBox], list[ScoredBox]]:
        """``detect_single_image``: one forward pass and a split into bubble and text boxes."""
        pixel_values = self._processor.preprocess(image, self._device).to(self._dtype)
        outputs = self._model(pixel_values=pixel_values)
        results = self._processor.post_process_object_detection(
            outputs.logits,
            outputs.pred_boxes,
            self.settings.confidence_threshold,
            [(image.height, image.width)],
            self._config.use_focal_loss,
        )
        result = results[0]
        bubbles: list[ScoredBox] = []
        texts: list[ScoredBox] = []
        for box, score, label in zip(result.boxes, result.scores, result.labels):
            # Koharu truncates each coordinate to a whole pixel (``as i32``).
            xyxy = tuple(float(int(value)) for value in box)
            if label == LABEL_BUBBLE:
                bubbles.append(ScoredBox(xyxy, score))
            elif label in (LABEL_TEXT_BUBBLE, LABEL_TEXT_FREE):
                texts.append(ScoredBox(xyxy, score))
        return bubbles, texts

    def _layout(self, image: Image.Image, bubble_boxes: list[ScoredBox], text_boxes: list[ScoredBox]) -> LayoutDetections:
        width, height = image.width, image.height
        detections: list[LayoutDetection] = []
        # Bubbles are output as detected (after Koharu's bounds filter); text is output as merged text blocks.
        for bubble in filter_and_fix_bboxes(bubble_boxes, width, height):
            detections.append(_rectangle(LABEL_BUBBLE, "bubble", bubble.score, bubble.xyxy))
        for block in create_text_blocks(width, height, text_boxes, bubble_boxes):
            label_id = TEXT_LABEL_IDS[block.text_class]
            detections.append(_rectangle(label_id, "text", block.score, block.xyxy))
        return LayoutDetections(image_width=width, image_height=height, detections=detections)


def _rectangle(label_id: int, label: str, score: float, xyxy) -> LayoutDetection:
    """A rectangular instance: the mask covers its box and is filled with 255."""
    x0, y0, x1, y1 = (int(round(value)) for value in xyxy)
    width = max(x1 - x0, 0)
    height = max(y1 - y0, 0)
    return LayoutDetection(
        label_id=label_id,
        label=label,
        score=float(score),
        bbox=(float(x0), float(y0), float(x1), float(y1)),
        area=width * height,
        mask=LayoutMask(x=x0, y=y0, width=width, height=height, pixels=np.full((height, width), 255, dtype=np.uint8)),
    )
