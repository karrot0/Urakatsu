"""Text detector: DBNet with a ResNet34 backbone, manga-image-translator's default detector.

Weights: ``detect-20241225.ckpt`` from manga-image-translator's ``beta-0.3`` GitHub release (the model
is not on Hugging Face), loaded strictly. The network and its input preparation live in ``dbnet/``.
Torch is imported only in ``_load`` and ``_detect``; the checkpoint is resolved in ``_load``, so
nothing is downloaded unless this detector is used.

This detector is not in Koharu. Its decoding reuses the Comic Text Detector port (``ctd/processor.py``)
rather than manga-image-translator's GPL-licensed post-processing:

- The DB probability map becomes text-line quads through CTD's ``extract_line_polygons``, with
  manga-image-translator's defaults: binarization at ``text_threshold`` 0.5, lines kept above
  ``box_threshold`` 0.7, unclip ratio 2.3 (CTD's unclip is the same area * ratio / perimeter offset).
- Lines are grouped into blocks and the mask head's output is refined around them with CTD's
  ``group_text_lines`` / ``refine_mask`` (``finalize_detection``), so blocks and masks follow Koharu's
  CTD rules, not manga-image-translator's textline merge and mask refinement.
- Very tall or wide pages (manga-image-translator's rule: long side over 2.5 x ``detect_size`` and an
  aspect ratio over 3) are cut into overlapping strips about twice as long as the short side. Each strip
  owns the rows up to the middle of its overlaps; lines are kept by their centre and the mask by row.
  manga-image-translator packs such strips side by side into square batches instead.

Like CTD, this detector finds text only: no ``onomatopoeia``, ``bubble`` or ``panel`` detections. A
block's score is the mean of its lines' DB scores.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np
from PIL import Image
from pydantic import Field

from urakatsu.core.errors import ProcessingError
from urakatsu.core.registry import register
from urakatsu.core.results import LayoutDetections
from urakatsu.providers.base import DetectionProvider, ProviderSettings
from urakatsu.runtimes.store import GitHubReleaseFile

from .comic_text_detector import text_detections

log = logging.getLogger(__name__)

RELEASE = GitHubReleaseFile("zyddnys/manga-image-translator", "beta-0.3", "detect-20241225.ckpt")

# manga-image-translator's DetectorConfig defaults.
DEFAULT_DETECT_SIZE = 2048
DEFAULT_TEXT_THRESHOLD = 0.5
DEFAULT_BOX_THRESHOLD = 0.7
DEFAULT_UNCLIP_RATIO = 2.3
# The network's total stride; prepare() pads to it.
MIN_DETECT_SIZE = 256
# Strips (det_rearrange_forward's trigger).
STRIP_SCALE_TRIGGER = 2.5
STRIP_ASPECT_TRIGGER = 3.0


class DBNetResNet34Settings(ProviderSettings):
    """manga-image-translator's detector options."""

    detect_size: int = Field(default=DEFAULT_DETECT_SIZE, description="Long side of the image the network sees, in pixels (default 2048).")
    text_threshold: float = Field(default=DEFAULT_TEXT_THRESHOLD, description="Probability above which a pixel is text (default 0.5).")
    box_threshold: float = Field(default=DEFAULT_BOX_THRESHOLD, description="Mean probability a text line needs to be kept (default 0.7).")
    unclip_ratio: float = Field(default=DEFAULT_UNCLIP_RATIO, description="How far each text line is grown around its core (default 2.3).")


def _usable(name: str, value, ok: bool, default):
    """An out-of-range stored option is a preference and falls back to its default (as RF-DETR's thresholds do)."""
    if ok:
        return value
    log.warning("detector option is out of range; using the default", extra={"option": name, "value": value})
    return default


def _probability(value: float) -> bool:
    return math.isfinite(value) and 0.0 <= value <= 1.0


@register
class DBNetResNet34Provider(DetectionProvider):
    id = "dbnet-resnet34"
    name = "DBNet ResNet34"
    description = "manga-image-translator's default text detector (text only): DBNet text lines and a text mask."
    runtimes = ("torch",)
    requires = {"torch": ("torchvision",)}
    Settings = DBNetResNet34Settings

    def __init__(self, settings: Any = None, runtimes=None, runtime: Optional[str] = None):
        super().__init__(settings, runtimes, runtime)
        s = self.settings
        self.settings = s.model_copy(
            update={
                "detect_size": _usable("detect_size", s.detect_size, s.detect_size >= MIN_DETECT_SIZE, DEFAULT_DETECT_SIZE),
                "text_threshold": _usable("text_threshold", s.text_threshold, _probability(s.text_threshold), DEFAULT_TEXT_THRESHOLD),
                "box_threshold": _usable("box_threshold", s.box_threshold, _probability(s.box_threshold), DEFAULT_BOX_THRESHOLD),
                "unclip_ratio": _usable("unclip_ratio", s.unclip_ratio, math.isfinite(s.unclip_ratio) and s.unclip_ratio > 0.0, DEFAULT_UNCLIP_RATIO),
            }
        )
        self._network = None
        self._device = None

    @classmethod
    def weight_sources(cls, runtime: Optional[str] = None):
        return (RELEASE,)

    def _load(self) -> None:
        import torch

        from .dbnet.network import load_network

        (weights_path,) = self.resolve_weights()
        device = self.runtime_instance().torch_device
        # manga-image-translator runs this network in float32.
        self._network = load_network(weights_path, device, torch.float32)
        self._device = device

    def _unload(self) -> None:
        self._network = None
        self._device = None

    def _detect(self, image: Image.Image) -> LayoutDetections:
        from .ctd.processor import finalize_detection

        if self._network is None:
            raise ProcessingError(f"{self.id} is not loaded")
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        height, width = rgb.shape[:2]
        if width == 0 or height == 0:
            raise ProcessingError("cannot detect text in an empty image")
        transpose = width > height
        page = np.ascontiguousarray(rgb.transpose(1, 0, 2)) if transpose else rgb
        mask, lines = self._detect_tall(page)
        if transpose:
            mask = np.ascontiguousarray(mask.T)
            lines = [_transposed(line) for line in lines]
        mask, blocks = finalize_detection(rgb, mask, lines)
        return text_detections(width, height, mask, blocks)

    def _detect_tall(self, page: np.ndarray):
        """The page (height >= width) as one pass, or as overlapping strips when it is very long."""
        height, width = page.shape[:2]
        size = self.settings.detect_size
        if not (height / size > STRIP_SCALE_TRIGGER and height / width > STRIP_ASPECT_TRIGGER):
            return self._detect_once(page)
        strip = 2 * width
        overlap = width // 2
        starts = list(range(0, height - strip, strip - overlap)) + [height - strip]
        mask = np.zeros((height, width), dtype=np.uint8)
        lines = []
        for index, start in enumerate(starts):
            end = start + strip
            # Each strip owns the rows from the middle of its previous overlap to the middle of its next one.
            own_top = 0 if index == 0 else (start + starts[index - 1] + strip) // 2
            own_bottom = height if index == len(starts) - 1 else (end + starts[index + 1]) // 2
            strip_mask, strip_lines = self._detect_once(page[start:end])
            mask[own_top:own_bottom] = strip_mask[own_top - start : own_bottom - start]
            for line in strip_lines:
                centre = float(line.quad[:, 1].mean()) + start
                if own_top <= centre < own_bottom:
                    lines.append(_shifted(line, start))
        return mask, lines

    def _detect_once(self, page: np.ndarray):
        """One network pass: the uint8 text mask and the text lines, both in ``page`` pixels."""
        import torch

        from .ctd.processor import extract_line_polygons
        from .dbnet.network import forward
        from .dbnet.preprocess import prepare

        height, width = page.shape[:2]
        prepared = prepare(page, self.settings.detect_size)
        with torch.inference_mode():
            raw = forward(self._network, prepared, self._device, torch.float32)
        db = raw.db[: prepared.resized_height, : prepared.resized_width]
        lines = extract_line_polygons(
            db,
            prepared.resized_width,
            prepared.resized_height,
            width,
            height,
            binary_threshold=self.settings.text_threshold,
            score_threshold=self.settings.box_threshold,
            unclip_ratio=self.settings.unclip_ratio,
        )
        # manga-image-translator's raw mask: probability x 255, truncated to uint8.
        mask = (np.clip(raw.mask, 0.0, 1.0) * 255.0).astype(np.uint8)
        return mask, lines


def _shifted(line, dy: int):
    from .ctd.processor import LinePolygon

    quad = line.quad.copy()
    quad[:, 1] += dy
    return LinePolygon(quad=quad, score=line.score)


def _transposed(line):
    from .ctd.processor import LinePolygon

    # Swapping x and y mirrors the quad; reversing the corners keeps their winding.
    return LinePolygon(quad=np.ascontiguousarray(line.quad[::-1, ::-1]), score=line.score)
