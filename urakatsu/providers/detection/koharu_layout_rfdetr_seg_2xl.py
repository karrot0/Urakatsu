"""Layout detector: RF-DETR Seg 2XL.

Weights: ``mayocream/koharu-layout-rfdetr-seg-2xl-1152`` at a pinned revision (``inference_config.json``
and ``model.safetensors``). The network is upstream ``rfdetr``; pre/postprocessing lives in
``rfdetr/processor.py``. Torch is imported only in ``_load``/``_detect``.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from pydantic import Field

from urakatsu.core.errors import ProcessingError
from urakatsu.core.registry import register
from urakatsu.core.results import LayoutDetections, LayoutThresholds
from urakatsu.providers.base import DetectionProvider, ProviderSettings
from urakatsu.runtimes.store import HuggingFaceFile

log = logging.getLogger(__name__)

REPOSITORY = "mayocream/koharu-layout-rfdetr-seg-2xl-1152"
REVISION = "aed55fdb8ca953c6bec33cf6ed6dd52a9b72bfa2"

THRESHOLD_FIELDS = ("text_threshold", "bubble_threshold", "panel_threshold")


class KoharuLayoutRFDetrSeg2XLSettings(ProviderSettings):
    """``KoharuLayoutRFDetrSeg2XLConfig``: optional per-class confidence overrides."""

    text_threshold: Optional[float] = Field(default=None, description="Text confidence threshold (checkpoint default if unset).")
    bubble_threshold: Optional[float] = Field(default=None, description="Bubble confidence threshold (checkpoint default if unset).")
    panel_threshold: Optional[float] = Field(default=None, description="Panel confidence threshold (checkpoint default if unset).")


def _usable_threshold(name: str, value: Optional[float]) -> Optional[float]:
    """``Processor::new``: a stored threshold outside [0, 1] is a preference and falls back to the model default."""
    if value is None:
        return None
    if math.isfinite(value) and 0.0 <= value <= 1.0:
        return value
    log.warning("confidence threshold is not between zero and one; using the model default", extra={"label": name, "threshold": value})
    return None


@register
class KoharuLayoutRFDetrSeg2XLProvider(DetectionProvider):
    id = "koharu-layout-rfdetr-seg-2xl"
    name = "Koharu Layout RF-DETR Seg 2XL"
    description = "High-resolution manga layout instance segmentation (text, onomatopoeia, bubble, panel)."
    runtimes = ("torch",)

    catalog_queries = ('koharu-layout',)

    @classmethod
    def accepts(cls, facts) -> Optional[str]:
        """RF-DETR segmentation checkpoints that ship the inference_config.json layout."""
        is_rfdetr = facts.library == "rfdetr" or "rf-detr" in facts.tags or "rfdetr" in facts.tags
        if facts.runtime == "torch" and is_rfdetr and facts.has_file("inference_config.json") and facts.has_file("model.safetensors"):
            return "torch"
        return None
    requires = {"torch": ("rfdetr", "safetensors")}
    Settings = KoharuLayoutRFDetrSeg2XLSettings

    def __init__(self, settings: Any = None, runtimes=None, runtime: Optional[str] = None):
        super().__init__(settings, runtimes, runtime)
        # Normalise once at construction.
        self.settings = self.settings.model_copy(
            update={name: _usable_threshold(name, getattr(self.settings, name)) for name in THRESHOLD_FIELDS}
        )
        self._processor = None
        self._network = None
        self._thresholds: Optional[LayoutThresholds] = None
        self._device = None
        self._dtype = None

    @classmethod
    def weight_sources(cls, runtime: Optional[str] = None):
        return (
            HuggingFaceFile(REPOSITORY, "inference_config.json", REVISION),
            HuggingFaceFile(REPOSITORY, "model.safetensors", REVISION),
        )

    def _load(self) -> None:
        config_path, weights_path = self.resolve_weights()
        self._build(config_path, weights_path)

    def _build(self, config_path: Path, weights_path: Path) -> None:
        """``load`` + ``DetectionModel`` thresholds in the stage's ``Model::load``."""
        from .rfdetr.config import InferenceConfig
        from .rfdetr.network import load_network
        from .rfdetr.processor import KoharuLayoutRFDetrImageProcessor

        config = InferenceConfig.from_file(config_path)
        processor = KoharuLayoutRFDetrImageProcessor(config)
        runtime = self.runtime_instance()
        device = runtime.torch_device
        dtype = runtime.dtype()
        network = load_network(weights_path, config, device, dtype)

        recommended = processor.recommended_thresholds
        thresholds = LayoutThresholds(
            text=self.settings.text_threshold if self.settings.text_threshold is not None else recommended.text,
            onomatopoeia=recommended.onomatopoeia,
            bubble=self.settings.bubble_threshold if self.settings.bubble_threshold is not None else recommended.bubble,
            panel=self.settings.panel_threshold if self.settings.panel_threshold is not None else recommended.panel,
        )
        thresholds.validate()
        self._processor = processor
        self._network = network
        self._thresholds = thresholds
        self._device = device
        self._dtype = dtype

    def _unload(self) -> None:
        self._processor = None
        self._network = None
        self._thresholds = None
        self._device = None
        self._dtype = None

    def _detect(self, image: Image.Image) -> LayoutDetections:
        """``inference_with_thresholds``: no-grad preprocess, forward, postprocess."""
        import torch

        from .rfdetr.network import run_network

        if self._network is None or self._processor is None or self._thresholds is None:
            raise ProcessingError(f"{self.id} is not loaded")
        with torch.inference_mode():
            pixel_values = self._processor.preprocess(image, self._device)
            output = run_network(self._network, pixel_values, self._dtype)
            return self._processor.postprocess(output, image.width, image.height, self._thresholds)
