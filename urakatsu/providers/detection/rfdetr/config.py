"""Inference configuration of the layout checkpoint (port of ``config.rs``).

Parsing fails closed: a missing or malformed field is an error, never a silent default.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from urakatsu.core.errors import ModelLoadError
from urakatsu.core.results import LayoutThresholds


class _Thresholds(BaseModel):
    text: float
    onomatopoeia: float
    bubble: float
    panel: float

    def to_thresholds(self) -> LayoutThresholds:
        return LayoutThresholds(text=self.text, onomatopoeia=self.onomatopoeia, bubble=self.bubble, panel=self.panel)


class InferenceConfig(BaseModel):
    """``KoharuLayoutRFDetrSeg2XLConfig``: fields of ``inference_config.json``."""

    architecture: str
    rfdetr_version: str
    resolution: int
    num_select: int
    classes: dict[str, str]
    recommended_thresholds: _Thresholds
    checkpoint: str
    checkpoint_sha256: str
    source_checkpoint_sha256: str

    @classmethod
    def from_file(cls, path: Path | str) -> "InferenceConfig":
        path = Path(path)
        try:
            return cls.model_validate_json(path.read_bytes())
        except (OSError, ValueError) as error:
            raise ModelLoadError(f"failed to parse {path}: {error}") from error

    def class_names(self) -> list[str]:
        # config.rs class_names: keys "0".."n-1" in index order.
        return [self.classes[str(index)] for index in range(len(self.classes))]

    def thresholds(self) -> LayoutThresholds:
        return self.recommended_thresholds.to_thresholds()
