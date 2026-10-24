"""Model result types shared by providers and stages.

``LayoutDetections`` mirrors the upstream ``detections``:
any detection provider returns instance-segmented regions labelled ``text``,
``onomatopoeia``, ``bubble`` or ``panel`` with a cropped mask per instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LayoutMask:
    x: int
    y: int
    width: int
    height: int
    pixels: np.ndarray  # (height, width) uint8, non-zero inside the instance

    def contains(self, x: int, y: int) -> bool:
        local_x, local_y = x - self.x, y - self.y
        if local_x < 0 or local_y < 0 or local_x >= self.width or local_y >= self.height:
            return False
        return bool(self.pixels[local_y, local_x] != 0)


@dataclass
class LayoutDetection:
    label_id: int
    label: str
    score: float
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    area: int
    mask: LayoutMask


@dataclass
class LayoutDetections:
    image_width: int
    image_height: int
    detections: list[LayoutDetection] = field(default_factory=list)


@dataclass
class LayoutThresholds:
    text: float
    onomatopoeia: float
    bubble: float
    panel: float

    def validate(self) -> None:
        for name in ("text", "onomatopoeia", "bubble", "panel"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} confidence threshold must be between 0 and 1")
