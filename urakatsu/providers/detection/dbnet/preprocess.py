"""Input preparation for the DBNet text detector.

Follows manga-image-translator's default detector for the non-rearranged path: a bilateral filter
(d=17, sigmaColor=80, sigmaSpace=80) on the full-resolution page, then an aspect-preserving resize whose
long side is ``detect_size`` (rounded as Python ``round``), zero (black) padding on the bottom and right
up to a multiple of 256, and ``x / 127.5 - 1`` normalization. The padding is added in uint8 before
normalization, so padded pixels become -1.0.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

# Total stride of the network: ResNet34 (32) times three stride-2 down blocks (8).
STRIDE_MULTIPLE = 256
BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE = 17, 80, 80


@dataclass(frozen=True)
class Prepared:
    """A page ready for the network.

    ``tensor`` is ``(1, 3, padded_height, padded_width)`` float32 on the CPU. The resized image occupies
    ``[:resized_height, :resized_width]``; the remaining ``pad_height`` rows and ``pad_width`` columns
    are padding. ``ratio`` is the uniform scale from original pixels to resized pixels.
    """

    tensor: torch.Tensor
    orig_width: int
    orig_height: int
    resized_width: int
    resized_height: int
    pad_width: int
    pad_height: int
    ratio: float

    @property
    def padded_width(self) -> int:
        return self.resized_width + self.pad_width

    @property
    def padded_height(self) -> int:
        return self.resized_height + self.pad_height


def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        raise ValueError(f"expected a uint8 image, got {image.dtype}")
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"expected an HxWx3 RGB image, got shape {image.shape}")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return np.ascontiguousarray(image)


def prepare(image_rgb: np.ndarray, detect_size: int = 2048) -> Prepared:
    """Filters, resizes (long side to ``detect_size``), pads to a multiple of 256, and normalizes a page.

    ``image_rgb`` is HxWx3 (or HxW, or HxWx4 with alpha dropped) uint8 in RGB order.
    """
    image = _as_rgb_uint8(image_rgb)
    orig_h, orig_w = image.shape[:2]
    filtered = cv2.bilateralFilter(image, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)

    ratio = detect_size / max(orig_h, orig_w)
    res_h = max(1, int(round(orig_h * ratio)))
    res_w = max(1, int(round(orig_w * ratio)))
    resized = cv2.resize(filtered, (res_w, res_h), interpolation=cv2.INTER_LINEAR)

    pad_h = (-res_h) % STRIDE_MULTIPLE
    pad_w = (-res_w) % STRIDE_MULTIPLE
    canvas = np.zeros((res_h + pad_h, res_w + pad_w, 3), dtype=np.uint8)
    canvas[:res_h, :res_w] = resized

    normalized = canvas.astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).unsqueeze(0)
    return Prepared(
        tensor=tensor,
        orig_width=orig_w,
        orig_height=orig_h,
        resized_width=res_w,
        resized_height=res_h,
        pad_width=pad_w,
        pad_height=pad_h,
        ratio=ratio,
    )
