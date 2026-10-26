"""Preprocessing helpers shared by the inpainting processors.

Each helper names the upstream function it mirrors. Resampling filters follow the
fast_image_resize kernels upstream selects: ``Convolution(CatmullRom)`` and
``Convolution(Lanczos3)`` are antialiased Pillow filters; ``Interpolation(Bilinear)``
is a plain bilinear sample (no antialiasing), which OpenCV's INTER_LINEAR matches.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

CATMULL_ROM = Image.Resampling.BICUBIC  # Keys cubic, a = -0.5 (Catmull-Rom)
LANCZOS3 = Image.Resampling.LANCZOS
NEAREST = Image.Resampling.NEAREST


def symmetric_index(index: int, length: int) -> int:
    """Index of the mirrored sample: ``0 1 2 2 1 0 0 1 ...`` for period ``2 * length``."""
    index %= length * 2
    return index if index < length else length * 2 - index - 1


def symmetric_indices(length: int, output_length: int) -> np.ndarray:
    return np.fromiter((symmetric_index(i, length) for i in range(output_length)), dtype=np.int64, count=output_length)


def ceil_modulo(value: int, modulo: int) -> int:
    return value if value % modulo == 0 else (value // modulo + 1) * modulo


def pad_symmetric(tensor: torch.Tensor, out_height: int, out_width: int) -> torch.Tensor:
    """Symmetric padding of an NCHW tensor to ``out_height x out_width`` (bottom and right)."""
    height, width = tensor.shape[2], tensor.shape[3]
    rows = torch.from_numpy(symmetric_indices(height, out_height)).to(tensor.device)
    cols = torch.from_numpy(symmetric_indices(width, out_width)).to(tensor.device)
    return tensor.index_select(2, rows).index_select(3, cols)


def python_round(value: float) -> int:
    """Round half to even, as Python's ``round``."""
    return int(round(value))


def resize_pil(image: Image.Image, width: int, height: int, resample) -> Image.Image:
    if image.size == (width, height):
        return image.copy()
    return image.resize((width, height), resample)


def resize_bilinear_nonaliased(array: np.ndarray, width: int, height: int) -> np.ndarray:
    """``fast_image_resize::ResizeAlg::Interpolation(Bilinear)``: no antialiasing."""
    import cv2

    if array.shape[1] == width and array.shape[0] == height:
        return array.copy()
    return cv2.resize(array, (width, height), interpolation=cv2.INTER_LINEAR)


def to_device_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array)).to(device)


def apply_overlay(mask: np.ndarray, init_image: np.ndarray, generated: np.ndarray) -> np.ndarray:
    """``Flux2ImageProcessor::apply_overlay`` without a crop region: a soft blend by the (unbinarized) mask."""
    amount = (mask.astype(np.float32) / np.float32(255.0))[..., None]
    original = init_image.astype(np.float32)
    blended = generated.astype(np.float32) * amount + original * (np.float32(1.0) - amount)
    return np.floor(blended + np.float32(0.5)).clip(0.0, 255.0).astype(np.uint8)
