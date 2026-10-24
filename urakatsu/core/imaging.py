"""Image conversions shared by providers and stages."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def to_rgb(image: Image.Image) -> Image.Image:
    return image if image.mode == "RGB" else image.convert("RGB")


def to_array(image: Image.Image, mode: str = "RGB") -> np.ndarray:
    if image.mode != mode:
        image = image.convert(mode)
    return np.asarray(image)


def from_array(array: np.ndarray, mode: str | None = None) -> Image.Image:
    array = np.ascontiguousarray(array)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if mode is None:
        if array.ndim == 2:
            mode = "L"
        else:
            mode = "RGBA" if array.shape[2] == 4 else "RGB"
    return Image.fromarray(array, mode)


def encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=3)
    return buffer.getvalue()


def encode(image: Image.Image, format: str = "PNG", **kwargs) -> bytes:
    buffer = io.BytesIO()
    fmt = format.upper()
    if fmt in ("JPEG", "JPG"):
        fmt = "JPEG"
        image = to_rgb(image)
        kwargs.setdefault("quality", 90)
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


def decode(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    image.load()
    return image


def mask_from_png(data: bytes) -> np.ndarray:
    return to_array(decode(data), "L")


def mask_to_png(mask: np.ndarray) -> bytes:
    return encode_png(from_array((mask > 0).astype(np.uint8) * 255, "L"))


def resize_longest(image: Image.Image, longest: int) -> Image.Image:
    width, height = image.size
    scale = longest / max(width, height)
    if scale >= 1.0:
        return image
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.LANCZOS)


def pad_to_multiple(array: np.ndarray, multiple: int) -> np.ndarray:
    """Reflect-pads an HxW(xC) array so both spatial sizes are multiples of ``multiple``."""
    height, width = array.shape[:2]
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    if pad_h == 0 and pad_w == 0:
        return array
    pads = [(0, pad_h), (0, pad_w)] + [(0, 0)] * (array.ndim - 2)
    return np.pad(array, pads, mode="reflect")


def thumbnail_png(image: Image.Image, longest: int = 256) -> bytes:
    return encode(resize_longest(to_rgb(image), longest), "JPEG", quality=80)
