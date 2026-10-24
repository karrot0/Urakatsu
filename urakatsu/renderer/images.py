"""Bounded decoded-image storage and validation.

Port of ``images.rs``: an LRU cache of decoded RGBA images with a byte
budget (``DEFAULT_IMAGE_CACHE_BYTES``), and ``decode`` with the expected-size check.
"""

from __future__ import annotations

import io
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image as PILImage

DEFAULT_IMAGE_CACHE_BYTES = 512 * 1024 * 1024


@dataclass
class DecodedImage:
    width: int
    height: int
    encoded: bytes
    pixels: np.ndarray  # (height, width, 4) uint8, unpremultiplied RGBA

    def byte_len(self) -> int:
        return len(self.encoded) + int(self.pixels.nbytes)


class ImageCache:
    """``ImageCache``: least-recently-used eviction once ``max_bytes`` is exceeded."""

    def __init__(self, max_bytes: int = DEFAULT_IMAGE_CACHE_BYTES) -> None:
        self.max_bytes = max_bytes
        self.bytes = 0
        self._entries: "OrderedDict[str, DecodedImage]" = OrderedDict()

    def get(self, blob: str) -> Optional[DecodedImage]:
        image = self._entries.get(blob)
        if image is not None:
            self._entries.move_to_end(blob)
        return image

    def insert(self, blob: str, image: DecodedImage) -> None:
        previous = self._entries.pop(blob, None)
        if previous is not None:
            self.bytes -= previous.byte_len()
        if image.byte_len() > self.max_bytes:
            return
        self.bytes += image.byte_len()
        self._entries[blob] = image
        while self.bytes > self.max_bytes and self._entries:
            _, removed = self._entries.popitem(last=False)
            self.bytes -= removed.byte_len()


def decode(blob: str, encoded: bytes, expected: Optional[tuple[int, int]]) -> DecodedImage:
    """``decode``: RGBA8 decode; fails when the decoded size differs from ``expected``."""
    pil = PILImage.open(io.BytesIO(encoded)).convert("RGBA")
    if expected is not None and expected != pil.size:
        raise ValueError(f"blob {blob} decoded as {pil.width}x{pil.height}, expected {expected}")
    pixels = np.ascontiguousarray(np.asarray(pil, dtype=np.uint8))
    return DecodedImage(width=pil.width, height=pil.height, encoded=encoded, pixels=pixels)
