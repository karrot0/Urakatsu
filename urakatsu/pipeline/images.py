"""Decoded image cache keyed by blob."""

from __future__ import annotations

import io
import threading
from typing import Optional

from PIL import Image

from urakatsu.scene import Snapshot


class ImageCache:
    def __init__(self) -> None:
        self._decoded: dict[str, Image.Image] = {}
        self._lock = threading.Lock()

    def get(self, scene: Snapshot, entity: str, role: str) -> Optional[Image.Image]:
        asset = scene.asset(entity, role)
        if asset is None:
            return None
        with self._lock:
            cached = self._decoded.get(asset.blob)
        if cached is not None:
            return cached
        data = scene.blobs.read(asset.blob)
        image = Image.open(io.BytesIO(data))
        image.load()
        with self._lock:
            self._decoded[asset.blob] = image
        return image
