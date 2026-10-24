"""Series covers for the Sources page, kept on disk at tile size.

Sources hand out full-size covers (MangaDex's average ~900 KB), and Suwayomi downloads a
cover again every time for series that are not in its library. The browser shows them as
small tiles over six connections, so every search waited seconds per cover and held up the
next request. Here each cover is fetched once, shrunk to tile size, and stored under
``<home>/cache/covers``; concurrent requests for one cover share its download. A search
prefetches its covers in parallel, so they are ready before the browser asks.
"""

from __future__ import annotations

import hashlib
import io
import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from PIL import Image as PILImage

from .base import Image

log = logging.getLogger(__name__)

# Tiles are ~130 px wide; 2x that for high-density screens, in WebP (every browser shows it).
TILE = (320, 480)
QUALITY = 82
MAX_AGE = 7 * 24 * 3600  # a cover is fetched again after a week; sources rarely change them
PREFETCHERS = 16  # a search page has ~20 covers; sources answer them in parallel


class CoverCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._inflight: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(PREFETCHERS, thread_name_prefix="cover")

    @staticmethod
    def key(system: str, source: str, url: str) -> str:
        return hashlib.blake2b(f"{system}\n{source}\n{url}".encode(), digest_size=16).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / key

    def cached(self, key: str) -> Optional[Image]:
        path = self._path(key)
        try:
            if time.time() - path.stat().st_mtime > MAX_AGE:
                return None
            data = path.read_bytes()
        except OSError:
            return None
        return Image(data, _media_type(data))

    def get(self, key: str, fetch: Callable[[], Optional[Image]]) -> Optional[Image]:
        """The cached cover for ``key``, else ``fetch()`` it (once, however many ask), shrink and store it."""
        found = self.cached(key)
        if found is not None:
            return found
        with self._lock:
            future = self._inflight.get(key)
            owner = future is None
            if owner:
                future = self._inflight[key] = Future()
        if not owner:
            return future.result()
        try:
            image = fetch()
            if image is not None:
                image = shrink(image)
                self._store(key, image.data)
            future.set_result(image)
            return image
        except BaseException as error:
            future.set_exception(error)
            raise
        finally:
            with self._lock:
                self._inflight.pop(key, None)

    def prefetch(self, covers: list[tuple[str, Callable[[], Optional[Image]]]]) -> None:
        """Starts fetching ``(key, fetch)`` covers that are neither cached nor on their way."""
        for key, fetch in covers:
            with self._lock:
                if key in self._inflight:
                    continue
            if self.cached(key) is None:
                self._pool.submit(self._quietly, key, fetch)

    def _quietly(self, key: str, fetch: Callable[[], Optional[Image]]) -> None:
        try:
            self.get(key, fetch)
        except Exception as error:  # noqa: BLE001 - the browser's own request reports it
            log.debug("prefetching a cover failed: %s", error)

    def _store(self, key: str, data: bytes) -> None:
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{key}.{uuid.uuid4().hex}.tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        except OSError as error:  # a full or read-only disk only costs the cache
            log.warning("could not cache a cover: %s", error)


def shrink(image: Image) -> Image:
    """``image`` at tile size as WebP; the original when it is already small or Pillow cannot read it."""
    try:
        picture = PILImage.open(io.BytesIO(image.data))
        if picture.width <= TILE[0] and picture.height <= TILE[1] and len(image.data) < 64 * 1024:
            return image
        picture.draft("RGB", TILE)  # JPEG decodes at a reduced scale directly
        picture = picture.convert("RGBA" if picture.mode in ("RGBA", "LA", "P") else "RGB")
        picture.thumbnail(TILE, PILImage.Resampling.LANCZOS)
        buffer = io.BytesIO()
        picture.save(buffer, format="WEBP", quality=QUALITY, method=4)
        return Image(buffer.getvalue(), "image/webp")
    except Exception:  # noqa: BLE001 - an unreadable cover is still served as the source sent it
        return image


def _media_type(data: bytes) -> str:
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "application/octet-stream"
