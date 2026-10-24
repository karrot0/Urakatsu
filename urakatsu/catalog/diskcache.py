"""A small on-disk JSON cache for catalog lookups, so they survive a restart.

The model catalog asks the Hugging Face Hub and GitHub the same questions every time the app
starts (searches, a repo's files, GGUF headers, the newest runtime release). Answers are kept
here as one JSON file per key; callers decide how old an answer may be.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


class DiskCache:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).hexdigest()
        return self.root / digest[:2] / f"{digest}.json"

    def get(self, key: str) -> Optional[tuple[float, Any]]:
        """``(age in seconds, value)`` for ``key``, or None."""
        path = self._path(key)
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if stored.get("key") != key:  # a digest collision: treat as a miss
            return None
        return max(0.0, time.time() - float(stored.get("at", 0))), stored.get("value")

    def put(self, key: str, value: Any) -> None:
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
            tmp.write_text(json.dumps({"key": key, "at": time.time(), "value": value}), encoding="utf-8")
            tmp.replace(path)
        except (OSError, TypeError, ValueError) as error:  # the cache is an optimization only
            log.debug("could not cache %s: %s", key, error)
