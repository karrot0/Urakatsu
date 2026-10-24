"""Atomic file replacement that also holds on Windows.

POSIX replaces a file at once, even while another thread or process has it open. Windows
refuses with PermissionError until every handle on the target is closed, so a reader in the
middle of ``read_text`` makes a concurrent save fail. ``replace`` waits those moments out.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

ATTEMPTS = 100
DELAY = 0.01  # seconds; a read holds the file for well under a millisecond


def replace(source: Path, target: Path) -> None:
    """``os.replace``, retried for up to a second while Windows reports ``target`` as in use."""
    for attempt in range(ATTEMPTS):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == ATTEMPTS - 1:
                raise
            time.sleep(DELAY)


def write_text(path: Path, text: str) -> None:
    """Writes ``path`` atomically: a uniquely named temporary file, then ``replace``."""
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
