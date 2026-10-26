"""Atomic file replacement that also holds on Windows, and names safe on every file system.

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


def write_bytes(path: Path, data: bytes) -> None:
    """Writes ``path`` atomically, as :func:`write_text`."""
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_bytes(data)
        replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def safe_stem(label: str, extension: bool = True) -> str:
    """A file or folder name safe on every file system: no reserved characters, and no extension.

    With ``extension`` False, ``label`` is a name rather than a file name, and keeps its dots
    ("Chapter 12.5", "Dr.STONE")."""
    name = label.strip().rstrip(". ")
    stem, dot, suffix = name.rpartition(".")
    if extension and dot and stem and 1 <= len(suffix) <= 5 and suffix.isalnum():
        name = stem
    return "".join("_" if ch in '<>:"/\\|?*' or ord(ch) < 32 else ch for ch in name).strip()
