"""Where model weights live, and moving them to a new folder.

The folder defaults to ``URAKATSU_HOME/models``; a chosen one is kept in
``URAKATSU_HOME/storage.json``. Every ``ModelStore`` without an explicit root reads it live,
so a change applies to the catalog, the runtimes and every provider at once.

Moving copies first and switches the folder only once every file arrived, so a failed move
(a full disk, say) leaves the old folder in use and untouched. Files on the same drive are
renamed instead of copied. Originals are deleted after the switch; one a running model holds
open stays behind and is reported.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from urakatsu.core.config import home

log = logging.getLogger(__name__)

SETTINGS = "storage.json"

_lock = threading.Lock()
_root: Optional[Path] = None


def default_root() -> Path:
    return home() / "models"


def _settings_path() -> Path:
    return home() / SETTINGS


def _chosen() -> Optional[Path]:
    try:
        value = json.loads(_settings_path().read_text(encoding="utf-8-sig")).get("models_dir")  # -sig: a hand-edited file may carry a BOM
    except (OSError, ValueError):
        return None
    return Path(value) if value else None


def models_root() -> Path:
    global _root
    with _lock:
        if _root is None:
            _root = _chosen() or default_root()
        return _root


def set_models_root(path: Optional[Path]) -> Path:
    """Use ``path`` from now on (None means the default) and remember it."""
    global _root
    settings = _settings_path()
    settings.parent.mkdir(parents=True, exist_ok=True)
    chosen = None if path is None or _same(path, default_root()) else str(path)
    settings.write_text(json.dumps({"models_dir": chosen}, indent=1), encoding="utf-8")
    with _lock:
        _root = Path(chosen) if chosen else default_root()
        return _root


def _same(a: Path, b: Path) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _inside(child: Path, parent: Path) -> bool:
    child_s, parent_s = os.path.normcase(os.path.abspath(child)), os.path.normcase(os.path.abspath(parent))
    return child_s.startswith(parent_s.rstrip("\\/") + os.sep)


def existing_ancestor(path: Path) -> Path:
    """The nearest folder of ``path`` that exists, for disk-space questions about a folder not made yet."""
    path = Path(os.path.abspath(path))
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def folder_size(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            pass
    return total


def validate(target: Path) -> Path:
    """An absolute, writable folder that does not nest with the current one."""
    if not target.is_absolute():
        raise ValueError("Choose a full folder path, such as D:\\Urakatsu\\models.")
    current = models_root()
    if _inside(target, current) or _inside(current, target):
        raise ValueError("The new folder cannot be inside the current models folder, or contain it.")
    if target.exists() and not target.is_dir():
        raise ValueError(f"{target} is a file, not a folder.")
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".urakatsu-write-test"
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as error:
        raise ValueError(f"Urakatsu cannot write to {target}: {error.strerror or error}") from error
    return target


class Move(BaseModel):
    source: str
    target: str
    status: str = "moving"  # moving | completed | failed
    done: int = 0
    total: int = 0
    current: str = ""
    error: Optional[str] = None
    left_behind: int = 0  # originals that could not be deleted (held open by a running model)
    started_at: float = 0.0
    finished_at: float = 0.0


class Mover:
    """One move at a time, in a background thread."""

    def __init__(self) -> None:
        self.move: Optional[Move] = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self.move is not None and self.move.status == "moving"

    def start(self, target: Path) -> Move:
        with self._lock:
            if self.running:
                raise ValueError("Models are already moving to a new folder.")
            source = models_root()
            files = [p for p in source.rglob("*") if p.is_file()] if source.exists() else []
            total = sum(p.stat().st_size for p in files)
            same_drive = os.path.splitdrive(os.path.abspath(source))[0].lower() == os.path.splitdrive(os.path.abspath(target))[0].lower()
            if not same_drive:
                free = shutil.disk_usage(existing_ancestor(target)).free
                if total > free:
                    raise ValueError(f"The new folder's drive has {free / 1024**3:.1f} GB free; the models need {total / 1024**3:.1f} GB.")
            self.move = Move(source=str(source), target=str(target), total=total, started_at=time.time())
            threading.Thread(target=self._run, args=(self.move, source, target, files), name="models-move", daemon=True).start()
            return self.move

    def _run(self, move: Move, source: Path, target: Path, files: list[Path]) -> None:
        renamed: list[tuple[Path, Path]] = []
        copied: list[tuple[Path, Path]] = []
        present: list[Path] = []  # already in the target with the same size
        try:
            for path in files:
                destination = target / path.relative_to(source)
                move.current = str(path.relative_to(source))
                size = path.stat().st_size
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and destination.stat().st_size == size:
                    present.append(path)  # already there; drop the original afterwards
                else:
                    try:
                        os.replace(path, destination)
                        renamed.append((path, destination))
                    except OSError:  # another drive, or held open: copy and delete later
                        shutil.copy2(path, destination)
                        copied.append((path, destination))
                move.done += size
        except Exception as error:  # noqa: BLE001 - reported to the UI
            for original, moved in reversed(renamed):
                try:
                    os.replace(moved, original)
                except OSError:
                    log.warning("could not return %s to %s", moved, original)
            for _, duplicate in copied:
                duplicate.unlink(missing_ok=True)
            move.status, move.error = "failed", str(error)
            move.finished_at = time.time()
            return
        set_models_root(target)
        for original in [original for original, _ in copied] + present:
            try:
                original.unlink()
            except OSError:
                move.left_behind += 1
        _prune_empty(source)
        move.status, move.current = "completed", ""
        move.finished_at = time.time()


def _prune_empty(directory: Path) -> None:
    if not directory.exists():
        return
    for path in sorted(directory.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        try:
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        except OSError:
            pass


mover = Mover()


_picker = threading.Lock()


def pick_folder(initial: Optional[Path] = None) -> Optional[str]:
    """Opens the system folder picker on this machine (Urakatsu runs locally); None when cancelled."""
    import tkinter
    from tkinter import filedialog

    if not _picker.acquire(blocking=False):
        raise ValueError("A folder picker is already open.")
    root = None
    try:
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askdirectory(parent=root, initialdir=str(existing_ancestor(initial)) if initial else None, title="Choose the models folder", mustexist=False)
        return os.path.normpath(chosen) if chosen else None
    finally:
        if root is not None:
            root.destroy()
        _picker.release()
