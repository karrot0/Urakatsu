"""Source-system registry: the plug-in point for scraper runtimes.

Built-in systems are listed in ``BUILTIN``; third-party packages add systems through the
``urakatsu.source_systems`` entry-point group (``name = "package.module:SystemClass"``).
Each system gets its own folder under ``URAKATSU_HOME/sources/<id>``.

Which systems are enabled lives in ``URAKATSU_HOME/sources/systems.json``. An enabled
system starts in the background with the service; any system also starts on first use.
"""

from __future__ import annotations

import atexit
import importlib
import importlib.metadata
import json
import logging
import threading
from pathlib import Path
from typing import Optional

from urakatsu.core.config import home
from urakatsu.runtimes.packages import PackageManager

from .base import SourceSystem, SystemStatus

log = logging.getLogger(__name__)

BUILTIN = (
    "urakatsu.sources.suwayomi:SuwayomiSystem",
    "urakatsu.sources.mana:ManaSystem",
)
ENTRY_POINTS = "urakatsu.source_systems"


def _load(spec: str) -> type[SourceSystem]:
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)


def discover() -> list[type[SourceSystem]]:
    classes = [_load(spec) for spec in BUILTIN]
    for entry in importlib.metadata.entry_points(group=ENTRY_POINTS):
        try:
            classes.append(entry.load())
        except Exception as error:  # noqa: BLE001 - one broken plug-in must not hide the others
            log.warning("could not load source system %s: %s", entry.name, error)
    return classes


class SourceSystems:
    def __init__(self, root: Optional[Path] = None, packages: Optional[PackageManager] = None, classes: Optional[list[type[SourceSystem]]] = None):
        self.root = Path(root) if root else home() / "sources"
        self.root.mkdir(parents=True, exist_ok=True)
        self.packages = packages or PackageManager()
        self._systems: dict[str, SourceSystem] = {}
        self._starters: dict[str, threading.Thread] = {}
        self._starters_lock = threading.Lock()
        for cls in classes if classes is not None else discover():
            if cls.id in self._systems:
                log.warning("source system %s registered twice; keeping the first", cls.id)
                continue
            self._systems[cls.id] = cls(self.root / cls.id, self.packages)
        # Systems run child processes (a Java server, Bun); never leave them behind.
        atexit.register(self.close)

    def get(self, system_id: str) -> SourceSystem:
        try:
            return self._systems[system_id]
        except KeyError as error:
            raise KeyError(f"no source system named {system_id!r}; known: {', '.join(self._systems)}") from error

    def all(self) -> list[SourceSystem]:
        return list(self._systems.values())

    # -- enabled systems ---------------------------------------------------------------------------

    def _settings_path(self) -> Path:
        return self.root / "systems.json"

    def enabled(self) -> set[str]:
        try:
            return set(json.loads(self._settings_path().read_text(encoding="utf-8")).get("enabled", []))
        except (OSError, ValueError):
            return set()

    def set_enabled(self, system_id: str, enabled: bool) -> SystemStatus:
        system = self.get(system_id)
        current = self.enabled()
        current.add(system_id) if enabled else current.discard(system_id)
        path = self._settings_path()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"enabled": sorted(current)}), encoding="utf-8")
        tmp.replace(path)
        if not enabled:
            system.stop()
        return self.status(system_id)

    def status(self, system_id: str) -> SystemStatus:
        status = self.get(system_id).status()
        status.enabled = system_id in self.enabled()
        if self.starting(system_id) and status.state != "running":
            status.state = "starting"
        return status

    def statuses(self) -> list[SystemStatus]:
        return [self.status(system.id) for system in self.all()]

    def start_background(self, system_id: str) -> threading.Thread:
        """Starts one system on its own thread (once at a time); failures are logged, not raised."""
        system = self.get(system_id)
        with self._starters_lock:
            running = self._starters.get(system_id)
            if running is not None and running.is_alive():
                return running

            def run() -> None:
                try:
                    system.start()
                except Exception as error:  # noqa: BLE001
                    log.warning("could not start source system %s: %s", system_id, error)

            thread = threading.Thread(target=run, name=f"source-system-{system_id}", daemon=True)
            self._starters[system_id] = thread
            thread.start()
            return thread

    def starting(self, system_id: str) -> bool:
        thread = self._starters.get(system_id)
        return thread is not None and thread.is_alive()

    def start_enabled(self) -> list[threading.Thread]:
        """Starts every enabled system, each on its own thread, so a slow one delays no other."""
        return [self.start_background(system_id) for system_id in sorted(self.enabled())]

    def close(self) -> None:
        for system in self.all():
            try:
                system.close()
            except Exception:  # noqa: BLE001
                log.debug("closing %s failed", system.id, exc_info=True)
