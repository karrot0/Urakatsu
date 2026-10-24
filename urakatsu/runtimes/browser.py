"""The browser runtime: a Chromium the source systems drive headlessly (Bun.WebView, over CDP).

Sites behind Cloudflare and similar checks only answer a real browser, so source requests
to them run inside one. Urakatsu uses, in order:

1. ``URAKATSU_BROWSER``: a Chrome, Edge, Brave or Chromium executable, or ``managed`` to
   always use Urakatsu's own Chrome for Testing;
2. an installed Chrome, Edge or Brave: they update themselves, so they are current;
3. the newest stable Chrome for Testing, installed on demand into
   ``URAKATSU_HOME/runtimes/chrome`` and kept on the newest stable release.

Chrome for Testing is downloaded over HTTPS from Google's storage and checked against the
MD5 that storage reports (``x-goog-hash``); Google publishes no other digest for it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import platform
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import httpx

from urakatsu.core.config import home

from .packages import CHECK_INTERVAL, KEEP_INSTALLS, _extract

log = logging.getLogger(__name__)

VERSIONS = "https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json"


@dataclass(frozen=True)
class Browser:
    path: Path
    kind: str  # "configured" | "system" | "managed"
    name: str
    version: Optional[str] = None


def _cft_platform() -> Optional[str]:
    system, machine = platform.system(), platform.machine().lower()
    if system == "Windows":
        return "win64" if machine in ("amd64", "x86_64", "arm64") else "win32"
    if system == "Linux":
        return "linux64" if machine in ("x86_64", "amd64") else None
    if system == "Darwin":
        return "mac-arm64" if machine == "arm64" else "mac-x64"
    return None


def _cft_executable(root: Path) -> Optional[Path]:
    names = ("chrome.exe", "chrome", "Google Chrome for Testing")
    for name in names:
        for found in sorted(root.rglob(name)):
            if found.is_file():
                return found
    return None


def system_browsers() -> list[Browser]:
    """Installed Chromium browsers, most preferred first."""
    found: list[Browser] = []
    if os.name == "nt":
        roots = [os.environ.get(v) for v in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")]
        candidates = [
            ("Google Chrome", r"Google\Chrome\Application\chrome.exe"),
            ("Microsoft Edge", r"Microsoft\Edge\Application\msedge.exe"),
            ("Brave", r"BraveSoftware\Brave-Browser\Application\brave.exe"),
        ]
        for name, relative in candidates:
            for root in roots:
                if root and (Path(root) / relative).is_file():
                    found.append(Browser(Path(root) / relative, "system", name))
                    break
    elif platform.system() == "Darwin":
        for name, app in (("Google Chrome", "Google Chrome"), ("Microsoft Edge", "Microsoft Edge"), ("Brave", "Brave Browser"), ("Chromium", "Chromium")):
            path = Path(f"/Applications/{app}.app/Contents/MacOS/{app}")
            if path.is_file():
                found.append(Browser(path, "system", name))
    else:
        for name, command in (("Google Chrome", "google-chrome-stable"), ("Google Chrome", "google-chrome"), ("Chromium", "chromium"), ("Chromium", "chromium-browser"), ("Microsoft Edge", "microsoft-edge")):
            path = shutil.which(command)
            if path and not any(b.name == name for b in found):
                found.append(Browser(Path(path), "system", name))
    return found


class BrowserRuntime:
    def __init__(self, root: Optional[Path] = None, offline: Optional[bool] = None):
        self.root = Path(root) if root else home() / "runtimes" / "chrome"
        if offline is None:
            offline = os.environ.get("URAKATSU_OFFLINE", "").lower() in ("1", "true", "yes")
        self.offline = offline
        self._lock = threading.Lock()

    # -- lookup ------------------------------------------------------------------------------------

    def _state(self) -> Optional[dict]:
        try:
            state = json.loads((self.root / "current.json").read_text(encoding="utf-8"))
            if (self.root / state["path"]).is_dir():
                return state
        except (OSError, ValueError, KeyError):
            pass
        return None

    def managed(self) -> Optional[Browser]:
        state = self._state()
        if state is None:
            return None
        executable = _cft_executable(self.root / state["path"])
        return Browser(executable, "managed", "Chrome for Testing", state["version"]) if executable else None

    def find(self) -> Optional[Browser]:
        """The browser to use, without installing anything."""
        configured = os.environ.get("URAKATSU_BROWSER", "")
        if configured == "managed":
            return self.managed()
        if configured and Path(configured).is_file():
            return Browser(Path(configured), "configured", Path(configured).stem)
        system = system_browsers()
        if system:
            return system[0]
        return self.managed()

    # -- install -----------------------------------------------------------------------------------

    def ensure(self, progress: Optional[Callable[[str, int, int], None]] = None) -> Browser:
        """The browser to use, installing or updating Chrome for Testing when no other is present."""
        found = self.find()
        if found is not None and found.kind != "managed":
            return found
        with self._lock:
            state = self._state()
            if state is not None and (self.offline or time.time() - state.get("checked_at", 0) < CHECK_INTERVAL):
                return self.managed()  # type: ignore[return-value]
            if self.offline:
                raise RuntimeError("no Chrome, Edge or Brave is installed and Urakatsu is offline, so Chrome for Testing cannot be installed")
            try:
                version, url = self.latest()
            except Exception as error:  # noqa: BLE001 - an unreachable index must not block an installed browser
                if state is not None:
                    log.warning("could not check for a newer Chrome for Testing: %s", error)
                    return self.managed()  # type: ignore[return-value]
                raise
            if state is not None and state["version"] == version:
                state["checked_at"] = time.time()
                self._write_state(state)
                return self.managed()  # type: ignore[return-value]
            self._install(version, url, progress)
            log.info("installed Chrome for Testing %s", version)
            return self.managed()  # type: ignore[return-value]

    def latest(self) -> tuple[str, str]:
        target = _cft_platform()
        if target is None:
            raise RuntimeError(f"Chrome for Testing has no build for {platform.system()} {platform.machine()}")
        response = httpx.get(VERSIONS, timeout=30.0, follow_redirects=True)
        response.raise_for_status()
        stable = response.json()["channels"]["Stable"]
        download = next(d["url"] for d in stable["downloads"]["chrome"] if d["platform"] == target)
        return stable["version"], download

    def _write_state(self, state: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / "current.tmp"
        tmp.write_text(json.dumps(state), encoding="utf-8")
        tmp.replace(self.root / "current.json")

    def _install(self, version: str, url: str, progress: Optional[Callable[[str, int, int], None]]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.root))
        try:
            archive = staging / url.rsplit("/", 1)[-1]
            digest = hashlib.md5()  # noqa: S324 - integrity check against the digest Google's storage reports
            expected: Optional[str] = None
            with httpx.stream("GET", url, timeout=httpx.Timeout(30.0, read=120.0), follow_redirects=True) as response:
                response.raise_for_status()
                for value in response.headers.get_list("x-goog-hash"):
                    for part in value.split(","):
                        key, _, encoded = part.strip().partition("=")
                        if key == "md5":
                            expected = base64.b64decode(encoded).hex()
                total = int(response.headers.get("content-length") or 0)
                done = 0
                with archive.open("wb") as handle:
                    for chunk in response.iter_bytes(1 << 20):
                        handle.write(chunk)
                        digest.update(chunk)
                        done += len(chunk)
                        if progress:
                            progress(archive.name, done, total)
            if expected is None:
                raise RuntimeError(f"{archive.name} came without a digest; refusing to install it")
            if digest.hexdigest() != expected:
                raise RuntimeError(f"{archive.name} failed its integrity check")
            _extract(archive, staging / "files")
            if _cft_executable(staging / "files") is None:
                raise RuntimeError(f"{archive.name} holds no Chrome executable")
            target = self.root / version
            if target.exists():
                shutil.rmtree(target)
            (staging / "files").rename(target)
            if os.name != "nt":
                for path in target.rglob("*"):
                    if path.is_file() and path.name in ("chrome", "Google Chrome for Testing", "chrome_crashpad_handler"):
                        path.chmod(path.stat().st_mode | 0o111)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        now = time.time()
        self._write_state({"version": version, "path": version, "installed_at": now, "checked_at": now})
        installs = sorted((p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")), key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in [p for p in installs if p.name != version][KEEP_INSTALLS - 1 :]:
            shutil.rmtree(stale, ignore_errors=True)
