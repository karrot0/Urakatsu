"""App updates from GitHub releases: which build is running, what is newer, and installing it.

Releases (``.github/workflows/release.yml``) come in two channels:

- stable: a ``vX.Y.Z`` tag makes the release "Urakatsu X.Y.Z" (GitHub's "latest" release);
- nightly: the ``nightly`` pre-release, rebuilt from main each night, at ``target_commitish``.

``packaging/build.py`` writes ``urakatsu/_build.py`` into the wheel with the channel and commit
it was built from; a source checkout has no such file and reports the "dev" channel with its
git commit. The chosen channel lives in ``URAKATSU_HOME/updates.json``, defaulting to the
running build's own channel (stable for a dev build).

The service checks once an hour on a background thread. An update is available when the chosen
channel has a newer build: a higher version on stable, another commit on nightly, or simply
the other channel's build when the user switched channels. On Windows the desktop app installs
it: the installer is downloaded (its SHA-256 checked against GitHub's digest), and the desktop
app hands it to ``installer`` to run once the app has quit. Elsewhere the UI offers the download.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal, Optional

import httpx
from pydantic import BaseModel

from urakatsu import __version__
from urakatsu.core.config import home

log = logging.getLogger(__name__)

REPO = "karrot0/Urakatsu"
API = f"https://api.github.com/repos/{REPO}"
CHECK_EVERY = 3600  # seconds
CHANNELS = ("stable", "nightly")
Channel = Literal["stable", "nightly"]


# ---- the running build -----------------------------------------------------------------------------


class BuildInfo(BaseModel):
    version: str
    channel: str  # stable | nightly | dev
    commit: str = ""
    built: str = ""  # ISO 8601, UTC


@lru_cache(maxsize=1)
def build_info() -> BuildInfo:
    try:
        from urakatsu import _build  # written by packaging/build.py

        return BuildInfo(version=__version__, channel=_build.CHANNEL, commit=_build.COMMIT, built=_build.BUILT)
    except ImportError:
        return BuildInfo(version=__version__, channel="dev", commit=_git_commit())


def _git_commit() -> str:
    try:
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5, check=False)
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _version_key(version: str) -> tuple:
    """1.2.10 > 1.2.9; anything after the numbers (rc1, dev) sorts before the plain release."""
    numbers = [int(part) for part in re.findall(r"\d+", version.split("-")[0].split("+")[0])[:3]]
    return (*numbers, *([0] * (3 - len(numbers))), 0 if re.search(r"[a-z]", version, re.I) else 1)


# ---- releases ----------------------------------------------------------------------------------------


class Asset(BaseModel):
    name: str
    url: str
    size: int
    sha256: Optional[str] = None


class Release(BaseModel):
    id: str  # "v0.2.0", or "nightly@<commit>"
    channel: str
    version: str  # "0.2.0"; a nightly's version is its short commit
    commit: str = ""
    published: str = ""
    notes: str = ""
    url: str = ""  # the release page
    asset: Optional[Asset] = None  # the installer for this system


def _platform_asset(assets: list[dict]) -> Optional[Asset]:
    system = platform.system()
    machine = platform.machine().lower()
    arch = {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    for asset in assets:
        name = asset.get("name", "")
        wanted = (
            name.startswith("Urakatsu-Setup-") and name.endswith(".exe")
            if system == "Windows"
            else name.endswith(f"-macos-{arch}.dmg")
            if system == "Darwin"
            else name.endswith(f"-linux-{arch}.tar.gz")
        )
        if wanted:
            digest = asset.get("digest") or ""
            return Asset(
                name=name,
                url=asset["browser_download_url"],
                size=int(asset.get("size") or 0),
                sha256=digest.removeprefix("sha256:") if digest.startswith("sha256:") else None,
            )
    return None


def fetch_release(channel: Channel, client: Optional[httpx.Client] = None) -> Optional[Release]:
    """The newest release of ``channel``, or None when it has none yet."""
    headers = {"Accept": "application/vnd.github+json", "User-Agent": f"urakatsu/{__version__}", "X-GitHub-Api-Version": "2022-11-28"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    own = client is None
    client = client or httpx.Client(timeout=20.0, follow_redirects=True)
    try:
        response = client.get(f"{API}/releases/latest" if channel == "stable" else f"{API}/releases/tags/nightly", headers=headers)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
    finally:
        if own:
            client.close()
    notes = (data.get("body") or "").strip()
    if channel == "stable":
        tag = data.get("tag_name") or ""
        return Release(
            id=tag,
            channel="stable",
            version=tag.removeprefix("v"),
            published=data.get("published_at") or "",
            notes=notes[:4000],
            url=data.get("html_url") or "",
            asset=_platform_asset(data.get("assets") or []),
        )
    commit = data.get("target_commitish") or ""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):  # a branch name: read the commit from the notes
        found = re.search(r"Built from ([0-9a-f]{7,40})", notes)
        commit = found.group(1) if found else ""
    return Release(
        id=f"nightly@{commit[:12]}",
        channel="nightly",
        version=commit[:7],
        commit=commit,
        published=data.get("published_at") or "",
        notes=notes[:4000],
        url=data.get("html_url") or "",
        asset=_platform_asset(data.get("assets") or []),
    )


def is_newer(release: Release, build: BuildInfo) -> bool:
    """Whether ``release`` is an update for ``build``: always when it is another channel's build."""
    if build.channel in CHANNELS and release.channel != build.channel:
        return True
    if release.channel == "stable":
        return _version_key(release.version) > _version_key(build.version)
    if release.commit and build.commit:
        return not (release.commit.startswith(build.commit) or build.commit.startswith(release.commit))
    return bool(release.published and build.built and release.published > build.built)


# ---- the service ------------------------------------------------------------------------------------


class Install(BaseModel):
    status: Literal["idle", "downloading", "ready", "installing", "failed"] = "idle"
    done: int = 0
    total: int = 0
    error: Optional[str] = None


class UpdateStatus(BaseModel):
    current: BuildInfo
    channel: str  # the chosen channel
    checked_at: Optional[float] = None
    checking: bool = False
    error: Optional[str] = None
    latest: Optional[Release] = None
    available: bool = False
    switching: bool = False  # the update moves to the other channel
    dismissed: bool = False  # the user said "Later" to this release
    can_install: bool = False  # this app can install it itself (the Windows desktop app)
    install: Install = Install()


@dataclass
class _Settings:
    channel: Optional[str] = None
    dismissed: Optional[str] = None


class Updater:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.checked_at: Optional[float] = None
        self.checking = False
        self.error: Optional[str] = None
        self.latest: Optional[Release] = None
        self.install_state = Install()
        # Set by the desktop app: runs a downloaded installer once the app has quit.
        self.installer: Optional[Callable[[Path], None]] = None

    # -- settings --

    @property
    def _path(self) -> Path:
        return home() / "updates.json"

    def _settings(self) -> _Settings:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8-sig"))
            return _Settings(channel=data.get("channel"), dismissed=data.get("dismissed"))
        except (OSError, ValueError):
            return _Settings()

    def _save(self, settings: _Settings) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps({"channel": settings.channel, "dismissed": settings.dismissed}, indent=1), encoding="utf-8")

    def channel(self) -> Channel:
        chosen = self._settings().channel
        if chosen in CHANNELS:
            return chosen  # type: ignore[return-value]
        running = build_info().channel
        return running if running in CHANNELS else "stable"  # type: ignore[return-value]

    def set_channel(self, channel: str) -> UpdateStatus:
        if channel not in CHANNELS:
            raise ValueError(f"channel must be one of {', '.join(CHANNELS)}")
        settings = self._settings()
        if settings.channel != channel:
            settings.channel = channel
            self._save(settings)
            with self._lock:
                self.latest = None
                self.install_state = Install()
        return self.check()

    def dismiss(self) -> UpdateStatus:
        if self.latest is not None:
            settings = self._settings()
            settings.dismissed = self.latest.id
            self._save(settings)
        return self.status()

    # -- checking --

    def start(self) -> None:
        """Checks now and then every hour, on a daemon thread."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="urakatsu-updates", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            self.check()
            self._wake.wait(CHECK_EVERY)
            self._wake.clear()

    def check(self) -> UpdateStatus:
        with self._lock:
            if self.checking:
                return self.status()
            self.checking = True
        try:
            latest = fetch_release(self.channel())
            with self._lock:
                self.latest, self.error = latest, None
        except (httpx.HTTPError, ValueError, KeyError) as error:
            log.info("update check failed: %s", error)
            with self._lock:
                self.error = _describe(error)
        finally:
            with self._lock:
                self.checking = False
                self.checked_at = time.time()
        return self.status()

    def status(self) -> UpdateStatus:
        build = build_info()
        latest = self.latest
        available = latest is not None and is_newer(latest, build)
        return UpdateStatus(
            current=build,
            channel=self.channel(),
            checked_at=self.checked_at,
            checking=self.checking,
            error=self.error,
            latest=latest,
            available=available,
            switching=available and build.channel in CHANNELS and latest.channel != build.channel,
            dismissed=latest is not None and self._settings().dismissed == latest.id,
            can_install=available and self.installer is not None and latest.asset is not None and latest.asset.name.endswith(".exe"),
            install=self.install_state,
        )

    # -- installing --

    def install(self) -> UpdateStatus:
        """Downloads the installer in the background; the desktop app runs it and quits."""
        status = self.status()
        if not status.can_install or status.latest is None or status.latest.asset is None:
            raise ValueError("This update cannot be installed from the app; download it from the release page.")
        with self._lock:
            if self.install_state.status in ("downloading", "installing"):
                return self.status()
            self.install_state = Install(status="downloading", total=status.latest.asset.size)
        threading.Thread(target=self._install, args=(status.latest.asset,), name="urakatsu-update-install", daemon=True).start()
        return self.status()

    def _install(self, asset: Asset) -> None:
        folder = home() / "updates"
        target = folder / asset.name
        try:
            folder.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            digest = hashlib.sha256()
            with httpx.Client(timeout=httpx.Timeout(30.0, read=120.0), follow_redirects=True) as client:
                with client.stream("GET", asset.url, headers={"User-Agent": f"urakatsu/{__version__}"}) as response:
                    response.raise_for_status()
                    with partial.open("wb") as handle:
                        for chunk in response.iter_bytes(1 << 20):
                            handle.write(chunk)
                            digest.update(chunk)
                            self.install_state.done += len(chunk)
            if asset.sha256 and digest.hexdigest() != asset.sha256.lower():
                partial.unlink(missing_ok=True)
                raise ValueError("The download does not match its published SHA-256; try again.")
            partial.replace(target)
            self.install_state.status = "installing"
            installer = self.installer
            if installer is None:
                raise ValueError("This app cannot run the installer.")
            installer(target)
        except Exception as error:  # noqa: BLE001 - reported to the UI
            log.exception("update install failed")
            self.install_state.status, self.install_state.error = "failed", _describe(error)


def _describe(error: Exception) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        if status == 403:
            return "GitHub's rate limit was reached; Urakatsu tries again within the hour."
        return f"GitHub answered {status}."
    if isinstance(error, httpx.HTTPError):
        return "Could not reach GitHub."
    return str(error)


updater = Updater()
