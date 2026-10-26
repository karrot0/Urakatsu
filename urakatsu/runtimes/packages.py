"""Native runtime packages: the newest official builds of llama.cpp, stable-diffusion.cpp,
and the runtimes behind the source systems (Bun for Mana bundles, Suwayomi Server).

Urakatsu always runs the latest runtimes (AGENTS.md). Python bindings vendor old
snapshots of these projects, so instead Urakatsu installs each project's own release
binaries and talks to its server (``llama-server``, ``sd-server``) over HTTP, the way
The upstream runtime installs its native packages.

Layout under ``URAKATSU_HOME/runtimes/<package>/``:

    <tag>-<variant>/      extracted release assets (binaries and their CUDA runtime DLLs)
    current.json          {"tag", "variant", "path", "installed_at", "checked_at"}

    pins.json             {"<package>": "<tag>"} for packages the user pinned

A newer release replaces the current one; the previous install is kept for rollback.
A pinned package stays on its tag and is never auto-updated; models that need a newer
release get a warning on the Models page instead of failing to load.
Every asset is checked against the SHA-256 digest GitHub publishes for it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import shutil
import ssl
import tarfile
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import httpx

from urakatsu.core.config import home

from .device import Backend, Device

log = logging.getLogger(__name__)

CHECK_INTERVAL = 6 * 3600  # seconds between update checks
RELEASES_PER_PAGE = 30
KEEP_INSTALLS = 2  # current plus the previous one
CUDA13_MIN_DRIVER = 580  # NVIDIA driver branch that supports CUDA 13


@dataclass(frozen=True)
class Variant:
    id: str
    assets: tuple[str, ...]  # one regex per required asset


@dataclass(frozen=True)
class Package:
    id: str
    name: str
    repository: str
    executable: str
    variants: tuple[Variant, ...]
    # Installed the first time something needs it; the updater only refreshes it once present.
    on_demand: bool = False

    def variant(self, variant_id: str) -> Variant:
        return next(v for v in self.variants if v.id == variant_id)


LLAMA_CPP = Package(
    id="llama.cpp",
    name="llama.cpp",
    repository="ggml-org/llama.cpp",
    executable="llama-server",
    variants=(
        Variant("win-cuda-13", (r"llama-b\d+-bin-win-cuda-13\.[\d.]+-x64\.zip", r"cudart-llama-bin-win-cuda-13\.[\d.]+-x64\.zip")),
        Variant("win-cuda-12", (r"llama-b\d+-bin-win-cuda-12\.[\d.]+-x64\.zip", r"cudart-llama-bin-win-cuda-12\.[\d.]+-x64\.zip")),
        Variant("win-rocm", (r"llama-b\d+-bin-win-rocm-[\d.]+-x64\.zip",)),
        Variant("win-vulkan", (r"llama-b\d+-bin-win-vulkan-x64\.zip",)),
        Variant("win-cpu", (r"llama-b\d+-bin-win-cpu-x64\.zip",)),
        Variant(
            "linux-cuda-13",
            (r"llama-b\d+-bin-ubuntu-cuda-13\.[\d.]+-x64\.tar\.gz", r"cudart-llama-b\d+-bin-ubuntu-cuda-13\.[\d.]+-x64\.tar\.gz"),
        ),
        Variant(
            "linux-cuda-12",
            (r"llama-b\d+-bin-ubuntu-cuda-12\.[\d.]+-x64\.tar\.gz", r"cudart-llama-b\d+-bin-ubuntu-cuda-12\.[\d.]+-x64\.tar\.gz"),
        ),
        Variant("linux-rocm", (r"llama-b\d+-bin-ubuntu-rocm-[\d.]+-x64\.tar\.gz",)),
        Variant("linux-vulkan", (r"llama-b\d+-bin-ubuntu-vulkan-x64\.tar\.gz",)),
        Variant("linux-cpu", (r"llama-b\d+-bin-ubuntu-x64\.tar\.gz",)),
        Variant("macos-arm64", (r"llama-b\d+-bin-macos-arm64\.tar\.gz",)),
        Variant("macos-x64", (r"llama-b\d+-bin-macos-x64\.tar\.gz",)),
    ),
)

STABLE_DIFFUSION_CPP = Package(
    id="stable-diffusion.cpp",
    name="stable-diffusion.cpp",
    repository="leejet/stable-diffusion.cpp",
    executable="sd-server",
    variants=(
        Variant("win-cuda-12", (r"sd-.+-bin-win-cuda12-x64\.zip", r"cudart-sd-bin-win-cu12-x64\.zip")),
        Variant("win-rocm", (r"sd-.+-bin-win-rocm-[\d.]+-x64\.zip",)),
        Variant("win-vulkan", (r"sd-.+-bin-win-vulkan-x64\.zip",)),
        Variant("win-cpu", (r"sd-.+-bin-win-cpu-x64\.zip",)),
        Variant("linux-rocm", (r"sd-.+-bin-Linux-.+-x86_64-rocm-[\d.]+\.zip",)),
        Variant("linux-vulkan", (r"sd-.+-bin-Linux-.+-x86_64-vulkan\.zip",)),
        Variant("linux-cpu", (r"sd-.+-bin-Linux-Ubuntu-[\d.]+-x86_64\.zip",)),
        Variant("macos-arm64", (r"sd-.+-bin-Darwin-.+-arm64\.zip",)),
    ),
)

# Mana bundles are JavaScriptCore code; Bun is JavaScriptCore too, so it runs them as the app does.
BUN = Package(
    id="bun",
    name="Bun",
    repository="oven-sh/bun",
    executable="bun",
    variants=(
        Variant("win-cpu", (r"bun-windows-x64-baseline\.zip",)),
        Variant("linux-cpu", (r"bun-linux-x64-baseline\.zip",)),
        Variant("macos-arm64", (r"bun-darwin-aarch64\.zip",)),
        Variant("macos-x64", (r"bun-darwin-x64-baseline\.zip",)),
    ),
    on_demand=True,
)

# The platform bundles ship their own Java runtime, so the executable is that java.
SUWAYOMI = Package(
    id="suwayomi-server",
    name="Suwayomi Server",
    repository="Suwayomi/Suwayomi-Server",
    executable="java",
    variants=(
        Variant("win-cpu", (r"Suwayomi-Server-v[\d.]+-windows-x64\.zip",)),
        Variant("linux-cpu", (r"Suwayomi-Server-v[\d.]+-linux-x64\.tar\.gz",)),
        Variant("macos-arm64", (r"Suwayomi-Server-v[\d.]+-macOS-arm64\.tar\.gz",)),
        Variant("macos-x64", (r"Suwayomi-Server-v[\d.]+-macOS-x64\.tar\.gz",)),
    ),
    on_demand=True,
)

PACKAGES = {package.id: package for package in (LLAMA_CPP, STABLE_DIFFUSION_CPP, BUN, SUWAYOMI)}


@lru_cache(maxsize=1)
def _tls_context() -> ssl.SSLContext:
    """One TLS context for every GitHub request: creating one reads the whole certificate store."""
    return httpx.create_ssl_context()


def preferred_variants(devices: list[Device]) -> list[str]:
    """Variant ids to try, best first, for the GPU the pipeline picks (as Koharu's ``discover``).

    CUDA for NVIDIA, ROCm for a supported AMD GPU once the ROCm libraries the builds load are
    present (``rocm.available``), then Vulkan for any GPU, then the CPU. On macOS, the build for
    the Mac's processor: Apple silicon, or Intel (``macos-x64``: llama.cpp, Bun and Suwayomi
    Server publish one; stable-diffusion.cpp does not).
    """
    system = platform.system()
    prefix = {"Windows": "win", "Linux": "linux", "Darwin": "macos"}.get(system, "")
    if prefix == "macos":
        return ["macos-arm64"] if platform.machine().lower() in ("arm64", "aarch64") else ["macos-x64"]
    from . import rocm
    from .hardware import rank

    accelerators = sorted((d for d in devices if d.is_accelerator), key=rank)
    first = accelerators[0] if accelerators else None
    order: list[str] = []
    if first is not None and first.backend is Backend.CUDA:
        driver = max((_driver_major(d.driver_version) for d in accelerators if d.backend is Backend.CUDA), default=0)
        if driver >= CUDA13_MIN_DRIVER:
            order.append(f"{prefix}-cuda-13")
        order.append(f"{prefix}-cuda-12")
    elif first is not None and first.backend is Backend.ROCM and rocm.available():
        order.append(f"{prefix}-rocm")
    if first is not None and first.backend is not Backend.DIRECTML:
        order.append(f"{prefix}-vulkan")
    order.append(f"{prefix}-cpu")
    return order


def _driver_major(version: str) -> int:
    try:
        return int(version.split(".")[0])
    except ValueError:
        return 0


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    size: int
    digest: str  # "sha256:<hex>"


@dataclass(frozen=True)
class Release:
    tag: str
    variant: str
    published: str
    assets: tuple[Asset, ...]


@dataclass
class Installed:
    package: str
    tag: str
    variant: str
    path: Path
    installed_at: float
    checked_at: float

    @property
    def executable(self) -> Path:
        name = PACKAGES[self.package].executable
        return find_executable(self.path, name)


def find_executable(root: Path, name: str) -> Path:
    candidates = [name + ".exe", name] if os.name == "nt" else [name]
    for candidate in candidates:
        for found in sorted(root.rglob(candidate)):
            if found.is_file():
                return found
    raise FileNotFoundError(f"{name} not found under {root}")


ProgressFn = Callable[[str, int, int], None]


class PackageManager:
    """Installs and updates the native runtime packages."""

    def __init__(self, root: Optional[Path] = None, offline: Optional[bool] = None, transport: Optional[httpx.BaseTransport] = None):
        self.root = Path(root) if root else home() / "runtimes"
        if offline is None:
            offline = os.environ.get("URAKATSU_OFFLINE", "").lower() in ("1", "true", "yes")
        self.offline = offline
        self._transport = transport
        self._lock = threading.Lock()

    # -- state ---------------------------------------------------------------------------------

    def _state_path(self, package: Package) -> Path:
        return self.root / package.id / "current.json"

    def installed(self, package: Package) -> Optional[Installed]:
        path = self._state_path(package)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            installed = Installed(package=package.id, path=self.root / package.id / data["path"], **{k: data[k] for k in ("tag", "variant", "installed_at", "checked_at")})
            installed.executable  # noqa: B018 - validates the install is complete
            return installed
        except (OSError, ValueError, KeyError, FileNotFoundError):
            return None

    def _write_state(self, package: Package, installed: Installed) -> None:
        path = self._state_path(package)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "tag": installed.tag,
                    "variant": installed.variant,
                    "path": installed.path.name,
                    "installed_at": installed.installed_at,
                    "checked_at": installed.checked_at,
                }
            ),
            encoding="utf-8",
        )
        tmp.replace(path)

    # -- pins ------------------------------------------------------------------------------------

    def _pins(self) -> dict[str, str]:
        try:
            return dict(json.loads((self.root / "pins.json").read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return {}

    def pin(self, package: Package) -> Optional[str]:
        return self._pins().get(package.id)

    def set_pin(self, package: Package, tag: Optional[str]) -> None:
        pins = self._pins()
        if tag:
            pins[package.id] = tag
        else:
            pins.pop(package.id, None)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "pins.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(pins), encoding="utf-8")
        tmp.replace(path)

    def local_installs(self, package: Package) -> list[tuple[str, str, Path]]:
        """(tag, variant, path) of every complete install on disk, newest first."""
        base = self.root / package.id
        if not base.exists():
            return []
        found = []
        for path in sorted((p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")), key=lambda p: p.stat().st_mtime, reverse=True):
            variant = next((v.id for v in package.variants if path.name.endswith("-" + v.id)), None)
            if variant is None:
                continue
            try:
                find_executable(path, package.executable)
            except FileNotFoundError:
                continue
            found.append((path.name[: -len(variant) - 1], variant, path))
        return found

    # -- releases --------------------------------------------------------------------------------

    def releases(self, package: Package, variants: list[str], limit: int = RELEASES_PER_PAGE) -> list[dict]:
        """Recent releases, newest first, each with the best build this machine could install from it."""
        with self._client() as client:
            response = client.get(f"https://api.github.com/repos/{package.repository}/releases", params={"per_page": limit})
            response.raise_for_status()
            releases = [r for r in response.json() if not r.get("draft")]
        releases.sort(key=lambda r: r.get("published_at") or "", reverse=True)
        result = []
        for release in releases:
            best = _best_build(package, release, variants)
            result.append({"tag": release["tag_name"], "published": release.get("published_at") or "", "variant": best.variant if best else None})
        return result

    def release(self, package: Package, tag: str, variants: list[str]) -> Release:
        with self._client() as client:
            response = client.get(f"https://api.github.com/repos/{package.repository}/releases/tags/{tag}")
            response.raise_for_status()
        best = _best_build(package, response.json(), variants)
        if best is None:
            raise RuntimeError(f"{package.name} {tag} has no build for {', '.join(variants) or 'this platform'}")
        return best

    def _client(self) -> httpx.Client:
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "urakatsu"}
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        # Building a TLS context loads the certificate store (~0.5 s on Windows); share one.
        verify = True if self._transport is not None else _tls_context()
        return httpx.Client(headers=headers, follow_redirects=True, timeout=httpx.Timeout(30.0, read=120.0), transport=self._transport, verify=verify)

    def latest(self, package: Package, variants: list[str]) -> Release:
        """The best variant this machine can run, from the newest release that ships all of its assets."""
        with self._client() as client:
            response = client.get(f"https://api.github.com/repos/{package.repository}/releases", params={"per_page": RELEASES_PER_PAGE})
            response.raise_for_status()
            releases = [r for r in response.json() if not r.get("draft")]
        releases.sort(key=lambda r: r.get("published_at") or "", reverse=True)
        known = [v for v in variants if any(variant.id == v for variant in package.variants)]
        # Best build for the hardware first, then the newest release that has all of its files:
        # a release whose assets are still uploading must not downgrade a GPU machine to CPU.
        for variant_id in known:
            for release in releases:
                best = _best_build(package, release, [variant_id])
                if best is not None:
                    return best
        raise RuntimeError(f"no {package.name} release has a build for {', '.join(known) or 'this platform'}")

    # -- install ---------------------------------------------------------------------------------

    def ensure(self, package: Package, devices: list[Device], progress: Optional[ProgressFn] = None, force: bool = False) -> Installed:
        """The installed package, updated first when a check is due and the network is allowed."""
        with self._lock:
            current = self.installed(package)
            pinned = self.pin(package)
            if pinned is not None:
                if current is not None and current.tag == pinned:
                    return current
                return self._switch(package, pinned, devices, progress, current)
            variants = preferred_variants(devices)
            best = next((v for v in variants if any(variant.id == v for variant in package.variants)), None)
            # A build worse than this machine supports (a GPU missed at an earlier check) is
            # re-checked now rather than at the next interval.
            settled = current is not None and current.variant == best
            if current is not None and (self.offline or (settled and not force and time.time() - current.checked_at < CHECK_INTERVAL)):
                return current
            if self.offline:
                raise RuntimeError(f"{package.name} is not installed and Urakatsu is offline; run `urakatsu runtimes update` online")
            try:
                release = self.latest(package, variants)
            except Exception as error:  # noqa: BLE001 - an unreachable GitHub must not block an installed runtime
                if current is not None:
                    log.warning("could not check for a newer %s: %s", package.name, error)
                    return current
                raise
            if current is not None and current.tag == release.tag and current.variant == release.variant:
                current.checked_at = time.time()
                self._write_state(package, current)
                return current
            installed = self._install(package, release, progress)
            log.info("installed %s %s (%s)", package.name, release.tag, release.variant)
            return installed

    def install_tag(self, package: Package, tag: str, devices: list[Device], progress: Optional[ProgressFn] = None) -> Installed:
        """Switches to ``tag`` (a rollback or a pin), reusing a kept install when there is one."""
        with self._lock:
            return self._switch(package, tag, devices, progress, self.installed(package))

    def _switch(self, package: Package, tag: str, devices: list[Device], progress: Optional[ProgressFn], current: Optional[Installed]) -> Installed:
        variants = preferred_variants(devices)
        for kept_tag, variant, path in self.local_installs(package):
            if kept_tag == tag and variant in variants:
                now = time.time()
                installed = Installed(package.id, tag, variant, path, now, now)
                os.utime(path)  # most recent, so pruning keeps it
                self._write_state(package, installed)
                return installed
        if self.offline:
            if current is not None:
                log.warning("%s %s is not on disk and Urakatsu is offline; staying on %s", package.name, tag, current.tag)
                return current
            raise RuntimeError(f"{package.name} {tag} is not installed and Urakatsu is offline")
        installed = self._install(package, self.release(package, tag, variants), progress)
        log.info("switched %s to %s (%s)", package.name, tag, installed.variant)
        return installed

    def _install(self, package: Package, release: Release, progress: Optional[ProgressFn]) -> Installed:
        base = self.root / package.id
        base.mkdir(parents=True, exist_ok=True)
        target = base / f"{release.tag}-{release.variant}"
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=base))
        try:
            with self._client() as client:
                for asset in release.assets:
                    archive = staging / asset.name
                    self._download(client, asset, archive, progress)
                    _extract(archive, staging / "files")
                    archive.unlink()
            find_executable(staging / "files", package.executable)
            if target.exists():
                shutil.rmtree(target)
            (staging / "files").rename(target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        now = time.time()
        installed = Installed(package.id, release.tag, release.variant, target, now, now)
        self._write_state(package, installed)
        self._prune(package, keep=target)
        return installed

    def _download(self, client: httpx.Client, asset: Asset, path: Path, progress: Optional[ProgressFn]) -> None:
        algorithm, _, expected = asset.digest.partition(":")
        if algorithm != "sha256" or not expected:
            raise RuntimeError(f"{asset.name} has no SHA-256 digest; refusing to install it")
        digest = hashlib.sha256()
        done = 0
        with client.stream("GET", asset.url, headers={"Accept": "application/octet-stream"}) as response:
            response.raise_for_status()
            with path.open("wb") as handle:
                for chunk in response.iter_bytes(1 << 20):
                    handle.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
                    if progress:
                        progress(asset.name, done, asset.size)
        if digest.hexdigest() != expected.lower():
            raise RuntimeError(f"{asset.name} failed its SHA-256 check")

    def _prune(self, package: Package, keep: Path) -> None:
        base = self.root / package.id
        # Only "<tag>-<variant>" folders are installs; "support/" and other caches are not.
        installs = sorted(
            (p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".") and any(p.name.endswith("-" + v.id) for v in package.variants)),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        survivors = [keep] + [p for p in installs if p != keep][: KEEP_INSTALLS - 1]
        for path in installs:
            if path not in survivors:
                shutil.rmtree(path, ignore_errors=True)


def _best_build(package: Package, release: dict, variants: list[str]) -> Optional[Release]:
    assets = {a["name"]: a for a in release.get("assets", [])}
    for variant_id in variants:
        if not any(v.id == variant_id for v in package.variants):
            continue
        matched = _match_assets(package.variant(variant_id), assets)
        if matched is not None:
            return Release(release["tag_name"], variant_id, release.get("published_at") or "", matched)
    return None


def _match_assets(variant: Variant, assets: dict[str, dict]) -> Optional[tuple[Asset, ...]]:
    matched: list[Asset] = []
    for pattern in variant.assets:
        regex = re.compile(pattern)
        hits = [a for name, a in assets.items() if regex.fullmatch(name)]
        if len(hits) != 1:
            return None
        hit = hits[0]
        matched.append(Asset(hit["name"], hit["browser_download_url"], int(hit.get("size") or 0), hit.get("digest") or ""))
    return tuple(matched)


def _extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()

    def safe(name: str) -> Path:
        target = (destination / name).resolve()
        if root not in target.parents and target != root:
            raise RuntimeError(f"{archive.name} contains an unsafe path: {name}")
        return target

    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                safe(member.filename)
            zf.extractall(destination)
    elif archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as tf:
            for member in tf.getmembers():
                safe(member.name)
            tf.extractall(destination, filter="data")
    else:
        raise RuntimeError(f"unsupported archive {archive.name}")
    # Executables in tarballs may lose their mode bits in zip extraction on POSIX.
    if os.name != "nt":
        for path in destination.rglob("*"):
            if path.is_file() and (path.name.startswith(("llama-", "sd-")) or path.name in ("bun", "java")) and "." not in path.name:
                path.chmod(path.stat().st_mode | 0o111)


def start_updater(manager: Optional[PackageManager] = None) -> Optional[threading.Thread]:
    """Keeps every package on its newest release while a service runs: checks now, then every interval."""
    manager = manager or PackageManager()
    if manager.offline:
        return None

    def loop() -> None:
        from .hardware import discover

        while True:
            devices = discover().candidates()
            for package in PACKAGES.values():
                if package.on_demand and manager.installed(package) is None:
                    continue
                try:
                    manager.ensure(package, devices)
                except Exception as error:  # noqa: BLE001 - a failed update never stops the service
                    log.warning("runtime update for %s failed: %s", package.name, error)
            time.sleep(CHECK_INTERVAL)

    thread = threading.Thread(target=loop, name="runtime-updater", daemon=True)
    thread.start()
    return thread
