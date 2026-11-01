"""Immutable model store with HuggingFace downloads.

Weights are addressed by ``(repository, revision, filename)`` and cached under the models
folder (``URAKATSU_HOME/models`` unless one is chosen, see ``models_folder``). A download streams to a ``.part`` file and is renamed
atomically on success, so a partially downloaded file is never mistaken for a
complete one.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol

import httpx

from urakatsu.core.errors import ModelLoadError
from urakatsu.runtimes.models_folder import models_root

DownloadProgress = Callable[[str, int, int], None]


class WeightSource(Protocol):
    """Anything that can resolve to a local file path."""

    def describe(self) -> str: ...

    def resolve(self, store: "ModelStore", progress: Optional[DownloadProgress] = None) -> Path: ...


@dataclass(frozen=True)
class LocalFile:
    path: str

    def describe(self) -> str:
        return self.path

    def resolve(self, store: "ModelStore", progress: Optional[DownloadProgress] = None) -> Path:
        path = Path(self.path).expanduser()
        if not path.exists():
            raise ModelLoadError(f"weights not found at {path}")
        return path


@dataclass(frozen=True)
class HuggingFaceFile:
    repository: str
    filename: str
    revision: str = "main"
    # "model", or "dataset" for a dataset repository.
    repo_type: str = "model"

    def describe(self) -> str:
        kind = "datasets/" if self.repo_type == "dataset" else ""
        return f"hf://{kind}{self.repository}@{self.revision}/{self.filename}"

    def url(self) -> str:
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        kind = "datasets/" if self.repo_type == "dataset" else ""
        return f"{endpoint}/{kind}{self.repository}/resolve/{self.revision}/{self.filename}"

    def local_path(self, store: "ModelStore") -> Path:
        safe_revision = self.revision.replace("/", "_")
        root = store.root / "hf" / "datasets" if self.repo_type == "dataset" else store.root / "hf"
        return root / self.repository / safe_revision / self.filename

    def resolve(self, store: "ModelStore", progress: Optional[DownloadProgress] = None) -> Path:
        target = self.local_path(store)
        if target.exists():
            return target
        if store.offline:
            raise ModelLoadError(f"{self.describe()} is not cached and the store is offline")
        store.download(self.url(), target, label=self.describe(), progress=progress)
        return target


@dataclass(frozen=True)
class GitHubReleaseFile:
    """One asset of a GitHub release, pinned by tag."""

    repository: str
    tag: str
    asset: str

    def describe(self) -> str:
        return f"github://{self.repository}@{self.tag}/{self.asset}"

    def url(self) -> str:
        return f"https://github.com/{self.repository}/releases/download/{self.tag}/{self.asset}"

    def local_path(self, store: "ModelStore") -> Path:
        return store.root / "github" / self.repository / self.tag / self.asset

    def resolve(self, store: "ModelStore", progress: Optional[DownloadProgress] = None) -> Path:
        target = self.local_path(store)
        if target.exists():
            return target
        if store.offline:
            raise ModelLoadError(f"{self.describe()} is not cached and the store is offline")
        store.download(self.url(), target, label=self.describe(), progress=progress)
        return target


class _Progressing(io.RawIOBase):
    """Reads ``response``, reporting the bytes read so far: ``tarfile`` only sees a stream."""

    def __init__(self, response, report: Callable[[int], None]):
        self.response = response
        self.report = report
        self.done = 0

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        chunk = self.response.read(size)
        self.done += len(chunk)
        self.report(self.done)
        return chunk


@dataclass(frozen=True)
class ContainerLayerFile:
    """One file inside a layer of a public Docker Hub image, pinned by layer digest and file hash.

    For weights published nowhere else. The layer is streamed and only ``member`` is saved, under
    its own name: nothing in the image runs, and no path from the archive is used. A file whose
    hash is not ``sha256`` is discarded.
    """

    repository: str
    digest: str
    member: str
    sha256: str
    size: int = 0  # of the layer, for progress

    def describe(self) -> str:
        return f"docker://{self.repository}@{self.digest}/{self.member}"

    def local_path(self, store: "ModelStore") -> Path:
        return store.root / "docker" / self.repository / self.digest.removeprefix("sha256:") / Path(self.member).name

    def resolve(self, store: "ModelStore", progress: Optional[DownloadProgress] = None) -> Path:
        target = self.local_path(store)
        if target.exists():
            return target
        if store.offline:
            raise ModelLoadError(f"{self.describe()} is not cached and the store is offline")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        report = (lambda done: progress(self.describe(), done, self.size)) if progress else (lambda done: None)
        try:
            self._extract(partial, report)
            shutil.move(str(partial), str(target))
        except (OSError, tarfile.TarError, ValueError, KeyError) as error:
            partial.unlink(missing_ok=True)
            raise ModelLoadError(f"failed to download {self.describe()}: {error}") from error
        return target

    def _extract(self, partial: Path, report: Callable[[int], None]) -> None:
        scope = f"repository:{self.repository}:pull"
        with urllib.request.urlopen(f"https://auth.docker.io/token?service=registry.docker.io&scope={scope}", timeout=60) as response:
            token = json.load(response)["token"]
        request = urllib.request.Request(f"https://registry-1.docker.io/v2/{self.repository}/blobs/{self.digest}", headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(request, timeout=120) as response, tarfile.open(fileobj=_Progressing(response, report), mode="r|gz") as layer:
            for member in layer:
                if member.isfile() and member.name == self.member:
                    digest = hashlib.sha256()
                    source = layer.extractfile(member)
                    with partial.open("wb") as handle:
                        while chunk := source.read(1 << 20):
                            digest.update(chunk)
                            handle.write(chunk)
                    if digest.hexdigest() != self.sha256:
                        raise ValueError(f"unexpected SHA-256 {digest.hexdigest()}")
                    return
        raise ValueError(f"{self.member} is not in the layer")


class ModelStore:
    def __init__(self, root: Path | None = None, offline: bool | None = None):
        self._root = Path(root) if root else None
        env_offline = os.environ.get("URAKATSU_OFFLINE", "").lower() in ("1", "true", "yes")
        self.offline = env_offline if offline is None else offline

    @property
    def root(self) -> Path:
        """An explicit root, else the models folder chosen in Settings (read live, so a change applies everywhere)."""
        return self._root or models_root()

    def cached(self, source: WeightSource) -> bool:
        if isinstance(source, (HuggingFaceFile, GitHubReleaseFile, ContainerLayerFile)):
            return source.local_path(self).exists()
        if isinstance(source, LocalFile):
            return Path(source.path).expanduser().exists()
        return False

    def download(
        self,
        url: str,
        target: Path,
        label: str = "",
        progress: Optional[DownloadProgress] = None,
    ) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        headers = {}
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token and "huggingface" in url:
            headers["Authorization"] = f"Bearer {token}"
        try:
            with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, read=120.0)) as client:
                with client.stream("GET", url, headers=headers) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("content-length", "0") or 0)
                    done = 0
                    with partial.open("wb") as handle:
                        for chunk in response.iter_bytes(chunk_size=1 << 20):
                            handle.write(chunk)
                            done += len(chunk)
                            if progress:
                                progress(label or url, done, total)
            shutil.move(str(partial), str(target))
        except (httpx.HTTPError, OSError) as error:  # OSError: a full disk, or a full Google Drive
            partial.unlink(missing_ok=True)
            raise ModelLoadError(f"failed to download {label or url}: {error}") from error
        return target

    def usage(self) -> int:
        if not self.root.exists():
            return 0
        return sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file())
