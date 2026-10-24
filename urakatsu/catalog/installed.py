"""Installed catalog models: a registry plus background downloads.

An installed model is one variant of one Hugging Face repository at an exact commit:
a GGUF quantization (with its shards and vision projector), a diffusion weight file,
or a Transformers / ONNX snapshot. Files live in the same store layout as the defaults
(``<models folder>/hf/<repo>/<commit>/<path>``), and the registry,
``<models folder>/installed.json``, remembers each file's role so providers can swap them in.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

import httpx
from pydantic import BaseModel, Field

from urakatsu.runtimes.store import HuggingFaceFile, ModelStore

REGISTRY = "installed.json"


class InstalledFile(BaseModel):
    path: str
    role: str  # model | shard | projector | file, or a stable-diffusion.cpp slot (diffusion_model, llm, vae, ...)
    size: int = 0
    # Companions from another repository (a text encoder or VAE a diffusion model needs).
    repo: Optional[str] = None
    revision: Optional[str] = None


class InstalledModel(BaseModel):
    id: str
    repo: str
    revision: str
    runtime: str
    variant: str = ""
    files: list[InstalledFile] = Field(default_factory=list)
    architecture: Optional[str] = None
    family: Optional[str] = None  # stable-diffusion.cpp family doc id
    model_type: Optional[str] = None
    pipeline_tag: Optional[str] = None
    chat_template: bool = False
    projector: bool = False
    reasoning: bool = False
    size: int = 0
    installed_at: float = 0.0
    usable_by: list[dict] = Field(default_factory=list)  # [{"stage", "provider", "runtime"}]
    recipe: Optional[str] = None  # stable-diffusion.cpp recipe id

    def source(self, path: str) -> HuggingFaceFile:
        entry = next((f for f in self.files if f.path == path and f.repo), None)
        if entry is not None:
            return HuggingFaceFile(entry.repo, path, entry.revision or "main")
        return HuggingFaceFile(self.repo, path, self.revision)

    def source_for_role(self, role: str) -> Optional[HuggingFaceFile]:
        found = next((f for f in self.files if f.role == role), None)
        return self.source(found.path) if found else None

    def source_for_name(self, name: str) -> Optional[HuggingFaceFile]:
        if not name:
            return None
        found = next((f for f in self.files if f.path == name or f.path.endswith("/" + name)), None)
        return self.source(found.path) if found else None


class InstalledModels:
    def __init__(self, store: Optional[ModelStore] = None):
        self.store = store or ModelStore()

    @property
    def path(self) -> Path:
        return self.store.root / REGISTRY

    def all(self) -> list[InstalledModel]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [InstalledModel.model_validate(item) for item in data.get("models", [])]

    def get(self, model_id: str) -> Optional[InstalledModel]:
        return next((m for m in self.all() if m.id == model_id), None)

    def _write(self, models: list[InstalledModel]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"models": [m.model_dump(mode="json") for m in models]}, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, model: InstalledModel) -> None:
        models = [m for m in self.all() if m.id != model.id]
        models.append(model)
        self._write(models)

    def remove(self, model_id: str) -> None:
        models = self.all()
        doomed = next((m for m in models if m.id == model_id), None)
        if doomed is None:
            raise KeyError(f"model {model_id} is not installed")
        survivors = [m for m in models if m.id != model_id]

        def key(model: InstalledModel, file: InstalledFile) -> tuple[str, str, str]:
            source = model.source(file.path)
            return (source.repository, source.revision, file.path)

        shared = {key(m, f) for m in survivors for f in m.files}
        for file in doomed.files:
            if key(doomed, file) not in shared:
                doomed.source(file.path).local_path(self.store).unlink(missing_ok=True)
        for repo in {doomed.source(f.path).repository for f in doomed.files} | {doomed.repo}:
            _prune_empty(self.store.root / "hf" / repo)
        self._write(survivors)

    def complete(self, model: InstalledModel) -> bool:
        return all(model.source(f.path).local_path(self.store).exists() for f in model.files)


def _prune_empty(directory: Path) -> None:
    if not directory.exists():
        return
    for path in sorted(directory.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    if directory.exists() and not any(directory.iterdir()):
        directory.rmdir()


# ---- downloads -----------------------------------------------------------------------------------


class Download(BaseModel):
    id: str
    model: InstalledModel
    status: str = "queued"  # queued | downloading | completed | failed | cancelled
    done: int = 0
    total: int = 0
    current: str = ""
    error: Optional[str] = None
    started_at: float = 0.0
    finished_at: float = 0.0


class Downloader:
    """Runs installs one at a time in a background thread; resumes partial files."""

    def __init__(self, registry: InstalledModels, on_change: Optional[Callable[[Download], None]] = None):
        self.registry = registry
        self.on_change = on_change
        self.downloads: dict[str, Download] = {}
        self._cancel: set[str] = set()
        self._queue: list[str] = []
        self._lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None

    def start(self, model: InstalledModel) -> Download:
        with self._lock:
            for existing in self.downloads.values():
                if existing.model.id == model.id and existing.status in ("queued", "downloading"):
                    return existing
            download = Download(id=uuid.uuid4().hex[:12], model=model, total=sum(f.size for f in model.files))
            self.downloads[download.id] = download
            self._queue.append(download.id)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run, name="model-downloads", daemon=True)
                self._worker.start()
        return download

    def cancel(self, download_id: str) -> None:
        with self._lock:
            if download_id not in self.downloads:
                raise KeyError(f"download {download_id} not found")
            self._cancel.add(download_id)

    def _changed(self, download: Download) -> None:
        if self.on_change:
            self.on_change(download)

    def _run(self) -> None:
        while True:
            with self._lock:
                if not self._queue:
                    self._worker = None
                    return
                download = self.downloads[self._queue.pop(0)]
            self._install(download)

    def _install(self, download: Download) -> None:
        store = self.registry.store
        model = download.model
        download.status, download.started_at = "downloading", time.time()
        self._changed(download)
        headers = {"User-Agent": "urakatsu"}
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            with httpx.Client(follow_redirects=True, headers=headers, timeout=httpx.Timeout(30.0, read=120.0)) as client:
                base = 0
                for file in model.files:
                    source = model.source(file.path)
                    target = source.local_path(store)
                    download.current = file.path
                    if target.exists():
                        base += file.size
                        download.done = base
                        self._changed(download)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    partial = target.with_name(target.name + ".part")
                    have = partial.stat().st_size if partial.exists() else 0
                    request_headers = {"Range": f"bytes={have}-"} if have else {}
                    with client.stream("GET", source.url(), headers=request_headers) as response:
                        if response.status_code == 416:  # already complete
                            pass
                        else:
                            response.raise_for_status()
                            if have and response.status_code != 206:
                                have = 0  # server ignored the range; start over
                            with partial.open("ab" if have else "wb") as handle:
                                last = 0.0
                                for chunk in response.iter_bytes(1 << 20):
                                    if download.id in self._cancel:
                                        raise _Cancelled
                                    handle.write(chunk)
                                    have += len(chunk)
                                    download.done = base + have
                                    if time.monotonic() - last > 0.25:
                                        last = time.monotonic()
                                        self._changed(download)
                    shutil.move(str(partial), str(target))
                    base += file.size
            model.installed_at = time.time()
            self.registry.add(model)
            download.status = "completed"
        except _Cancelled:
            download.status = "cancelled"
        except Exception as error:  # noqa: BLE001 - reported to the UI
            download.status, download.error = "failed", str(error)
        finally:
            download.finished_at = time.time()
            self._cancel.discard(download.id)
            self._changed(download)


class _Cancelled(Exception):
    pass
