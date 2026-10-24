"""Hugging Face Hub client for the model catalog: search, model details, GGUF headers.

Only public endpoints are used (``/api/models``); ``HF_TOKEN`` is sent when set, for
gated or private repositories. Responses are reused for CACHE_SECONDS. With a cache folder
they also survive a restart: an answer up to STALE_SECONDS old is shown at once and
refreshed in the background, and GGUF headers (fixed for a commit) are kept for good.
"""

from __future__ import annotations

import os
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx

from .diskcache import DiskCache

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
CACHE_SECONDS = 300
STALE_SECONDS = 24 * 3600  # an older saved answer is refetched before it is shown
HEADER_CHUNK = 256 << 10  # bytes of a GGUF file fetched first to read its header (doubled as needed)
HEADER_LIMIT = 64 << 20  # give up rather than download most of a model

# Search filters per runtime. The Hub's `filter` matches tags and the library name.
RUNTIME_FILTERS: dict[str, dict[str, Any]] = {
    "llama": {"filter": "gguf"},
    "diffusion": {"filter": ["gguf", "diffusion-single-file", "diffusers"]},
    "torch": {"filter": ["transformers", "safetensors"]},
    "onnx": {"filter": "onnx"},
}


@dataclass
class RepoFile:
    path: str
    size: int


@dataclass
class RepoInfo:
    id: str
    sha: str
    author: str
    pipeline_tag: Optional[str]
    library: Optional[str]
    tags: list[str]
    downloads: int
    likes: int
    last_modified: str
    gated: bool
    base_models: list[str]
    gguf: dict[str, Any]
    config: dict[str, Any]
    files: list[RepoFile] = field(default_factory=list)


class HubClient:
    def __init__(self, transport: Optional[httpx.BaseTransport] = None, cache_dir: Optional[Path] = None):
        headers = {"User-Agent": "urakatsu"}
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(base_url=ENDPOINT, headers=headers, follow_redirects=True, timeout=httpx.Timeout(20.0, read=60.0), transport=transport)
        self._cache: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()
        self._disk = DiskCache(cache_dir) if cache_dir else None
        self._refreshing: set[str] = set()
        self._refresher = ThreadPoolExecutor(2, thread_name_prefix="hub-refresh")

    def _get_json(self, path: str, params: Any = None) -> Any:
        key = path + "?" + repr(params)
        with self._lock:
            hit = self._cache.get(key)
            if hit and time.monotonic() - hit[0] < CACHE_SECONDS:
                return hit[1]
        saved = self._disk.get(ENDPOINT + key) if self._disk else None
        if saved and saved[0] < STALE_SECONDS:
            age, data = saved
            with self._lock:
                self._cache[key] = (time.monotonic() - age, data)
            if age >= CACHE_SECONDS:
                self._refresh(key, path, params)  # shown now, fresh next time
            return data
        return self._fetch_json(key, path, params)

    def _fetch_json(self, key: str, path: str, params: Any) -> Any:
        response = self._client.get(path, params=params)
        response.raise_for_status()
        data = response.json()
        with self._lock:
            self._cache[key] = (time.monotonic(), data)
        if self._disk:  # a search answer is large JSON (~0.1 s to write): not on the request's time
            self._refresher.submit(self._disk.put, ENDPOINT + key, data)
        return data

    def _refresh(self, key: str, path: str, params: Any) -> None:
        with self._lock:
            if key in self._refreshing:
                return
            self._refreshing.add(key)

        def run() -> None:
            try:
                self._fetch_json(key, path, params)
            except Exception:  # noqa: BLE001 - the saved answer stays until the Hub answers
                pass
            finally:
                with self._lock:
                    self._refreshing.discard(key)

        self._refresher.submit(run)

    # -- search ----------------------------------------------------------------------------------

    def search(self, runtime: str, query: str = "", sort: str = "downloads", limit: int = 30, pipeline_tag: Optional[str] = None) -> list[RepoInfo]:
        filters = RUNTIME_FILTERS.get(runtime, {}).get("filter", [])
        filters = [filters] if isinstance(filters, str) else list(filters)
        params: list[tuple[str, Any]] = [("sort", sort), ("direction", "-1"), ("limit", str(limit)), ("full", "true")]
        params += [("expand[]", e) for e in ("gguf", "downloads", "likes", "lastModified", "pipeline_tag", "library_name", "tags", "cardData", "config", "gated", "sha", "author", "siblings")]
        if query:
            params.append(("search", query))
        if pipeline_tag:
            params.append(("pipeline_tag", pipeline_tag))
        results: dict[str, RepoInfo] = {}
        # The Hub ANDs repeated filters, so a runtime with several acceptable libraries searches each.
        for value in filters or [None]:
            batch = self._get_json("/api/models", params + ([("filter", value)] if value else []))
            for item in batch:
                info = _repo(item)
                results.setdefault(info.id, info)
        ranked = sorted(results.values(), key=lambda r: (r.downloads if sort == "downloads" else r.likes if sort == "likes" else 0), reverse=True)
        return ranked[:limit]

    # -- details ---------------------------------------------------------------------------------

    def model(self, repo: str, revision: Optional[str] = None) -> RepoInfo:
        path = f"/api/models/{repo}" + (f"/revision/{revision}" if revision else "")
        params = [("blobs", "true")] + [("expand[]", e) for e in ("gguf", "downloads", "likes", "lastModified", "pipeline_tag", "library_name", "tags", "cardData", "config", "gated", "sha", "author", "siblings")]
        return _repo(self._get_json(path, params))

    def readme(self, repo: str, revision: str = "main") -> str:
        # At a commit (40 hex digits) the card never changes, so it is kept on disk.
        pinned = self._disk is not None and len(revision) == 40
        key = f"{ENDPOINT}/readme:{repo}@{revision}"
        saved = self._disk.get(key) if pinned else None
        if saved is not None and isinstance(saved[1], str):
            return saved[1]
        response = self._client.get(f"/{repo}/raw/{revision}/README.md")
        text = response.text if response.status_code == 200 else ""
        if pinned and response.status_code in (200, 404):
            self._disk.put(key, text)
        return text

    def file_url(self, repo: str, revision: str, path: str) -> str:
        return f"{ENDPOINT}/{repo}/resolve/{revision}/{path}"

    # -- GGUF header ---------------------------------------------------------------------------

    def gguf_header(self, repo: str, revision: str, path: str, keys: set[str]) -> dict[str, Any]:
        """Reads only the GGUF header (ranged requests) and returns ``keys`` that are present."""
        cache_key = f"gguf:{repo}@{revision}/{path}:{sorted(keys)}"
        with self._lock:
            hit = self._cache.get(cache_key)
            if hit:
                return hit[1]
        saved = self._disk.get(ENDPOINT + cache_key) if self._disk else None
        if saved is not None and revision:  # a file at a commit never changes
            with self._lock:
                self._cache[cache_key] = (time.monotonic(), saved[1])
            return saved[1]
        data = b""
        size = HEADER_CHUNK
        while True:
            response = self._client.get(self.file_url(repo, revision, path), headers={"Range": f"bytes={len(data)}-{size - 1}"})
            if response.status_code not in (200, 206):
                response.raise_for_status()
            data += response.content if response.status_code == 206 else response.content[len(data) :]
            try:
                result = parse_gguf_header(data, keys)
                break
            except _NeedMore:
                if size >= HEADER_LIMIT or response.status_code == 200:
                    result = parse_gguf_header(data, keys, partial=True)
                    break
                size *= 2
        with self._lock:
            self._cache[cache_key] = (time.monotonic(), result)
        if self._disk and revision:
            self._disk.put(ENDPOINT + cache_key, result)
        return result


def _repo(item: dict[str, Any]) -> RepoInfo:
    card = item.get("cardData") or {}
    base = card.get("base_model") or []
    base = [base] if isinstance(base, str) else [b for b in base if isinstance(b, str)]
    files = [RepoFile(s["rfilename"], int((s.get("lfs") or {}).get("size") or s.get("size") or 0)) for s in item.get("siblings") or []]
    return RepoInfo(
        id=item.get("id") or item.get("modelId") or "",
        sha=item.get("sha") or "",
        author=item.get("author") or (item.get("id", "").split("/")[0]),
        pipeline_tag=item.get("pipeline_tag"),
        library=item.get("library_name"),
        tags=list(item.get("tags") or []),
        downloads=int(item.get("downloads") or 0),
        likes=int(item.get("likes") or 0),
        last_modified=item.get("lastModified") or "",
        gated=bool(item.get("gated")),
        base_models=base,
        gguf=item.get("gguf") or {},
        config=item.get("config") or {},
        files=files,
    )


# ---- GGUF header parsing over a byte buffer ------------------------------------------------------


class _NeedMore(Exception):
    pass


_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def parse_gguf_header(data: bytes, keys: set[str], partial: bool = False) -> dict[str, Any]:
    """Parses GGUF key/values from ``data``; raises ``_NeedMore`` if the buffer ends early."""
    offset = 0

    def take(fmt: str) -> Any:
        nonlocal offset
        size = struct.calcsize(fmt)
        if offset + size > len(data):
            raise _NeedMore
        value = struct.unpack_from(fmt, data, offset)[0]
        offset += size
        return value

    def string() -> str:
        nonlocal offset
        length = take("<Q")
        if offset + length > len(data):
            raise _NeedMore
        text = data[offset : offset + length].decode("utf-8", errors="replace")
        offset += length
        return text

    def value(kind: int, keep: bool) -> Any:
        nonlocal offset
        if kind in _SCALARS:
            return take(_SCALARS[kind])
        if kind == 8:
            return string()
        if kind == 9:
            item, count = take("<I"), take("<Q")
            if not keep and item in _SCALARS:  # skip large numeric arrays without decoding them
                size = struct.calcsize(_SCALARS[item]) * count
                if offset + size > len(data):
                    raise _NeedMore
                offset += size
                return None
            return [value(item, keep) for _ in range(count)]
        raise ValueError(f"unknown GGUF value type {kind}")

    result: dict[str, Any] = {}
    try:
        if data[:4] != b"GGUF":
            if len(data) < 4:
                raise _NeedMore
            raise ValueError("not a GGUF file")
        offset = 4
        take("<I")
        take("<Q")
        tokenizer_wanted = any(k.startswith("tokenizer.") for k in keys)
        for _ in range(take("<Q")):
            key = string()
            # Writers put general.* and the architecture's keys before the tokenizer, whose
            # vocabulary is most of the header (~6 MB, half a million strings): stop there.
            if key.startswith("tokenizer.") and keys and not tokenizer_wanted:
                break
            kind = take("<I")
            wanted = key in keys or any(k.endswith(".*") and key.startswith(k[:-1]) for k in keys)
            parsed = value(kind, wanted)
            if wanted:
                result[key] = parsed
            if keys and all(k in result for k in keys if not k.endswith(".*")) and not any(k.endswith(".*") for k in keys):
                break
    except _NeedMore:
        if not partial:
            raise
    return result
