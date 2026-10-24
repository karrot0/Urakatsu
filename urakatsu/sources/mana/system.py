"""Mana as a source system: built ``.mana`` bundles, run natively under Bun.

A Mana repository is a folder of static files (``kakarot-extensions-mana`` builds one):

    <repo>/sources.json            {"repositoryName", "sources": [{"id", "name", "path", "intents", ...}]}
    <repo>/sources/<path>.mana     the bundle
    <repo>/assets/<thumbnail>      the icon

Installing a source downloads its bundle to ``URAKATSU_HOME/sources/mana/bundles``. The
bundles run in ``host.js`` under the newest Bun (an on-demand runtime package): Mana itself
runs sources on JavaScriptCore, which is Bun's engine too. Each source keeps its preferences
in ``stores/<id>.json``.

Sites behind checks (Cloudflare and the like) are handled by the host's built-in browser, a
headless Chromium (see ``urakatsu.runtimes.browser``) with its profile in ``browser/``. A check
it cannot clear alone surfaces as :class:`SourceChallenge`; a person clears it in a browser
session (``browser_*``), and the request works after that.
"""

from __future__ import annotations

import base64
import itertools
import json
import logging
import math
import os
import subprocess
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Optional

import httpx

from urakatsu.runtimes.browser import Browser, BrowserRuntime
from urakatsu.runtimes.packages import BUN

from ..base import (
    BrowserFrame,
    Extension,
    Image,
    RemoteChapter,
    Repository,
    SearchPage,
    Series,
    SourceChallenge,
    SourceInfo,
    SourceSystem,
    SourceSystemError,
    SystemStatus,
)

log = logging.getLogger(__name__)

HOST = Path(__file__).with_name("host.js")
REQUEST_TIMEOUT = 120.0
INDEX_TTL = 3600.0

# Intent bits the mana-dev toolchain computes (kakarot-extensions-mana scripts/harness/runtime.mjs).
PROVIDES_SEARCH = 1 << 10
CHAPTERS_IN_CONTENT = 1 << 20
PROVIDES_CHAPTERS = 1 << 21
REQUIRES_SETUP = 1 << 1
EXPLICIT_RATING = 2  # CatalogRating: SAFE, MIXED, EXPLICIT


def repository_root(url: str) -> str:
    """The repository folder for whatever URL the user pasted (the folder, sources.json, or the page)."""
    root = url.strip().rstrip("/")
    for suffix in ("/sources.json", "/index.html"):
        if root.endswith(suffix):
            root = root[: -len(suffix)]
    if not root.startswith(("http://", "https://")):
        raise ValueError("a Mana repository URL starts with http:// or https://")
    return root


def usable(entry: dict[str, Any]) -> bool:
    """Sources Urakatsu can link: they search, serve chapters, and need no in-app setup."""
    intents = int(entry.get("intents") or 0)
    return bool(intents & PROVIDES_SEARCH) and bool(intents & (PROVIDES_CHAPTERS | CHAPTERS_IN_CONTENT)) and not intents & REQUIRES_SETUP


def _language(entry: dict[str, Any]) -> str:
    languages = entry.get("supportedLanguages") or []
    return str(languages[0]).split("_")[0] if languages else ""


def _tags(raw: Any) -> list[str]:
    tags: list[str] = []
    for item in raw or []:
        if isinstance(item, dict) and isinstance(item.get("tags"), list):
            tags += _tags(item["tags"])
        elif isinstance(item, dict):
            tags.append(str(item.get("title") or item.get("id") or ""))
        elif isinstance(item, str):
            tags.append(item)
    return [tag for tag in tags if tag]


class _BrowserNeeded(SourceSystemError):
    """A site needs the browser and the host has none yet."""


class _Host:
    """The Bun process running host.js, with request ids matched to replies on a reader thread."""

    def __init__(self, bun: Path, log_path: Path, env: dict[str, str]):
        flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        self._log = log_path.open("ab")
        self.process = subprocess.Popen(
            [str(bun), str(HOST)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            creationflags=flags,
            env={**os.environ, **env},
        )
        self._ids = itertools.count(1)
        self._pending: dict[int, Future] = {0: Future()}
        self._write = threading.Lock()
        self.version: Optional[str] = None
        threading.Thread(target=self._read, name="mana-host", daemon=True).start()
        self.version = self._pending[0].result(timeout=30).get("bun")

    def alive(self) -> bool:
        return self.process.poll() is None

    def _read(self) -> None:
        assert self.process.stdout is not None
        for raw in self.process.stdout:
            try:
                message = json.loads(raw)
            except ValueError:
                continue
            future = self._pending.pop(message.get("id"), None)
            if future is None:
                continue
            if "error" in message:
                if message.get("challenge"):
                    future.set_exception(SourceChallenge(message["error"], message["challenge"]))
                elif message.get("needsBrowser"):
                    future.set_exception(_BrowserNeeded(message["error"]))
                else:
                    future.set_exception(SourceSystemError(message["error"]))
            else:
                future.set_result(message.get("result"))
        for future in self._pending.values():
            if not future.done():
                future.set_exception(SourceSystemError("the Mana host exited"))
        self._pending.clear()

    def call(self, method: str, timeout: float = REQUEST_TIMEOUT, **params: Any) -> Any:
        if not self.alive():
            raise SourceSystemError("the Mana host is not running")
        request_id = next(self._ids)
        future: Future = Future()
        self._pending[request_id] = future
        line = json.dumps({"id": request_id, "method": method, "params": params}) + "\n"
        with self._write:
            assert self.process.stdin is not None
            self.process.stdin.write(line.encode("utf-8"))
            self.process.stdin.flush()
        try:
            return future.result(timeout=timeout)
        except TimeoutError as error:
            self._pending.pop(request_id, None)
            raise SourceSystemError(f"the source did not answer {method} within {timeout:.0f} s") from error

    def close(self) -> None:
        if self.alive():
            try:
                assert self.process.stdin is not None
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
        self._log.close()


class ManaSystem(SourceSystem):
    id = "mana"
    name = "Mana"
    description = "Runs Mana source bundles. Add a Mana repository URL, then install its sources."
    homepage = "https://github.com/Mana-iOS/mana-dev"

    def __init__(self, root: Path, packages):
        super().__init__(root, packages)
        self._host: Optional[_Host] = None
        self._loaded: set[str] = set()
        self._indexes: dict[str, tuple[float, dict[str, Any]]] = {}
        self._raw_pages: dict[str, str] = {}
        self._error: Optional[str] = None
        self._starting = False
        self._lock = threading.RLock()
        self.browsers = BrowserRuntime(packages.root / "chrome")
        self._browser: Optional[Browser] = None

    # -- state on disk -----------------------------------------------------------------------------

    def _read(self, name: str, default: Any) -> Any:
        try:
            return json.loads((self.root / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return default

    def _write_json(self, name: str, value: Any) -> None:
        path = self.root / name
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, indent=1), encoding="utf-8")
        tmp.replace(path)

    def _repo_urls(self) -> list[str]:
        return list(self._read("repositories.json", []))

    def _installed(self) -> dict[str, dict[str, Any]]:
        return dict(self._read("installed.json", {}))

    def _bundle(self, source_id: str) -> Path:
        return self.root / "bundles" / f"{source_id}.mana"

    # -- lifecycle ---------------------------------------------------------------------------------

    def status(self) -> SystemStatus:
        installed = self.packages.installed(BUN)
        if self._starting:
            state = "starting"
        elif self._host is not None and self._host.alive():
            state = "running"
        elif self._error:
            state = "error"
        else:
            state = "stopped" if installed else "not_installed"
        version = self._host.version if self._host is not None and self._host.alive() else None
        return SystemStatus(
            id=self.id,
            name=self.name,
            description=self.description,
            homepage=self.homepage,
            state=state,
            runtime=BUN.id,
            version=f"Bun {version or (installed.tag.removeprefix('bun-v') if installed else '')}".strip() if (version or installed) else None,
            message=self._error if state == "error" else None,
            browser=self._describe_browser(),
        )

    def _describe_browser(self) -> str:
        found = self._browser or self.browsers.find()
        if found is None:
            return "Chrome for Testing installs the first time a site needs a browser"
        if found.kind == "managed":
            return f"Chrome for Testing {found.version}, kept up to date by Urakatsu"
        return f"{found.name}, installed on this computer" if found.kind == "system" else f"{found.name} ({found.path})"

    def start(self) -> SystemStatus:
        with self._lock:
            if self._host is not None and self._host.alive():
                return self.status()
            self._starting = True
            self._error = None
            try:
                from urakatsu.runtimes.hardware import discover

                bun = self.packages.ensure(BUN, discover().candidates()).executable
                self._browser = self.browsers.find()
                env = {"URAKATSU_BROWSER_PROFILE": str(self.root / "browser")}
                if self._browser is not None:
                    env["URAKATSU_BROWSER_PATH"] = str(self._browser.path)
                self._host = _Host(bun, self.root / "host.log", env)
                self._loaded.clear()
            except Exception as error:
                self._error = str(error)
                raise SourceSystemError(f"the Mana host did not start: {error}") from error
            finally:
                self._starting = False
        return self.status()

    def stop(self) -> None:
        with self._lock:
            host, self._host = self._host, None
            self._loaded.clear()
        if host is not None:
            host.close()

    def _call(self, source_id: str, method: str, **params: Any) -> Any:
        with self._lock:
            if self._host is None or not self._host.alive():
                self.start()
            host = self._host
            assert host is not None
            if source_id not in self._loaded:
                if not self._bundle(source_id).exists():
                    raise KeyError(f"Mana source {source_id} is not installed")
                host.call("load", source=source_id, bundle=str(self._bundle(source_id)), store=str(self.root / "stores" / f"{source_id}.json"))
                self._loaded.add(source_id)
        try:
            try:
                return host.call(method, source=source_id, **params)
            except _BrowserNeeded:
                self._ensure_browser()
                return host.call(method, source=source_id, **params)
        except SourceChallenge as challenge:
            challenge.system = self.id
            raise

    def _ensure_browser(self) -> None:
        """Gives the host a browser, installing Chrome for Testing when the computer has none."""
        with self._lock:
            if self._host is None or not self._host.alive():
                self.start()
            browser = self.browsers.ensure()
            if self._browser != browser:
                assert self._host is not None
                self._host.call("browser.configure", path=str(browser.path), profile=str(self.root / "browser"))
                self._browser = browser

    # -- built-in browser --------------------------------------------------------------------------

    def _browser_call(self, method: str, **params: Any) -> Any:
        if self._browser is None or self._host is None or not self._host.alive():
            self._ensure_browser()
        assert self._host is not None
        return self._host.call(method, **params)

    def browser_open(self, url: str) -> str:
        if not url.startswith(("http://", "https://")):
            raise ValueError("the browser opens http:// and https:// pages")
        return self._browser_call("browser.open", url=url)["id"]

    def browser_frame(self, session: str) -> BrowserFrame:
        return BrowserFrame(**self._browser_call("browser.frame", id=session))

    def browser_input(self, session: str, event: dict) -> None:
        self._browser_call("browser.input", id=session, **event)

    def browser_close(self, session: str) -> None:
        if self._host is not None and self._host.alive():
            self._host.call("browser.close", id=session)

    # -- repositories ------------------------------------------------------------------------------

    def _index(self, url: str, refresh: bool = False) -> dict[str, Any]:
        cached = self._indexes.get(url)
        if cached and not refresh and time.time() - cached[0] < INDEX_TTL:
            return cached[1]
        response = httpx.get(f"{url}/sources.json", timeout=30.0, follow_redirects=True)
        if response.status_code >= 400:
            raise SourceSystemError(f"{url}/sources.json answered {response.status_code}")
        try:
            index = response.json()
            index["sources"] = list(index.get("sources") or [])
        except (ValueError, AttributeError) as error:
            raise SourceSystemError(f"{url}/sources.json is not a Mana repository index") from error
        self._indexes[url] = (time.time(), index)
        return index

    def repositories(self) -> list[Repository]:
        repositories = []
        for url in self._repo_urls():
            try:
                index = self._index(url)
                repositories.append(Repository(url=url, name=index.get("repositoryName") or url, extensions=len(index["sources"])))
            except (httpx.HTTPError, SourceSystemError) as error:
                log.warning("Mana repository %s is unreachable: %s", url, error)
                repositories.append(Repository(url=url, name=url))
        return repositories

    def add_repository(self, url: str) -> list[Repository]:
        root = repository_root(url)
        self._index(root, refresh=True)
        urls = self._repo_urls()
        if root not in urls:
            self._write_json("repositories.json", urls + [root])
        return self.repositories()

    def remove_repository(self, url: str) -> list[Repository]:
        self._write_json("repositories.json", [u for u in self._repo_urls() if u != url])
        self._indexes.pop(url, None)
        return self.repositories()

    # -- extensions (for Mana, one extension is one source) ---------------------------------------

    def _extension(self, repo: Optional[str], entry: dict[str, Any], installed: dict[str, dict[str, Any]]) -> Extension:
        local = installed.get(entry["id"])
        icon = entry.get("thumbnail")
        if icon and not str(icon).startswith("http") and repo:
            icon = f"{repo}/assets/{icon}"
        return Extension(
            id=entry["id"],
            name=entry.get("name") or entry["id"],
            version=str(entry.get("version") or ""),
            language=_language(entry),
            icon=icon,
            nsfw=entry.get("rating") == EXPLICIT_RATING,
            installed=local is not None,
            has_update=local is not None and str(local.get("version")) != str(entry.get("version")),
            repository=repo,
            description=entry.get("description") or "",
        )

    def extensions(self, refresh: bool = False) -> list[Extension]:
        installed = self._installed()
        found: dict[str, Extension] = {}
        for url in self._repo_urls():
            try:
                index = self._index(url, refresh)
            except (httpx.HTTPError, SourceSystemError) as error:
                log.warning("Mana repository %s is unreachable: %s", url, error)
                continue
            for entry in index["sources"]:
                if entry.get("environment", "source") == "source" and entry.get("id") and entry["id"] not in found:
                    found[entry["id"]] = self._extension(url, entry, installed)
        for source_id, entry in installed.items():
            if source_id not in found:
                found[source_id] = self._extension(entry.get("repository"), entry, installed)
                found[source_id].has_update = False
        return sorted(found.values(), key=lambda e: (not e.installed, e.name.lower()))

    def install_extension(self, extension_id: str) -> list[Extension]:
        for url in self._repo_urls():
            entry = next((s for s in self._index(url)["sources"] if s.get("id") == extension_id), None)
            if entry is None:
                continue
            response = httpx.get(f"{url}/sources/{entry['path']}.mana", timeout=60.0, follow_redirects=True)
            if response.status_code >= 400:
                raise SourceSystemError(f"could not download {entry['name']} ({response.status_code})")
            bundle = self._bundle(extension_id)
            bundle.parent.mkdir(parents=True, exist_ok=True)
            tmp = bundle.with_suffix(".tmp")
            tmp.write_bytes(response.content)
            tmp.replace(bundle)
            installed = self._installed()
            installed[extension_id] = {**entry, "repository": url}
            self._write_json("installed.json", installed)
            self._unload(extension_id)
            return self.extensions()
        raise KeyError(f"no repository offers a Mana source named {extension_id!r}")

    def uninstall_extension(self, extension_id: str) -> list[Extension]:
        installed = self._installed()
        if installed.pop(extension_id, None) is None:
            raise KeyError(f"Mana source {extension_id} is not installed")
        self._write_json("installed.json", installed)
        self._unload(extension_id)
        self._bundle(extension_id).unlink(missing_ok=True)
        return self.extensions()

    def _unload(self, source_id: str) -> None:
        with self._lock:
            if source_id in self._loaded and self._host is not None and self._host.alive():
                self._host.call("unload", source=source_id)
            self._loaded.discard(source_id)

    # -- scraping ----------------------------------------------------------------------------------

    def sources(self) -> list[SourceInfo]:
        extensions = {e.id: e for e in self.extensions()}
        result = []
        for source_id, entry in self._installed().items():
            if not usable(entry):
                continue
            extension = extensions.get(source_id)
            result.append(
                SourceInfo(
                    id=source_id,
                    name=entry.get("name") or source_id,
                    language=_language(entry),
                    icon=extension.icon if extension else None,
                    nsfw=entry.get("rating") == EXPLICIT_RATING,
                    extension=source_id,
                    website=entry.get("website"),
                )
            )
        return sorted(result, key=lambda s: s.name.lower())

    @staticmethod
    def _series(series_id: str, raw: dict[str, Any]) -> Series:
        authors = [str(a) for a in (raw.get("authors") or []) if a] or [str(raw[k]) for k in ("author", "artist") if raw.get(k)]
        status = raw.get("status")
        return Series(
            id=series_id,
            title=raw.get("title") or series_id,
            cover=raw.get("cover") or None,
            url=raw.get("webUrl") or None,
            summary=raw.get("summary") or "",
            authors=list(dict.fromkeys(authors)),
            status=status if isinstance(status, str) and status else None,
            tags=_tags(raw.get("tags")),
        )

    def search(self, source_id: str, query: str, page: int = 1) -> SearchPage:
        result = self._call(source_id, "search", query=query.strip(), page=page)
        return SearchPage(
            results=[self._series(str(item["id"]), item) for item in result["results"] if item.get("id")],
            has_more=not result["isLastPage"],
        )

    def series(self, source_id: str, series_id: str) -> Series:
        return self._series(series_id, self._call(source_id, "content", contentId=series_id))

    def chapters(self, source_id: str, series_id: str) -> list[RemoteChapter]:
        raw = self._call(source_id, "chapters", contentId=series_id)
        numbered = [c for c in raw if c.get("number") is not None and math.isfinite(c["number"]) and c["number"] >= 0]
        if len(numbered) >= 0.8 * len(raw):
            # Sources disagree on newest-first or oldest-first; chapter numbers settle it.
            raw = sorted(raw, key=lambda c: (c["number"] if c in numbered else math.inf, c.get("index") or 0))
        return [
            RemoteChapter(
                id=c["chapterId"],
                name=c.get("title") or (f"Chapter {c['number']:g}" if c.get("number") is not None else f"Chapter {position + 1}"),
                number=c.get("number"),
                volume=c.get("volume"),
                scanlator=c.get("scanlator"),
                language=c.get("language"),
                date=c.get("date"),
            )
            for position, c in enumerate(raw)
        ]

    def pages(self, source_id: str, series_id: str, chapter_id: str) -> list[str]:
        handles = []
        for number, page in enumerate(self._call(source_id, "pages", contentId=series_id, chapterId=chapter_id)):
            if page.get("url"):
                handles.append(page["url"])
            else:
                handle = f"raw:{source_id}:{chapter_id}:{number}"
                self._raw_pages[handle] = page["raw"]
                handles.append(handle)
        return handles

    def page(self, source_id: str, series_id: str, chapter_id: str, handle: str) -> Image:
        if handle.startswith("raw:"):
            return Image(base64.b64decode(self._raw_pages.pop(handle)), "application/octet-stream")
        return self._image(source_id, handle, "page")

    def cover(self, source_id: str, url: str) -> Optional[Image]:
        return self._image(source_id, url, "cover") if url else None

    def _image(self, source_id: str, url: str, origin: str) -> Image:
        found = self._call(source_id, "image", url=url, origin=origin)
        return Image(base64.b64decode(found["data"]), found.get("contentType") or "application/octet-stream")
