"""Suwayomi Server as a source system: Tachiyomi / Mihon extensions, driven over GraphQL.

Urakatsu installs the newest Suwayomi-Server release (it bundles its own Java runtime) as
an on-demand runtime package and runs it on a free localhost port, with its data under
``URAKATSU_HOME/sources/suwayomi/server``. Repositories are Suwayomi's extension stores;
installing an extension makes its sources (one per language) available.

Suwayomi keeps every series it has seen in its own database, so a series id here is
Suwayomi's manga id and a chapter id is its chapter id. Both stay valid as long as the
server's data folder does.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from urakatsu.runtimes.packages import SUWAYOMI

from .base import (
    Extension,
    Image,
    RemoteChapter,
    Repository,
    SearchPage,
    Series,
    SourceInfo,
    SourceSystem,
    SourceSystemError,
    SystemStatus,
)

log = logging.getLogger(__name__)

STARTUP_TIMEOUT = 180.0  # seconds; the first start also unpacks the server's own data
LOCAL_SOURCE = "0"
CONFIG_PREFIX = "-Dsuwayomi.tachidesk.config.server."

EXTENSION_FIELDS = "pkgName name versionName lang iconUrl contentWarning isInstalled hasUpdate isObsolete extensionStore { indexUrl }"
MANGA_FIELDS = "id title author artist description genre status thumbnailUrl realUrl"
CHAPTER_FIELDS = "id name chapterNumber sourceOrder uploadDate scanlator"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class SuwayomiSystem(SourceSystem):
    id = "suwayomi"
    name = "Suwayomi Server"
    description = "Runs Tachiyomi and Mihon extensions. Add an extension repository, then install the sites you need."
    homepage = "https://github.com/Suwayomi/Suwayomi-Server"

    def __init__(self, root: Path, packages):
        super().__init__(root, packages)
        self._process: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        self._version: Optional[str] = None
        self._error: Optional[str] = None
        self._starting = False
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------------------------------

    def _running(self) -> bool:
        return self._process is not None and self._process.poll() is None and self._port is not None and not self._starting

    def status(self) -> SystemStatus:
        installed = self.packages.installed(SUWAYOMI)
        if self._starting:
            state = "starting"
        elif self._running():
            state = "running"
        elif self._error:
            state = "error"
        else:
            state = "stopped" if installed else "not_installed"
        return SystemStatus(
            id=self.id,
            name=self.name,
            description=self.description,
            homepage=self.homepage,
            state=state,
            runtime=SUWAYOMI.id,
            version=self._version or (installed.tag if installed else None),
            message=self._error if state == "error" else None,
        )

    def start(self) -> SystemStatus:
        with self._lock:
            if self._running():
                return self.status()
            self._starting = True
            self._error = None
            try:
                self._launch()
            except Exception as error:
                self._error = str(error)
                self._terminate()
                raise SourceSystemError(f"Suwayomi Server did not start: {error}") from error
            finally:
                self._starting = False
        return self.status()

    def _launch(self) -> None:
        from urakatsu.runtimes.hardware import discover

        installed = self.packages.ensure(SUWAYOMI, discover().candidates())
        jar = next(iter(sorted(installed.path.rglob("Suwayomi-Server.jar"))), None)
        if jar is None:
            raise FileNotFoundError(f"Suwayomi-Server.jar not found under {installed.path}")
        data = self.root / "server"
        data.mkdir(parents=True, exist_ok=True)
        self._reap_orphan()
        port = _free_port()
        settings = {
            "rootDir": data.as_posix(),
            "ip": "127.0.0.1",
            "port": port,
            "webUIEnabled": "false",
            "initialOpenInBrowserEnabled": "false",
            "systemTrayEnabled": "false",
        }
        command = [str(installed.executable), *(f"{CONFIG_PREFIX}{key}={value}" for key, value in settings.items()), "-jar", str(jar)]
        log_file = (self.root / "server.log").open("ab")
        flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        self._process = subprocess.Popen(command, cwd=jar.parent.parent, stdout=log_file, stderr=subprocess.STDOUT, creationflags=flags)
        log_file.close()
        self._port = port
        (self.root / "server.json").write_text(json.dumps({"pid": self._process.pid, "port": port}), encoding="utf-8")
        deadline = time.time() + STARTUP_TIMEOUT
        while time.time() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(f"the server exited with code {self._process.returncode}; see {self.root / 'server.log'}")
            try:
                about = self._gql("{ aboutServer { version } }", timeout=5)["aboutServer"]
                self._version = about["version"]
                log.info("Suwayomi Server %s listening on 127.0.0.1:%d", self._version, port)
                return
            except (httpx.HTTPError, SourceSystemError):
                time.sleep(1.0)
        raise TimeoutError(f"the server did not answer within {STARTUP_TIMEOUT:.0f} s")

    def _reap_orphan(self) -> None:
        """Stops a server a previous Urakatsu left running (killed before it could clean up).

        The pid and port it recorded must still answer as Suwayomi, so a reused pid is never hit.
        """
        record = self.root / "server.json"
        try:
            previous = json.loads(record.read_text(encoding="utf-8"))
            answer = httpx.post(f"http://127.0.0.1:{previous['port']}/api/graphql", json={"query": "{ aboutServer { version } }"}, timeout=3)
            if answer.status_code == 200 and "aboutServer" in answer.text:
                log.info("stopping the Suwayomi Server a previous run left on port %s", previous["port"])
                os.kill(int(previous["pid"]), signal.SIGTERM)
                time.sleep(2.0)
        except (OSError, ValueError, KeyError, httpx.HTTPError):
            pass
        record.unlink(missing_ok=True)

    def _terminate(self) -> None:
        (self.root / "server.json").unlink(missing_ok=True)
        process, self._process, self._port = self._process, None, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    def stop(self) -> None:
        with self._lock:
            self._terminate()

    # -- transport ---------------------------------------------------------------------------------

    def _base(self) -> str:
        if not self._running():
            self.start()
        return f"http://127.0.0.1:{self._port}"

    def _gql(self, query: str, variables: Optional[dict[str, Any]] = None, timeout: float = 120.0) -> dict[str, Any]:
        base = f"http://127.0.0.1:{self._port}" if self._starting else self._base()
        response = httpx.post(f"{base}/api/graphql", json={"query": query, "variables": variables or {}}, timeout=timeout)
        response.raise_for_status()
        body = response.json()
        if body.get("errors"):
            raise SourceSystemError("; ".join(str(e.get("message", e)) for e in body["errors"]))
        return body["data"]

    def _get(self, path: str) -> Image:
        url = path if path.startswith("http") else self._base() + path
        for attempt in range(2):
            response = httpx.get(url, timeout=120.0, follow_redirects=True)
            # A thumbnail can 404 while Suwayomi is still fetching it the first time.
            if response.status_code == 404 and attempt == 0:
                time.sleep(1.0)
                continue
            if response.status_code >= 400:
                raise SourceSystemError(f"Suwayomi answered {response.status_code} for {path}")
            return Image(response.content, response.headers.get("content-type", "application/octet-stream").split(";")[0])
        raise SourceSystemError(f"Suwayomi has nothing at {path}")

    # -- repositories and extensions ---------------------------------------------------------------

    def repositories(self) -> list[Repository]:
        nodes = self._gql("{ extensionStores { nodes { name indexUrl extensions { totalCount } } } }")["extensionStores"]["nodes"]
        return [Repository(url=n["indexUrl"], name=n["name"], extensions=n["extensions"]["totalCount"]) for n in nodes]

    def add_repository(self, url: str) -> list[Repository]:
        self._gql("mutation($url: String!) { addExtensionStore(input: { indexUrl: $url }) { extensionStore { indexUrl } } }", {"url": url.strip()})
        self._gql("mutation { fetchExtensions(input: {}) { extensions { pkgName } } }")
        return self.repositories()

    def remove_repository(self, url: str) -> list[Repository]:
        self._gql("mutation($url: String!) { removeExtensionStore(input: { indexUrl: $url }) { clientMutationId } }", {"url": url})
        return self.repositories()

    def _extension(self, node: dict[str, Any]) -> Extension:
        store = node.get("extensionStore") or {}
        return Extension(
            id=node["pkgName"],
            name=node["name"],
            version=node["versionName"],
            language=node["lang"],
            icon=f"/api/v1/sources/systems/{self.id}/extensions/{node['pkgName']}/icon",
            nsfw=node["contentWarning"] == "NSFW",
            installed=node["isInstalled"],
            has_update=node["hasUpdate"],
            repository=store.get("indexUrl"),
        )

    def extensions(self, refresh: bool = False) -> list[Extension]:
        if refresh:
            nodes = self._gql(f"mutation {{ fetchExtensions(input: {{}}) {{ extensions {{ {EXTENSION_FIELDS} }} }} }}")["fetchExtensions"]["extensions"]
        else:
            nodes = self._gql(f"{{ extensions {{ nodes {{ {EXTENSION_FIELDS} }} }} }}")["extensions"]["nodes"]
        nodes = [n for n in nodes if not n["isObsolete"] or n["isInstalled"]]
        return sorted((self._extension(n) for n in nodes), key=lambda e: (not e.installed, e.name.lower(), e.language))

    def _patch_extension(self, extension_id: str, patch: str) -> list[Extension]:
        self._gql(
            f"mutation($id: String!) {{ updateExtension(input: {{ id: $id, patch: {{ {patch}: true }} }}) {{ extension {{ pkgName }} }} }}",
            {"id": extension_id},
        )
        return self.extensions()

    def install_extension(self, extension_id: str) -> list[Extension]:
        return self._patch_extension(extension_id, "install")

    def update_extension(self, extension_id: str) -> list[Extension]:
        return self._patch_extension(extension_id, "update")

    def uninstall_extension(self, extension_id: str) -> list[Extension]:
        return self._patch_extension(extension_id, "uninstall")

    def extension_icon(self, extension_id: str) -> Optional[Image]:
        return self._get(f"/api/v1/extension/icon/{extension_id}")

    # -- scraping ----------------------------------------------------------------------------------

    def sources(self) -> list[SourceInfo]:
        nodes = self._gql("{ sources { nodes { id displayName lang contentWarning homeUrl extension { pkgName } } } }")["sources"]["nodes"]
        return [
            SourceInfo(
                id=n["id"],
                name=n["displayName"],
                language=n["lang"],
                icon=f"/api/v1/sources/systems/{self.id}/extensions/{n['extension']['pkgName']}/icon",
                nsfw=n["contentWarning"] == "NSFW",
                extension=n["extension"]["pkgName"],
                website=n.get("homeUrl"),
            )
            for n in nodes
            if n["id"] != LOCAL_SOURCE
        ]

    def source_icon(self, source_id: str) -> Optional[Image]:
        pkg = self._gql("query($id: LongString!) { source(id: $id) { extension { pkgName } } }", {"id": source_id})["source"]["extension"]["pkgName"]
        return self.extension_icon(pkg)

    @staticmethod
    def _series(node: dict[str, Any]) -> Series:
        authors = [name for name in (node.get("author"), node.get("artist")) if name]
        status = node.get("status")
        return Series(
            id=str(node["id"]),
            title=node["title"],
            cover=node.get("thumbnailUrl"),
            url=node.get("realUrl"),
            summary=node.get("description") or "",
            authors=list(dict.fromkeys(authors)),
            status=None if status in (None, "UNKNOWN") else status.replace("_", " ").capitalize(),
            tags=node.get("genre") or [],
        )

    def search(self, source_id: str, query: str, page: int = 1) -> SearchPage:
        kind = "SEARCH" if query.strip() else "POPULAR"
        result = self._gql(
            "mutation($source: LongString!, $type: FetchSourceMangaType!, $page: Int!, $query: String) {"
            " fetchSourceManga(input: { source: $source, type: $type, page: $page, query: $query }) {"
            " hasNextPage mangas { id title thumbnailUrl realUrl } } }",
            {"source": source_id, "type": kind, "page": page, "query": query.strip() or None},
        )["fetchSourceManga"]
        return SearchPage(results=[self._series(n) for n in result["mangas"]], has_more=result["hasNextPage"])

    def _fetch(self, series_id: str) -> dict[str, Any]:
        return self._gql(
            "mutation($id: Int!) { fetchMangaAndChapters(input: { id: $id, fetchManga: true, fetchChapters: true }) {"
            f" manga {{ {MANGA_FIELDS} }} chapters {{ {CHAPTER_FIELDS} }} }} }}",
            {"id": int(series_id)},
        )["fetchMangaAndChapters"]

    def series(self, source_id: str, series_id: str) -> Series:
        return self._series(self._fetch(series_id)["manga"])

    def chapters(self, source_id: str, series_id: str) -> list[RemoteChapter]:
        nodes = sorted(self._fetch(series_id)["chapters"], key=lambda n: n["sourceOrder"])
        return [
            RemoteChapter(
                id=str(n["id"]),
                name=n["name"],
                number=n["chapterNumber"] if n["chapterNumber"] >= 0 else None,
                scanlator=n.get("scanlator") or None,
                date=int(n["uploadDate"]) / 1000 if int(n["uploadDate"] or 0) > 0 else None,
            )
            for n in nodes
        ]

    def pages(self, source_id: str, series_id: str, chapter_id: str) -> list[str]:
        return self._gql("mutation($id: Int!) { fetchChapterPages(input: { chapterId: $id }) { pages } }", {"id": int(chapter_id)})[
            "fetchChapterPages"
        ]["pages"]

    def page(self, source_id: str, series_id: str, chapter_id: str, handle: str) -> Image:
        return self._get(handle)

    def cover(self, source_id: str, url: str) -> Optional[Image]:
        return self._get(url) if url else None
