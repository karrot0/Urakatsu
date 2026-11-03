"""Suwayomi Server as a source system: Tachiyomi / Mihon extensions, driven over GraphQL.

Urakatsu installs the newest Suwayomi-Server release (it bundles its own Java runtime) as
an on-demand runtime package and runs it on a free localhost port, with its data under
``URAKATSU_HOME/sources/suwayomi/server``. Repositories are Suwayomi's extension stores;
installing an extension makes its sources (one per language) available.

Suwayomi keeps every series it has seen in its own database, so a series id here is
Suwayomi's manga id and a chapter id is its chapter id. Both stay valid as long as the
server's data folder does.

Urakatsu can also use a Suwayomi Server that already runs elsewhere (``connect``): an always-on
server on a NAS or in Docker, signed in to with basic authentication when it asks for it. The
address is kept in ``URAKATSU_HOME/sources/suwayomi/remote.json``. Starting and stopping then
only connect and disconnect: that server is never installed, started or stopped from here. Its
ids are its own, so projects linked through one server are not found through the other.
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
from pydantic import BaseModel, field_validator

from urakatsu.core import files
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


REMOTE_FILE = "remote.json"
REMOTE_TIMEOUT = 15.0  # seconds to wait for a server elsewhere to answer when connecting


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class RemoteServer(BaseModel):
    """A Suwayomi Server that runs elsewhere, used instead of Urakatsu's own."""

    url: str  # http(s)://host:port
    username: str = ""
    password: str = ""

    @field_validator("url")
    @classmethod
    def _url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if value.endswith("/api/graphql"):
            value = value[: -len("/api/graphql")]
        if not value.startswith(("http://", "https://")):
            raise ValueError("the Suwayomi address starts with http:// or https:// (such as http://192.168.1.10:4567)")
        return value

    @field_validator("username")
    @classmethod
    def _username(cls, value: str) -> str:
        return value.strip()


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
        self._connected = False  # to a remote server
        self._lock = threading.Lock()
        self._remote = self._read_remote()

    # -- remote server -----------------------------------------------------------------------------

    def _read_remote(self) -> Optional[RemoteServer]:
        try:
            return RemoteServer.model_validate_json((self.root / REMOTE_FILE).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except ValueError as error:
            log.warning("ignoring unreadable %s: %s", self.root / REMOTE_FILE, error)
            return None

    def remote(self) -> Optional[RemoteServer]:
        """The server elsewhere this system uses, or None while it runs its own."""
        return self._remote

    def connect(self, url: Optional[str], username: str = "", password: Optional[str] = None) -> SystemStatus:
        current = self.remote()
        server = None
        if url and url.strip():
            server = RemoteServer(url=url, username=username, password=password or "")
            # The saved password is only ever sent to the address and user it was saved for.
            if password is None and current is not None and (current.url, current.username) == (server.url, server.username):
                server.password = current.password
        with self._lock:
            self._terminate()
            self._connected = False
            self._version = None
            self._error = None
            if server is None:
                (self.root / REMOTE_FILE).unlink(missing_ok=True)
            else:
                files.write_text(self.root / REMOTE_FILE, server.model_dump_json(indent=1))
            self._remote = server
        if server is not None:
            try:
                self.start()
            except SourceSystemError:
                pass  # the status carries the error; the address stays saved to try again
        return self.status()

    def _auth(self) -> Optional[httpx.BasicAuth]:
        remote = self.remote()
        return httpx.BasicAuth(remote.username, remote.password) if remote is not None and remote.username else None

    # -- lifecycle ---------------------------------------------------------------------------------

    def _running(self) -> bool:
        if self.remote() is not None:
            return self._connected and not self._starting
        return self._process is not None and self._process.poll() is None and self._port is not None and not self._starting

    def status(self) -> SystemStatus:
        remote = self.remote()
        installed = self.packages.installed(SUWAYOMI) if remote is None else None
        if self._starting:
            state = "starting"
        elif self._running():
            state = "running"
        elif self._error:
            state = "error"
        else:
            state = "stopped" if installed or remote else "not_installed"
        return SystemStatus(
            id=self.id,
            name=self.name,
            description=self.description,
            homepage=self.homepage,
            state=state,
            runtime=SUWAYOMI.id if remote is None else None,
            version=self._version or (installed.tag if installed else None),
            message=self._error if state == "error" else None,
            connectable=True,
            server=remote.url if remote else None,
            username=(remote.username or None) if remote else None,
        )

    def start(self) -> SystemStatus:
        with self._lock:
            if self._running():
                return self.status()
            remote = self.remote()
            self._starting = True
            self._error = None
            try:
                if remote is not None:
                    self._reach(remote)
                else:
                    self._launch()
            except Exception as error:
                self._error = self._explain(error, remote)
                self._terminate()
                if remote is not None:
                    raise SourceSystemError(f"Could not reach Suwayomi Server at {remote.url}: {self._error}") from error
                raise SourceSystemError(f"Suwayomi Server did not start: {error}") from error
            finally:
                self._starting = False
        return self.status()

    def _reach(self, remote: RemoteServer) -> None:
        about = self._gql("{ aboutServer { version } }", timeout=REMOTE_TIMEOUT)["aboutServer"]
        self._version = about["version"]
        self._connected = True
        log.info("using Suwayomi Server %s at %s", self._version, remote.url)

    @staticmethod
    def _explain(error: Exception, remote: Optional[RemoteServer]) -> str:
        if remote is not None and isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 401:
            return "it asks for a sign-in; enter the user name and password set in its basic authentication" if not remote.username else "it did not accept the user name and password"
        return str(error)

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
        """Stops Urakatsu's own server; a remote one keeps running, and is only let go of."""
        with self._lock:
            self._terminate()
            self._connected = False

    # -- transport ---------------------------------------------------------------------------------

    def _address(self) -> str:
        remote = self.remote()
        return remote.url if remote is not None else f"http://127.0.0.1:{self._port}"

    def _base(self) -> str:
        if not self._running():
            self.start()
        return self._address()

    def _gql(self, query: str, variables: Optional[dict[str, Any]] = None, timeout: float = 120.0) -> dict[str, Any]:
        base = self._address() if self._starting else self._base()
        response = httpx.post(f"{base}/api/graphql", json={"query": query, "variables": variables or {}}, timeout=timeout, auth=self._auth())
        response.raise_for_status()
        body = response.json()
        if body.get("errors"):
            raise SourceSystemError("; ".join(str(e.get("message", e)) for e in body["errors"]))
        return body["data"]

    def _get(self, path: str) -> Image:
        url = path if path.startswith("http") else self._base() + path
        # The sign-in goes to the Suwayomi server only, never to a site an absolute URL points at.
        auth = self._auth() if url.startswith(self._address() + "/") else None
        for attempt in range(2):
            response = httpx.get(url, timeout=120.0, follow_redirects=True, auth=auth)
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

    def _series(self, node: dict[str, Any]) -> Series:
        authors = [name for name in (node.get("author"), node.get("artist")) if name]
        status = node.get("status")
        cover = node.get("thumbnailUrl")
        remote = self.remote()
        if cover and remote is not None and cover.startswith("/"):
            # Covers are cached by URL, and another server's manga 5 is another series.
            cover = remote.url + cover
        return Series(
            id=str(node["id"]),
            title=node["title"],
            cover=cover,
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
