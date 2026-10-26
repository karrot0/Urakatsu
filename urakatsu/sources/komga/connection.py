"""The Komga connection: the server, its API key, and where each library's folder is here.

Kept in ``URAKATSU_HOME/komga.json``. Komga only reads books from its libraries' folders, so
Urakatsu writes them there itself and needs each folder as this computer sees it:

- Komga on this computer: the library's own path, found without asking.
- Komga in Docker, or on a NAS: the folder the library's path is mounted from, or a share
  mounted here (``/data/manga`` in the container may be ``~/komga/manga`` here). Set once per
  library; :meth:`KomgaConnection.set_folder` checks it holds that library's series.

Exporting needs an admin account's key: only admins may scan a library.
"""

from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, Optional

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

from urakatsu.core import files
from urakatsu.core.config import home

from .client import KomgaClient, KomgaError, Library, Series

log = logging.getLogger(__name__)

FILE = "komga.json"
SAMPLE_SERIES = 20  # series looked for when checking a library's folder


class KomgaSettings(BaseModel):
    url: str = ""  # http(s)://host:port
    api_key: str = ""
    folders: dict[str, str] = Field(default_factory=dict)  # library id -> its folder on this computer

    @field_validator("url")
    @classmethod
    def _url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if value and not value.startswith(("http://", "https://")):
            raise ValueError("the Komga address starts with http:// or https:// (such as http://localhost:25600)")
        return value

    @field_validator("api_key")
    @classmethod
    def _key(cls, value: str) -> str:
        return value.strip()

    @property
    def configured(self) -> bool:
        return bool(self.url and self.api_key)


class LibraryView(BaseModel):
    id: str
    name: str
    root: str
    folder: Optional[str] = None  # where Urakatsu writes this library's books
    folder_source: Literal["set", "komga", "none"] = "none"  # set here, Komga's own path, or unknown
    problem: Optional[str] = None


class KomgaStatus(BaseModel):
    configured: bool
    connected: bool = False
    url: str = ""
    user: Optional[str] = None
    version: Optional[str] = None
    error: Optional[str] = None
    libraries: list[LibraryView] = Field(default_factory=list)


def relative_folder(root: str, url: str) -> str:
    """A series' folder relative to its library's root, both as Komga gives them (POSIX or Windows)."""
    flavor = PureWindowsPath if "\\" in root or (len(root) > 1 and root[1] == ":") else PurePosixPath
    try:
        relative = flavor(url).relative_to(flavor(root))
    except ValueError as error:
        raise ValueError(f"the series folder {url} is not inside the library folder {root}") from error
    if not relative.parts:
        raise ValueError("a series cannot be the library's own folder")
    return "/".join(relative.parts)


def _problem(folder: Path) -> Optional[str]:
    if not folder.is_dir():
        return f"{folder} does not exist on this computer"
    probe = folder / f".urakatsu-{uuid.uuid4().hex}.tmp"
    try:
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as error:
        return f"cannot write to {folder}: {error.strerror or error}"
    return None


class KomgaConnection:
    def __init__(self, path: Optional[Path] = None, transport: Optional[httpx.BaseTransport] = None):
        self.path = path or home() / FILE
        self.transport = transport  # tests talk to a fake Komga
        self._lock = threading.Lock()

    # -- settings ----------------------------------------------------------------------------------

    def settings(self) -> KomgaSettings:
        try:
            return KomgaSettings.model_validate_json(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return KomgaSettings()
        except ValueError as error:
            log.warning("ignoring unreadable %s: %s", self.path, error)
            return KomgaSettings()

    def _write(self, settings: KomgaSettings) -> None:
        files.write_text(self.path, settings.model_dump_json(indent=1))

    def update(self, url: str, api_key: Optional[str] = None) -> KomgaStatus:
        """Saves the server and key, then reports on them. None keeps the saved key, for the same
        server only: the key is never sent to another address. Folders set for the same server's
        libraries stay."""
        with self._lock:
            current = self.settings()
            same = url.strip().rstrip("/") == current.url
            try:
                settings = KomgaSettings(url=url, api_key=current.api_key if api_key is None else api_key, folders=current.folders if same else {})
            except ValidationError as error:
                raise ValueError(error.errors()[0]["msg"].removeprefix("Value error, ")) from None
            if api_key is None and current.api_key and not same:
                raise ValueError("enter the API key for the new address; the saved key stays with the old one")
            self._write(settings)
        return self.status()

    def client(self) -> KomgaClient:
        settings = self.settings()
        if not settings.configured:
            raise KomgaError("connect Komga first: Settings, Sources, Komga")
        return KomgaClient(settings.url, settings.api_key, self.transport)

    # -- libraries and their folders -----------------------------------------------------------------

    def status(self) -> KomgaStatus:
        settings = self.settings()
        status = KomgaStatus(configured=settings.configured, url=settings.url)
        if not settings.configured:
            return status
        try:
            with KomgaClient(settings.url, settings.api_key, self.transport) as client:
                status.user = client.user()
                status.version = client.version()
                status.libraries = [self._view(library, settings) for library in client.libraries()]
            status.connected = True
        except KomgaError as error:
            status.error = str(error)
        except (KeyError, TypeError) as error:  # an answer shaped unlike Komga's
            status.error = f"{settings.url} did not answer like Komga ({error.__class__.__name__}: {error}); check the address"
        return status

    @staticmethod
    def _view(library: Library, settings: KomgaSettings) -> LibraryView:
        view = LibraryView(id=library.id, name=library.name, root=library.root)
        chosen = settings.folders.get(library.id)
        if chosen:
            view.folder, view.folder_source = chosen, "set"
            view.problem = _problem(Path(chosen))
        elif Path(library.root).is_absolute() and Path(library.root).is_dir():
            view.folder, view.folder_source = library.root, "komga"  # Komga runs here: its path works as is
            view.problem = _problem(Path(library.root))
        else:
            view.problem = f"Komga reads this library from {library.root}; choose where that folder is on this computer"
        return view

    def folder(self, library_id: str, client: Optional[KomgaClient] = None) -> Path:
        """The library's folder on this computer; raises when it is unknown or unusable."""
        settings = self.settings()
        own = client is None
        client = client or self.client()
        try:
            view = self._view(client.library(library_id), settings)
        finally:
            if own:
                client.close()
        if view.folder is None or view.problem:
            raise KomgaError(f"the Komga library {view.name}: {view.problem}")
        return Path(view.folder)

    def set_folder(self, library_id: str, folder: Optional[str]) -> KomgaStatus:
        """Sets (or with None, forgets) where the library's folder is on this computer.

        The folder must exist and be writable, and when the library has series, hold some of
        them: a folder that holds none is not the one Komga reads."""
        with self.client() as client:
            library = client.library(library_id)
            if folder:
                path = Path(folder).expanduser()
                problem = _problem(path)
                if problem:
                    raise ValueError(problem)
                sample = [series for _, series in zip(range(SAMPLE_SERIES), client.all_series(library_id))]
                if sample and not any((path / relative_folder(library.root, s.url)).is_dir() for s in sample if s.url):
                    raise ValueError(f"{path} holds none of {library.name}'s series; choose the folder Komga reads as {library.root}")
        with self._lock:
            settings = self.settings()
            if folder:
                settings.folders[library_id] = str(Path(folder).expanduser())
            else:
                settings.folders.pop(library_id, None)
            self._write(settings)
        return self.status()

    def search(self, library_id: str, query: str) -> list[Series]:
        with self.client() as client:
            return client.search(library_id, query)


__all__ = ["KomgaConnection", "KomgaSettings", "KomgaStatus", "LibraryView", "relative_folder"]
