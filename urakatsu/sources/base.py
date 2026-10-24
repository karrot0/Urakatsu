"""The source-system layer: what every scraper runtime offers Urakatsu.

Three levels, from the outside in:

* a **source system** is a runtime that runs scrapers (Suwayomi Server, Mana bundles);
* a **source** is one website a source system can scrape;
* a **series** is one title (manga, manhua, manhwa) on a source.

A source system also installs its own scrapers: it keeps a list of repositories, and
each repository offers extensions that provide sources. Everything above this module
(project links, chapter downloads, the UI) talks to :class:`SourceSystem` only, so a new
runtime plugs in by subclassing it and registering (``urakatsu.sources.registry``).

Calls block on the network; callers run them on worker threads.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, Optional

from pydantic import BaseModel, Field

from urakatsu.core.errors import UrakatsuError

if TYPE_CHECKING:  # pragma: no cover
    from urakatsu.runtimes.packages import PackageManager

SystemState = Literal["not_installed", "stopped", "starting", "running", "error"]


class SourceSystemError(UrakatsuError):
    """A source system or the site behind a source failed a request."""


class SourceChallenge(SourceSystemError):
    """The site asks for a check (Cloudflare and the like) that a person has to clear.

    ``url`` is the page to open in the system's browser; the request works once it is cleared.
    """

    def __init__(self, message: str, url: str, system: Optional[str] = None):
        super().__init__(message)
        self.url = url
        self.system = system


class SystemStatus(BaseModel):
    id: str
    name: str
    description: str
    homepage: str
    state: SystemState
    runtime: Optional[str] = None  # runtime package id (urakatsu.runtimes.packages)
    version: Optional[str] = None
    message: Optional[str] = None
    enabled: bool = False
    browser: Optional[str] = None  # the built-in browser, for systems that have one


class Repository(BaseModel):
    url: str
    name: str
    extensions: int = 0


class Extension(BaseModel):
    """An installable scraper. Its sources appear once it is installed."""

    id: str
    name: str
    version: str = ""
    language: str = ""
    icon: Optional[str] = None  # URL, possibly relative to /api/v1/sources
    nsfw: bool = False
    installed: bool = False
    has_update: bool = False
    repository: Optional[str] = None
    description: str = ""


class SourceInfo(BaseModel):
    id: str
    name: str
    language: str = ""
    icon: Optional[str] = None
    nsfw: bool = False
    extension: Optional[str] = None
    website: Optional[str] = None


class Series(BaseModel):
    id: str
    title: str
    cover: Optional[str] = None  # as the source gives it; fetch it with SourceSystem.cover
    url: Optional[str] = None  # the series page on the site
    summary: str = ""
    authors: list[str] = Field(default_factory=list)
    status: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class SearchPage(BaseModel):
    results: list[Series]
    has_more: bool = False


class RemoteChapter(BaseModel):
    """A chapter as the source lists it."""

    id: str
    name: str
    number: Optional[float] = None
    volume: Optional[float] = None
    scanlator: Optional[str] = None
    language: Optional[str] = None
    date: Optional[float] = None  # unix seconds


class BrowserFrame(BaseModel):
    """One look at an interactive browser session: a JPEG of the page and where it is."""

    image: str  # base64 JPEG
    url: str
    title: str
    loading: bool
    challenge: bool  # the page still shows a site check
    width: int
    height: int


@dataclass(frozen=True)
class Image:
    data: bytes
    media_type: str


class SourceSystem(ABC):
    """One scraper runtime. Subclasses register with :func:`urakatsu.sources.registry.register`."""

    id: ClassVar[str]
    name: ClassVar[str]
    description: ClassVar[str]
    homepage: ClassVar[str]

    def __init__(self, root: Path, packages: "PackageManager"):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.packages = packages

    # -- lifecycle ---------------------------------------------------------------------------------

    @abstractmethod
    def status(self) -> SystemStatus: ...

    @abstractmethod
    def start(self) -> SystemStatus:
        """Installs the runtime when missing and starts it. Idempotent."""

    @abstractmethod
    def stop(self) -> None: ...

    def close(self) -> None:
        self.stop()

    # -- installer: repositories and extensions ----------------------------------------------------

    @abstractmethod
    def repositories(self) -> list[Repository]: ...

    @abstractmethod
    def add_repository(self, url: str) -> list[Repository]: ...

    @abstractmethod
    def remove_repository(self, url: str) -> list[Repository]: ...

    @abstractmethod
    def extensions(self, refresh: bool = False) -> list[Extension]: ...

    @abstractmethod
    def install_extension(self, extension_id: str) -> list[Extension]: ...

    @abstractmethod
    def uninstall_extension(self, extension_id: str) -> list[Extension]: ...

    def update_extension(self, extension_id: str) -> list[Extension]:
        return self.install_extension(extension_id)

    def extension_icon(self, extension_id: str) -> Optional[Image]:
        return None

    # -- scraping ----------------------------------------------------------------------------------

    @abstractmethod
    def sources(self) -> list[SourceInfo]:
        """Installed sources that can search and serve chapters."""

    @abstractmethod
    def search(self, source_id: str, query: str, page: int = 1) -> SearchPage: ...

    @abstractmethod
    def series(self, source_id: str, series_id: str) -> Series: ...

    @abstractmethod
    def chapters(self, source_id: str, series_id: str) -> list[RemoteChapter]:
        """The series' chapters in reading order, oldest first."""

    @abstractmethod
    def pages(self, source_id: str, series_id: str, chapter_id: str) -> list[str]:
        """Opaque page handles, in reading order, for :meth:`page`."""

    @abstractmethod
    def page(self, source_id: str, series_id: str, chapter_id: str, handle: str) -> Image: ...

    @abstractmethod
    def cover(self, source_id: str, url: str) -> Optional[Image]:
        """A cover image by the URL a :class:`Series` carries (sites may need the source's headers)."""

    def source_icon(self, source_id: str) -> Optional[Image]:
        return None

    # -- built-in browser (optional) ---------------------------------------------------------------
    # A system with a browser lets a person clear site checks and sign in, the way the app's
    # WebView does. Sessions are driven remotely: the UI shows frames and sends input.

    def browser_open(self, url: str) -> str:
        raise NotImplementedError(f"{self.name} has no built-in browser")

    def browser_frame(self, session: str) -> BrowserFrame:
        raise NotImplementedError(f"{self.name} has no built-in browser")

    def browser_input(self, session: str, event: dict) -> None:
        raise NotImplementedError(f"{self.name} has no built-in browser")

    def browser_close(self, session: str) -> None:
        raise NotImplementedError(f"{self.name} has no built-in browser")
