"""Komga: projects exported as series of a Komga server (https://komga.org).

``client`` speaks Komga's REST API, ``connection`` keeps the server, its key and each
library's folder on this computer, ``archive`` packs a chapter as a CBZ book with
``ComicInfo.xml``, and ``links`` ties a project to one series and exports its chapters.
"""

from .archive import BookInfo, comic_info, pack_cbz
from .client import KomgaClient, KomgaError, Library, Series
from .connection import KomgaConnection, KomgaSettings, KomgaStatus, LibraryView
from .links import ExportReport, ExportState, KomgaLink, KomgaLinks, KomgaLinkView

__all__ = [
    "BookInfo",
    "ExportReport",
    "ExportState",
    "KomgaClient",
    "KomgaConnection",
    "KomgaError",
    "KomgaLink",
    "KomgaLinkView",
    "KomgaLinks",
    "KomgaSettings",
    "KomgaStatus",
    "Library",
    "LibraryView",
    "Series",
    "comic_info",
    "pack_cbz",
]
