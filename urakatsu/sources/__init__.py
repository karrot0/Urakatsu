"""Source systems: scraper runtimes that link projects to series on manga sites.

``base`` defines the layer every system implements, ``registry`` loads the systems, and
``links`` ties a project to one series and downloads its chapters.
"""

from .base import (
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
from .registry import SourceSystems

__all__ = [
    "BrowserFrame",
    "Extension",
    "Image",
    "RemoteChapter",
    "Repository",
    "SearchPage",
    "Series",
    "SourceChallenge",
    "SourceInfo",
    "SourceSystem",
    "SourceSystemError",
    "SourceSystems",
    "SystemStatus",
]
