"""A client for the part of Komga's REST API that exporting needs.

Komga serves the books it finds in its libraries' folders and has no upload route, so the
books themselves are written to the library's folder (``urakatsu.sources.komga.links``). The API
lists libraries and series, starts scans, and sets a series' metadata and cover. Requests
send a Komga API key (``X-API-Key``; Komga 1.13 or newer: Account Settings, API Keys).

Calls block on the network; callers run them on worker threads.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional

import httpx
from pydantic import BaseModel

from urakatsu.core.errors import UrakatsuError

TIMEOUT = 15.0
PAGE_SIZE = 100


class KomgaError(UrakatsuError):
    """Komga refused a request, or could not be reached."""


class Library(BaseModel):
    id: str
    name: str
    root: str  # the library's folder, as Komga sees it


class Series(BaseModel):
    id: str
    library_id: str
    title: str
    url: str  # the series' folder, as Komga sees it
    books: int = 0


def _series(data: dict[str, Any]) -> Series:
    metadata = data.get("metadata") or {}
    return Series(
        id=data["id"],
        library_id=data["libraryId"],
        title=metadata.get("title") or data.get("name") or "",
        url=data.get("url") or "",
        books=int(data.get("booksCount") or 0),
    )


class KomgaClient:
    def __init__(self, url: str, api_key: str, transport: Optional[httpx.BaseTransport] = None, timeout: float = TIMEOUT):
        self.url = url.rstrip("/")
        self._client = httpx.Client(base_url=self.url, headers={"X-API-Key": api_key}, transport=transport, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "KomgaClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as error:
            raise KomgaError(f"cannot reach Komga at {self.url}: {error}") from error
        if response.status_code == 401:
            raise KomgaError("Komga rejected the API key; make one in Komga under Account Settings, API Keys")
        if response.status_code == 403:
            raise KomgaError("the Komga account behind this API key may not do that (exporting needs an admin account)")
        if response.status_code >= 400:
            try:
                message = response.json().get("message") or response.text
            except (ValueError, AttributeError):  # not JSON, or not an object
                message = response.text
            raise KomgaError(f"Komga answered {response.status_code} to {method} {path}: {str(message)[:200]}")
        return response

    def _json(self, method: str, path: str, **kwargs) -> Any:
        """The answer's JSON; an answer that is not JSON (a web page at a wrong address) is a KomgaError."""
        response = self._request(method, path, **kwargs)
        try:
            return response.json()
        except ValueError as error:
            raise KomgaError(f"{self.url} did not answer like Komga (not JSON); check the address") from error

    # -- reading -----------------------------------------------------------------------------------

    def user(self) -> str:
        """The email of the account the key belongs to; also checks the address and the key."""
        return str(self._json("GET", "/api/v2/users/me").get("email", ""))

    def version(self) -> Optional[str]:
        """Komga's version, when the server tells it."""
        try:
            return self._json("GET", "/actuator/info")["build"]["version"]
        except (KomgaError, ValueError, KeyError, TypeError):
            return None

    def libraries(self) -> list[Library]:
        return [Library(id=item["id"], name=item["name"], root=item["root"]) for item in self._json("GET", "/api/v1/libraries")]

    def library(self, library_id: str) -> Library:
        item = self._json("GET", f"/api/v1/libraries/{library_id}")
        return Library(id=item["id"], name=item["name"], root=item["root"])

    def search(self, library_id: str, query: str, size: int = 20) -> list[Series]:
        params = {"library_id": library_id, "search": query, "size": size} if query.strip() else {"library_id": library_id, "size": size, "sort": "lastModified,desc"}
        return [_series(item) for item in self._json("GET", "/api/v1/series", params=params)["content"]]

    def series(self, series_id: str) -> Series:
        return _series(self._json("GET", f"/api/v1/series/{series_id}"))

    def all_series(self, library_id: str) -> Iterator[Series]:
        """Every series in the library, a page at a time."""
        page = 0
        while True:
            data = self._json("GET", "/api/v1/series", params={"library_id": library_id, "page": page, "size": PAGE_SIZE})
            yield from (_series(item) for item in data["content"])
            if data.get("last", True):
                return
            page += 1

    # -- writing -----------------------------------------------------------------------------------

    def scan(self, library_id: str) -> None:
        """Asks Komga to scan the library now; the scan runs in the background."""
        self._request("POST", f"/api/v1/libraries/{library_id}/scan")

    def update_series(self, series_id: str, metadata: dict[str, Any]) -> None:
        """Sets series metadata (Komga's field names: ``title``, ``summary``, ``status``...)."""
        self._request("PATCH", f"/api/v1/series/{series_id}/metadata", json=metadata)

    def upload_series_cover(self, series_id: str, data: bytes, media_type: str) -> None:
        """Adds a cover to the series and makes it the one shown."""
        extension = {"image/png": "png", "image/webp": "webp"}.get(media_type, "jpg")
        self._request("POST", f"/api/v1/series/{series_id}/thumbnails", params={"selected": "true"}, files={"file": (f"cover.{extension}", data, media_type)})

    def series_page(self, series_id: str) -> str:
        """The series in Komga's web reader."""
        return f"{self.url}/series/{series_id}"


__all__ = ["KomgaClient", "KomgaError", "Library", "Series"]
