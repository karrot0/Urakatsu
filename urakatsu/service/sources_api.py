"""JSON routes for source systems, their sources, and project links, under ``/api/v1``.

``/sources/systems/*`` manages each system (start, repositories, extensions);
``/sources/*`` browses the installed sources; ``/projects/{id}/source`` links a project to a
series and downloads its chapters. Scraping blocks on the network, so every route is a
plain ``def`` (FastAPI runs it on a thread).

Series ids, chapter ids and image URLs travel as query parameters: Mana ids contain slashes.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel

from urakatsu.core.config import home
from urakatsu.sources import Image, SourceChallenge, SourceSystemError
from urakatsu.sources.covers import CoverCache

from .workspace import Workspace


class EnabledBody(BaseModel):
    enabled: bool


class RepositoryBody(BaseModel):
    url: str


class LinkBody(BaseModel):
    system: str
    source: str
    series: str


class DownloadBody(BaseModel):
    chapter_ids: list[str]


class BrowserOpenBody(BaseModel):
    url: str


class BrowserInputBody(BaseModel):
    action: str  # click | type | press | scroll | navigate | back | forward | reload
    x: Optional[float] = None
    y: Optional[float] = None
    text: Optional[str] = None
    key: Optional[str] = None
    dx: Optional[float] = None
    dy: Optional[float] = None
    url: Optional[str] = None


def _guard(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except HTTPException:
        raise
    except KeyError as error:
        raise HTTPException(404, str(error.args[0]) if error.args else "not found") from error
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except NotImplementedError as error:
        raise HTTPException(501, str(error)) from error
    except SourceChallenge as error:
        # The UI opens the system's browser at ``url``, then retries.
        raise HTTPException(409, {"message": str(error), "challenge": {"system": error.system, "url": error.url}}) from error
    except SourceSystemError as error:
        raise HTTPException(502, str(error)) from error
    except httpx.HTTPError as error:
        raise HTTPException(502, f"Could not reach the source: {error}") from error


def _image(image: Optional[Image], max_age: int = 86400) -> Response:
    if image is None:
        raise HTTPException(404, "no image")
    return Response(content=image.data, media_type=image.media_type, headers={"cache-control": f"max-age={max_age}"})


SOURCES_TTL = 600.0  # seconds; also refreshed by any extension or system change made here


def sources_router(ws: Workspace) -> APIRouter:
    router = APIRouter()
    systems = ws.systems
    links = ws.links
    covers = CoverCache(home() / "cache" / "covers")
    # Each running system's source list (a GraphQL round trip to Suwayomi, ~0.4 s). Installing,
    # updating or removing an extension, changing a repository, or starting or stopping a system drops it.
    listed: dict[str, tuple[float, list[Any]]] = {}

    def changed(system_id: str, result: Any = None) -> Any:
        listed.pop(system_id, None)
        return result

    def sources_of(system) -> list[Any]:
        cached = listed.get(system.id)
        if cached and time.monotonic() - cached[0] < SOURCES_TTL:
            return cached[1]
        found = system.sources()
        listed[system.id] = (time.monotonic(), found)
        return found

    # -- source systems ------------------------------------------------------------------------------

    @router.get("/sources/systems")
    def list_systems():
        return systems.statuses()

    @router.put("/sources/systems/{system_id}/enabled")
    def set_enabled(system_id: str, body: EnabledBody):
        changed(system_id)
        status = _guard(systems.set_enabled, system_id, body.enabled)
        if body.enabled:
            _guard(systems.get(system_id).start)
            status = systems.status(system_id)
        return status

    @router.post("/sources/systems/{system_id}/start")
    def start(system_id: str):
        _guard(lambda: systems.get(system_id).start())
        return changed(system_id, systems.status(system_id))

    @router.post("/sources/systems/{system_id}/stop")
    def stop(system_id: str):
        _guard(systems.get, system_id).stop()
        return changed(system_id, systems.status(system_id))

    @router.get("/sources/systems/{system_id}/repositories")
    def repositories(system_id: str):
        return _guard(lambda: systems.get(system_id).repositories())

    @router.post("/sources/systems/{system_id}/repositories")
    def add_repository(system_id: str, body: RepositoryBody):
        return changed(system_id, _guard(lambda: systems.get(system_id).add_repository(body.url)))

    @router.delete("/sources/systems/{system_id}/repositories")
    def remove_repository(system_id: str, url: str = Query(...)):
        return changed(system_id, _guard(lambda: systems.get(system_id).remove_repository(url)))

    @router.get("/sources/systems/{system_id}/extensions")
    def extensions(system_id: str, refresh: bool = Query(False)):
        return _guard(lambda: systems.get(system_id).extensions(refresh))

    @router.post("/sources/systems/{system_id}/extensions/{extension_id}/install")
    def install(system_id: str, extension_id: str):
        return changed(system_id, _guard(lambda: systems.get(system_id).install_extension(extension_id)))

    @router.post("/sources/systems/{system_id}/extensions/{extension_id}/update")
    def update(system_id: str, extension_id: str):
        return changed(system_id, _guard(lambda: systems.get(system_id).update_extension(extension_id)))

    @router.delete("/sources/systems/{system_id}/extensions/{extension_id}")
    def uninstall(system_id: str, extension_id: str):
        return changed(system_id, _guard(lambda: systems.get(system_id).uninstall_extension(extension_id)))

    @router.get("/sources/systems/{system_id}/extensions/{extension_id}/icon")
    def extension_icon(system_id: str, extension_id: str):
        return _image(_guard(lambda: systems.get(system_id).extension_icon(extension_id)))

    # -- built-in browser ------------------------------------------------------------------------------

    @router.post("/sources/systems/{system_id}/browser")
    def browser_open(system_id: str, body: BrowserOpenBody):
        return {"id": _guard(lambda: systems.get(system_id).browser_open(body.url))}

    @router.get("/sources/systems/{system_id}/browser/{session}/frame")
    def browser_frame(system_id: str, session: str):
        return _guard(lambda: systems.get(system_id).browser_frame(session))

    @router.post("/sources/systems/{system_id}/browser/{session}/input", status_code=204)
    def browser_input(system_id: str, session: str, body: BrowserInputBody):
        _guard(lambda: systems.get(system_id).browser_input(session, body.model_dump(exclude_none=True)))
        return Response(status_code=204)

    @router.delete("/sources/systems/{system_id}/browser/{session}", status_code=204)
    def browser_close(system_id: str, session: str):
        _guard(lambda: systems.get(system_id).browser_close(session))
        return Response(status_code=204)

    # -- sources ---------------------------------------------------------------------------------------

    @router.get("/sources")
    def list_sources():
        """Every enabled system with its installed sources; a failing system reports its error."""
        result = []
        enabled = systems.enabled()
        for system in systems.all():
            if system.id in enabled and system.status().state != "running":
                changed(system.id)
                systems.start_background(system.id)  # booting (Suwayomi takes a while) holds up nothing
            entry = {"system": systems.status(system.id), "sources": [], "error": None}
            if system.id in enabled and entry["system"].state == "running":
                try:
                    entry["sources"] = sources_of(system)
                except Exception as error:  # noqa: BLE001 - one broken system must not hide the others
                    entry["error"] = str(error) or error.__class__.__name__
                entry["system"] = systems.status(system.id)
            result.append(entry)
        return result

    @router.get("/sources/systems/{system_id}/sources/{source_id}/search")
    def search(system_id: str, source_id: str, q: str = Query(""), page: int = Query(1, ge=1)):
        system = _guard(systems.get, system_id)
        found = _guard(lambda: system.search(source_id, q, page))
        # The browser asks for these covers next, six at a time; start them all now.
        covers.prefetch(
            [(CoverCache.key(system_id, source_id, s.cover), lambda url=s.cover: system.cover(source_id, url)) for s in found.results if s.cover]
        )
        return found

    @router.get("/sources/systems/{system_id}/sources/{source_id}/series")
    def series(system_id: str, source_id: str, id: str = Query(...)):
        return _guard(lambda: systems.get(system_id).series(source_id, id))

    @router.get("/sources/systems/{system_id}/sources/{source_id}/chapters")
    def chapters(system_id: str, source_id: str, series: str = Query(...)):
        return _guard(lambda: systems.get(system_id).chapters(source_id, series))

    @router.get("/sources/systems/{system_id}/sources/{source_id}/cover")
    def cover(system_id: str, source_id: str, url: str = Query(...)):
        # Fetched once, at tile size (urakatsu/sources/covers.py); the browser keeps it a week.
        key = CoverCache.key(system_id, source_id, url)
        return _image(_guard(lambda: covers.get(key, lambda: systems.get(system_id).cover(source_id, url))), max_age=7 * 24 * 3600)

    # -- project links ---------------------------------------------------------------------------------

    @router.get("/projects/{project_id}/source")
    def get_link(project_id: str):
        return _guard(links.view, project_id)

    @router.put("/projects/{project_id}/source")
    def link(project_id: str, body: LinkBody):
        return _guard(links.link, project_id, body.system, body.source, body.series)

    @router.delete("/projects/{project_id}/source", status_code=204)
    def unlink(project_id: str):
        _guard(links.unlink, project_id)
        return Response(status_code=204)

    @router.post("/projects/{project_id}/source/refresh")
    def refresh(project_id: str):
        return _guard(links.refresh, project_id)

    @router.post("/projects/{project_id}/source/downloads")
    def download(project_id: str, body: DownloadBody):
        return _guard(links.download, project_id, body.chapter_ids)

    @router.delete("/projects/{project_id}/source/downloads")
    def cancel(project_id: str, chapter: str = Query(...)):
        return _guard(links.cancel, project_id, chapter)

    @router.get("/projects/{project_id}/cover")
    def project_cover(project_id: str):
        """The linked series' cover, else the first page's thumbnail."""
        found = _guard(links.cover, project_id)
        if found is not None:
            return _image(found)
        pages = _guard(lambda: ws.session(project_id).snapshot().pages())
        if not pages:
            raise HTTPException(404, "the project has no cover")
        thumbnail = _guard(ws.asset, project_id, pages[0], "thumbnail")
        if thumbnail is None:
            raise HTTPException(404, "the project has no cover")
        return Response(content=thumbnail[0], media_type=thumbnail[1], headers={"cache-control": "no-cache"})

    return router
