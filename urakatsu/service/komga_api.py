"""JSON routes for Komga (``urakatsu.sources.komga``), under ``/api/v1``.

``/komga`` is the connection (the server, its key, each library's folder here) and searching
its series; ``/projects/{id}/komga`` puts a project on Komga and exports its chapters. Every
call may wait on Komga, so every route is a plain ``def`` (FastAPI runs it on a thread). The
API key is written, never read back.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel

from urakatsu.sources.komga import KomgaError

from .workspace import Workspace


class ConnectionBody(BaseModel):
    url: str
    api_key: Optional[str] = None  # None keeps the saved key


class FolderBody(BaseModel):
    folder: Optional[str] = None  # None goes back to Komga's own path


class LinkBody(BaseModel):
    series_id: str


class AddBody(BaseModel):
    library_id: str
    title: Optional[str] = None


class ExportBody(BaseModel):
    chapter_ids: Optional[list[str]] = None  # None exports every chapter with pages


def _guard(fn, *args):
    try:
        return fn(*args)
    except KeyError as error:
        raise HTTPException(404, str(error.args[0]) if error.args else "not found") from error
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except KomgaError as error:
        raise HTTPException(502, str(error)) from error


def komga_router(workspace: Workspace) -> APIRouter:
    router = APIRouter()
    connection = workspace.komga
    links = workspace.komga_links

    # -- the connection --------------------------------------------------------------------------------

    @router.get("/komga")
    def status():
        return connection.status()

    @router.put("/komga")
    def update(body: ConnectionBody):
        return _guard(connection.update, body.url, body.api_key)

    @router.put("/komga/libraries/{library_id}/folder")
    def set_folder(library_id: str, body: FolderBody):
        return _guard(connection.set_folder, library_id, body.folder)

    @router.get("/komga/libraries/{library_id}/series")
    def search(library_id: str, q: str = Query("")):
        return {"series": [s.model_dump(mode="json") for s in _guard(connection.search, library_id, q)]}

    # -- a project's series ----------------------------------------------------------------------------

    @router.get("/projects/{project_id}/komga")
    def get_link(project_id: str):
        return _guard(links.view, project_id)

    @router.put("/projects/{project_id}/komga")
    def link(project_id: str, body: LinkBody):
        return _guard(links.link, project_id, body.series_id)

    @router.post("/projects/{project_id}/komga")
    def add(project_id: str, body: AddBody):
        return _guard(links.add, project_id, body.library_id, body.title)

    @router.delete("/projects/{project_id}/komga", status_code=204)
    def unlink(project_id: str):
        _guard(links.unlink, project_id)
        return Response(status_code=204)

    @router.post("/projects/{project_id}/komga/export")
    def export(project_id: str, body: ExportBody):
        return _guard(links.start_export, project_id, body.chapter_ids)

    return router


__all__ = ["komga_router"]
