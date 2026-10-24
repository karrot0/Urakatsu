"""JSON routes for the model catalog and runtime versions, under ``/api/v1``.

``/catalog/*`` searches Hugging Face, rates each model against this machine and
installs variants; ``/runtimes`` lists runtime versions and pins or updates them.
Hub and GitHub calls block, so every route is a plain ``def`` (FastAPI runs it on a thread).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from urakatsu.catalog.service import Catalog

RUNTIMES = ("llama", "diffusion", "torch", "onnx")


class InstallBody(BaseModel):
    runtime: str
    repo: str
    variant: str


class MoveBody(BaseModel):
    position: int  # 0 downloads next


class PinBody(BaseModel):
    tag: Optional[str] = None  # None unpins


class StorageBody(BaseModel):
    models_dir: Optional[str] = None  # None uses the default folder
    move: bool = True  # move the installed models along


@lru_cache(maxsize=1)
def catalog() -> Catalog:
    return Catalog()


def _guard(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except HTTPException:
        raise
    except KeyError as error:
        raise HTTPException(404, str(error).strip("'\"")) from error
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except httpx.HTTPStatusError as error:
        status = error.response.status_code
        if status in (401, 403):
            raise HTTPException(403, "This repository is gated or private. Accept its terms on Hugging Face and set HF_TOKEN.") from error
        if status == 404:
            raise HTTPException(404, "Not found on Hugging Face.") from error
        raise HTTPException(502, f"Hugging Face or GitHub answered {status}.") from error
    except httpx.HTTPError as error:
        raise HTTPException(502, f"Could not reach the network: {error}") from error


def _runtime(runtime: str) -> str:
    if runtime not in RUNTIMES:
        raise HTTPException(400, f"runtime must be one of {', '.join(RUNTIMES)}")
    return runtime


def catalog_router() -> APIRouter:
    router = APIRouter()

    @router.get("/catalog/hardware")
    def hardware():
        return _guard(catalog().hardware)

    @router.get("/catalog/search")
    def search(
        runtime: str = Query("llama"),
        q: str = Query(""),
        sort: str = Query("downloads"),
        limit: int = Query(30, ge=1, le=100),
        usable: bool = Query(False),
    ):
        return _guard(catalog().search, _runtime(runtime), q, sort=sort, limit=limit, usable_only=usable)

    @router.get("/catalog/models/{repo:path}")
    def details(repo: str, runtime: str = Query("llama")):
        return _guard(catalog().details, _runtime(runtime), repo)

    @router.post("/catalog/install", status_code=202)
    def install(body: InstallBody):
        return _guard(catalog().install, _runtime(body.runtime), body.repo, body.variant)

    @router.get("/catalog/downloads")
    def downloads():
        return catalog().list_downloads()

    @router.post("/catalog/downloads/{download_id}/cancel", status_code=202)
    def cancel(download_id: str):
        _guard(catalog().downloads.cancel, download_id)
        return {"ok": True}

    @router.post("/catalog/downloads/{download_id}/pause", status_code=202)
    def pause(download_id: str):
        _guard(catalog().downloads.pause, download_id)
        return {"ok": True}

    @router.post("/catalog/downloads/{download_id}/resume", status_code=202)
    def resume(download_id: str):
        _guard(catalog().downloads.resume, download_id)
        return {"ok": True}

    @router.post("/catalog/downloads/{download_id}/move")
    def move(download_id: str, body: MoveBody):
        _guard(catalog().downloads.move, download_id, body.position)
        return catalog().list_downloads()

    @router.get("/catalog/installed")
    def installed():
        return _guard(catalog().list_installed)

    @router.delete("/catalog/installed/{model_id:path}", status_code=204)
    def delete(model_id: str):
        _guard(catalog().delete, model_id)

    @router.get("/storage")
    def storage():
        return _guard(catalog().storage)

    @router.put("/storage")
    def set_storage(body: StorageBody):
        return _guard(catalog().set_models_dir, body.models_dir, body.move)

    @router.post("/storage/browse")
    def browse():
        """The system folder picker, on the machine Urakatsu runs on; {"path": null} when cancelled."""
        from urakatsu.runtimes.models_folder import pick_folder

        try:
            return {"path": pick_folder(catalog().store.root)}
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        except Exception as error:  # no display, or Tk missing: the UI falls back to typing a path
            raise HTTPException(501, f"No folder picker is available here: {error}") from error

    @router.get("/runtimes")
    def runtimes():
        return _guard(catalog().runtimes)

    @router.get("/runtimes/{package_id}/releases")
    def releases(package_id: str):
        return _guard(catalog().releases, package_id)

    @router.put("/runtimes/{package_id}/pin")
    def pin(package_id: str, body: PinBody):
        return _guard(catalog().set_pin, package_id, body.tag)

    @router.post("/runtimes/{package_id}/update")
    def update(package_id: str):
        return _guard(catalog().update, package_id)

    return router
