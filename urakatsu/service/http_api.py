"""JSON HTTP API (the ``listen`` surface) mirroring the gRPC contract.

Routes live under ``/api/v1``. Progress streams as Server-Sent Events. The
one-shot ``POST /api/v1/process`` runs the whole pipeline on an uploaded image
and returns the rendered page.
"""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import quote

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from urakatsu.core.config import PipelineConfig, home
from urakatsu.core.stage import Operation, Stage
from urakatsu.pipeline.scope import Bounds
from urakatsu.pipeline.stitch import ProjectFormat

from .servicer import message_for, status_for
from .workspace import Workspace

HTTP_STATUS = {
    "NOT_FOUND": 404,
    "INVALID_ARGUMENT": 400,
    "FAILED_PRECONDITION": 412,
    "ABORTED": 409,
    "INTERNAL": 500,
}


class BoxBody(BaseModel):
    x: float
    y: float
    width: float
    height: float


class RunBody(BaseModel):
    operation: str = "full"  # full | through | only | stages
    stage: Optional[str] = None
    stages: list[str] = Field(default_factory=list)
    page_ids: list[str] = Field(default_factory=list)
    region: Optional[BoxBody] = None
    entity_ids: list[str] = Field(default_factory=list)
    # Redo: stages run on pages they already did, keeping what the user edited.
    force: bool = False


class TextBlockPatch(BaseModel):
    model_config = {"extra": "forbid"}
    source_text: Optional[str] = None
    translation: Optional[str] = None
    typography: Optional[dict[str, Any]] = None
    frame: Optional[list[dict[str, float]]] = None
    visible: Optional[bool] = None


class VisibilityBody(BaseModel):
    layer_ids: list[str]
    visible: Optional[bool] = None
    opacity: Optional[float] = None


class MoveBody(BaseModel):
    delta: int


class ChapterBody(BaseModel):
    name: str = ""
    index: Optional[int] = None


class ChapterLayoutBody(BaseModel):
    id: str
    page_ids: list[str]


class ArrangeBody(BaseModel):
    chapters: list[ChapterLayoutBody]


def operation(body: RunBody) -> Operation:
    if body.operation == "full":
        return Operation.full()
    if body.operation == "through":
        return Operation.through(Stage.parse(body.stage or ""))
    if body.operation == "only":
        return Operation.only(Stage.parse(body.stage or ""))
    if body.operation == "stages":
        return Operation.select([Stage.parse(s) for s in body.stages])
    raise ValueError(f"unknown operation {body.operation!r}")


def http_error(error: Exception) -> HTTPException:
    return HTTPException(HTTP_STATUS.get(status_for(error).name, 500), message_for(error))


def api_router(ws: Workspace) -> APIRouter:
    router = APIRouter()

    def guard(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as error:
            raise http_error(error)

    # -- capabilities / configuration --------------------------------------------------------------

    @router.get("/capabilities")
    def capabilities():
        return ws.capabilities()

    @router.get("/memory")
    def memory():
        return guard(ws.memory)

    @router.get("/data-folder")
    def data_folder() -> dict[str, str]:
        return {"path": str(home())}

    @router.get("/config")
    def get_config() -> PipelineConfig:
        return ws.get_config()

    @router.put("/config")
    def put_config(config: PipelineConfig) -> PipelineConfig:
        return guard(ws.set_config, config)

    @router.patch("/config")
    def patch_config(patch: dict[str, Any]) -> PipelineConfig:
        merged = ws.get_config().model_dump(mode="json")
        _deep_update(merged, patch)
        return guard(lambda: ws.set_config(PipelineConfig.model_validate(merged)))

    @router.get("/translation/models")
    def translation_models():
        return guard(ws.translation_models)

    @router.get("/fonts")
    def fonts():
        return [{"name": name, "faces": faces} for name, faces in guard(ws.fonts)]

    @router.get("/fonts/{family}/face")
    def font_face(family: str, weight: int = 400, italic: bool = False):
        found = guard(ws.font_face, family, weight, italic)
        if found is None:
            raise HTTPException(status_code=404, detail=f"no loadable face for {family!r}")
        data, media = found
        return Response(content=data, media_type=media, headers={"cache-control": "public, max-age=86400"})

    # -- projects ----------------------------------------------------------------------------------

    @router.get("/projects")
    def list_projects():
        return ws.list_projects()

    @router.post("/projects", status_code=201)
    def create_project(body: dict[str, Any]):
        return guard(ws.create_project, str(body.get("name", "Untitled")))

    @router.get("/projects/{project_id}")
    def get_project(project_id: str):
        return guard(ws.project_view, project_id)

    @router.patch("/projects/{project_id}")
    def rename_project(project_id: str, body: dict[str, Any]):
        return guard(ws.rename_project, project_id, str(body.get("name", "Untitled")))

    @router.put("/projects/{project_id}/format")
    def set_project_format(project_id: str, body: ProjectFormat):
        return guard(ws.set_project_format, project_id, body)

    @router.delete("/projects/{project_id}", status_code=204)
    def delete_project(project_id: str):
        ws.delete_project(project_id)
        return Response(status_code=204)

    @router.post("/projects/{project_id}/undo")
    def undo(project_id: str):
        return guard(ws.undo, project_id)

    @router.post("/projects/{project_id}/redo")
    def redo(project_id: str):
        return guard(ws.redo, project_id)

    # -- chapters ----------------------------------------------------------------------------------------

    @router.post("/projects/{project_id}/chapters", status_code=201)
    def create_chapter(project_id: str, body: ChapterBody):
        return guard(ws.create_chapter, project_id, body.name, body.index)

    @router.patch("/projects/{project_id}/chapters/{chapter_id}")
    def rename_chapter(project_id: str, chapter_id: str, body: ChapterBody):
        return guard(ws.rename_chapter, project_id, chapter_id, body.name)

    @router.get("/projects/{project_id}/chapters/{chapter_id}/export")
    def export_chapter(project_id: str, chapter_id: str, format: str = "cbz"):
        data, filename = guard(ws.export_chapter, project_id, chapter_id, format)
        media = {"cbz": "application/vnd.comicbook+zip", "strip": "image/png"}.get(format, "application/zip")
        ascii_name = filename.encode("ascii", "replace").decode().replace("?", "_").replace('"', "_")
        disposition = f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"
        return Response(content=data, media_type=media, headers={"content-disposition": disposition})

    @router.post("/projects/{project_id}/chapters/{chapter_id}/restitch")
    async def restitch_chapter(project_id: str, chapter_id: str):
        try:
            return await ws.restitch_chapter(project_id, chapter_id)
        except Exception as error:
            raise http_error(error)

    @router.delete("/projects/{project_id}/chapters/{chapter_id}")
    def delete_chapter(project_id: str, chapter_id: str):
        return guard(ws.delete_chapter, project_id, chapter_id)

    @router.put("/projects/{project_id}/arrangement")
    def arrange(project_id: str, body: ArrangeBody):
        return guard(ws.arrange, project_id, [(chapter.id, chapter.page_ids) for chapter in body.chapters])

    # -- pages -------------------------------------------------------------------------------------------

    @router.post("/projects/{project_id}/pages", status_code=201)
    async def add_pages(
        project_id: str,
        files: list[UploadFile] = File(...),
        index: Optional[int] = Form(None),
        chapter_id: Optional[str] = Form(None),
    ):
        pages = []
        for offset, upload in enumerate(files):
            data = await upload.read()
            position = index + offset if index is not None else None
            pages.append(guard(ws.add_page, project_id, upload.filename or "page", data, position, chapter_id))
        return pages

    @router.get("/projects/{project_id}/pages/{page_id}")
    def get_page(project_id: str, page_id: str):
        return guard(ws.page, project_id, page_id)

    @router.delete("/projects/{project_id}/pages/{page_id}")
    def remove_page(project_id: str, page_id: str):
        return guard(ws.remove_page, project_id, page_id)

    @router.get("/projects/{project_id}/pages/{page_id}/text-layout")
    def text_layout(project_id: str, page_id: str, v: Optional[str] = None):
        return guard(ws.text_layout, project_id, page_id)

    @router.get("/projects/{project_id}/pages/{page_id}/assets/{kind}")
    def asset(project_id: str, page_id: str, kind: str, v: Optional[str] = None):
        found = guard(ws.asset, project_id, page_id, kind)
        if found is None:
            raise HTTPException(404, f"page {page_id} has no {kind} asset")
        data, content_type, _, _ = found
        # ``v`` is the page revision the URL was built for (assetHref in the UI); every edit
        # bumps it, so a versioned URL always names the same image and the browser may keep it.
        cache = "private, max-age=31536000, immutable" if v else "no-cache"
        return Response(content=data, media_type=content_type, headers={"cache-control": cache})

    # -- text blocks -------------------------------------------------------------------------------------

    @router.get("/projects/{project_id}/pages/{page_id}/blocks/{layer_id}/lift")
    def lift_block(project_id: str, page_id: str, layer_id: str, frame: Optional[str] = None):
        # ``frame`` ("x,y,x,y,..."): lift the block as it would render with that frame.
        values = [float(v) for v in frame.split(",")] if frame else None
        outline = [{"x": x, "y": y} for x, y in zip(values[::2], values[1::2])] if values else None
        return guard(ws.lift, project_id, page_id, layer_id, outline)

    @router.patch("/projects/{project_id}/pages/{page_id}/blocks/{layer_id}")
    def update_block(project_id: str, page_id: str, layer_id: str, patch: TextBlockPatch):
        return guard(ws.update_text_block, project_id, page_id, layer_id, patch.model_dump(exclude_unset=True))

    @router.post("/projects/{project_id}/pages/{page_id}/blocks", status_code=201)
    def create_block(project_id: str, page_id: str, bounds: BoxBody):
        return guard(ws.create_text_block, project_id, page_id, bounds.model_dump())

    @router.delete("/projects/{project_id}/pages/{page_id}/blocks/{layer_id}")
    def delete_block(project_id: str, page_id: str, layer_id: str):
        return guard(ws.delete_text_block, project_id, page_id, layer_id)

    # -- layers ------------------------------------------------------------------------------------------

    @router.patch("/projects/{project_id}/pages/{page_id}/layers/visibility")
    def layer_visibility(project_id: str, page_id: str, body: VisibilityBody):
        return guard(ws.set_layer_visibility, project_id, page_id, body.layer_ids, body.visible, body.opacity)

    @router.post("/projects/{project_id}/pages/{page_id}/layers/{layer_id}/move")
    def move_layer(project_id: str, page_id: str, layer_id: str, body: MoveBody):
        return guard(ws.move_layer, project_id, page_id, layer_id, body.delta)

    @router.put("/projects/{project_id}/pages/{page_id}/rasters/{kind}")
    async def set_raster(project_id: str, page_id: str, kind: str, file: UploadFile = File(...), layer_id: Optional[str] = Form(None)):
        return guard(ws.set_raster_layer, project_id, page_id, kind, layer_id, await file.read())

    @router.delete("/projects/{project_id}/pages/{page_id}/layers/{layer_id}")
    def delete_layer(project_id: str, page_id: str, layer_id: str):
        return guard(ws.delete_layer, project_id, page_id, layer_id)

    # -- jobs -------------------------------------------------------------------------------------------

    @router.post("/projects/{project_id}/run", status_code=202)
    async def run(project_id: str, body: RunBody):
        try:
            region = Bounds(**body.region.model_dump()) if body.region else None
            job = await ws.start_run(
                project_id, operation(body), page_ids=body.page_ids or None, region=region, entities=body.entity_ids or None, force=body.force
            )
        except Exception as error:
            raise http_error(error)
        return {"job_id": job.id, "events": f"/api/v1/jobs/{job.id}/events"}

    @router.get("/jobs")
    def jobs():
        return ws.job_info()

    @router.get("/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request):
        if job_id not in ws.jobs:
            raise HTTPException(404, f"unknown job {job_id}")

        async def stream():
            async for event in ws.events(job_id):
                if await request.is_disconnected():
                    return
                yield f"event: progress\ndata: {event.model_dump_json()}\n\n"
            yield "event: end\ndata: {}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"cache-control": "no-cache"})

    @router.post("/jobs/{job_id}/stop", status_code=202)
    def stop(job_id: str):
        guard(ws.stop, job_id)
        return {"job_id": job_id, "stopping": True}

    # -- one-shot processing ---------------------------------------------------------------------------

    @router.post("/process")
    async def process(
        file: UploadFile = File(...),
        operation_kind: str = Query("full", alias="operation"),
        stage: Optional[str] = Query(None),
        output: str = Query("render", pattern="^(render|cleanup|json)$"),
        keep: bool = Query(False, description="keep the temporary project"),
    ):
        """Runs the pipeline on one image and returns the rendered page (or the page view as JSON)."""
        data = await file.read()
        project = ws.create_project(f"one-shot {file.filename}")
        try:
            page = ws.add_page(project.id, file.filename or "page", data)
            job = await ws.start_run(project.id, operation(RunBody(operation=operation_kind, stage=stage)))
            await job.task
            if job.status == "failed":
                raise HTTPException(500, job.error or "pipeline failed")
            if output == "json":
                return JSONResponse(ws.page(project.id, page.id).model_dump(mode="json"))
            found = ws.asset(project.id, page.id, output)
            if found is None:
                raise HTTPException(409, f"pipeline produced no {output} image")
            return Response(content=found[0], media_type=found[1])
        except HTTPException:
            raise
        except Exception as error:
            raise http_error(error)
        finally:
            if not keep:
                ws.delete_project(project.id)

    return router


def _deep_update(target: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
