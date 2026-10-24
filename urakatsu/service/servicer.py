"""gRPC servicer: thin adapter from protobuf messages to the workspace."""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Awaitable, Callable

import grpc

from urakatsu.core.errors import ProjectBusy, ProviderUnavailable, UrakatsuError
from urakatsu.pipeline import PipelineError
from urakatsu.pipeline.scope import Bounds
from urakatsu.scene import AuthorshipError, SceneError, StalePatch

from . import convert
from .gen import urakatsu_pb2 as pb
from .gen import urakatsu_pb2_grpc as rpc
from .workspace import Workspace

log = logging.getLogger(__name__)


def status_for(error: BaseException) -> grpc.StatusCode:
    if isinstance(error, KeyError):
        return grpc.StatusCode.NOT_FOUND
    if isinstance(error, (ValueError, TypeError)):
        return grpc.StatusCode.INVALID_ARGUMENT
    if isinstance(error, (ProviderUnavailable, AuthorshipError, ProjectBusy)):
        return grpc.StatusCode.FAILED_PRECONDITION
    if isinstance(error, StalePatch):
        return grpc.StatusCode.ABORTED
    if isinstance(error, (SceneError, PipelineError, UrakatsuError)):
        return grpc.StatusCode.INTERNAL
    return grpc.StatusCode.INTERNAL


def message_for(error: BaseException) -> str:
    if isinstance(error, KeyError) and error.args:
        return str(error.args[0])
    return str(error) or error.__class__.__name__


def guarded(method: Callable[..., Awaitable[Any]]):
    @functools.wraps(method)
    async def wrapper(self, request, context):
        try:
            return await method(self, request, context)
        except grpc.RpcError:
            raise
        except Exception as error:
            log.debug("%s failed", method.__name__, exc_info=True)
            await context.abort(status_for(error), message_for(error))

    return wrapper


class UrakatsuServicer(rpc.UrakatsuServicer):
    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    # -- capabilities / configuration ------------------------------------------------------------

    @guarded
    async def GetCapabilities(self, request, context):
        ws = self.workspace
        return convert.capabilities_to_proto(ws.report(), ws.providers(), ws.loaded(), ws.languages())

    @guarded
    async def GetConfig(self, request, context):
        return convert.config_to_proto(self.workspace.get_config())

    @guarded
    async def SetConfig(self, request, context):
        config = convert.config_from_proto(request, self.workspace.get_config())
        return convert.config_to_proto(self.workspace.set_config(config))

    @guarded
    async def ListTranslationModels(self, request, context):
        # Hosted providers list their models over the network (seconds); off the event loop, or
        # every other request (the project, its page images) waits behind it.
        return convert.translation_models_to_proto(await asyncio.to_thread(self.workspace.translation_models))

    # -- projects ------------------------------------------------------------------------------------

    @guarded
    async def ListProjects(self, request, context):
        return pb.ProjectList(projects=[convert.summary_to_proto(s) for s in self.workspace.list_projects()])

    @guarded
    async def CreateProject(self, request, context):
        return convert.project_to_proto(self.workspace.create_project(request.name))

    @guarded
    async def GetProject(self, request, context):
        return convert.project_to_proto(self.workspace.project_view(request.project_id))

    @guarded
    async def RenameProject(self, request, context):
        return convert.project_to_proto(self.workspace.rename_project(request.project_id, request.name))

    @guarded
    async def DeleteProject(self, request, context):
        self.workspace.delete_project(request.project_id)
        return pb.Empty()

    @guarded
    async def Undo(self, request, context):
        return convert.project_to_proto(self.workspace.undo(request.project_id))

    @guarded
    async def Redo(self, request, context):
        return convert.project_to_proto(self.workspace.redo(request.project_id))

    # -- chapters --------------------------------------------------------------------------------------

    @guarded
    async def CreateChapter(self, request, context):
        index = request.index if request.HasField("index") else None
        return convert.project_to_proto(self.workspace.create_chapter(request.project_id, request.name, index))

    @guarded
    async def RenameChapter(self, request, context):
        return convert.project_to_proto(self.workspace.rename_chapter(request.project_id, request.chapter_id, request.name))

    @guarded
    async def DeleteChapter(self, request, context):
        return convert.project_to_proto(self.workspace.delete_chapter(request.project_id, request.chapter_id))

    @guarded
    async def ArrangeProject(self, request, context):
        layout = [(chapter.chapter_id, list(chapter.page_ids)) for chapter in request.chapters]
        return convert.project_to_proto(self.workspace.arrange(request.project_id, layout))

    # -- pages -----------------------------------------------------------------------------------------

    @guarded
    async def AddPage(self, request, context):
        index = request.index if request.HasField("index") else None
        page = self.workspace.add_page(request.project_id, request.name, bytes(request.data), index, request.chapter_id or None)
        return convert.page_to_proto(page)

    @guarded
    async def RemovePage(self, request, context):
        return convert.project_to_proto(self.workspace.remove_page(request.project_id, request.page_id))

    @guarded
    async def GetPage(self, request, context):
        return convert.page_to_proto(self.workspace.page(request.project_id, request.page_id))

    @guarded
    async def GetAsset(self, request, context):
        # Thumbnails and renders decode images; off the event loop so other requests keep moving.
        found = await asyncio.to_thread(self.workspace.asset, request.project_id, request.page_id, request.kind)
        if found is None:
            return pb.Asset(found=False)
        data, content_type, width, height = found
        return pb.Asset(data=data, content_type=content_type, width=width, height=height, found=True)

    # -- text blocks ---------------------------------------------------------------------------------

    @guarded
    async def UpdateTextBlock(self, request, context):
        fields = set(request.fields)
        patch: dict[str, Any] = {}
        if "source_text" in fields:
            patch["source_text"] = request.source_text
        if "translation" in fields:
            patch["translation"] = request.translation
            patch["translation_styles"] = [convert.text_style_from_proto(style) for style in request.translation_styles]
        if "typography" in fields:
            typography = convert.typography_from_proto(request.typography)
            # "typography.<key>" entries name the properties the user changed; only those merge.
            # Without them the whole style is replaced (older clients, the agent).
            changed = {field.removeprefix("typography.") for field in fields if field.startswith("typography.")}
            patch["typography"] = {key: value for key, value in typography.items() if key in changed} if changed else typography
        if "frame" in fields:
            patch["frame"] = [{"x": p.x, "y": p.y} for p in request.frame]
        if "visible" in fields:
            patch["visible"] = request.visible
        page = self.workspace.update_text_block(request.project_id, request.page_id, request.layer_id, patch)
        return convert.page_to_proto(page)

    @guarded
    async def CreateTextBlock(self, request, context):
        b = request.bounds
        page = self.workspace.create_text_block(request.project_id, request.page_id, {"x": b.x, "y": b.y, "width": b.width, "height": b.height})
        return convert.page_to_proto(page)

    @guarded
    async def DeleteTextBlock(self, request, context):
        return convert.page_to_proto(self.workspace.delete_text_block(request.project_id, request.page_id, request.layer_id))

    # -- layers --------------------------------------------------------------------------------------

    @guarded
    async def SetLayerVisibility(self, request, context):
        visible = request.visible if request.HasField("visible") else None
        opacity = request.opacity if request.HasField("opacity") else None
        page = self.workspace.set_layer_visibility(request.project_id, request.page_id, list(request.layer_ids), visible, opacity)
        return convert.page_to_proto(page)

    @guarded
    async def MoveLayer(self, request, context):
        return convert.page_to_proto(self.workspace.move_layer(request.project_id, request.page_id, request.layer_id, request.delta))

    @guarded
    async def SetRasterLayer(self, request, context):
        # Decoding (and for a patch, compositing and encoding) a page-sized layer takes tens of ms: off
        # the event loop, so the UI's other requests keep moving. Commits are safe from any thread.
        page = await asyncio.to_thread(
            self.workspace.set_raster_layer,
            request.project_id,
            request.page_id,
            request.kind or "paint",
            request.layer_id or None,
            bytes(request.png),
            (request.x, request.y) if request.patch else None,
        )
        return convert.page_to_proto(page)

    @guarded
    async def AddRasterLayer(self, request, context):
        page = self.workspace.add_empty_layer(request.project_id, request.page_id, request.name or None, request.above_layer_id or None)
        return convert.page_to_proto(page)

    @guarded
    async def DeleteLayer(self, request, context):
        return convert.page_to_proto(self.workspace.delete_layer(request.project_id, request.page_id, request.layer_id))

    # -- processing -----------------------------------------------------------------------------------

    async def Run(self, request, context):
        try:
            region = None
            if request.HasField("region"):
                r = request.region
                region = Bounds(r.x, r.y, r.width, r.height)
            job = await self.workspace.start_run(
                request.project_id,
                convert.operation_from_proto(request.operation if request.HasField("operation") else None),
                page_ids=list(request.page_ids) or None,
                region=region,
                entities=list(request.entity_ids) or None,
                inpainting_mask=bytes(request.inpainting_mask) or None,
                fill_prompt=request.fill_prompt if request.HasField("fill_prompt") else None,
            )
        except Exception as error:
            await context.abort(status_for(error), message_for(error))
            return
        async for progress in self.workspace.events(job.id):
            page = None
            if progress.page and progress.kind.value in ("finished", "skipped"):
                try:
                    page = self.workspace.page(request.project_id, progress.page)
                except Exception:
                    page = None
            yield convert.progress_to_proto(job.id, progress, page)

    @guarded
    async def Stop(self, request, context):
        self.workspace.stop(request.job_id)
        return pb.Empty()

    @guarded
    async def ListJobs(self, request, context):
        return pb.JobList(
            jobs=[
                pb.Job(id=j.id, project_id=j.project_id, status=j.status, completed=j.completed, total=j.total, started_at=j.started_at, error=j.error or "")
                for j in self.workspace.job_info()
            ]
        )

    @guarded
    async def ListFonts(self, request, context):
        return pb.FontList(families=[pb.FontFamily(name=name, faces=faces) for name, faces in self.workspace.fonts()])
