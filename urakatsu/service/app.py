"""Assembles the single service: HTTP (UI + gRPC-web + JSON API) and native gRPC."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from pathlib import Path
from typing import Optional

import grpc
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from urakatsu import __version__
from urakatsu.agents import AgentService

from .agent_api import agent_router
from .catalog_api import catalog_router
from .gen import urakatsu_pb2_grpc as rpc
from .grpc_web import GrpcWebBridge
from .http_api import api_router
from .servicer import UrakatsuServicer
from .sources_api import sources_router
from .updates_api import updates_router
from .workspace import Workspace

log = logging.getLogger(__name__)

# A source checkout serves web/dist; an installed wheel (the desktop app) ships the build in service/ui.
_CHECKOUT_UI = Path(__file__).resolve().parents[2] / "web" / "dist"
DEFAULT_UI_DIR = _CHECKOUT_UI if _CHECKOUT_UI.exists() else Path(__file__).resolve().parent / "ui"


def build_app(
    workspace: Workspace,
    servicer: UrakatsuServicer,
    ui_dir: Optional[Path] = None,
    serve_ui: bool = True,
    cors_origins: Optional[list[str]] = None,
) -> FastAPI:
    app = FastAPI(title="Urakatsu", version=__version__, docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["grpc-status", "grpc-message", "content-type"],
    )
    app.include_router(api_router(workspace), prefix="/api/v1")
    app.include_router(catalog_router(), prefix="/api/v1")
    app.include_router(sources_router(workspace), prefix="/api/v1")
    app.include_router(updates_router(), prefix="/api/v1")
    app.state.agents = AgentService(workspace)
    app.include_router(agent_router(app.state.agents), prefix="/api/v1")

    bridge = GrpcWebBridge(servicer)
    app.add_api_route(bridge.route, bridge.handle, methods=["POST", "OPTIONS"], include_in_schema=False)

    @app.get("/api/health", include_in_schema=False)
    def health():
        return {"ok": True, "version": __version__}

    ui = ui_dir or DEFAULT_UI_DIR
    if serve_ui and ui.exists() and (ui / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=ui / "assets"), name="ui-assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            # An API path that matched no route is a 404, never the app page: the UI parses API answers as JSON.
            if path == "api" or path.startswith("api/"):
                return JSONResponse({"detail": f"no API route /{path}"}, status_code=404)
            candidate = ui / path
            if path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(ui / "index.html")

    else:

        @app.get("/", include_in_schema=False)
        def index():
            return JSONResponse(
                {
                    "service": "urakatsu",
                    "version": __version__,
                    "ui": "not built" if serve_ui else "disabled (api mode)",
                    "api": "/api/v1",
                    "docs": "/api/docs",
                    "grpc_web": bridge.route.replace("{method}", "<Method>"),
                }
            )

    return app


async def serve(
    workspace: Workspace,
    host: str = "127.0.0.1",
    port: int = 8080,
    grpc_port: Optional[int] = 50051,
    ui_dir: Optional[Path] = None,
    serve_ui: bool = True,
    cors_origins: Optional[list[str]] = None,
    log_level: str = "info",
    shutdown: Optional[asyncio.Event] = None,
) -> None:
    """Runs until SIGINT / SIGTERM, or until ``shutdown`` is set (the desktop app runs this on a thread)."""
    servicer = UrakatsuServicer(workspace)
    workspace.systems.start_enabled()
    if not os.environ.get("URAKATSU_NO_UPDATE_CHECK"):
        from urakatsu.updates import updater

        updater.start()  # checks GitHub for a newer release now and every hour
    app = build_app(workspace, servicer, ui_dir, serve_ui, cors_origins)

    grpc_server = None
    if grpc_port:
        grpc_server = grpc.aio.server(
            options=[("grpc.max_receive_message_length", 256 * 1024 * 1024), ("grpc.max_send_message_length", 256 * 1024 * 1024)]
        )
        rpc.add_UrakatsuServicer_to_server(servicer, grpc_server)
        grpc_server.add_insecure_port(f"{host}:{grpc_port}")
        await grpc_server.start()
        log.info("gRPC listening on %s:%d", host, grpc_port)

    config = uvicorn.Config(app, host=host, port=port, log_level=log_level, access_log=False)
    server = uvicorn.Server(config)
    mode = "workspace" if serve_ui else "api"
    log.info("HTTP (%s mode) listening on http://%s:%d", mode, host, port)

    stop_event = shutdown or asyncio.Event()

    def _request_stop(*_):
        stop_event.set()

    loop = asyncio.get_running_loop()
    if shutdown is None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _request_stop)
            except (NotImplementedError, RuntimeError):  # Windows / non-main thread
                signal.signal(sig, _request_stop)

    server_task = asyncio.create_task(server.serve())
    stop_task = asyncio.create_task(stop_event.wait())
    try:
        done, _ = await asyncio.wait({server_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        if stop_task in done:
            server.should_exit = True
            await server_task
    finally:
        stop_task.cancel()
        if grpc_server is not None:
            await grpc_server.stop(grace=2.0)
        await app.state.agents.close()
        await workspace.close()
