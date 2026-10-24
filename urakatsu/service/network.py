"""The network server: the workspace shared on another address, like LM Studio's local server.

The desktop app's own server listens on 127.0.0.1 for its window only. From Settings > About
the user can also share the workspace on a host and port of their choice (``0.0.0.0`` for the
local network), with its own rules, kept in ``URAKATSU_HOME/server.json``:

- CORS: the origins allowed to call it from a browser page; none by default. The workspace app
  answers every origin for its window, so the listener drops ``Origin`` before handing a
  request on and applies its own list instead.
- API key: when required, every API and gRPC-web call needs ``Authorization: Bearer <key>``,
  ``X-API-Key: <key>`` or the key cookie. Opening ``/?key=<key>`` sets the cookie, so the
  workspace UI (whose image and event-stream requests cannot send headers) works in a browser.
- The workspace UI can be left out, sharing only the API.

``/api/v1/server`` (these settings) is never served on the listener: only the app on this
computer can change who may reach it.

The listener is a second uvicorn server in the same event loop, serving the same app, so it
shares the workspace, its jobs and the accelerator lane with the window.
"""

from __future__ import annotations

import asyncio
import errno
import hmac
import ipaddress
import json
import logging
import secrets
import socket
import sys
from contextlib import contextmanager
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode

import uvicorn
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, RedirectResponse

from urakatsu.core.config import home

log = logging.getLogger(__name__)

COOKIE = "urakatsu_key"
SETTINGS_PATH = "/api/v1/server"


class NetworkSettings(BaseModel):
    autostart: bool = False  # start the listener whenever Urakatsu starts
    host: str = "127.0.0.1"
    port: int = Field(default=8686, ge=1, le=65535)  # not 8090: `urakatsu serve` takes it
    cors_origins: list[str] = Field(default_factory=list)  # "*" allows any origin
    require_key: bool = False
    api_key: str = ""
    serve_ui: bool = True

    @field_validator("host")
    @classmethod
    def _host(cls, value: str) -> str:
        value = value.strip()
        try:
            ipaddress.ip_address(value)
        except ValueError:
            if value != "localhost":
                raise ValueError(f"{value!r} is not an IP address") from None
        return value

    @field_validator("cors_origins")
    @classmethod
    def _origins(cls, value: list[str]) -> list[str]:
        origins = []
        for origin in (o.strip().rstrip("/") for o in value):
            if origin and origin != "*" and not origin.startswith(("http://", "https://")):
                raise ValueError(f"{origin!r} is not an origin (http://host:port or *)")
            if origin and origin not in origins:
                origins.append(origin)
        return origins


class NetworkStatus(BaseModel):
    settings: NetworkSettings
    running: bool
    error: Optional[str] = None
    urls: list[str] = Field(default_factory=list)


def new_key() -> str:
    return f"uk-{secrets.token_urlsafe(24)}"


def lan_addresses() -> list[str]:
    """This computer's IPv4 addresses other machines can reach, for the URLs shown."""
    import psutil

    found = []
    for addresses in psutil.net_if_addrs().values():
        for address in addresses:
            if address.family != socket.AF_INET:
                continue
            ip = ipaddress.ip_address(address.address)
            if not (ip.is_loopback or ip.is_link_local) and address.address not in found:
                found.append(address.address)
    return found


class _Server(uvicorn.Server):
    """A uvicorn server that leaves signals alone: the workspace server owns Ctrl+C."""

    @contextmanager
    def capture_signals(self):
        yield


def _bind(host: str, port: int) -> socket.socket:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        if sys.platform != "win32":  # on Windows SO_REUSEADDR lets two servers share a port
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(2048)
    except OSError:
        sock.close()
        raise
    sock.set_inheritable(True)
    return sock


class NetworkServer:
    """Starts, stops and guards the network listener for one workspace app."""

    def __init__(self, app: Any, path: Optional[Path] = None, grpc_web_prefix: str = "/urakatsu."):
        self.app = app
        self.path = path or home() / "server.json"
        self.grpc_web_prefix = grpc_web_prefix
        self.settings = self._load()
        self.error: Optional[str] = None
        self._server: Optional[_Server] = None
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._gate = self._build_gate()

    # ---- settings --------------------------------------------------------------------------------

    def _load(self) -> NetworkSettings:
        try:
            return NetworkSettings.model_validate(json.loads(self.path.read_text(encoding="utf-8-sig")))
        except (OSError, ValueError):
            return NetworkSettings()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.settings.model_dump_json(indent=1), encoding="utf-8")

    async def update(self, settings: NetworkSettings) -> NetworkStatus:
        """Saves new settings; a running listener moves to a new address, other rules apply at once."""
        if settings.require_key and not settings.api_key:
            settings = settings.model_copy(update={"api_key": new_key()})
        moved = (settings.host, settings.port) != (self.settings.host, self.settings.port)
        self.settings = settings
        self.error = None  # an error belongs to the old settings
        self._save()
        self._gate = self._build_gate()
        if moved and self.running:
            await self.start()
        return self.status()

    async def regenerate_key(self) -> NetworkStatus:
        return await self.update(self.settings.model_copy(update={"api_key": new_key()}))

    # ---- the listener ----------------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> NetworkStatus:
        async with self._lock:
            await self._stop()
            self.error = None
            try:
                sock = _bind(self.settings.host, self.settings.port)
            except OSError as error:
                where = f"{self.settings.host}:{self.settings.port}"
                in_use = error.errno in (errno.EADDRINUSE, getattr(errno, "WSAEADDRINUSE", None), 10048)
                self.error = f"Could not listen on {where}: another program uses port {self.settings.port}." if in_use else f"Could not listen on {where}: {error.strerror or error}"
                log.warning(self.error)
                return self.status()
            # No log config: the workspace server already set up logging.
            config = uvicorn.Config(self, log_config=None, log_level=None, access_log=False, lifespan="off")
            self._server = _Server(config)
            self._task = asyncio.create_task(self._serve(self._server, sock))
            log.info("network server listening on http://%s:%d", self.settings.host, self.settings.port)
        return self.status()

    async def _serve(self, server: _Server, sock: socket.socket) -> None:
        try:
            await server.serve(sockets=[sock])
        except Exception as error:  # noqa: BLE001 - reported in Settings
            log.exception("network server stopped")
            self.error = str(error)

    async def stop(self) -> NetworkStatus:
        async with self._lock:
            await self._stop()
            self.error = None
        return self.status()

    async def _stop(self) -> None:
        if self._server is not None and self._task is not None:
            self._server.should_exit = True
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except asyncio.TimeoutError:
                self._task.cancel()
        self._server = None
        self._task = None

    def status(self) -> NetworkStatus:
        host, port = self.settings.host, self.settings.port
        if host in ("0.0.0.0", "::"):
            hosts = ["127.0.0.1", *lan_addresses()]
        else:
            hosts = [host]
        urls = [f"http://[{h}]:{port}" if ":" in h else f"http://{h}:{port}" for h in hosts]
        return NetworkStatus(settings=self.settings, running=self.running, error=self.error, urls=urls if self.running else [])

    # ---- the gate --------------------------------------------------------------------------------

    def _build_gate(self) -> Any:
        return CORSMiddleware(
            self._guard,
            allow_origins=self.settings.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["grpc-status", "grpc-message", "content-type"],
        )

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            await self._gate(scope, receive, send)

    def _protected(self, path: str) -> bool:
        return path == "/api" or path.startswith("/api/") or path.startswith(self.grpc_web_prefix)

    def _key_ok(self, headers: dict[str, str]) -> bool:
        expected = self.settings.api_key.encode()
        candidates = [headers.get("x-api-key", "")]
        authorization = headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            candidates.append(authorization[7:].strip())
        cookie = SimpleCookie()
        try:
            cookie.load(headers.get("cookie", ""))
        except Exception:  # noqa: BLE001 - a malformed cookie header is no key
            pass
        if COOKIE in cookie:
            candidates.append(cookie[COOKIE].value)
        return any(c and hmac.compare_digest(c.encode(), expected) for c in candidates)

    async def _guard(self, scope, receive, send) -> None:
        settings = self.settings
        path: str = scope["path"]
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        protected = self._protected(path)

        if path == SETTINGS_PATH or path.startswith(SETTINGS_PATH + "/"):
            response = JSONResponse({"detail": "Server settings can only be changed in Urakatsu on this computer."}, status_code=403)
        elif not protected and not settings.serve_ui:
            response = JSONResponse({"detail": "This server shares the API only.", "api": "/api/v1", "docs": "/api/docs"}, status_code=404)
        elif not protected and settings.require_key and scope.get("query_string"):
            response = self._login(scope)
        elif protected and settings.require_key and not self._key_ok(headers):
            response = JSONResponse(
                {"detail": "This server needs an API key: send it as 'Authorization: Bearer <key>' or 'X-API-Key: <key>'."},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        else:
            response = None
        if response is not None:
            await response(scope, receive, send)
            return
        # The workspace app answers every origin (its window needs no CORS); the gate decided already.
        inner = dict(scope, headers=[(k, v) for k, v in scope["headers"] if k.lower() != b"origin"])
        await self.app(inner, receive, send)

    def _login(self, scope) -> Any:
        """``/?key=<key>`` stores the key in a cookie and reloads the page without it."""
        query = parse_qsl(scope["query_string"].decode("latin-1"), keep_blank_values=True)
        keys = [v for k, v in query if k == "key"]
        if not keys:
            return None
        if not hmac.compare_digest(keys[-1].encode(), self.settings.api_key.encode()):
            return JSONResponse({"detail": "Wrong API key."}, status_code=401)
        rest = urlencode([(k, v) for k, v in query if k != "key"])
        response = RedirectResponse(scope["path"] + (f"?{rest}" if rest else ""), status_code=303)
        response.set_cookie(COOKIE, keys[-1], httponly=True, samesite="strict", max_age=400 * 86400)
        return response


# ---- routes ------------------------------------------------------------------------------------------


def network_router(network: NetworkServer) -> APIRouter:
    router = APIRouter()

    @router.get("/server")
    def status() -> NetworkStatus:
        return network.status()

    @router.put("/server")
    async def update(settings: NetworkSettings) -> NetworkStatus:
        return await network.update(settings)

    @router.post("/server/start")
    async def start() -> NetworkStatus:
        result = await network.start()
        if result.error:
            raise HTTPException(409, result.error)
        return result

    @router.post("/server/stop")
    async def stop() -> NetworkStatus:
        return await network.stop()

    @router.post("/server/key")
    async def regenerate_key() -> NetworkStatus:
        return await network.regenerate_key()

    return router

