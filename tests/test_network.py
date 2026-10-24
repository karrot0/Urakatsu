"""The network server (urakatsu.service.network): its gate (API key, CORS, settings routes) and listener."""

import socket

import httpx
import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from urakatsu.service.network import COOKIE, NetworkServer, NetworkSettings, network_router


def workspace_app() -> FastAPI:
    """Stands in for the workspace app: answers every origin, like build_app's."""
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/api/v1/projects")
    def projects():
        return []

    @app.post("/urakatsu.v1.Urakatsu/{method}")
    def grpc_web(method: str):
        return {"method": method}

    @app.get("/{path:path}")
    def page(path: str):
        return {"page": path}

    app.state.network = NetworkServer(app, grpc_web_prefix="/urakatsu.v1.Urakatsu/")
    app.include_router(network_router(app.state.network), prefix="/api/v1")
    return app


def over_network(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app.state.network), base_url="http://lan")


def over_window(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.asyncio
async def test_api_key_guards_the_api_but_not_the_page():
    app = workspace_app()
    async with over_window(app) as window:
        status = (await window.put("/api/v1/server", json={"require_key": True})).json()
    key = status["settings"]["api_key"]
    assert key.startswith("uk-")  # requiring a key makes one

    async with over_network(app) as lan:
        assert (await lan.get("/api/v1/projects")).status_code == 401
        assert (await lan.post("/urakatsu.v1.Urakatsu/ListProjects")).status_code == 401
        assert (await lan.get("/api/v1/projects", headers={"Authorization": f"Bearer {key}"})).status_code == 200
        assert (await lan.get("/api/v1/projects", headers={"X-API-Key": key})).status_code == 200
        assert (await lan.get("/api/v1/projects", headers={"X-API-Key": key + "x"})).status_code == 401
        assert (await lan.get("/projects/abc")).json() == {"page": "projects/abc"}

        assert (await lan.get("/?key=wrong")).status_code == 401
        login = await lan.get("/projects?key=" + key + "&tab=1")
        assert login.status_code == 303 and login.headers["location"] == "/projects?tab=1"
        assert login.cookies[COOKIE] == key
        assert (await lan.get("/api/v1/projects", headers={"Cookie": f"{COOKIE}={key}"})).status_code == 200


@pytest.mark.asyncio
async def test_settings_are_not_reachable_over_the_network():
    app = workspace_app()
    async with over_network(app) as lan:
        assert (await lan.get("/api/v1/server")).status_code == 403
        assert (await lan.put("/api/v1/server", json={"require_key": False})).status_code == 403
    async with over_window(app) as window:
        assert (await window.get("/api/v1/server")).status_code == 200


@pytest.mark.asyncio
async def test_cors_follows_the_listeners_own_origins():
    app = workspace_app()
    async with over_network(app) as lan:
        response = await lan.get("/api/v1/projects", headers={"Origin": "http://evil.example"})
        assert response.status_code == 200 and "access-control-allow-origin" not in response.headers
        preflight = {"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"}
        assert (await lan.options("/api/v1/projects", headers=preflight)).status_code == 400

    await app.state.network.update(NetworkSettings(cors_origins=["http://tool.example/"]))
    async with over_network(app) as lan:
        response = await lan.get("/api/v1/projects", headers={"Origin": "http://tool.example"})
        assert response.headers["access-control-allow-origin"] == "http://tool.example"
        preflight = {"Origin": "http://tool.example", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization"}
        assert (await lan.options("/api/v1/projects", headers=preflight)).status_code == 200


@pytest.mark.asyncio
async def test_api_only_hides_the_ui():
    app = workspace_app()
    await app.state.network.update(NetworkSettings(serve_ui=False))
    async with over_network(app) as lan:
        assert (await lan.get("/")).status_code == 404
        assert (await lan.get("/api/v1/projects")).status_code == 200


def test_settings_validation():
    with pytest.raises(ValueError):
        NetworkSettings(host="my laptop")
    with pytest.raises(ValueError):
        NetworkSettings(cors_origins=["tool.example"])
    assert NetworkSettings(cors_origins=["*", "https://a.example/", "https://a.example"]).cors_origins == ["*", "https://a.example"]


@pytest.mark.asyncio
async def test_listener_starts_moves_and_stops(isolated_home):
    app = workspace_app()
    network: NetworkServer = app.state.network
    first, second = free_port(), free_port()
    await network.update(NetworkSettings(port=first, autostart=True))
    status = await network.start()
    try:
        assert status.running and status.urls == [f"http://127.0.0.1:{first}"]
        async with httpx.AsyncClient() as client:
            assert (await client.get(f"http://127.0.0.1:{first}/api/v1/projects")).json() == []
            assert (await client.get(f"http://127.0.0.1:{first}/api/v1/server")).status_code == 403

            status = await network.update(network.settings.model_copy(update={"port": second}))
            assert status.running
            assert (await client.get(f"http://127.0.0.1:{second}/api/v1/projects")).status_code == 200
            with pytest.raises(httpx.ConnectError):
                await client.get(f"http://127.0.0.1:{first}/api/v1/projects")

        # A port in use is reported, not raised.
        with socket.socket() as taken:
            taken.bind(("127.0.0.1", 0))
            taken.listen()
            await network.update(network.settings.model_copy(update={"port": taken.getsockname()[1]}))
            assert not network.running and network.error and "another program uses port" in network.error
    finally:
        status = await network.stop()
    assert not status.running and status.urls == []
    assert NetworkServer(app, path=isolated_home / "server.json").settings.autostart  # saved
