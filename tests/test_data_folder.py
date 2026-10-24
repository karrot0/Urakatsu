"""The About page shows the data folder: ``GET /api/v1/data-folder`` answers ``core.config.home()``."""

from pathlib import Path


def test_data_folder_is_the_home_folder(isolated_home: Path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from urakatsu.service.http_api import api_router
    from urakatsu.service.workspace import Workspace

    app = FastAPI()
    app.include_router(api_router(Workspace(device="cpu")), prefix="/api/v1")
    client = TestClient(app)

    assert client.get("/api/v1/data-folder").json() == {"path": str(isolated_home)}
