"""JSON routes for app updates (``urakatsu.updates``), under ``/api/v1/updates``.

GitHub calls block, so every route is a plain ``def`` (FastAPI runs it on a thread).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from urakatsu.updates import Updater, updater as default_updater


class ChannelBody(BaseModel):
    channel: str


def updates_router(updater: Updater = default_updater) -> APIRouter:
    router = APIRouter()

    def guard(fn, *args):
        try:
            return fn(*args)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @router.get("/updates")
    def status():
        return updater.status()

    @router.post("/updates/check")
    def check():
        return updater.check()

    @router.put("/updates/channel")
    def set_channel(body: ChannelBody):
        return guard(updater.set_channel, body.channel)

    @router.post("/updates/dismiss")
    def dismiss():
        return updater.dismiss()

    @router.post("/updates/install", status_code=202)
    def install():
        return guard(updater.install)

    return router
