"""JSON routes for the app's workers (Settings, Workers), under ``/api/v1/workers``.

``urakatsu.remote.workers`` keeps the list and finds each worker; these routes add, change, order,
start and stop them. Status checks call the workers and Kaggle, so every route is a
plain ``def`` (FastAPI runs it on a thread). Kaggle tokens never leave the app: replies say only
whether one is set. Colab and other-machine workers take the connection line the worker prints
(``https://<address>#<token>``); a Colab notebook that restarts needs its new line pasted again.
"""

from __future__ import annotations

import logging
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from urakatsu.remote import kaggle, workers
from urakatsu.remote.tunnel import parse_connection
from urakatsu.remote.workers import KaggleAccount, WorkerEntry

log = logging.getLogger(__name__)


class NewWorker(BaseModel):
    kind: Literal["kaggle", "colab", "manual"]
    name: str = ""
    connection: str = ""  # the line the worker printed: "https://<address>#<token>"
    url: str = ""
    token: str = ""
    kaggle_token: str = ""
    kaggle_username: str = ""


class WorkerPatch(BaseModel):
    name: Optional[str] = None
    auto_start: Optional[bool] = None
    connection: Optional[str] = None
    url: Optional[str] = None
    token: Optional[str] = None
    kaggle_token: Optional[str] = None
    kaggle_username: Optional[str] = None


class Order(BaseModel):
    ids: list[str]


NAMES = {"kaggle": "Kaggle", "colab": "Colab", "manual": "Worker"}


ROLE_NAMES = {"detection": "Detection", "ocr": "Text recognition", "inpainting": "Inpainting", "fill": "Generative fill", "sound_effects": "Sound effects"}


def describe(entry: WorkerEntry, status: Optional[dict[str, Any]] = None, runs: Optional[list[dict[str, str]]] = None) -> dict[str, Any]:
    return {
        "id": entry.id,
        "name": entry.name or NAMES[entry.kind],
        "kind": entry.kind,
        "url": (entry.url or None) if entry.kind != "kaggle" else None,
        "token_set": bool(entry.token),
        "auto_start": entry.auto_start,
        "kaggle_username": entry.kaggle.username if entry.kaggle else None,
        "kaggle_token_set": bool(entry.kaggle and entry.kaggle.token),
        "status": status,
        # What the runtime menu sends here (Settings, Pipeline): [{role, provider, name}].
        "runs": runs or [],
    }


def _kaggle_account(token: str, username: str) -> KaggleAccount:
    """Checks the token with Kaggle before keeping it, so a typo shows at once."""
    try:
        client = kaggle.Kaggle(token, username)
        return KaggleAccount(token=client.token, username=client.whoami())
    except kaggle.KaggleError as error:
        raise HTTPException(400, str(error)) from error


def _connect(entry: WorkerEntry, connection: str) -> None:
    """Sets a Colab or other-machine worker's address and token from the line it printed."""
    parsed = parse_connection(connection)
    if parsed is None:
        raise HTTPException(400, "paste the whole line the worker printed, starting with Connection: https://")
    entry.url, entry.token = parsed


def _status(entry: WorkerEntry) -> dict[str, Any]:
    """One worker's status; a check that fails shows on that worker rather than failing the list."""
    try:
        return workers.status(entry)
    except Exception as error:  # noqa: BLE001
        log.warning("could not check worker %s: %s", entry.name or entry.id, error)
        return {"state": "offline", "device": None, "url": None, "detail": f"could not check it: {error}"}


def workers_router() -> APIRouter:
    router = APIRouter()

    @router.get("/workers")
    def list_workers(status: bool = True) -> dict[str, Any]:
        """Every worker, with what it runs; ``status=false`` skips the live checks (the runtime menu)."""
        entries = workers.load()
        states: list[Optional[dict[str, Any]]] = [None] * len(entries)
        if status and entries:
            with ThreadPoolExecutor(max_workers=min(8, len(entries))) as pool:
                states = list(pool.map(_status, entries))
        runs: dict[str, list[dict[str, str]]] = {}
        try:
            for assignment in workers.assignments():
                runs.setdefault(assignment.worker, []).append({"role": ROLE_NAMES[assignment.role], "provider": assignment.provider, "name": assignment.name})
        except Exception as error:  # noqa: BLE001 - the list still shows without it
            log.warning("could not work out what the workers run: %s", error)
        return {
            "workers": [describe(e, s, runs.get(e.id)) for e, s in zip(entries, states)],
            "colab_url": workers.COLAB_URL,
            "kaggle_settings_url": kaggle.SETTINGS_URL,
        }

    @router.post("/workers")
    def add_worker(body: NewWorker) -> dict[str, Any]:
        entry = WorkerEntry(kind=body.kind, name=body.name.strip())
        if body.connection.strip():
            _connect(entry, body.connection)
        elif body.kind == "manual":
            if not body.url.strip():
                raise HTTPException(400, "a worker on another machine needs its address")
            entry.url, entry.token = body.url.strip().rstrip("/"), body.token.strip()
        if body.kind == "kaggle":
            entry.kaggle = _kaggle_account(body.kaggle_token, body.kaggle_username)
            entry.token = secrets.token_urlsafe(24)
        with workers._lock:
            workers.save([*workers.load(), entry])
        return describe(entry)

    @router.patch("/workers/{identifier}")
    def change_worker(identifier: str, body: WorkerPatch) -> dict[str, Any]:
        entry = _find(identifier)
        if body.name is not None:
            entry.name = body.name.strip()
        if body.auto_start is not None:
            entry.auto_start = body.auto_start
        pasted = body.connection is not None and entry.kind != "kaggle"
        if pasted:
            _connect(entry, body.connection)
        elif entry.kind == "manual":
            if body.url is not None:
                entry.url = body.url.strip().rstrip("/")
            if body.token is not None:
                entry.token = body.token.strip()
        if entry.kind == "kaggle" and (body.kaggle_token or body.kaggle_username is not None):
            current = entry.kaggle or KaggleAccount()
            account = _kaggle_account(body.kaggle_token or current.token, body.kaggle_username if body.kaggle_username is not None else current.username)
            account.ref = current.ref if account.username == current.username else ""
            entry.kaggle = account
        workers.update(entry)
        if pasted:  # a notebook that just came up starts on its models while you go back to work
            threading.Thread(target=workers.preload, args=(entry,), name="worker-preload", daemon=True).start()
        return describe(entry)

    @router.delete("/workers/{identifier}")
    def remove_worker(identifier: str) -> dict[str, Any]:
        _find(identifier)
        with workers._lock:
            workers.save([e for e in workers.load() if e.id != identifier])
        return {"ok": True}

    @router.put("/workers/order")
    def order_workers(body: Order) -> dict[str, Any]:
        with workers._lock:
            entries = {e.id: e for e in workers.load()}
            ordered = [entries.pop(i) for i in body.ids if i in entries]
            workers.save(ordered + list(entries.values()))
        return {"ok": True}

    @router.post("/workers/{identifier}/start")
    def start_worker(identifier: str) -> dict[str, Any]:
        entry = _find(identifier)
        try:
            entry = workers.start(entry)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except kaggle.KaggleError as error:
            raise HTTPException(502, str(error)) from error
        return describe(entry, workers.status(entry))

    @router.post("/workers/{identifier}/stop")
    def stop_worker(identifier: str) -> dict[str, Any]:
        entry = _find(identifier)
        if not workers.stop(entry):
            raise HTTPException(409, "the worker does not answer, or was started without a way to stop it")
        return describe(entry)

    @router.get("/workers/{identifier}/quota")
    def worker_quota(identifier: str) -> dict[str, float]:
        entry = _find(identifier)
        if entry.kind != "kaggle":
            raise HTTPException(400, "only Kaggle workers have a GPU quota")
        try:
            return workers.account(entry).quota()
        except kaggle.KaggleError as error:
            raise HTTPException(502, str(error)) from error

    def _find(identifier: str) -> WorkerEntry:
        try:
            return workers.find(identifier)
        except KeyError as error:
            raise HTTPException(404, str(error.args[0])) from error

    return router
