"""``urakatsu worker``: serves this machine's detection, OCR and inpainting models to Urakatsu
apps elsewhere (their ``remote`` runtime, ``client.RemoteModel``).

API (every route needs ``Authorization: Bearer <token>`` unless the worker runs without one):

    GET    /v1/info                     protocol, version, device, the providers it can run
    POST   /v1/models                   {stage, provider, settings} -> {id, state}; loads in the background
    GET    /v1/models/{id}?wait=s       {id, state: loading | ready | failed, runtime, weights, failure}
    DELETE /v1/models/{id}              unloads it
    POST   /v1/models/{id}/detect       image (PNG)                             -> job
    POST   /v1/models/{id}/recognize    crops (PNG, repeated)                   -> job
    POST   /v1/models/{id}/inpaint      image (PNG), mask (npz), fill?, prompt? -> job
    GET    /v1/jobs/{job}?wait=s        {job, state: running | done | failed, failure}
    GET    /v1/jobs/{job}/result        detections JSON (``wire``), {texts}, or PNG; then the job is gone

No request runs longer than ``MAX_WAIT``: loads and inference are background jobs, and ``wait``
holds a poll open until the job ends or ``MAX_WAIT`` passes, so the client hears at once when
it is done. Tunnels and proxies close requests that take longer (Cloudflare after 100 s), and a
diffusion model on a notebook GPU can take that long for one crop. A result nobody fetches is
dropped after ``JOB_TTL``.

Like the pipeline, it keeps one model per stage (loading another for a stage unloads the
first), runs one load or inference at a time (one accelerator lane), and on out-of-memory
unloads the other stages' models and retries once. A model's id is a hash of its stage,
provider and settings, so apps that ask for the same model share it. Each provider runs on the
runtime this machine's configuration picks for it (``pipeline.toml``), never ``remote``.

Failures come back as ``{"error", "kind"}``; ``kind`` (unavailable, load, out_of_memory,
processing) lets the client raise the error the pipeline expects.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ValidationError

from urakatsu import __version__
from urakatsu.core.config import PipelineConfig
from urakatsu.core.errors import ModelLoadError, ProviderUnavailable, is_out_of_memory
from urakatsu.core.registry import Registry
from urakatsu.core.registry import registry as default_registry
from urakatsu.core.stage import Stage
from urakatsu.providers.base import Provider
from urakatsu.runtimes.manager import RuntimeManager

from . import PROTOCOL, REMOTE, wire

log = logging.getLogger(__name__)

STAGES = (Stage.DETECTION, Stage.OCR, Stage.INPAINTING)
# Seconds a poll may wait for its job: well under the 100 s that Cloudflare keeps a request open.
MAX_WAIT = 25.0
JOB_TTL = 600.0


class LoadRequest(BaseModel):
    stage: str
    provider: str
    settings: dict[str, Any] = {}


class ModelUnloaded(Exception):
    """The job's model was unloaded while the job waited for the lane; the client loads it again."""


def error_kind(error: BaseException) -> str:
    if isinstance(error, ModelUnloaded):
        return "gone"
    if is_out_of_memory(error):
        return "out_of_memory"
    if isinstance(error, ProviderUnavailable):
        return "unavailable"
    if isinstance(error, ModelLoadError):
        return "load"
    return "processing"


def failure(error: BaseException) -> dict[str, str]:
    return {"error": str(error) or type(error).__name__, "kind": error_kind(error)}


def failed(error: BaseException, status: int = 500) -> JSONResponse:
    body = failure(error)
    return JSONResponse(body, status_code=503 if body["kind"] == "unavailable" else status)


def model_id(stage: Stage, provider: str, settings: dict[str, Any]) -> str:
    key = json.dumps([stage.value, provider, settings], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


@dataclass
class Model:
    id: str
    stage: Stage
    provider: Provider
    state: str = "loading"
    failure: Optional[dict[str, str]] = None
    task: Optional[asyncio.Task] = field(default=None, repr=False)

    def describe(self) -> dict[str, Any]:
        runtime = self.provider.runtime_name
        try:
            weights = [w.describe() for w in self.provider.weight_sources(runtime)]
        except Exception:  # noqa: BLE001
            weights = []
        state = {"id": self.id, "state": self.state, "provider": self.provider.id, "runtime": runtime, "weights": weights, "failure": self.failure}
        if self.stage is Stage.INPAINTING:
            # The generic diffusion provider knows whether it takes edit instructions only once its
            # recipe is loaded (``sound_effects._instructed``); the client has no recipe to ask.
            edits = getattr(getattr(self.provider, "_processor", None), "edits", None)
            state["instructed"] = bool(edits) if edits is not None else bool(getattr(self.provider, "instructed", False))
        return state


@dataclass
class Job:
    id: str
    task: asyncio.Task = field(repr=False)
    body: bytes = b""
    media_type: str = "application/json"
    failure: Optional[dict[str, str]] = None
    finished: Optional[float] = None

    @property
    def state(self) -> str:
        if not self.task.done():
            return "running"
        return "failed" if self.failure is not None else "done"

    def describe(self) -> dict[str, Any]:
        return {"job": self.id, "state": self.state, "failure": self.failure}


async def settle(task: Optional[asyncio.Task], wait: float) -> None:
    """Waits up to ``wait`` seconds (at most ``MAX_WAIT``) for ``task`` to end, without cancelling it."""
    if task is not None and not task.done() and wait > 0:
        await asyncio.wait({task}, timeout=min(wait, MAX_WAIT))


class Worker:
    def __init__(self, config: PipelineConfig, runtimes: RuntimeManager, registry: Registry = default_registry):
        self.config = config
        self.runtimes = runtimes
        self.registry = registry
        self.models: dict[str, Model] = {}
        self.jobs: dict[str, Job] = {}
        self.lane = asyncio.Lock()

    def info(self) -> dict[str, Any]:
        providers = [
            {"id": cls.id, "name": cls.name, "stage": Stage.parse(cls.stage).value, "runtimes": list(cls.runtimes)}
            for stage in STAGES
            for cls in self.registry.list(stage)
        ]
        return {"protocol": PROTOCOL, "version": __version__, "device": self.runtimes.device.label(), "providers": providers}

    def runtime_for(self, cls: type[Provider]) -> Optional[str]:
        """This machine's choice for ``cls``; ``remote`` here would only send the work on again."""
        runtime = self.config.runtime_for(cls.id)
        return runtime if runtime and runtime != REMOTE and runtime in cls.runtimes else None

    async def load(self, request: LoadRequest) -> Model:
        stage = Stage.parse(request.stage)
        if stage not in STAGES:
            raise ValueError(f"a worker runs detection, OCR and inpainting models, not {stage.value}")
        cls = self.registry.get(stage, request.provider)
        identifier = model_id(stage, cls.id, request.settings)
        existing = self.models.get(identifier)
        if existing is not None and existing.state != "failed":
            return existing
        provider = cls(request.settings, self.runtimes, self.runtime_for(cls))
        model = Model(identifier, stage, provider)
        for other in [m for m in self.models.values() if m.stage is stage and m.id != identifier]:
            await self.unload(other.id)
        self.models[identifier] = model
        model.task = asyncio.create_task(self._load(model))
        return model

    async def _load(self, model: Model) -> None:
        try:
            async with self.lane:
                await model.provider.load()
            model.state = "ready"
            log.info("loaded %s on %s", model.provider.id, model.provider.runtime_name)
        except Exception as error:  # noqa: BLE001 - reported to the client that asked for it
            log.warning("could not load %s: %s", model.provider.id, error)
            model.state, model.failure = "failed", failure(error)

    async def unload(self, identifier: str) -> bool:
        model = self.models.pop(identifier, None)
        if model is None:
            return False
        if model.task is not None and not model.task.done():
            await asyncio.wait({model.task})
        async with self.lane:
            model.provider.unload()
        gc.collect()
        return True

    async def unload_all(self) -> None:
        for identifier in list(self.models):
            await self.unload(identifier)

    def ready(self, identifier: str) -> Optional[Model]:
        model = self.models.get(identifier)
        return model if model is not None and model.state == "ready" else None

    async def run(self, model: Model, function, *args: Any) -> Any:
        """``function(*args)`` on the lane; on out-of-memory, again once with the other stages unloaded."""
        async with self.lane:
            if self.models.get(model.id) is not model or model.state != "ready":
                raise ModelUnloaded(f"{model.provider.id} was unloaded before the job ran")
            try:
                return await asyncio.to_thread(function, *args)
            except Exception as error:
                if not is_out_of_memory(error):
                    raise
                log.warning("retrying %s after memory pressure: %s", model.provider.id, error)
                for other in self.models.values():
                    if other is not model and other.state == "ready":
                        other.provider.unload()
                        other.state = "failed"  # the client's next call loads it again
                        other.failure = {"error": "unloaded to free memory", "kind": "load"}
                gc.collect()
                return await asyncio.to_thread(function, *args)

    def submit(self, model: Model, function: Callable[..., Any], args: tuple, encode: Callable[[Any], tuple[bytes, str]]) -> Job:
        """Runs ``function(*args)`` as a job whose result is ``encode``d (body, media type)."""
        now = time.monotonic()
        for stale in [j for j in self.jobs.values() if j.finished is not None and now - j.finished > JOB_TTL]:
            del self.jobs[stale.id]
        identifier = secrets.token_hex(8)

        async def work() -> None:
            job = self.jobs[identifier]
            try:
                job.body, job.media_type = encode(await self.run(model, function, *args))
            except Exception as error:  # noqa: BLE001 - reported to the client that polls
                job.failure = failure(error)
            finally:
                job.finished = time.monotonic()

        self.jobs[identifier] = Job(identifier, asyncio.create_task(work()))
        return self.jobs[identifier]


def create_app(worker: Worker, token: Optional[str]) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await worker.unload_all()  # stops the llama-server / sd-server processes it started

    app = FastAPI(title="Urakatsu worker", version=__version__, docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if token is not None:
            sent = request.headers.get("authorization", "")
            if not hmac.compare_digest(sent.encode("utf-8"), f"Bearer {token}".encode("utf-8")):
                return JSONResponse({"error": "missing or wrong token", "kind": "unavailable"}, status_code=401)
        return await call_next(request)

    def gone(identifier: str) -> JSONResponse:
        return JSONResponse({"error": f"no loaded model {identifier}", "kind": "load"}, status_code=404)

    @app.get("/v1/info")
    async def info():
        return worker.info()

    @app.post("/v1/models")
    async def load(request: LoadRequest):
        try:
            model = await worker.load(request)
        except (KeyError, ValueError, ValidationError) as error:
            return JSONResponse({"error": str(error), "kind": "unavailable"}, status_code=400)
        return model.describe()

    @app.get("/v1/models/{identifier}")
    async def state(identifier: str, wait: float = 0.0):
        model = worker.models.get(identifier)
        if model is None:
            return gone(identifier)
        await settle(model.task, wait)
        return model.describe()

    @app.delete("/v1/models/{identifier}")
    async def unload(identifier: str):
        return Response(status_code=204) if await worker.unload(identifier) else gone(identifier)

    def started(job: Job) -> JSONResponse:
        return JSONResponse(job.describe(), status_code=202)

    def as_json(value: Any) -> tuple[bytes, str]:
        return json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json"

    @app.post("/v1/models/{identifier}/detect")
    async def detect(identifier: str, image: UploadFile = File(...)):
        model = worker.ready(identifier)
        if model is None:
            return gone(identifier)
        try:
            page = wire.decode_image(await image.read())
        except Exception as error:  # noqa: BLE001
            return failed(error, 400)
        return started(worker.submit(model, model.provider._detect, (page,), lambda result: as_json(wire.encode_detections(result))))

    @app.post("/v1/models/{identifier}/recognize")
    async def recognize(identifier: str, crops: list[UploadFile] = File(...)):
        model = worker.ready(identifier)
        if model is None:
            return gone(identifier)
        try:
            images = [wire.decode_image(await crop.read()) for crop in crops]
        except Exception as error:  # noqa: BLE001
            return failed(error, 400)
        return started(worker.submit(model, model.provider._recognize, (images,), lambda texts: as_json({"texts": list(texts)})))

    @app.post("/v1/models/{identifier}/inpaint")
    async def inpaint(
        identifier: str,
        image: UploadFile = File(...),
        mask: UploadFile = File(...),
        fill: bool = Form(False),
        prompt: Optional[str] = Form(None),
    ):
        model = worker.ready(identifier)
        if model is None:
            return gone(identifier)
        try:
            pixels, region = wire.decode_image(await image.read()), wire.decode_array(await mask.read())
        except Exception as error:  # noqa: BLE001
            return failed(error, 400)
        if fill:
            function, args = model.provider._fill, (pixels, region, prompt or None)
        else:
            function, args = model.provider._inpaint, (pixels, region)
        return started(worker.submit(model, function, args, lambda result: (wire.encode_image(result), wire.PNG)))

    def no_job(identifier: str) -> JSONResponse:
        message = f"no job {identifier} (it finished more than {JOB_TTL:.0f} s ago, or the worker restarted)"
        return JSONResponse({"error": message, "kind": "processing"}, status_code=404)

    @app.get("/v1/jobs/{identifier}")
    async def job_state(identifier: str, wait: float = 0.0):
        job = worker.jobs.get(identifier)
        if job is None:
            return no_job(identifier)
        await settle(job.task, wait)
        return job.describe()

    @app.get("/v1/jobs/{identifier}/result")
    async def job_result(identifier: str):
        job = worker.jobs.get(identifier)
        if job is None:
            return no_job(identifier)
        if job.state != "done":
            return JSONResponse({**job.describe(), "error": f"job {identifier} is {job.state}", "kind": "processing"}, status_code=409)
        del worker.jobs[identifier]
        return Response(job.body, media_type=job.media_type)

    return app
