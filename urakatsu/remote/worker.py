"""``urakatsu worker``: serves this machine's detection, OCR, inpainting, censor bar and
colorization models to Urakatsu apps elsewhere (their ``remote`` runtime, ``client.RemoteModel``).

API (every route needs ``Authorization: Bearer <token>`` unless the worker runs without one):

    GET    /v1/info                     protocol, version, device, the providers it can run
    POST   /v1/models                   {stage, provider, settings, installed?} -> {id, state}; loads in the background
    POST   /v1/preload                  [{stage, provider, settings, installed?}] -> 202; loads or downloads them early
    GET    /v1/models/{id}?wait=s       {id, state: loading | ready | failed, runtime, weights, failure}
    DELETE /v1/models/{id}              unloads it
    POST   /v1/models/{id}/detect       image (PNG)                             -> job
    POST   /v1/models/{id}/recognize    crops (PNG, repeated)                   -> job
    POST   /v1/models/{id}/inpaint      image (PNG), mask (npz), fill?, prompt? -> job
    POST   /v1/models/{id}/colorize     image (PNG), reference? (PNG)           -> job
    POST   /v1/models/{id}/detect_censorship  image (PNG)                       -> job (npz mask)
    POST   /v1/models/{id}/decensor     image (PNG), mask (npz), white          -> job
    GET    /v1/files/{sha256}           {received, complete}
    PUT    /v1/files/{sha256}?name&size&offset   the next part of a file    -> {received, complete}
    GET    /v1/jobs/{job}?wait=s        {job, state: running | done | failed, failure}
    GET    /v1/jobs/{job}/result        detections JSON (``wire``), {texts}, or PNG; then the job is gone
    POST   /v1/shutdown                 ends the worker (the app's Stop button), when it was started to allow it

No request runs longer than ``MAX_WAIT``: loads and inference are background jobs, and ``wait``
holds a poll open until the job ends or ``MAX_WAIT`` passes, so the client hears at once when
it is done. Tunnels and proxies close requests that take longer (Cloudflare after 100 s), and a
diffusion model on a notebook GPU can take that long for one crop. A result nobody fetches is
dropped after ``JOB_TTL``.

``installed`` is a model the app installed from its Models page (the registry entry,
``catalog.installed.InstalledModel``): the worker registers it in its own store and downloads its
files, since only the app knows what it installed.

A setting that names a file on the app's machine (a LoRA, ``Provider.remote_files``) reaches the
worker as ``upload:<sha256>/<name>``: the client sends the file first through ``/v1/files``, in
parts small enough for a tunnel (Cloudflare refuses request bodies over 100 MB), and skips one
the worker already has. The worker checks the digest and gives the provider its own copy.

Preloading (``/v1/preload``, ``urakatsu worker --preload``) takes the models the app assigned to
this worker: the first of each stage loads, the others only download (one model per stage stays
loaded), so the first page does not wait for weights.

Like the pipeline, it keeps one model per stage (loading another for a stage unloads the
first), runs one load or inference at a time (one accelerator lane), and on out-of-memory
unloads the other stages' models and retries once. A model's id is a hash of its stage,
provider and settings, so apps that ask for the same model share it. Each provider runs on the
runtime this machine's configuration picks for it (``pipeline.toml``), never ``remote`` or ``cloud``.

A worker on a notebook (``urakatsu worker --tunnel``) opens its own tunnel (``tunnel``) and can
end itself after a while without requests (``--idle-exit``), so a session nobody uses stops
spending the account's GPU quota.

Failures come back as ``{"error", "kind"}``; ``kind`` (unavailable, load, out_of_memory,
processing) lets the client raise the error the pipeline expects.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
import re
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
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
from urakatsu.runtimes.torch_runtime import release_memory

from . import OFFLOADED, PROTOCOL, wire

log = logging.getLogger(__name__)

STAGES = (Stage.DETECTION, Stage.OCR, Stage.INPAINTING, Stage.CENSOR_DETECTION, Stage.DECENSOR, Stage.COLORIZATION)
# Seconds a poll may wait for its job: well under the 100 s that Cloudflare keeps a request open.
MAX_WAIT = 25.0
JOB_TTL = 600.0
UPLOAD = "upload:"
UPLOAD_LIMIT = 8 * 2**30
DIGEST = re.compile(r"[0-9a-f]{64}")


class LoadRequest(BaseModel):
    stage: str
    provider: str
    settings: dict[str, Any] = {}
    # The app's Models page entry for ``settings["model"]``, when it names an installed model.
    installed: Optional[dict[str, Any]] = None


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


def local(cls: type[Provider]) -> bool:
    """Whether a worker can run ``cls`` itself (not a model only Urakatsu Cloud runs)."""
    return not cls.runtimes or any(runtime not in OFFLOADED for runtime in cls.runtimes)


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
        self.active = time.monotonic()
        # Ends the process; set by ``urakatsu worker``, so ``/v1/shutdown`` and ``--idle-exit`` work.
        self.stop: Optional[Callable[[], None]] = None

    def touch(self) -> None:
        self.active = time.monotonic()

    def idle(self) -> float:
        """Seconds since the last request that used a model, or 0 while one runs."""
        return 0.0 if self.lane.locked() else time.monotonic() - self.active

    def info(self) -> dict[str, Any]:
        providers = [
            {"id": cls.id, "name": cls.name, "stage": Stage.parse(cls.stage).value, "runtimes": list(cls.runtimes)}
            for stage in STAGES
            for cls in self.registry.list(stage)
            if local(cls)
        ]
        return {
            "protocol": PROTOCOL,
            "version": __version__,
            "device": self.runtimes.device.label(),
            "providers": providers,
            "stoppable": self.stop is not None,
        }

    def runtime_for(self, cls: type[Provider]) -> Optional[str]:
        """This machine's choice for ``cls``; ``remote`` or ``cloud`` here would only send the work on again."""
        runtime = self.config.runtime_for(cls.id)
        return runtime if runtime and runtime not in OFFLOADED and runtime in cls.runtimes else None

    async def load(self, request: LoadRequest) -> Model:
        stage = Stage.parse(request.stage)
        if stage not in STAGES:
            raise ValueError(f"a worker runs detection, OCR, inpainting, censor bar and colorization models, not {stage.value}")
        cls = self.registry.get(stage, request.provider)
        if not local(cls):
            raise ValueError(f"{cls.id} runs only on Urakatsu Cloud, not on a worker")
        identifier = model_id(stage, cls.id, request.settings)
        existing = self.models.get(identifier)
        if existing is not None and existing.state != "failed":
            return existing
        provider = self.provider(cls, request)
        model = Model(identifier, stage, provider)
        for other in [m for m in self.models.values() if m.stage is stage and m.id != identifier]:
            await self.unload(other.id)
        self.models[identifier] = model
        model.task = asyncio.create_task(self._load(model))
        return model

    def provider(self, cls: type[Provider], request: LoadRequest) -> Provider:
        """``cls`` with the request's settings on this machine's runtime, its installed model registered."""
        if request.installed is not None:
            from urakatsu.catalog.installed import InstalledModel, InstalledModels

            installed = InstalledModel.model_validate(request.installed)
            if installed.id != request.settings.get("model"):
                raise ValueError(f"the installed model sent is {installed.id}, not the one the settings name")
            InstalledModels(self.runtimes.store).add(installed)  # its files download when the model loads
        return cls(self.uploaded(request.settings), self.runtimes, self.runtime_for(cls))

    # -- files sent by the app ------------------------------------------------------------------

    def uploads(self) -> Path:
        return self.runtimes.store.root / "uploads"

    def _upload(self, digest: str) -> tuple[Optional[Path], Path]:
        """The finished file sent under ``digest``, if there is one, and where its parts go."""
        if not DIGEST.fullmatch(digest):
            raise ValueError(f"{digest!r} is not a SHA-256 digest")
        folder = self.uploads() / digest
        done = next((path for path in folder.iterdir() if path.is_file()), None) if folder.is_dir() else None
        return done, self.uploads() / f"{digest}.part"

    def upload_state(self, digest: str) -> dict[str, Any]:
        done, part = self._upload(digest)
        if done is not None:
            return {"received": done.stat().st_size, "complete": True}
        return {"received": part.stat().st_size if part.exists() else 0, "complete": False}

    def receive(self, digest: str, name: str, size: int, offset: int, data: bytes) -> dict[str, Any]:
        """Writes the part of a file at ``offset``; once all ``size`` bytes are in, checks them."""
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError(f"{name!r} is not a file name")
        if not 0 < size <= UPLOAD_LIMIT:
            raise ValueError(f"a file sent to a worker must be under {UPLOAD_LIMIT // 2**30} GiB")
        state = self.upload_state(digest)
        if state["complete"]:
            return state
        _, part = self._upload(digest)
        if offset != state["received"] or offset + len(data) > size:
            raise ValueError(f"expected bytes from {state['received']}, got {len(data)} at {offset}")
        part.parent.mkdir(parents=True, exist_ok=True)
        with part.open("ab") as file:
            file.write(data)
        received = offset + len(data)
        if received < size:
            return {"received": received, "complete": False}
        hasher = hashlib.sha256()
        with part.open("rb") as file:
            for block in iter(lambda: file.read(2**20), b""):
                hasher.update(block)
        if hasher.hexdigest() != digest:
            part.unlink()
            raise ValueError(f"{name} arrived damaged (its SHA-256 differs); send it again")
        target = self.uploads() / digest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        part.replace(target)
        log.info("received %s (%d bytes)", name, size)
        return {"received": size, "complete": True}

    def uploaded(self, settings: dict[str, Any]) -> dict[str, Any]:
        """``settings`` with each ``upload:<sha256>/<name>`` replaced by this worker's copy."""
        resolved = dict(settings)
        for key, value in settings.items():
            if isinstance(value, str) and value.startswith(UPLOAD):
                digest, _, name = value[len(UPLOAD) :].partition("/")
                done, _ = self._upload(digest)
                if done is None:
                    raise ValueError(f"{name or digest} was not sent to this worker; the app sends it before loading the model")
                resolved[key] = str(done)
        return resolved

    async def fetch(self, request: LoadRequest) -> None:
        """Downloads a model's weights without loading it."""
        cls = self.registry.get(Stage.parse(request.stage), request.provider)
        provider = self.provider(cls, request)
        await asyncio.to_thread(provider.resolve_weights)

    async def preload(self, requests: list[LoadRequest]) -> None:
        """Loads the first model of each stage and downloads the rest, one after another."""
        loaded: set[Stage] = set()
        for request in requests:
            try:
                stage = Stage.parse(request.stage)
                if stage in loaded:
                    await self.fetch(request)
                    log.info("downloaded %s ahead of use", request.provider)
                    continue
                loaded.add(stage)
                model = await self.load(request)
                if model.task is not None:
                    await asyncio.wait({model.task})
            except Exception as error:  # noqa: BLE001 - a preload that fails is retried when the app asks
                log.warning("could not preload %s: %s", request.provider, error)

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
        release_memory()
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
                log.warning("retrying %s after memory pressure: %s", model.provider.id, str(error))
                for other in self.models.values():
                    if other is not model and other.state == "ready":
                        other.provider.unload()
                        other.state = "failed"  # the client's next call loads it again
                        other.failure = {"error": "unloaded to free memory", "kind": "load"}
            # Outside the ``except``: while it runs, the exception holds the failed attempt's tensors.
            release_memory()
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


def create_app(worker: Worker, token: Optional[str], preload: Optional[list[LoadRequest]] = None) -> FastAPI:
    tasks: set[asyncio.Task] = set()

    def start_preload(requests: list[LoadRequest]) -> None:
        task = asyncio.create_task(worker.preload(requests))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        if preload:
            start_preload(preload)
        yield
        for task in tasks:
            task.cancel()
        await worker.unload_all()  # stops the llama-server / sd-server processes it started

    app = FastAPI(title="Urakatsu worker", version=__version__, docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if token is not None:
            sent = request.headers.get("authorization", "")
            if not hmac.compare_digest(sent.encode("utf-8"), f"Bearer {token}".encode("utf-8")):
                return JSONResponse({"error": "missing or wrong token", "kind": "unavailable"}, status_code=401)
        if request.url.path != "/v1/info":  # the app's status checks do not keep a worker alive
            worker.touch()
        return await call_next(request)

    def gone(identifier: str) -> JSONResponse:
        return JSONResponse({"error": f"no loaded model {identifier}", "kind": "load"}, status_code=404)

    @app.get("/v1/info")
    async def info():
        return worker.info()

    @app.post("/v1/preload")
    async def preload_models(requests: list[LoadRequest]):
        start_preload(requests)
        return Response(status_code=202)

    @app.post("/v1/shutdown")
    async def shutdown():
        if worker.stop is None:
            return JSONResponse({"error": "this worker was not started to be stopped remotely", "kind": "unavailable"}, status_code=501)
        asyncio.get_running_loop().call_later(0.5, worker.stop)  # after this answer is sent
        return Response(status_code=202)

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

    @app.post("/v1/models/{identifier}/colorize")
    async def colorize(identifier: str, image: UploadFile = File(...), reference: Optional[UploadFile] = File(None)):
        model = worker.ready(identifier)
        if model is None:
            return gone(identifier)
        try:
            page = wire.decode_image(await image.read())
            guide = wire.decode_image(await reference.read()) if reference is not None else None
        except Exception as error:  # noqa: BLE001
            return failed(error, 400)
        return started(worker.submit(model, model.provider._colorize, (page, guide), lambda result: (wire.encode_image(result), wire.PNG)))

    @app.post("/v1/models/{identifier}/detect_censorship")
    async def detect_censorship(identifier: str, image: UploadFile = File(...)):
        model = worker.ready(identifier)
        if model is None:
            return gone(identifier)
        try:
            page = wire.decode_image(await image.read())
        except Exception as error:  # noqa: BLE001
            return failed(error, 400)
        return started(worker.submit(model, model.provider._detect_censorship, (page,), lambda mask: (wire.encode_array(mask), wire.NPZ)))

    @app.post("/v1/models/{identifier}/decensor")
    async def decensor(identifier: str, image: UploadFile = File(...), mask: UploadFile = File(...), white: bool = Form(False)):
        model = worker.ready(identifier)
        if model is None:
            return gone(identifier)
        try:
            pixels, region = wire.decode_image(await image.read()), wire.decode_array(await mask.read())
        except Exception as error:  # noqa: BLE001
            return failed(error, 400)
        return started(worker.submit(model, model.provider._reconstruct, (pixels, region, white), lambda result: (wire.encode_image(result), wire.PNG)))

    @app.get("/v1/files/{digest}")
    async def file_state(digest: str):
        try:
            return worker.upload_state(digest)
        except ValueError as error:
            return failed(error, 400)

    @app.put("/v1/files/{digest}")
    async def upload(digest: str, request: Request, name: str, size: int, offset: int):
        try:
            data = await request.body()
            return await asyncio.to_thread(worker.receive, digest, name, size, offset, data)
        except ValueError as error:
            return failed(error, 400)

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
