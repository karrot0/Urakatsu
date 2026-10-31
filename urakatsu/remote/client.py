"""The ``remote`` runtime's client: one provider's model, loaded on a Urakatsu worker.

A provider whose runtime is ``remote`` keeps its class, settings and stage behavior (tiling,
``prompted``, ``instructed``); only ``_load`` / ``_unload`` and the inference methods go to the
worker (``Provider.__init__`` binds them here). The worker runs the provider's own methods on
its runtime, so a remote run produces what a local one would.

- Where: the app's worker the model is assigned to (``remote_worker``, picked in the runtime
  menu; Settings, Workers), which may start a Kaggle worker first. Without one, ``remote_url`` in
  the provider's settings, else ``URAKATSU_REMOTE_URL``; the token is ``remote_api_key``, else
  ``URAKATSU_REMOTE_API_KEY``, sent as a bearer token. With none of those, the first of the app's
  workers that answers (``workers.connect``).
- A model the user installed from the Models page goes along with the load request
  (``installed``): the worker has none of the app's installs. On the ``cloud`` runtime, Urakatsu Cloud and the account key
  instead (``cloud``).
- A Kaggle worker gets a new tunnel address when its session restarts: when it stops answering,
  the client reads its address from the session's log again and repeats the request there once.
- Loading and every inference call are jobs on the worker: the request returns at once, and the
  client polls with ``wait`` (the worker holds each poll open up to ``POLL_WAIT`` seconds, so it
  answers as soon as the job ends), then fetches the result. No request stays open longer than
  that, however long the model takes: tunnels and proxies close requests after a while
  (Cloudflare after 100 s), and a first load downloads weights, or a diffusion model on a
  notebook GPU takes over a minute for a crop.
- A worker that restarted (a notebook reconnecting) or unloaded the model for another client
  answers 404 for it; the client loads it again and repeats the call once.
- Stopping the run stops the waiting: inside a pipeline run the client polls in short holds
  (``STOP_POLL_WAIT``) and raises ``StoppedError`` once the run's token is set, so the
  accelerator lane frees within seconds. The job on the worker is abandoned, not cancelled
  (neither the worker nor Urakatsu Cloud can cancel one); its result expires unfetched.
- Step previews of diffusion models are not streamed from the worker.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Callable, Optional

import httpx
import numpy as np
from PIL import Image

from urakatsu.core.errors import ModelLoadError, OutOfMemory, ProcessingError, ProviderUnavailable
from urakatsu.core.results import LayoutDetections
from urakatsu.core.stop import current as current_stop

from . import CLOUD, PROTOCOL, cloud, wire, workers

if TYPE_CHECKING:  # pragma: no cover
    from urakatsu.providers.base import Provider

log = logging.getLogger(__name__)

URL_ENV = "URAKATSU_REMOTE_URL"
KEY_ENV = "URAKATSU_REMOTE_API_KEY"
TIMEOUT = httpx.Timeout(30.0, read=None)
# Seconds the worker may hold a poll open (it caps this at its ``MAX_WAIT``).
POLL_WAIT = 20.0
STOP_POLL_WAIT = 2.0
# Settings that say where to connect or hold the user's own keys, not how the model runs: the
# worker uses its own, and a key never leaves this machine.
CONNECTION_SETTINGS = frozenset({"remote_url", "remote_api_key", "remote_worker", "server_url", "server_api_key", "api_key"})

# Tests swap this for one that returns a client bound to an in-process worker.
client_factory: Callable[[str, dict[str, str]], httpx.Client] = lambda url, headers: httpx.Client(
    base_url=url, headers=headers, timeout=TIMEOUT
)


def error_for(kind: Optional[str], message: str) -> Exception:
    """The error a worker failure stands for, keeping its kind so the pipeline reacts the same way
    (an out-of-memory error makes it unload the other models and retry)."""
    text = f"worker: {message}"
    if kind == "out_of_memory":
        return OutOfMemory(text)
    if kind == "unavailable":
        return ProviderUnavailable(text)
    if kind == "load":
        return ModelLoadError(text)
    return ProcessingError(text)


def worker_error(response: httpx.Response, action: str, on_cloud: bool = False) -> Exception:
    if response.status_code == 401:
        if on_cloud:
            return ProviderUnavailable(f"Urakatsu Cloud rejected the key when asked to {action}; make a new one at {cloud.KEYS_URL}")
        return ProviderUnavailable(f"the worker rejected the token when asked to {action}; check the remote API key")
    if response.status_code == 402:
        try:
            message = str(response.json().get("error"))
        except (ValueError, AttributeError):
            message = response.text[:500]
        return ProviderUnavailable(f"Urakatsu Cloud: {message}")
    try:
        body = response.json()
        return error_for(body.get("kind"), str(body.get("error") or response.text))
    except (ValueError, AttributeError):
        return error_for("unavailable" if response.status_code == 503 else None, response.text[:500] or f"HTTP {response.status_code}")


def _poll_wait() -> float:
    """How long one poll may wait: short inside a run, so a stop is noticed quickly."""
    return POLL_WAIT if current_stop() is None else STOP_POLL_WAIT


def load_request(provider: "Provider") -> dict[str, Any]:
    """What ``POST /v1/models`` (and a preload) sends a worker for ``provider``: its stage, its
    settings without where it connects, and the installed model those settings name, if any."""
    body: dict[str, Any] = {
        "stage": provider.stage.value,
        "provider": provider.id,
        "settings": provider.settings.model_dump(mode="json", exclude=CONNECTION_SETTINGS),
    }
    model = getattr(provider.settings, "model", None)
    if model:
        from urakatsu.catalog.installed import InstalledModels

        installed = InstalledModels(provider.store()).get(model)
        if installed is None:
            raise ProviderUnavailable(f"{provider.id}: installed model {model!r} was removed; pick another on the Models page")
        body["installed"] = installed.model_dump(mode="json")
    return body


class ModelGone(Exception):
    """The worker no longer has the model (it restarted or swapped it out)."""


class RemoteModel:
    def __init__(self, provider: "Provider"):
        self.provider = provider
        self.client: Optional[httpx.Client] = None
        self.model_id: Optional[str] = None
        # One of the app's workers, when the settings name no address (``pooled``).
        self.endpoint: Optional[workers.Endpoint] = None
        # Set once a request moved to the worker's new address: its jobs did not move along.
        self.moved = False

    # -- connection -----------------------------------------------------------------------------

    @property
    def on_cloud(self) -> bool:
        return self.provider.runtime_name == CLOUD

    def _configured(self) -> str:
        return (getattr(self.provider.settings, "remote_url", None) or os.environ.get(URL_ENV) or "").strip().rstrip("/")

    @property
    def assigned(self) -> Optional[str]:
        """The app's worker picked for this model in the runtime menu, if one was."""
        return getattr(self.provider.settings, "remote_worker", None) or None

    @property
    def pooled(self) -> bool:
        """Whether this model runs on one of the app's workers rather than an address of its own."""
        return not self.on_cloud and (self.assigned is not None or not self._configured())

    def url(self) -> str:
        if self.on_cloud:
            return cloud.url()
        if self.pooled:
            return self.endpoint.url if self.endpoint is not None else ""
        return self._configured()

    def headers(self) -> dict[str, str]:
        if self.on_cloud:
            key = cloud.api_key()
        elif self.pooled:
            key = self.endpoint.token if self.endpoint is not None else None
        else:
            key = getattr(self.provider.settings, "remote_api_key", None) or os.environ.get(KEY_ENV)
        return {"Authorization": f"Bearer {key.strip()}"} if key and key.strip() else {}

    def model_settings(self) -> dict[str, Any]:
        """What the worker gets: the model's settings; on Urakatsu Cloud only which catalog model,
        since the cloud runs its models with its own settings."""
        if self.on_cloud:
            model = getattr(self.provider.settings, "model", None)
            return {"model": model} if model else {}
        return self.provider.settings.model_dump(mode="json", exclude=CONNECTION_SETTINGS)

    # -- lifecycle ------------------------------------------------------------------------------

    def load(self) -> None:
        self.close()
        if self.pooled:
            try:
                self.endpoint = workers.connect(self.assigned)
            except ProviderUnavailable as error:
                raise ProviderUnavailable(f"{self.provider.name}: {error}") from None
        url = self.url()
        if not url:
            raise ProviderUnavailable(f"{self.provider.name}: set the worker URL in Settings, Pipeline (runtime remote)")
        if self.on_cloud and cloud.api_key() is None:
            raise ProviderUnavailable(f"{self.provider.name}: add your Urakatsu Cloud key in Settings, Providers ({cloud.KEYS_URL})")
        self.client = client_factory(url, self.headers())
        try:
            info = self._request("GET", "/v1/info", action="describe itself")
        except ProcessingError as error:
            self.close()
            raise ModelLoadError(str(error)) from error
        if info.get("protocol") != PROTOCOL:
            self.close()
            raise ProviderUnavailable(
                f"the worker at {url} speaks protocol {info.get('protocol')}, this app {PROTOCOL}; update Urakatsu on one of them"
            )
        try:
            self._load_model()
        except BaseException:
            self.close()
            raise
        log.info("loaded %s on the worker at %s (%s)", self.provider.id, url, info.get("device", "unknown device"))

    def _load_model(self) -> None:
        if self.on_cloud:
            body = {"stage": self.provider.stage.value, "provider": self.provider.id, "settings": self.model_settings()}
        else:
            body = load_request(self.provider)
        state = self._request("POST", "/v1/models", json=body, action=f"load {self.provider.id}", error=ModelLoadError)
        self.model_id = state["id"]
        while state.get("state") == "loading":
            try:
                state = self._request(
                    "GET", f"/v1/models/{self.model_id}", params={"wait": _poll_wait()}, action=f"load {self.provider.id}", error=ModelLoadError
                )
            except ModelGone:
                raise ModelLoadError(f"the worker dropped {self.provider.id} while loading it (another client may have swapped it out)") from None
        if state.get("state") != "ready":
            failure = state.get("failure") or {}
            raise error_for(failure.get("kind", "load"), str(failure.get("error") or "the model failed to load"))
        if "instructed" in state:  # an inpainter whose recipe on the worker decides it
            self.provider.instructed = bool(state["instructed"])
            self.provider.crop_edits = bool(state["instructed"]) and not getattr(type(self.provider), "sees_mask", False)
        self._check_weights(state)

    def _check_weights(self, state: dict[str, Any]) -> None:
        """Weights stay pinned for reproducible results: say so when the worker's differ from ours."""
        runtime = state.get("runtime")
        try:
            ours = [w.describe() for w in self.provider.weight_sources(runtime)]
        except Exception:  # noqa: BLE001 - a provider that cannot list them for that runtime
            return
        theirs = state.get("weights") or []
        if ours and theirs and ours != theirs:
            log.warning("the worker runs %s on %s with other weights than this app pins: %s (here %s)", self.provider.id, runtime, theirs, ours)

    def unload(self) -> None:
        if self.client is not None and self.model_id is not None:
            try:
                self.client.delete(f"/v1/models/{self.model_id}")
            except httpx.HTTPError as error:  # the worker may be gone; nothing is held here
                log.info("could not unload %s on the worker: %s", self.provider.id, error)
        self.close()

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
        self.client = None
        self.model_id = None
        self.moved = False

    def _move(self) -> bool:
        """Follows a pooled worker to its current address; True if it moved."""
        if not self.pooled or self.endpoint is None:
            return False
        try:
            entry = workers.find(self.endpoint.worker)
        except KeyError:
            return False
        point = workers.endpoint(entry, fresh=True)
        if point is None or point.url == self.endpoint.url:
            return False
        log.info("the worker moved from %s to %s", self.endpoint.url, point.url)
        if self.client is not None:
            self.client.close()
        self.endpoint = point
        self.client = client_factory(point.url, self.headers())
        self.moved = True
        return True

    # -- calls ----------------------------------------------------------------------------------

    def _request(self, method: str, path: str, *, action: str, error: type[Exception] = ProcessingError, **kwargs: Any) -> Any:
        stop = current_stop()
        if stop is not None:
            stop.raise_if_stopped()
        if self.client is None:
            raise ProcessingError(f"{self.provider.name} is not loaded")
        try:
            response = self.client.request(method, path, **kwargs)
        except httpx.TransportError as failure:
            if not self._move():
                raise error(f"cannot reach the worker at {self.url()}: {failure}") from failure
            try:
                response = self.client.request(method, path, **kwargs)
            except httpx.HTTPError as again:
                raise error(f"cannot reach the worker at {self.url()}: {again}") from again
        except httpx.HTTPError as failure:
            raise error(f"cannot reach the worker at {self.url()}: {failure}") from failure
        if response.status_code == 404 and path.startswith("/v1/models/"):
            raise ModelGone()
        if response.status_code == 404 and path.startswith("/v1/jobs/") and self.moved:
            raise ModelGone()  # the restarted worker never had this job: load the model, run it again
        if response.status_code >= 400:
            raise worker_error(response, action, self.on_cloud)
        return response.json() if response.headers.get("content-type", "").startswith("application/json") else response.content

    def _call(self, operation: str, **kwargs: Any) -> Any:
        """``operation`` on the loaded model, loading it again once if the worker lost it."""
        try:
            return self._job(operation, **kwargs)
        except ModelGone:
            log.info("the worker no longer has %s; loading it again", self.provider.id)
            self._load_model()
            self.moved = False
        try:
            return self._job(operation, **kwargs)
        except ModelGone:
            raise ProcessingError(f"the worker unloaded {self.provider.id} again before it could run; is another app using it?") from None

    def _job(self, operation: str, **kwargs: Any) -> Any:
        """Submits ``operation``, waits for the job, and returns its result."""
        job = self._request("POST", f"/v1/models/{self.model_id}/{operation}", action=operation, **kwargs)
        identifier = job["job"]
        while job.get("state") == "running":
            job = self._request("GET", f"/v1/jobs/{identifier}", params={"wait": _poll_wait()}, action=operation)
        if job.get("state") == "failed":
            failure = job.get("failure") or {}
            if failure.get("kind") == "gone":
                raise ModelGone()
            raise error_for(failure.get("kind"), str(failure.get("error") or f"{operation} failed"))
        return self._request("GET", f"/v1/jobs/{identifier}/result", action=operation)

    def detect(self, image: Image.Image) -> LayoutDetections:
        return wire.decode_detections(self._call("detect", files={"image": ("page.png", wire.encode_image(image), wire.PNG)}))

    def recognize(self, crops: list[Image.Image]) -> list[str]:
        if not crops:
            return []
        files = [("crops", (f"{index}.png", wire.encode_image(crop), wire.PNG)) for index, crop in enumerate(crops)]
        texts = self._call("recognize", files=files)["texts"]
        if len(texts) != len(crops):
            raise ProcessingError(f"the worker returned {len(texts)} texts for {len(crops)} crops")
        return [str(text) for text in texts]

    def inpaint(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        return self._inpaint(image, mask, {})

    def fill(self, image: Image.Image, mask: np.ndarray, prompt: Optional[str]) -> Image.Image:
        return self._inpaint(image, mask, {"fill": "true", **({"prompt": prompt} if prompt else {})})

    def _inpaint(self, image: Image.Image, mask: np.ndarray, data: dict[str, str]) -> Image.Image:
        files = {"image": ("image.png", wire.encode_image(image), wire.PNG), "mask": ("mask.npz", wire.encode_array(mask), wire.NPZ)}
        return wire.decode_image(self._call("inpaint", files=files, data=data))
