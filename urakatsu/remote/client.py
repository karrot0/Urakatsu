"""The ``remote`` runtime's client: one provider's model, loaded on a Urakatsu worker.

A provider whose runtime is ``remote`` keeps its class, settings and stage behavior (tiling,
``prompted``, ``instructed``); only ``_load`` / ``_unload`` and the inference methods go to the
worker (``Provider.__init__`` binds them here). The worker runs the provider's own methods on
its runtime, so a remote run produces what a local one would.

- Where: ``remote_url`` in the provider's settings, else ``URAKATSU_REMOTE_URL``; the token is
  ``remote_api_key``, else ``URAKATSU_REMOTE_API_KEY``, sent as a bearer token. On the ``cloud``
  runtime, Urakatsu Cloud and the account key instead (``cloud``).
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

from . import CLOUD, PROTOCOL, cloud, wire

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
CONNECTION_SETTINGS = frozenset({"remote_url", "remote_api_key", "server_url", "server_api_key", "api_key"})

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


class ModelGone(Exception):
    """The worker no longer has the model (it restarted or swapped it out)."""


class RemoteModel:
    def __init__(self, provider: "Provider"):
        self.provider = provider
        self.client: Optional[httpx.Client] = None
        self.model_id: Optional[str] = None

    # -- connection -----------------------------------------------------------------------------

    @property
    def on_cloud(self) -> bool:
        return self.provider.runtime_name == CLOUD

    def url(self) -> str:
        if self.on_cloud:
            return cloud.url()
        url = getattr(self.provider.settings, "remote_url", None) or os.environ.get(URL_ENV) or ""
        return url.strip().rstrip("/")

    def headers(self) -> dict[str, str]:
        key = cloud.api_key() if self.on_cloud else getattr(self.provider.settings, "remote_api_key", None) or os.environ.get(KEY_ENV)
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
        url = self.url()
        if not url:
            raise ProviderUnavailable(f"{self.provider.name}: set the worker URL in Settings, Pipeline (runtime remote)")
        if self.on_cloud and cloud.api_key() is None:
            raise ProviderUnavailable(f"{self.provider.name}: add your Urakatsu Cloud key in Settings, Providers ({cloud.KEYS_URL})")
        self.close()
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
        body = {"stage": self.provider.stage.value, "provider": self.provider.id, "settings": self.model_settings()}
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

    # -- calls ----------------------------------------------------------------------------------

    def _request(self, method: str, path: str, *, action: str, error: type[Exception] = ProcessingError, **kwargs: Any) -> Any:
        stop = current_stop()
        if stop is not None:
            stop.raise_if_stopped()
        if self.client is None:
            raise ProcessingError(f"{self.provider.name} is not loaded")
        try:
            response = self.client.request(method, path, **kwargs)
        except httpx.HTTPError as failure:
            raise error(f"cannot reach the worker at {self.url()}: {failure}") from failure
        if response.status_code == 404 and path.startswith("/v1/models/"):
            raise ModelGone()
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
