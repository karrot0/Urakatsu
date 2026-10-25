"""stable-diffusion.cpp runtime: the newest official release, run as ``sd-server``.

Each loaded model gets its own server process,
preloaded from command-line arguments; generation goes through the native async
``/sdcpp/v1/img_gen`` API (submit, then poll the job).

When someone listens for steps (``urakatsu.core.progress.reporting_steps``), the request
asks for a ``proj`` preview (a linear latent-to-RGB projection, ComfyUI's latent2rgb: no extra
weights, next to no cost) and each new preview frame on the polled job is reported. Releases
whose ``/sdcpp/v1/capabilities`` lacks ``proj`` (a pinned older runtime) run without previews.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any, Sequence

from PIL import Image

from urakatsu.core.errors import ProcessingError
from urakatsu.core.progress import StepPreview, step_sink
from urakatsu.core.types import Availability

from .base import Runtime
from .device import Device
from .llama import _error_text, _hardware
from .packages import STABLE_DIFFUSION_CPP, Installed, PackageManager
from .server_process import ServerProcess, device_env, free_port

log = logging.getLogger(__name__)

POLL_INTERVAL = 0.2
PREVIEW_MODE = "proj"


def image_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class SdServer:
    """One ``sd-server`` process with one model preloaded."""

    def __init__(self, process: ServerProcess):
        self.process = process
        self._previews: bool | None = None

    def previews(self) -> bool:
        """Whether this release can stream ``proj`` step previews (asked once per process)."""
        if self._previews is None:
            try:
                response = self.process.client.get("/sdcpp/v1/capabilities")
                modes = (response.json().get("preview_modes") or []) if response.status_code < 400 else []
            except Exception:
                modes = []
            self._previews = PREVIEW_MODE in modes
        return self._previews

    def generate(self, request: dict[str, Any]) -> list[Image.Image]:
        client = self.process.client
        sink = step_sink()
        if sink is not None and "preview" not in request and self.previews():
            request = {**request, "preview": PREVIEW_MODE, "preview_interval": 1}
        else:
            sink = None
        shown = None
        response = client.post("/sdcpp/v1/img_gen", json=request)
        if response.status_code >= 400:
            raise ProcessingError(f"sd-server: {_error_text(response)}")
        job = response.json()
        poll = job.get("poll_url") or f"/sdcpp/v1/jobs/{job['id']}"
        while True:
            state = client.get(poll).raise_for_status().json()
            status = state.get("status")
            preview = state.get("preview")
            if sink is not None and preview and preview.get("b64_json") and (preview.get("pass"), preview.get("step")) != shown:
                shown = (preview.get("pass"), preview.get("step"))
                try:
                    frame = Image.open(io.BytesIO(base64.b64decode(preview["b64_json"]))).convert("RGB")
                    sink(StepPreview(int(preview.get("step") or 0), int(preview.get("total_steps") or 0), frame))
                except Exception:  # a preview must never fail the generation
                    log.debug("dropped an sd-server preview", exc_info=True)
            if status == "completed":
                images = (state.get("result") or {}).get("images") or []
                return [Image.open(io.BytesIO(base64.b64decode(i["b64_json"]))).copy() for i in sorted(images, key=lambda i: i.get("index", 0))]
            if status in ("failed", "cancelled"):
                error = state.get("error") or {}
                raise ProcessingError(f"sd-server job {status}: {error.get('message', error)}")
            if not self.process.running:
                raise ProcessingError(f"sd-server exited:\n{self.process.log_tail()}")
            time.sleep(POLL_INTERVAL)

    def close(self) -> None:
        self.process.stop()


class DiffusionRuntime(Runtime):
    name = "diffusion"
    description = "stable-diffusion.cpp, newest official release (sd-server)"

    def __init__(self, device: Device):
        super().__init__(device)
        self.packages = PackageManager()

    @classmethod
    def available(cls) -> Availability:
        manager = PackageManager()
        if manager.installed(STABLE_DIFFUSION_CPP) is None and manager.offline:
            return Availability.no("stable-diffusion.cpp is not installed; run `urakatsu runtimes update` while online")
        return Availability.yes()

    def package(self) -> Installed:
        return self.packages.ensure(STABLE_DIFFUSION_CPP, _hardware())

    def launch(self, name: str, model_args: Sequence[str]) -> SdServer:
        installed = self.package()
        port = free_port()
        env = device_env(self.device, installed.variant) if self.device.is_accelerator else {}
        process = ServerProcess(
            f"sd-server-{name}",
            installed.executable,
            [*model_args, "--listen-ip", "127.0.0.1", "--listen-port", str(port)],
            port,
            ready=lambda client: client.get("/sdcpp/v1/models/status").json().get("state") == "ready",
            env=env,
        )
        return SdServer(process)

    def describe(self) -> dict:
        installed = self.packages.installed(STABLE_DIFFUSION_CPP)
        return {
            "name": self.name,
            "version": installed.tag if installed else "not installed",
            "build": installed.variant if installed else None,
            "device": self.device.label(),
        }
