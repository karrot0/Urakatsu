"""llama.cpp runtime: the newest official llama.cpp release, run as ``llama-server``.

Each loaded GGUF model gets its own server process
(``LlamaServer``); providers render their prompts themselves and use the raw
``/completion`` endpoint, so prompts are byte-for-byte what the upstream builds. Images go
in ``multimodal_data`` with the server's media marker in the prompt.
"""

from __future__ import annotations

import base64
import functools
from pathlib import Path
from typing import Any, Optional, Sequence

from urakatsu.core.errors import ProcessingError
from urakatsu.core.types import Availability

from .base import Runtime
from .device import Device
from .packages import LLAMA_CPP, Installed, PackageManager
from .server_process import ServerProcess, device_env, free_port

# DEFAULT_GPU_LAYERS: offload everything when an accelerator is selected.
DEFAULT_GPU_LAYERS = 1000
# DEFAULT_SEED
DEFAULT_SEED = 299_792_458


@functools.lru_cache(maxsize=1)
def _hardware() -> list[Device]:
    from .hardware import discover

    return discover().candidates()


class LlamaServer:
    """One ``llama-server`` process serving one model."""

    def __init__(self, process: ServerProcess):
        self.process = process
        self.props: dict[str, Any] = process.client.get("/props").raise_for_status().json()

    @property
    def media_marker(self) -> str:
        return self.props.get("media_marker") or "<__media__>"

    @property
    def vision(self) -> bool:
        return bool((self.props.get("modalities") or {}).get("vision"))

    @functools.cached_property
    def bos_text(self) -> str:
        """The BOS token's text when the server inserts one for this model, else ""."""
        response = self.process.client.post("/tokenize", json={"content": "", "add_special": True, "with_pieces": True})
        tokens = response.raise_for_status().json().get("tokens") or []
        return tokens[0].get("piece", "") if tokens and isinstance(tokens[0], dict) else ""

    def strip_bos(self, prompt: str) -> str:
        """A rendered chat template starts with BOS; the server adds its own, so drop ours (one BOS)."""
        bos = self.bos_text
        return prompt[len(bos) :] if bos and prompt.startswith(bos) else prompt

    def complete(self, prompt: str, images: Sequence[bytes] = (), **params: Any) -> str:
        """``POST /completion`` with a raw prompt; returns the generated text."""
        if images and not self.vision:
            raise ProcessingError("this model has no vision projector")
        body: dict[str, Any] = {
            "prompt": {"prompt_string": prompt, "multimodal_data": [base64.b64encode(i).decode("ascii") for i in images]}
            if images
            else prompt,
            "stream": False,
            "cache_prompt": False,  # Every request is evaluated from scratch; this keeps results deterministic
            "seed": DEFAULT_SEED,
            **{k: v for k, v in params.items() if v is not None},
        }
        response = self.process.client.post("/completion", json=body)
        if response.status_code >= 400:
            raise ProcessingError(f"llama-server: {_error_text(response)}")
        return response.json().get("content", "")

    def close(self) -> None:
        self.process.stop()


def _error_text(response) -> str:
    try:
        error = response.json().get("error")
        return error.get("message", str(error)) if isinstance(error, dict) else str(error)
    except ValueError:
        return response.text[:500]


class LlamaRuntime(Runtime):
    name = "llama"
    description = "llama.cpp, newest official release (llama-server)"

    def __init__(self, device: Device):
        super().__init__(device)
        self.packages = PackageManager()

    @classmethod
    def available(cls) -> Availability:
        manager = PackageManager()
        if manager.installed(LLAMA_CPP) is None and manager.offline:
            return Availability.no("llama.cpp is not installed; run `urakatsu runtimes update` while online")
        return Availability.yes()

    def package(self) -> Installed:
        return self.packages.ensure(LLAMA_CPP, _hardware())

    def gpu_layers(self) -> int:
        if not self.device.is_accelerator:
            return 0
        installed = self.packages.installed(LLAMA_CPP)
        return 0 if installed is not None and installed.variant.endswith("-cpu") else DEFAULT_GPU_LAYERS

    def effective_device(self) -> Device:
        return self.device if self.gpu_layers() > 0 else Device.cpu()

    def launch(
        self,
        model: Path,
        projector: Optional[Path] = None,
        context: int = 8192,
        batch: Optional[int] = None,
        ubatch: Optional[int] = None,
    ) -> LlamaServer:
        installed = self.package()
        port = free_port()
        layers = self.gpu_layers()
        args = [
            "--model", str(model),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--ctx-size", str(context),
            "--n-gpu-layers", str(layers),
            "--parallel", "1",
            "--no-webui",
        ]  # fmt: skip
        if batch:
            args += ["--batch-size", str(batch)]
        if ubatch:
            args += ["--ubatch-size", str(ubatch)]
        if projector is not None:
            args += ["--mmproj", str(projector)]
            if layers == 0:
                args.append("--no-mmproj-offload")
        env = device_env(self.device, installed.variant) if layers else {}
        process = ServerProcess(
            f"llama-server-{model.stem}",
            installed.executable,
            args,
            port,
            ready=lambda client: client.get("/health").status_code == 200,
            env=env,
        )
        return LlamaServer(process)

    def describe(self) -> dict:
        installed = self.packages.installed(LLAMA_CPP)
        return {
            "name": self.name,
            "version": installed.tag if installed else "not installed",
            "build": installed.variant if installed else None,
            "device": self.effective_device().label(),
        }
