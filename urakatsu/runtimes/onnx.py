"""ONNX Runtime bound to the selected device.

On CUDA the execution provider needs the CUDA / cuDNN libraries at session
creation time. ONNX Runtime can locate them when they come from the ``nvidia-*``
pip packages (``onnxruntime.preload_dlls``); otherwise a session silently falls
back to CPU, which this wrapper detects and reports instead of hiding.

Every other GPU on Windows (AMD, Intel) runs ONNX models through DirectML
(``onnxruntime-directml``), addressed by its DXGI adapter index.
"""

from __future__ import annotations

import importlib.util
import logging
import platform
from pathlib import Path
from typing import Any

from urakatsu.core.types import Availability

from .base import Runtime
from .device import Backend, Device

log = logging.getLogger(__name__)

_PRELOADED = False


class OnnxRuntime(Runtime):
    name = "onnx"
    description = "ONNX Runtime (CPU, CUDA, DirectML, CoreML)"

    def __init__(self, device: Device):
        super().__init__(device)
        import onnxruntime as ort  # noqa: WPS433 - lazy by design

        self._ort = ort
        self._providers = self._select_providers()
        self._fallback_reason: str | None = None
        if self.device.backend is Backend.CUDA and self._wants_cuda():
            self._preload_cuda()

    @classmethod
    def available(cls) -> Availability:
        if importlib.util.find_spec("onnxruntime") is None:
            return Availability.no("install onnxruntime (pip install 'urakatsu[onnx]')")
        return Availability.yes()

    def _select_providers(self) -> list[Any]:
        installed = set(self._ort.get_available_providers())
        chosen: list[Any] = []
        if self.device.backend is Backend.CUDA and "CUDAExecutionProvider" in installed:
            chosen.append(("CUDAExecutionProvider", {"device_id": self.device.index}))
        elif self._directml() and "DmlExecutionProvider" in installed:
            chosen.append(("DmlExecutionProvider", {"device_id": max(self.device.adapter, 0)}))
        elif self.device.backend is Backend.METAL and "CoreMLExecutionProvider" in installed:
            chosen.append("CoreMLExecutionProvider")
        elif self.device.is_accelerator:
            self._fallback_reason = f"onnxruntime has no execution provider for {self.device.label()}"
            package = "onnxruntime-directml" if self._directml() else "onnxruntime-gpu"
            log.info("%s; falling back to CPU (install %s)", self._fallback_reason, package)
        chosen.append("CPUExecutionProvider")
        return chosen

    def _directml(self) -> bool:
        """A GPU other than NVIDIA's on Windows: DirectML reaches it whatever its native backend is."""
        return self.device.backend in (Backend.ROCM, Backend.VULKAN, Backend.DIRECTML) and platform.system() == "Windows"

    def _wants_cuda(self) -> bool:
        first = self._providers[0]
        return (first[0] if isinstance(first, tuple) else first) == "CUDAExecutionProvider"

    def _preload_cuda(self) -> None:
        """Loads CUDA / cuDNN DLLs from the nvidia pip packages (ORT >= 1.21) once per process."""
        global _PRELOADED
        if _PRELOADED:
            return
        _PRELOADED = True
        preload = getattr(self._ort, "preload_dlls", None)
        if preload is None:
            return
        try:
            preload(cuda=True, cudnn=True, msvc=True)
        except Exception as error:  # the session then reports the CPU fallback
            log.warning("onnxruntime.preload_dlls failed: %s", error)

    def effective_device(self) -> Device:
        if self._fallback_reason is not None:
            return Device.cpu()
        first = self._providers[0]
        name = first[0] if isinstance(first, tuple) else first
        return self.device if name != "CPUExecutionProvider" else Device.cpu()

    def session(self, path: Path | str, *, cpu: bool = False, **options: Any):
        """An inference session. ``cpu`` keeps a graph that a GPU provider runs incorrectly off the GPU."""
        session_options = self._ort.SessionOptions()
        session_options.graph_optimization_level = self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.log_severity_level = 3  # errors only; EP fallback is reported below
        for key, value in options.items():
            setattr(session_options, key, value)
        providers = ["CPUExecutionProvider"] if cpu else self._providers
        session = self._ort.InferenceSession(str(path), sess_options=session_options, providers=providers)
        wanted = providers[0]
        wanted_name = wanted[0] if isinstance(wanted, tuple) else wanted
        if wanted_name != "CPUExecutionProvider" and wanted_name not in session.get_providers():
            if self._fallback_reason is None:
                self._fallback_reason = (
                    f"{wanted_name} could not be created (missing CUDA / cuDNN libraries, or a GPU driver without DirectX 12?); "
                    "running ONNX models on CPU. See https://onnxruntime.ai/docs/execution-providers/"
                )
                log.warning(self._fallback_reason)
        return session

    def describe(self) -> dict:
        info = {
            "name": self.name,
            "version": self._ort.__version__,
            "providers": [p[0] if isinstance(p, tuple) else p for p in self._providers],
            "device": self.effective_device().label(),
        }
        if self._fallback_reason:
            info["fallback"] = self._fallback_reason
        return info
