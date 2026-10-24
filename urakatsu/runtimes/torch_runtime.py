"""PyTorch runtime bound to the selected device."""

from __future__ import annotations

import importlib.util
import logging

from urakatsu.core.types import Availability

from .base import Runtime
from .device import Backend, Device

log = logging.getLogger(__name__)


class TorchRuntime(Runtime):
    name = "torch"
    description = "PyTorch (CPU, CUDA, ROCm, Intel XPU, Metal)"

    def __init__(self, device: Device):
        super().__init__(device)
        import torch  # noqa: WPS433 - lazy by design

        self.torch = torch
        self.torch_device = self._resolve()

    @classmethod
    def available(cls) -> Availability:
        if importlib.util.find_spec("torch") is None:
            return Availability.no("install torch (pip install 'urakatsu[torch]')")
        return Availability.yes()

    def _resolve(self):
        torch = self.torch
        if self.device.backend in (Backend.CUDA, Backend.ROCM) and torch.cuda.is_available():
            if self.device.index < torch.cuda.device_count():
                return torch.device("cuda", self.device.index)
        # Intel GPUs run native models on Vulkan; PyTorch reaches them through XPU.
        xpu = getattr(torch, "xpu", None)
        if self.device.vendor == "intel" and xpu is not None and xpu.is_available():
            if self.device.index < xpu.device_count():
                return torch.device("xpu", self.device.index)
        if self.device.backend is Backend.METAL and getattr(torch.backends, "mps", None):
            if torch.backends.mps.is_available():
                return torch.device("mps")
        if self.device.is_accelerator:
            log.info("torch cannot address %s; falling back to CPU", self.device.label())
        return torch.device("cpu")

    def effective_device(self) -> Device:
        return self.device if self.torch_device.type != "cpu" else Device.cpu()

    def dtype(self):
        """bfloat16 on Ampere+ CUDA, float32 everywhere else."""
        if self.torch_device.type == "cuda" and self.device.compute_capability >= 80:
            return self.torch.bfloat16
        return self.torch.float32

    def describe(self) -> dict:
        return {
            "name": self.name,
            "version": self.torch.__version__,
            "device": str(self.torch_device),
            "cuda": bool(self.torch.cuda.is_available()),
            "rocm": getattr(self.torch.version, "hip", None),
        }


def release_cache(device=None) -> None:
    """Hands memory PyTorch cached back to the GPU (CUDA and ROCm, XPU, MPS), for the next model.

    ``device`` is a ``torch.device``; None releases every backend that is in use. PyTorch is not
    imported here when nothing imported it yet.
    """
    import sys

    torch = sys.modules.get("torch")
    if torch is None or (device is not None and device.type == "cpu"):
        return
    kind = None if device is None else device.type
    for name in ("cuda", "xpu", "mps"):
        backend = getattr(torch, name, None)
        if backend is None or kind not in (None, name):
            continue
        available = backend.is_available() if name != "mps" else torch.backends.mps.is_available()
        if available:
            backend.empty_cache()
