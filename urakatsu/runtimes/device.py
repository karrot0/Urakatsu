"""Device vocabulary shared by every runtime."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class Backend(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    DIRECTML = "dml"
    METAL = "metal"
    VULKAN = "vulkan"


class DeviceType(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    INTEGRATED_GPU = "integrated_gpu"
    ACCELERATOR = "accelerator"
    UNKNOWN = "unknown"


class Device(BaseModel):
    index: int = 0  # position among the GPUs the backend's API enumerates (CUDA, HIP, XPU)
    name: str = "CPU"
    description: str = "CPU"
    backend: Backend = Backend.CPU
    device_type: DeviceType = DeviceType.CPU
    memory_total: int = 0
    memory_free: int = 0
    compute_capability: int = 0
    driver_version: str = ""
    vendor: str = ""  # nvidia | amd | intel | qualcomm | apple
    target: str = ""  # AMD ISA (gfx1100), which picks the ROCm device libraries
    adapter: int = -1  # DXGI adapter index on Windows (DirectML's device_id)
    locator: str = ""  # Windows: the adapter LUID as GPU performance counters name it; Linux: sysfs device

    @property
    def is_integrated(self) -> bool:
        return self.device_type is DeviceType.INTEGRATED_GPU

    @classmethod
    def cpu(cls) -> "Device":
        return cls()

    @classmethod
    def gpu(cls, backend: Backend, index: int, name: str | None = None, **kwargs) -> "Device":
        label = name or f"{backend.value}:{index}"
        kwargs.setdefault("device_type", DeviceType.GPU)
        return cls(index=index, name=label, description=label, backend=backend, **kwargs)

    @property
    def is_cpu(self) -> bool:
        return self.backend is Backend.CPU

    @property
    def is_accelerator(self) -> bool:
        return not self.is_cpu

    def label(self) -> str:
        if self.is_cpu:
            return "cpu"
        return f"{self.backend.value}:{self.index}"

    @classmethod
    def parse(cls, spec: str, candidates: list["Device"]) -> "Device":
        """Resolves ``"auto" | "cpu" | "cuda" | "cuda:1" | "dml"`` against discovered devices."""
        spec = (spec or "auto").strip().lower()
        if spec == "cpu":
            return cls.cpu()
        if spec == "auto":
            for device in candidates:
                if device.is_accelerator:
                    return device
            return cls.cpu()
        backend_name, _, index = spec.partition(":")
        backend = Backend(backend_name)
        wanted = int(index) if index else None
        for device in candidates:
            if device.backend is backend and (wanted is None or device.index == wanted):
                return device
        raise ValueError(f"no device matches {spec!r}; discovered: {[d.label() for d in candidates]}")
