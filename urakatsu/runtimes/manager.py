"""Owns hardware discovery, device selection, runtimes, and the model store."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from urakatsu.core.errors import ProviderUnavailable
from urakatsu.core.types import Availability

from .base import Runtime
from .device import Device
from .hardware import Hardware, discover
from .store import ModelStore

log = logging.getLogger(__name__)


class RuntimeReport(BaseModel):
    hardware: Hardware
    device: Device
    runtimes: dict[str, Any] = Field(default_factory=dict)
    store_root: str = ""
    store_bytes: int = 0


class RuntimeManager:
    # Same process-wide capabilities as upstream's Feature enum (Torch, Llama, Diffusion) plus ONNX.
    RUNTIMES: dict[str, str] = {
        "torch": "urakatsu.runtimes.torch_runtime:TorchRuntime",
        "llama": "urakatsu.runtimes.llama:LlamaRuntime",
        "diffusion": "urakatsu.runtimes.diffusion:DiffusionRuntime",
        "onnx": "urakatsu.runtimes.onnx:OnnxRuntime",
    }

    def __init__(self, device: str = "auto", store: ModelStore | None = None):
        self.hardware = discover()
        self.device = Device.parse(device, self.hardware.candidates())
        self.store = store or ModelStore()
        self._runtimes: dict[str, Runtime] = {}

    def select_device(self, spec: str) -> Device:
        device = Device.parse(spec, self.hardware.candidates())
        if device != self.device:
            self.device = device
            self._runtimes.clear()
        return device

    @classmethod
    def runtime_class(cls, name: str) -> type[Runtime]:
        import importlib

        try:
            module_name, class_name = cls.RUNTIMES[name].split(":")
        except KeyError as error:
            raise KeyError(f"unknown runtime {name!r}; known: {sorted(cls.RUNTIMES)}") from error
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    @classmethod
    def availability(cls, name: str) -> Availability:
        return cls.runtime_class(name).available()

    def runtime(self, name: str) -> Runtime:
        if name in self._runtimes:
            return self._runtimes[name]
        runtime_cls = self.runtime_class(name)
        availability = runtime_cls.available()
        if not availability.ok:
            raise ProviderUnavailable(f"runtime {name!r} is unavailable: {availability.reason}")
        runtime = runtime_cls(self.device)
        self._runtimes[name] = runtime
        return runtime

    def describe(self) -> RuntimeReport:
        runtimes: dict[str, Any] = {}
        for name in self.RUNTIMES:
            availability = self.availability(name)
            entry: dict[str, Any] = {"available": availability.ok, "reason": availability.reason}
            if name in self._runtimes:
                entry.update(self._runtimes[name].describe())
            runtimes[name] = entry
        return RuntimeReport(
            hardware=self.hardware,
            device=self.device,
            runtimes=runtimes,
            store_root=str(self.store.root),
            store_bytes=self.store.usage(),
        )
