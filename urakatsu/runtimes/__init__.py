"""Hardware discovery, device selection, model store, and inference runtimes.

A *runtime* is a process-wide capability a provider needs (ONNX Runtime, Torch).
A *device* is where tensors live. The ``RuntimeManager`` discovers hardware once,
selects the device the configuration asks for, and hands providers a runtime that
is already bound to that device, the way the upstream runtime crate owns
the Torch / llama.cpp / diffusion backends.
"""

from .device import Backend, Device, DeviceType
from .hardware import Hardware, discover
from .store import GitHubReleaseFile, HuggingFaceFile, LocalFile, ModelStore, WeightSource
from .base import Runtime
from .manager import RuntimeManager, RuntimeReport

__all__ = [
    "Backend",
    "Device",
    "DeviceType",
    "Hardware",
    "discover",
    "GitHubReleaseFile",
    "HuggingFaceFile",
    "LocalFile",
    "ModelStore",
    "WeightSource",
    "Runtime",
    "RuntimeManager",
    "RuntimeReport",
]
