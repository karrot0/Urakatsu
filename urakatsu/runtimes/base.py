"""Runtime base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from urakatsu.core.types import Availability

from .device import Device


class Runtime(ABC):
    """A process-wide inference capability bound to one device."""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""

    def __init__(self, device: Device):
        self.device = device

    @classmethod
    @abstractmethod
    def available(cls) -> Availability:
        """Whether the runtime's Python package can be imported."""

    @abstractmethod
    def describe(self) -> dict:
        """Human-readable snapshot for the capabilities endpoint."""

    def effective_device(self) -> Device:
        """The device this runtime actually uses (falls back to CPU when it cannot address the GPU)."""
        return self.device
