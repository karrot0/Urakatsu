"""Cooperative cancellation shared across a pipeline run."""

from __future__ import annotations

import threading


class StoppedError(Exception):
    """Raised inside a stage when the run was stopped cooperatively."""


class StopToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def stop(self) -> None:
        self._event.set()

    @property
    def stopped(self) -> bool:
        return self._event.is_set()

    def raise_if_stopped(self) -> None:
        if self.stopped:
            raise StoppedError()
