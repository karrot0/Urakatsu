"""Cooperative cancellation shared across a pipeline run.

The pipeline opens ``stopping`` around a stage, so code that waits a long time inside it (the
remote runtime polling a worker or Urakatsu Cloud) can see the run's token through ``current``
without a provider learning about runs. ``asyncio.to_thread`` copies the context, so the
provider's worker thread sees it too.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from typing import Awaitable, Iterator, Optional, TypeVar

T = TypeVar("T")

CHECK_INTERVAL = 0.25


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


_current: ContextVar[Optional[StopToken]] = ContextVar("urakatsu_stop", default=None)


@contextmanager
def stopping(token: Optional[StopToken]) -> Iterator[None]:
    """Makes ``token`` the one ``current`` returns in this context (and threads started from it)."""
    reset = _current.set(token)
    try:
        yield
    finally:
        _current.reset(reset)


def current() -> Optional[StopToken]:
    """The token of the run this code is part of, or None outside one."""
    return _current.get()


async def interruptible(awaitable: Awaitable[T]) -> T:
    """Awaits ``awaitable``, cancelling it and raising ``StoppedError`` if the current run stops
    first. Only for work that cancelling really ends (an HTTP request on the event loop), never
    a thread, which would keep running on the accelerator after the lane is released."""
    token = current()
    if token is None:
        return await awaitable
    task = asyncio.ensure_future(awaitable)
    while True:
        done, _ = await asyncio.wait({task}, timeout=CHECK_INTERVAL)
        if done:
            return task.result()
        if token.stopped:
            task.cancel()
            with suppress(BaseException):
                await task
            raise StoppedError()
