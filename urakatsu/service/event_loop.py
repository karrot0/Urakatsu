"""The event loop every Urakatsu server runs on."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_server(main: Coroutine[Any, Any, T]) -> T:
    """``asyncio.run`` on a loop whose listeners survive a failed accept.

    Windows defaults to the proactor loop, which closes a listening socket for good when one
    accept raises OSError. A single bad connection (WinError 10014 from ``getpeername`` on one
    user's machine) then leaves a running server that nothing can reach. The selector loop logs
    the error and keeps listening, and no server here needs what only the proactor offers
    (asyncio subprocesses and pipes).
    """
    factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    with asyncio.Runner(loop_factory=factory) as runner:
        return runner.run(main)
