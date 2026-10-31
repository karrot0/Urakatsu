"""run_server: a failed accept must not take the listener down with it."""

from __future__ import annotations

import asyncio
import errno
import socket
import sys

import pytest

from urakatsu.service.event_loop import run_server


class _FailsOnce(socket.socket):
    """A listener whose first accept fails, as WinError 10014 did on a user's machine."""

    failed = False

    def accept(self):
        conn, addr = super().accept()
        if not self.failed:
            self.failed = True
            conn.close()
            raise OSError(errno.EFAULT, "invalid pointer address")
        return conn, addr


async def _fetch(port: int) -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        return await asyncio.wait_for(reader.read(), timeout=5)
    finally:
        writer.close()


def test_listener_survives_a_failed_accept():
    errors = []

    async def main() -> bytes:
        asyncio.get_running_loop().set_exception_handler(lambda _, context: errors.append(context))

        async def answer(_, writer) -> None:
            writer.write(b"ok")
            await writer.drain()
            writer.close()

        listener = _FailsOnce(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        server = await asyncio.start_server(answer, sock=listener)
        async with server:
            assert await _fetch(port) == b""  # the connection that failed
            return await _fetch(port)

    assert run_server(main()) == b"ok"
    assert len(errors) == 1


@pytest.mark.skipif(sys.platform != "win32", reason="the proactor loop is Windows only")
def test_servers_do_not_run_on_the_proactor_loop():
    async def main() -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    assert isinstance(run_server(main()), asyncio.SelectorEventLoop)
