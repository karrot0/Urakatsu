"""``urakatsu dev``: the service stops cleanly before a restart, and restarts on Python changes."""

import _thread
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from urakatsu.service import dev

# Stops the way serve() does: our handlers for STOP_SIGNALS, uvicorn catching (and re-raising) them.
SERVICE = """
import asyncio, signal, sys
import uvicorn
from urakatsu.service.app import STOP_SIGNALS

async def app(scope, receive, send):
    pass

async def main():
    stop = asyncio.Event()
    for sig in STOP_SIGNALS:
        signal.signal(sig, lambda *_: stop.set())
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    print("started", flush=True)
    await asyncio.wait({task, asyncio.create_task(stop.wait())}, return_when=asyncio.FIRST_COMPLETED)
    server.should_exit = True
    await task
    open(sys.argv[1], "w").write("closed")  # where serve() runs workspace.close()

asyncio.run(main())
"""


def test_stop_lets_the_service_close_its_workspace(tmp_path: Path):
    marker = tmp_path / "closed"
    process = subprocess.Popen([sys.executable, "-c", SERVICE, str(marker)], stdout=subprocess.PIPE, text=True, **dev._own_group())
    assert process.stdout.readline().strip() == "started"

    dev.stop(process, timeout=15)

    assert marker.read_text() == "closed"


def test_a_python_change_restarts_the_service(tmp_path: Path, monkeypatch):
    starts: list[subprocess.Popen] = []
    monkeypatch.setattr(dev, "PACKAGE", tmp_path)
    monkeypatch.setattr(dev, "service_command", lambda args: [sys.executable, "-c", "import time; time.sleep(60)"])
    real_start = dev.start_service
    monkeypatch.setattr(dev, "start_service", lambda args: starts.append(real_start(args)) or starts[-1])

    def edit_then_quit():
        deadline = time.monotonic() + 30
        while len(starts) < 2 and time.monotonic() < deadline:
            (tmp_path / "module.py").write_text(f"x = {time.monotonic()}\n")  # until the watcher is up
            time.sleep(1)
        _thread.interrupt_main()

    threading.Thread(target=edit_then_quit, daemon=True).start()
    assert dev.run(port=_free_port(), web=False) == 0

    assert len(starts) == 2
    assert all(process.poll() is not None for process in starts)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_a_taken_default_port_moves_to_the_next_free_one():
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]

        assert not dev.port_free("127.0.0.1", port)
        assert dev.pick_port("127.0.0.1", None, port, "--port") > port


def test_a_taken_port_that_was_asked_for_is_an_error():
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]

        with pytest.raises(dev.DevError, match=f"Port {port} is in use"):
            dev.pick_port("127.0.0.1", port, dev.DEFAULT_PORT, "--port")
