"""Catalog download queue: order, moving, pausing (with byte-range resume), resuming and cancelling."""

from __future__ import annotations

import threading
import time

import httpx
import pytest

from urakatsu.catalog import installed as installed_module
from urakatsu.catalog.installed import Downloader, InstalledFile, InstalledModel, InstalledModels
from urakatsu.runtimes.store import ModelStore

CHUNK = 1 << 20
SIZE = 3 * CHUNK


class _Server:
    """A fake Hugging Face: every file is SIZE bytes of its first letter, sent one 1 MiB chunk per ``step``."""

    def __init__(self):
        self.gates: dict[str, threading.Semaphore] = {}
        self.ranges: list[tuple[str, int]] = []
        self.order: list[str] = []

    def gate(self, name: str) -> threading.Semaphore:
        return self.gates.setdefault(name, threading.Semaphore(0))

    def step(self, name: str, chunks: int = 1) -> None:
        for _ in range(chunks):
            self.gate(name).release()

    def open(self, name: str) -> None:
        self.step(name, 1000)

    def handler(self, request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", 1)[-1]
        start = int(request.headers.get("range", "bytes=0-").split("=")[1].rstrip("-"))
        self.ranges.append((name, start))
        if not self.order or self.order[-1] != name:
            self.order.append(name)
        gate = self.gate(name)
        letter = name[0].encode()

        def body():
            sent = start
            while sent < SIZE:
                if not gate.acquire(timeout=10):
                    raise RuntimeError("test gate never opened")
                piece = min(CHUNK, SIZE - sent)
                sent += piece
                yield letter * piece

        return httpx.Response(206 if start else 200, content=body())


@pytest.fixture
def server(monkeypatch):
    fake = _Server()
    real = httpx.Client

    def client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(fake.handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(installed_module.httpx, "Client", client)
    return fake


@pytest.fixture
def downloader(tmp_path):
    return Downloader(InstalledModels(ModelStore(root=tmp_path, offline=False)))


def _model(name: str) -> InstalledModel:
    return InstalledModel(id=name, repo=f"org/{name}", revision="abc", runtime="torch", files=[InstalledFile(path=f"{name}.bin", role="file", size=SIZE)])


def _wait(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.01)


def _status(downloader: Downloader, download) -> str:
    return downloader.downloads[download.id].status


def test_downloads_run_in_queue_order_and_queued_ones_can_be_moved(server, downloader):
    first = downloader.start(_model("alpha"))
    _wait(lambda: _status(downloader, first) == "downloading" and server.order == ["alpha.bin"])
    second, third, fourth = (downloader.start(_model(n)) for n in ("bravo", "charlie", "delta"))
    assert downloader.queue() == [second.id, third.id, fourth.id]
    assert [second.position, third.position, fourth.position] == [0, 1, 2]

    downloader.move(fourth.id, 0)
    downloader.move(second.id, 99)  # clamped to the end
    assert downloader.queue() == [fourth.id, third.id, second.id]
    assert [fourth.position, third.position, second.position] == [0, 1, 2]
    with pytest.raises(ValueError):
        downloader.move(first.id, 0)  # the running one is not in the queue

    for name in ("alpha", "bravo", "charlie", "delta"):
        server.open(f"{name}.bin")
    _wait(lambda: all(_status(downloader, d) == "completed" for d in (first, second, third, fourth)))
    assert server.order == ["alpha.bin", "delta.bin", "charlie.bin", "bravo.bin"]
    assert all(d.position is None for d in (first, second, third, fourth))


def test_pausing_a_running_download_keeps_its_partial_file_and_resume_continues_it(server, downloader, tmp_path):
    download = downloader.start(_model("alpha"))
    server.step("alpha.bin", 1)
    _wait(lambda: downloader.downloads[download.id].done >= CHUNK)
    downloader.pause(download.id)
    server.step("alpha.bin", 1)  # lets the loop reach its stop check
    _wait(lambda: _status(downloader, download) == "paused")
    partial = tmp_path / "hf" / "org" / "alpha" / "abc" / "alpha.bin.part"
    assert partial.exists() and partial.stat().st_size >= CHUNK
    assert download.finished_at == 0.0

    kept = partial.stat().st_size
    downloader.resume(download.id)
    server.open("alpha.bin")
    _wait(lambda: _status(downloader, download) == "completed")
    assert server.ranges[-1] == ("alpha.bin", kept)  # continued with a byte range, not from zero
    target = tmp_path / "hf" / "org" / "alpha" / "abc" / "alpha.bin"
    assert target.read_bytes() == b"a" * SIZE


def test_queued_downloads_pause_resume_and_cancel_without_waiting(server, downloader):
    running = downloader.start(_model("alpha"))
    _wait(lambda: _status(downloader, running) == "downloading")
    paused, cancelled, last = (downloader.start(_model(n)) for n in ("bravo", "charlie", "delta"))

    downloader.pause(paused.id)
    downloader.cancel(cancelled.id)
    assert paused.status == "paused" and paused.position is None
    assert cancelled.status == "cancelled" and cancelled.finished_at > 0
    assert downloader.queue() == [last.id] and last.position == 0

    downloader.resume(paused.id)  # back at the end of the queue
    assert downloader.queue() == [last.id, paused.id]
    for name in ("alpha", "bravo", "delta"):
        server.open(f"{name}.bin")
    _wait(lambda: all(_status(downloader, d) == "completed" for d in (running, paused, last)))
    assert "charlie.bin" not in server.order
    assert server.order == ["alpha.bin", "delta.bin", "bravo.bin"]


def test_installing_a_paused_model_again_resumes_it(server, downloader):
    running = downloader.start(_model("alpha"))
    _wait(lambda: _status(downloader, running) == "downloading")
    queued = downloader.start(_model("bravo"))
    downloader.pause(queued.id)
    again = downloader.start(_model("bravo"))
    assert again.id == queued.id and again.status == "queued"
    server.open("alpha.bin")
    server.open("bravo.bin")
    _wait(lambda: _status(downloader, queued) == "completed")


def test_only_paused_or_failed_downloads_resume(server, downloader):
    running = downloader.start(_model("alpha"))
    _wait(lambda: _status(downloader, running) == "downloading")
    with pytest.raises(ValueError):
        downloader.resume(running.id)
    with pytest.raises(KeyError):
        downloader.pause("missing")
    server.open("alpha.bin")
    _wait(lambda: _status(downloader, running) == "completed")
    with pytest.raises(ValueError):
        downloader.pause(running.id)
