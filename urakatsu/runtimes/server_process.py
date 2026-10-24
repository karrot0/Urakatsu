"""A native runtime server (llama-server, sd-server) owned by one loaded model.

The process listens on a free localhost port, writes its log under
``URAKATSU_HOME/logs``, and is stopped when the provider unloads, which frees its
device memory the way dropping a model does.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import httpx

from urakatsu.core.config import home
from urakatsu.core.errors import ModelLoadError

from .device import Backend, Device

log = logging.getLogger(__name__)

STARTUP_TIMEOUT = 600.0  # seconds: large models load slowly from disk
STOP_TIMEOUT = 10.0


def device_env(device: Device, variant: str) -> dict[str, str]:
    """Environment that confines a native server to ``device``; a ROCm build also gets the ROCm libraries."""
    if device.backend is Backend.CUDA:
        return {"CUDA_VISIBLE_DEVICES": str(device.index)}
    if device.backend is Backend.ROCM and variant.endswith("-rocm"):
        from . import rocm

        return rocm.server_env(device.index)
    return {}


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ServerProcess:
    def __init__(
        self,
        name: str,
        executable: Path,
        args: Sequence[str],
        port: int,
        ready: Callable[[httpx.Client], bool],
        env: Optional[dict[str, str]] = None,
    ):
        self.name = name
        self.base_url = f"http://127.0.0.1:{port}"
        self.log_path = home() / "logs" / f"{name}-{port}.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w", encoding="utf-8", errors="replace")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        command = [str(executable), *args]
        log.info("starting %s: %s", name, " ".join(command))
        self._process = subprocess.Popen(
            command,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=str(executable.parent),
            env={**os.environ, **(env or {})},
            creationflags=flags,
        )
        self.client = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(10.0, read=None))
        try:
            self._wait(ready)
        except BaseException:
            self.stop()
            raise

    def _wait(self, ready: Callable[[httpx.Client], bool]) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise ModelLoadError(f"{self.name} exited with code {self._process.returncode}:\n{self.log_tail()}")
            try:
                if ready(self.client):
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        raise ModelLoadError(f"{self.name} did not become ready within {STARTUP_TIMEOUT:.0f} s:\n{self.log_tail()}")

    @property
    def running(self) -> bool:
        return self._process.poll() is None

    def log_tail(self, lines: int = 25) -> str:
        try:
            self._log.flush()
            return "\n".join(self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
        except OSError:
            return ""

    def stop(self) -> None:
        try:
            self.client.close()
        finally:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(STOP_TIMEOUT)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(STOP_TIMEOUT)
            self._log.close()

    def __del__(self) -> None:  # a provider that is garbage-collected must not leak its server
        try:
            if self._process.poll() is None:
                self._process.kill()
        except Exception:  # noqa: BLE001
            pass
