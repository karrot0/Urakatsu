"""A native runtime server (llama-server, sd-server) owned by one loaded model.

The process listens on a free localhost port, writes its log under
``URAKATSU_HOME/logs``, and is stopped when the provider unloads, which frees its
device memory the way dropping a model does.

A server never outlives Urakatsu: on Windows each one joins a job object that kills its
members when the Urakatsu process ends, however it ends (a crash or a killed process
included); elsewhere the servers still running at interpreter exit are killed.
"""

from __future__ import annotations

import atexit
import logging
import os
import socket
import subprocess
import sys
import time
import weakref
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


_JOB: Optional[int] = None  # the kill-on-close job object; its handle closes when this process ends
_LIVE: "weakref.WeakSet[ServerProcess]" = weakref.WeakSet()


def _kill_on_close_job() -> Optional[int]:
    """A Windows job object that kills every process in it when its last handle closes.

    The handle is never closed explicitly: Windows closes it when this process ends, which
    takes the servers with it."""
    global _JOB
    if _JOB is not None:
        return _JOB
    import ctypes
    from ctypes import wintypes

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("per_process_user_time", ctypes.c_int64),
            ("per_job_user_time", ctypes.c_int64),
            ("limit_flags", wintypes.DWORD),
            ("min_working_set", ctypes.c_size_t),
            ("max_working_set", ctypes.c_size_t),
            ("active_process_limit", wintypes.DWORD),
            ("affinity", ctypes.c_size_t),
            ("priority_class", wintypes.DWORD),
            ("scheduling_class", wintypes.DWORD),
        ]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("basic", BasicLimits),
            ("io", IoCounters),
            ("process_memory_limit", ctypes.c_size_t),
            ("job_memory_limit", ctypes.c_size_t),
            ("peak_process_memory", ctypes.c_size_t),
            ("peak_job_memory", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = ExtendedLimits()
    limits.basic.limit_flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):  # JobObjectExtendedLimitInformation
        raise ctypes.WinError(ctypes.get_last_error())
    _JOB = job
    return job


def _tie_to_this_process(process: subprocess.Popen) -> None:
    """Makes ``process`` end when Urakatsu does (Windows; see the module docstring)."""
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    try:
        job = _kill_on_close_job()
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        if not kernel32.AssignProcessToJobObject(job, int(process._handle)):  # type: ignore[attr-defined]
            raise ctypes.WinError(ctypes.get_last_error())
    except OSError as error:
        log.warning("could not tie server %d to this process; it may outlive a crash: %s", process.pid, error)


@atexit.register
def _kill_live_servers() -> None:
    for server in list(_LIVE):
        server.kill()


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
        _tie_to_this_process(self._process)
        _LIVE.add(self)
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
            _LIVE.discard(self)

    def kill(self) -> None:
        try:
            if self._process.poll() is None:
                self._process.kill()
        except Exception:  # noqa: BLE001
            pass

    def __del__(self) -> None:  # a provider that is garbage-collected must not leak its server
        self.kill()
