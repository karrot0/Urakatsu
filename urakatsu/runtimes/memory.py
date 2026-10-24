"""Live memory use of the machine Urakatsu runs on, and how much of it Urakatsu holds.

"Urakatsu" is this process plus every process it started (llama-server, sd-server, the source
system runtimes), so models served out of process count too. Two pools are reported:

- GPU memory of the active device. For CUDA, totals come from NVML. NVML also reports
  per-process GPU memory on Linux; Windows (WDDM) hides it there, so it is read from the
  "GPU Process Memory" performance counters instead. On Windows the same counters give
  ``spilled``: GPU allocations the driver moved into shared system memory because VRAM ran out,
  which is far slower. Other discrete GPUs on Windows (AMD, Intel) are read from the counters
  alone, matched by the adapter's LUID. An integrated GPU's memory is system RAM, so only RAM
  is reported for it.
- System RAM. Urakatsu's share is the resident memory of its processes, which includes model
  layers offloaded to the CPU (llama.cpp partial offload, sd-server --offload-to-cpu).
"""

from __future__ import annotations

import ctypes
import logging
import platform
import threading

import psutil
from pydantic import BaseModel

from .device import Backend, Device


log = logging.getLogger(__name__)


class Pool(BaseModel):
    name: str
    total: int
    used: int  # by every process
    app: int | None = None  # by Urakatsu and the processes it started; None when unknown


class GpuPool(Pool):
    spilled: int | None = None  # Urakatsu's GPU memory held in shared system memory (Windows)


class MemoryReport(BaseModel):
    device: str  # "cuda:0" or "cpu"
    gpu: GpuPool | None = None
    ram: Pool


def report(device: Device) -> MemoryReport:
    tree = _tree()
    gpu = None
    if device.backend is Backend.CUDA:
        try:
            gpu = _cuda(device, {process.pid for process in tree})
        except Exception as error:  # NVML missing or the driver refused; RAM still reports
            log.debug("NVML memory query failed: %s", error)
    elif device.is_accelerator and not device.is_integrated and platform.system() == "Windows":
        gpu = _windows_adapter(device, {process.pid for process in tree})
    return MemoryReport(device=device.label(), gpu=gpu, ram=_ram(tree))


def _tree() -> list[psutil.Process]:
    me = psutil.Process()
    try:
        return [me, *me.children(recursive=True)]
    except psutil.Error:
        return [me]


def _ram(tree: list[psutil.Process]) -> Pool:
    memory = psutil.virtual_memory()
    app = 0
    for process in tree:
        try:
            app += process.memory_info().rss
        except psutil.Error:
            pass
    return Pool(name="System memory", total=memory.total, used=memory.total - memory.available, app=min(app, memory.total))


_nvml_lock = threading.Lock()
_nvml_ready = False


def _cuda(device: Device, pids: set[int]) -> GpuPool:
    global _nvml_ready
    import pynvml

    with _nvml_lock:
        if not _nvml_ready:
            pynvml.nvmlInit()
            _nvml_ready = True
    handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    spilled = None
    if platform.system() == "Windows":
        app, spilled = _windows_gpu_usage(pids, info.used)
    else:
        app = 0
        for listing in (pynvml.nvmlDeviceGetComputeRunningProcesses, pynvml.nvmlDeviceGetGraphicsRunningProcesses):
            for process in listing(handle):
                if process.pid in pids and process.usedGpuMemory:
                    app += process.usedGpuMemory
    return GpuPool(
        name=device.name,
        total=info.total,
        used=info.used,
        app=None if app is None else min(app, info.used),
        spilled=spilled,
    )


# ---- Windows performance counters ----------------------------------------------------------------


class _CounterValue(ctypes.Structure):
    _fields_ = [("status", ctypes.c_ulong), ("value", ctypes.c_longlong)]  # PDH_FMT_COUNTERVALUE, large-value arm


class _CounterItem(ctypes.Structure):
    _fields_ = [("name", ctypes.c_wchar_p), ("value", _CounterValue)]  # PDH_FMT_COUNTERVALUE_ITEM_W


_PDH_FMT_LARGE = 0x00000400
_PDH_MORE_DATA = 0x800007D2


class _Counters:
    """One PDH query holding the wildcard GPU memory counters, collected on demand."""

    def __init__(self) -> None:
        self.pdh = ctypes.WinDLL("pdh")
        self.query = ctypes.c_void_p()
        self._check(self.pdh.PdhOpenQueryW(None, None, ctypes.byref(self.query)))
        self.dedicated = self._add(r"\GPU Process Memory(*)\Dedicated Usage")
        self.shared = self._add(r"\GPU Process Memory(*)\Shared Usage")
        self.adapters = self._add(r"\GPU Adapter Memory(*)\Dedicated Usage")
        self.lock = threading.Lock()

    @staticmethod
    def _check(status: int) -> None:
        if status:
            raise OSError(f"PDH error 0x{status & 0xFFFFFFFF:08X}")

    def _add(self, path: str) -> ctypes.c_void_p:
        counter = ctypes.c_void_p()
        self._check(self.pdh.PdhAddEnglishCounterW(self.query, path, None, ctypes.byref(counter)))
        return counter

    def _values(self, counter: ctypes.c_void_p) -> dict[str, int]:
        size, count = ctypes.c_ulong(0), ctypes.c_ulong(0)
        status = self.pdh.PdhGetFormattedCounterArrayW(counter, _PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), None)
        if status & 0xFFFFFFFF != _PDH_MORE_DATA:
            self._check(status)
            return {}
        buffer = ctypes.create_string_buffer(size.value)
        self._check(self.pdh.PdhGetFormattedCounterArrayW(counter, _PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), buffer))
        items = ctypes.cast(buffer, ctypes.POINTER(_CounterItem))
        return {items[i].name: items[i].value.value for i in range(count.value) if items[i].value.status in (0, 1)}

    def sample(self) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        with self.lock:
            self._check(self.pdh.PdhCollectQueryData(self.query))
            return self._values(self.dedicated), self._values(self.shared), self._values(self.adapters)


_counters: _Counters | None = None
_counters_failed = False


def _sample() -> tuple[dict[str, int], dict[str, int], dict[str, int]] | None:
    global _counters, _counters_failed
    if _counters_failed:
        return None
    try:
        if _counters is None:
            _counters = _Counters()
        return _counters.sample()
    except OSError as error:
        log.info("GPU memory counters are unavailable: %s", error)
        _counters_failed = True
        return None


def adapter_usage() -> dict[str, int]:
    """Dedicated memory in use on each GPU by every process, keyed by LUID ("luid_0x..._0x..."), on Windows."""
    sample = _sample()
    usage: dict[str, int] = {}
    for name, value in (sample[2] if sample else {}).items():
        luid = name.lower().partition("_phys")[0]
        usage[luid] = usage.get(luid, 0) + value
    return usage


def _ours(values: dict[str, int], adapter: str, pids: set[int]) -> int:
    total = 0
    for name, value in values.items():
        pid, _, luid = name.removeprefix("pid_").partition("_")
        if luid.lower().startswith(adapter) and pid.isdigit() and int(pid) in pids:
            total += value
    return total


def _windows_adapter(device: Device, pids: set[int]) -> GpuPool | None:
    """A non-NVIDIA GPU's memory from the Windows counters: used by every process, and by ours."""
    sample = _sample()
    if sample is None or not device.locator or not device.memory_total:
        return None
    dedicated, shared, _ = sample
    used = adapter_usage().get(device.locator)
    if used is None:
        return None
    app = _ours(dedicated, device.locator, pids)
    return GpuPool(name=device.name, total=device.memory_total, used=min(used, device.memory_total), app=min(app, used), spilled=_ours(shared, device.locator, pids))


def _windows_gpu_usage(pids: set[int], device_used: int) -> tuple[int | None, int | None]:
    """Dedicated and shared GPU memory of ``pids`` on the adapter whose total use best matches NVML's."""
    sample = _sample()
    if sample is None or not sample[2]:
        return None, None
    dedicated, shared, adapters = sample
    # Instances read "luid_0x..._0x..._phys_0"; the NVIDIA adapter is the one NVML agrees with.
    adapter = min(adapters, key=lambda name: abs(adapters[name] - device_used)).lower()
    return _ours(dedicated, adapter, pids), _ours(shared, adapter, pids)
