"""Hardware discovery without loading any accelerator runtime.

Discovery is kept separate from runtime activation so a cached runtime is
never loaded during the probe. NVIDIA GPUs come from ``nvidia-smi``, every other GPU from
``gpus`` (DXGI, OpenCL, sysfs); Torch and ONNX Runtime are imported lazily by their own
runtime classes.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from functools import lru_cache

from pydantic import BaseModel, Field

from .device import Backend, Device, DeviceType


log = logging.getLogger(__name__)

class Hardware(BaseModel):
    devices: list[Device] = Field(default_factory=list)
    platform: str = ""
    cpu: str = ""

    def candidates(self) -> list[Device]:
        """Accelerators best first, then the CPU fallback.

        As in Koharu: Metal or a discrete CUDA GPU, a discrete ROCm GPU, an integrated CUDA GPU,
        an integrated ROCm GPU, then Vulkan (discrete before integrated) and, on Windows, a GPU
        only DirectML can use; the larger GPU first within each.
        """
        accelerators = sorted((d for d in self.devices if d.is_accelerator), key=rank)
        return accelerators + [Device.cpu()]

    def device(self) -> Device:
        return self.candidates()[0]


_CATEGORY = {
    (Backend.METAL, False): 0, (Backend.METAL, True): 0,
    (Backend.CUDA, False): 0, (Backend.ROCM, False): 1,
    (Backend.CUDA, True): 2, (Backend.ROCM, True): 3,
    (Backend.VULKAN, False): 4, (Backend.VULKAN, True): 5,
    (Backend.DIRECTML, False): 6, (Backend.DIRECTML, True): 7,
}  # fmt: skip


def rank(device: Device) -> tuple[int, int, int]:
    return (_CATEGORY.get((device.backend, device.is_integrated), 8), -device.memory_total, -device.compute_capability)


def _nvidia_devices() -> list[Device]:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return []
    try:
        output = subprocess.run(
            [
                binary,
                "--query-gpu=index,name,memory.total,memory.free,compute_cap,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            # nvidia-smi can take many seconds while another process initializes CUDA (the
            # desktop app imports PyTorch as it starts); giving up early reports no GPU at all.
            timeout=30,
            check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired) as error:
        log.warning("nvidia-smi failed, so no NVIDIA GPU is reported: %s", error)
        return []
    devices: list[Device] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            index = int(parts[0])
            total = int(float(parts[2])) * 1024 * 1024
            free = int(float(parts[3])) * 1024 * 1024
        except ValueError:
            continue
        capability = 0
        if len(parts) >= 5 and parts[4]:
            try:
                major, _, minor = parts[4].partition(".")
                capability = int(major) * 10 + int(minor or 0)
            except ValueError:
                capability = 0
        devices.append(
            Device(
                index=index,
                name=parts[1],
                description=f"{parts[1]} ({total // (1024 * 1024)} MiB)",
                backend=Backend.CUDA,
                device_type=DeviceType.GPU,
                memory_total=total,
                memory_free=free,
                compute_capability=capability,
                driver_version=parts[5] if len(parts) >= 6 else "",
                vendor="nvidia",
            )
        )
    return devices


def _apple_devices() -> list[Device]:
    if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
        return [Device.gpu(Backend.METAL, 0, name="Apple GPU", device_type=DeviceType.INTEGRATED_GPU, vendor="apple")]
    return []


@lru_cache(maxsize=1)
def discover() -> Hardware:
    from . import gpus

    devices = _nvidia_devices() + _apple_devices() + gpus.probe()
    return Hardware(
        devices=devices,
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        cpu=platform.processor() or platform.machine(),
    )


# ---- system profile: memory and CPU, for model fit estimates ------------------------------------


class SystemProfile(BaseModel):
    """Memory and CPU facts the model catalog uses to judge whether a model will run."""

    ram_total: int = 0
    ram_available: int = 0
    cpu_name: str = ""
    cpu_threads: int = 0
    cpu_features: list[str] = Field(default_factory=list)  # avx, avx2, avx512f, neon
    unified_memory: bool = False  # Apple silicon: the GPU uses system RAM


def system_profile() -> SystemProfile:
    """Not cached: available memory changes while models load and unload."""
    import os

    system = platform.system()
    total, available = _memory(system)
    return SystemProfile(
        ram_total=total,
        ram_available=available,
        cpu_name=_cpu_name(system),
        cpu_threads=os.cpu_count() or 0,
        cpu_features=_cpu_features(system),
        unified_memory=system == "Darwin" and platform.machine() in ("arm64", "aarch64"),
    )


def _memory(system: str) -> tuple[int, int]:
    try:
        if system == "Windows":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys), int(status.ullAvailPhys)
        elif system == "Linux":
            info = {}
            with open("/proc/meminfo", encoding="ascii") as handle:
                for line in handle:
                    key, _, value = line.partition(":")
                    info[key] = int(value.split()[0]) * 1024
            return info.get("MemTotal", 0), info.get("MemAvailable", info.get("MemFree", 0))
        elif system == "Darwin":
            total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout.strip() or 0)
            pages = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            page_size, free = 16384, 0
            for line in pages.splitlines():
                if "page size of" in line:
                    page_size = int(line.split("page size of")[1].split()[0])
                if line.startswith(("Pages free", "Pages inactive", "Pages speculative")):
                    free += int(line.split(":")[1].strip().rstrip("."))
            return total, free * page_size
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return 0, 0


def _cpu_name(system: str) -> str:
    try:
        if system == "Windows":
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.lower().startswith(("model name", "hardware")):
                        return line.split(":", 1)[1].strip()
        if system == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def _cpu_features(system: str) -> list[str]:
    features: list[str] = []
    try:
        if system == "Windows":
            import ctypes

            present = ctypes.windll.kernel32.IsProcessorFeaturePresent
            # PF_AVX_INSTRUCTIONS_AVAILABLE, PF_AVX2_..., PF_AVX512F_...
            for name, flag in (("avx", 39), ("avx2", 40), ("avx512f", 41)):
                if present(flag):
                    features.append(name)
        elif system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.startswith(("flags", "Features")):
                        flags = set(line.split(":", 1)[1].split())
                        features = [f for f in ("avx", "avx2", "avx512f", "asimd") if f in flags]
                        break
        if platform.machine().lower() in ("arm64", "aarch64") and "neon" not in features:
            features.append("neon")
    except OSError:
        pass
    return features
