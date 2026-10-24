"""Discovery of the GPUs ``nvidia-smi`` does not report: AMD, Intel and any other Vulkan GPU.

Like Koharu (``koharu-runtime/src/hardware``), an AMD GPU whose ISA the ROCm builds support
runs on ROCm and every other GPU on Vulkan, found without loading a GPU runtime:

- Windows: DXGI lists the adapters with their memory and LUID; the system OpenCL loader names
  each AMD GPU's ISA (``gfx1100``), in the order HIP numbers them.
- Linux: sysfs lists the DRM cards with their PCI vendor and, for AMD, memory; the KFD topology
  gives each AMD GPU's ISA.

Vulkan counts as present when its loader library loads. On Windows a GPU without Vulkan still
runs ONNX models through DirectML.
"""

from __future__ import annotations

import ctypes
import logging
import platform
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import rocm
from .device import Backend, Device, DeviceType

log = logging.getLogger(__name__)

GIB = 1024**3
NVIDIA, AMD, AMD_CPU, INTEL, MICROSOFT, QUALCOMM = 0x10DE, 0x1002, 0x1022, 0x8086, 0x1414, 0x5143
# GPU makers only: virtual machines and servers also show display adapters (Hyper-V, VMware,
# QEMU, VirtualBox, a server's ASPEED or Matrox chip) that cannot run a model.
VENDORS = {AMD: "amd", AMD_CPU: "amd", INTEL: "intel", QUALCOMM: "qualcomm"}
# Integrated GPUs report a small carve-out (128 MB to 512 MB) as their own memory.
INTEGRATED_BELOW = 1 * GIB


def probe() -> list[Device]:
    system = platform.system()
    try:
        if system == "Windows":
            return _windows()
        if system == "Linux":
            return _linux()
    except Exception as error:  # discovery must never take the app down; the CPU still works
        log.warning("GPU discovery failed: %s", error)
    return []


def vulkan_available() -> bool:
    try:
        if platform.system() == "Windows":
            ctypes.WinDLL("vulkan-1.dll")
        else:
            ctypes.CDLL("libvulkan.so.1")
    except OSError:
        return False
    return True


def _backend(vendor: str, target: str, vulkan: bool) -> Backend | None:
    if vendor == "amd" and rocm.supported(target):
        return Backend.ROCM
    if vulkan:
        return Backend.VULKAN
    return Backend.DIRECTML if platform.system() == "Windows" else None


def _numbered(devices: list[Device]) -> list[Device]:
    """Indexes as each API counts: HIP over ROCm GPUs, and per vendor otherwise (XPU counts Intel GPUs)."""
    counts: dict[str, int] = {}
    for device in devices:
        key = "rocm" if device.backend is Backend.ROCM else device.vendor
        device.index = counts.get(key, 0)
        counts[key] = device.index + 1
    return devices


# ---- Windows: DXGI adapters, OpenCL ISA names ------------------------------------------------------


@dataclass
class Adapter:
    index: int
    name: str
    vendor_id: int
    dedicated: int
    shared: int
    free: int
    luid: str


def _windows() -> list[Device]:
    adapters = [a for a in dxgi_adapters() if a.vendor_id in VENDORS]  # nvidia-smi reports NVIDIA
    if not adapters:
        return []
    vulkan = vulkan_available()
    amd = [a for a in adapters if a.vendor_id in (AMD, AMD_CPU)]
    targets = _match_targets(amd, opencl_amd_gpus())
    # DXGI's budget ignores what other programs hold; the GPU performance counters include it.
    from .memory import adapter_usage

    usage = adapter_usage()
    devices: list[Device] = []
    for adapter in adapters:
        if adapter.luid in usage:
            adapter.free = max(0, adapter.dedicated - usage[adapter.luid])
        vendor = VENDORS[adapter.vendor_id]
        target = targets.get(adapter.index, "")
        integrated = rocm.integrated(target) if rocm.supported(target) else adapter.dedicated < INTEGRATED_BELOW
        backend = _backend(vendor, target, vulkan)
        if backend is None:
            continue
        # An integrated GPU works in shared system memory; a discrete one in its own.
        total = adapter.dedicated + adapter.shared if integrated else adapter.dedicated
        devices.append(
            Device(
                name=adapter.name,
                description=f"{adapter.name} ({total // 2**20} MiB)",
                backend=backend,
                device_type=DeviceType.INTEGRATED_GPU if integrated else DeviceType.GPU,
                memory_total=total,
                memory_free=min(adapter.free, total) if adapter.free else 0,
                vendor=vendor,
                target=target,
                adapter=adapter.index,
                locator=adapter.luid,
            )
        )
    # HIP numbers ROCm GPUs in OpenCL's order, which `targets` follows; the others keep DXGI's order.
    hip_order = {adapter: position for position, adapter in enumerate(targets)}
    devices.sort(key=lambda d: hip_order.get(d.adapter, len(hip_order)) if d.backend is Backend.ROCM else len(hip_order))
    return _numbered(devices)


def _match_targets(adapters: list[Adapter], gpus: list[tuple[str, str]]) -> dict[int, str]:
    """DXGI adapter index -> ISA, matched by board name, else by order; in OpenCL's order."""
    result: dict[int, str] = {}
    left = list(adapters)
    for target, board in gpus:
        hit = next((a for a in left if board and a.name.strip().lower() == board.strip().lower()), None) or (left[0] if left else None)
        if hit is None:
            break
        left.remove(hit)
        result[hit.index] = target
    return result


class _Luid(ctypes.Structure):
    _fields_ = [("low", ctypes.c_ulong), ("high", ctypes.c_long)]


class _AdapterDesc1(ctypes.Structure):
    _fields_ = [
        ("description", ctypes.c_wchar * 128),
        ("vendor_id", ctypes.c_uint),
        ("device_id", ctypes.c_uint),
        ("subsys_id", ctypes.c_uint),
        ("revision", ctypes.c_uint),
        ("dedicated_video", ctypes.c_size_t),
        ("dedicated_system", ctypes.c_size_t),
        ("shared_system", ctypes.c_size_t),
        ("luid", _Luid),
        ("flags", ctypes.c_uint),
    ]


class _VideoMemoryInfo(ctypes.Structure):
    _fields_ = [("budget", ctypes.c_uint64), ("usage", ctypes.c_uint64), ("available", ctypes.c_uint64), ("reserved", ctypes.c_uint64)]


_IID_FACTORY1 = uuid.UUID("770aae78-f26f-4dba-a829-253c83d1b387")
_IID_ADAPTER3 = uuid.UUID("645967a4-1392-4310-a798-8053ce3e93fd")
_SOFTWARE = 2  # DXGI_ADAPTER_FLAG_SOFTWARE


def _guid(value: uuid.UUID) -> ctypes.Array:
    return (ctypes.c_ubyte * 16).from_buffer_copy(value.bytes_le)


def _method(obj: ctypes.c_void_p, slot: int, *argtypes) -> Callable:
    """The COM method in vtable ``slot`` of ``obj``, bound to it."""
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    function = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtable[slot])
    return lambda *args: function(obj, *args)


def _release(obj: ctypes.c_void_p) -> None:
    if obj:
        _method(obj, 2)()  # IUnknown::Release


def dxgi_adapters() -> list[Adapter]:
    """Every hardware adapter, in DXGI order (DirectML's ``device_id``), with free memory where DXGI 1.4 reports it."""
    factory = ctypes.c_void_p()
    if ctypes.WinDLL("dxgi").CreateDXGIFactory1(ctypes.byref(_guid(_IID_FACTORY1)), ctypes.byref(factory)) != 0:
        return []
    adapters: list[Adapter] = []
    try:
        enum_adapters1 = _method(factory, 12, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))
        for index in range(64):
            adapter = ctypes.c_void_p()
            if enum_adapters1(index, ctypes.byref(adapter)) != 0:  # DXGI_ERROR_NOT_FOUND ends the list
                break
            try:
                desc = _AdapterDesc1()
                if _method(adapter, 10, ctypes.POINTER(_AdapterDesc1))(ctypes.byref(desc)) != 0 or desc.flags & _SOFTWARE:
                    continue
                adapters.append(
                    Adapter(
                        index=index,
                        name=desc.description.strip(),
                        vendor_id=desc.vendor_id,
                        dedicated=int(desc.dedicated_video),
                        shared=int(desc.shared_system),
                        free=_free_memory(adapter),
                        # GPU performance counter instances read "luid_0x00000000_0x0000D1B2_phys_0".
                        luid=f"luid_0x{desc.luid.high & 0xFFFFFFFF:08x}_0x{desc.luid.low:08x}",
                    )
                )
            finally:
                _release(adapter)
    finally:
        _release(factory)
    return adapters


def _free_memory(adapter: ctypes.c_void_p) -> int:
    """What this process may still place in the adapter's own memory (budget minus use), or 0."""
    adapter3 = ctypes.c_void_p()
    query_interface = _method(adapter, 0, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
    if query_interface(ctypes.byref(_guid(_IID_ADAPTER3)), ctypes.byref(adapter3)) != 0:
        return 0
    try:
        info = _VideoMemoryInfo()
        # QueryVideoMemoryInfo(node 0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL)
        if _method(adapter3, 14, ctypes.c_uint, ctypes.c_int, ctypes.POINTER(_VideoMemoryInfo))(0, 0, ctypes.byref(info)) != 0:
            return 0
        return max(0, int(info.budget) - int(info.usage))
    finally:
        _release(adapter3)


_CL_DEVICE_TYPE_GPU = 1 << 2
_CL_DEVICE_NAME = 0x102B
_CL_DEVICE_VENDOR_ID = 0x1001
_CL_DEVICE_BOARD_NAME_AMD = 0x4038
_ISA = re.compile(r"gfx[0-9a-f]+", re.IGNORECASE)


def opencl_amd_gpus() -> list[tuple[str, str]]:
    """(ISA, board name) of each AMD GPU the AMD OpenCL driver lists, in its order."""
    try:
        # The system loader by full path: a bare name could pick an application-local OpenCL.dll.
        cl = ctypes.WinDLL(str(Path(_system_directory()) / "OpenCL.dll"))
    except OSError:
        return []

    def ids(query: Callable[[int, object, object], int]) -> list[ctypes.c_void_p]:
        count = ctypes.c_uint(0)
        if query(0, None, ctypes.byref(count)) != 0 or not 0 < count.value <= 1024:
            return []
        values = (ctypes.c_void_p * count.value)()
        if query(count.value, values, ctypes.byref(count)) != 0:
            return []
        return [ctypes.c_void_p(v) for v in values[: count.value] if v]

    def info(device: ctypes.c_void_p, parameter: int) -> bytes:
        size = ctypes.c_size_t(0)
        if cl.clGetDeviceInfo(device, parameter, 0, None, ctypes.byref(size)) != 0 or not 0 < size.value <= 65536:
            return b""
        buffer = ctypes.create_string_buffer(size.value)
        if cl.clGetDeviceInfo(device, parameter, size, buffer, None) != 0:
            return b""
        return buffer.raw

    found: list[tuple[str, str]] = []
    for platform_id in ids(lambda n, out, count: cl.clGetPlatformIDs(n, out, count)):
        for device in ids(lambda n, out, count: cl.clGetDeviceIDs(platform_id, ctypes.c_uint64(_CL_DEVICE_TYPE_GPU), n, out, count)):
            vendor = int.from_bytes(info(device, _CL_DEVICE_VENDOR_ID)[:4] or b"\0", "little")
            name = info(device, _CL_DEVICE_NAME).split(b"\0")[0].decode(errors="replace")
            match = _ISA.match(name.strip())
            if vendor != AMD or not match:
                continue
            board = info(device, _CL_DEVICE_BOARD_NAME_AMD).split(b"\0")[0].decode(errors="replace")
            found.append((match.group(0).lower(), board))
    return found


def _system_directory() -> str:
    buffer = ctypes.create_unicode_buffer(260)
    ctypes.windll.kernel32.GetSystemDirectoryW(buffer, len(buffer))
    return buffer.value


# ---- Linux: sysfs DRM cards, KFD topology ----------------------------------------------------------

DRM = Path("/sys/class/drm")
KFD_NODES = Path("/sys/class/kfd/kfd/topology/nodes")


def _linux() -> list[Device]:
    targets = _kfd_targets()  # render minor -> ISA, in HIP's order
    hip_order = {minor: position for position, minor in enumerate(targets)}
    vulkan = vulkan_available()
    found: list[tuple[int, Device]] = []
    for card in sorted((p for p in DRM.glob("card*") if re.fullmatch(r"card\d+", p.name)), key=lambda p: int(p.name[4:])):
        node = card / "device"
        vendor_id = _read_int(node / "vendor", 16)
        if vendor_id not in VENDORS:
            continue  # NVIDIA is nvidia-smi's; the rest are no GPUs to run models on
        vendor = VENDORS[vendor_id]
        minor = _render_minor(node)
        target = targets.get(minor, "") if minor is not None else ""
        vram, used = _read_int(node / "mem_info_vram_total"), _read_int(node / "mem_info_vram_used")
        if rocm.supported(target):
            integrated = rocm.integrated(target)
        else:
            integrated = vram < INTEGRATED_BELOW
        backend = _backend(vendor, target, vulkan)
        if backend is None:
            continue
        total = vram + _read_int(node / "mem_info_gtt_total") if integrated else vram
        name = (_read_text(node / "product_name") or {"amd": "AMD Radeon", "intel": "Intel"}.get(vendor, "GPU")) + (f" ({target})" if target else "")
        # ROCm GPUs first, in HIP's order; the others after them, in card order (a stable sort).
        position = hip_order.get(minor, len(hip_order)) if backend is Backend.ROCM else len(hip_order)
        device = Device(
            name=name,
            description=f"{name} ({total // 2**20} MiB)" if total else name,
            backend=backend,
            device_type=DeviceType.INTEGRATED_GPU if integrated else DeviceType.GPU,
            memory_total=total,
            memory_free=max(0, vram - used) if vram and not integrated else 0,
            vendor=vendor,
            target=target,
            locator=str(node),
        )
        found.append((position, device))
    found.sort(key=lambda pair: pair[0])
    return _numbered([device for _, device in found])


def _render_minor(device: Path) -> int | None:
    """The card's render node number: ``device/drm`` holds ``cardN`` and ``renderD<minor>``."""
    for entry in (device / "drm").glob("renderD*"):
        if entry.name[7:].isdigit():
            return int(entry.name[7:])
    return None


def _kfd_targets() -> dict[int, str]:
    """Render minor -> ISA for each KFD GPU node, sorted as rocm-bootstrap sorts them (lexically)."""
    result: dict[int, str] = {}
    if not KFD_NODES.is_dir():
        return result
    for node in sorted((p for p in KFD_NODES.iterdir() if p.name.isdigit()), key=lambda p: p.name):
        properties = dict(line.split()[:2] for line in _read_text(node / "properties").splitlines() if len(line.split()) == 2)
        version, minor = int(properties.get("gfx_target_version", 0)), properties.get("drm_render_minor", "")
        if version and int(properties.get("simd_count", 0)) and minor.isdigit():
            result[int(minor)] = rocm.target_from_version(version)
    return result


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def _read_int(path: Path, base: int = 10) -> int:
    try:
        return int(_read_text(path), base)
    except ValueError:
        return 0
