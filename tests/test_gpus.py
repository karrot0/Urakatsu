"""GPUs beyond NVIDIA: discovery (DXGI, OpenCL, sysfs), ranking, native builds, PyTorch and ONNX choices."""

import sys
import types

import httpx
import pytest

from urakatsu.catalog import fit
from urakatsu.desktop import gpu_setup
from urakatsu.runtimes import gpus, memory, rocm
from urakatsu.runtimes.device import Backend, Device, DeviceType
from urakatsu.runtimes.hardware import Hardware
from urakatsu.runtimes.onnx import OnnxRuntime
from urakatsu.runtimes.packages import preferred_variants
from urakatsu.runtimes.server_process import device_env

GB = 1024**3
RTX = Device.gpu(Backend.CUDA, 0, "RTX 4090", memory_total=24 * GB, vendor="nvidia", driver_version="610.74")
RADEON = Device.gpu(Backend.ROCM, 0, "Radeon RX 7900 XTX", memory_total=24 * GB, vendor="amd", target="gfx1100", adapter=1)
RADEON_IGPU = Device.gpu(Backend.ROCM, 1, "Radeon 780M", memory_total=20 * GB, vendor="amd", target="gfx1103", device_type=DeviceType.INTEGRATED_GPU)
ARC = Device.gpu(Backend.VULKAN, 0, "Intel(R) Arc(TM) B580 Graphics", memory_total=12 * GB, vendor="intel", adapter=2)
IRIS = Device.gpu(Backend.VULKAN, 0, "Intel(R) Iris(R) Xe Graphics", memory_total=16 * GB, vendor="intel", device_type=DeviceType.INTEGRATED_GPU)


def test_kfd_versions_name_the_isa():
    assert rocm.target_from_version(110000) == "gfx1100"
    assert rocm.target_from_version(90010) == "gfx90a"
    assert rocm.target_from_version(100302) == "gfx1032"
    assert rocm.target_from_version(110501) == "gfx1151"


def test_gpus_rank_as_koharu_ranks_them():
    hardware = Hardware(devices=[IRIS, RADEON_IGPU, ARC, RADEON, RTX])
    assert [d.name for d in hardware.candidates()] == [RTX.name, RADEON.name, RADEON_IGPU.name, ARC.name, IRIS.name, "CPU"]
    # An integrated GPU's shared memory does not outrank a discrete GPU of the same backend.
    assert Hardware(devices=[IRIS, ARC]).device().name == ARC.name


def test_the_catalog_rates_against_the_gpu_the_pipeline_picks():
    from urakatsu.runtimes.hardware import SystemProfile

    system = SystemProfile(ram_total=64 * GB, ram_available=48 * GB)
    assert fit.usable_vram([IRIS, ARC], system) == int(12 * GB * 0.95) - fit.VRAM_RESERVE


@pytest.mark.parametrize("system", ["Windows", "Linux"])
def test_native_builds_follow_the_gpu(monkeypatch, system):
    monkeypatch.setattr("urakatsu.runtimes.packages.platform.system", lambda: system)
    prefix = "win" if system == "Windows" else "linux"
    monkeypatch.setattr(rocm, "available", lambda: True)
    assert preferred_variants([ARC, RADEON, Device.cpu()]) == [f"{prefix}-rocm", f"{prefix}-vulkan", f"{prefix}-cpu"]
    # Without hipBLAS and rocBLAS on the machine, the ROCm build cannot load: Vulkan instead.
    monkeypatch.setattr(rocm, "available", lambda: False)
    assert preferred_variants([RADEON, Device.cpu()]) == [f"{prefix}-vulkan", f"{prefix}-cpu"]
    assert preferred_variants([IRIS, Device.cpu()]) == [f"{prefix}-vulkan", f"{prefix}-cpu"]
    assert preferred_variants([Device.gpu(Backend.DIRECTML, 0), Device.cpu()]) == [f"{prefix}-cpu"]
    # An NVIDIA GPU next to an AMD one keeps CUDA.
    assert preferred_variants([RADEON, RTX, Device.cpu()])[0] == f"{prefix}-cuda-13"


def test_servers_see_only_their_gpu_and_rocm_gets_its_libraries(monkeypatch, tmp_path):
    assert device_env(RTX, "win-cuda-13") == {"CUDA_VISIBLE_DEVICES": "0"}
    monkeypatch.setattr(rocm, "library_dirs", lambda: [tmp_path])
    env = device_env(RADEON, "win-rocm")
    assert env["HIP_VISIBLE_DEVICES"] == "0"
    assert str(tmp_path) in env["PATH" if sys.platform == "win32" else "LD_LIBRARY_PATH"]
    assert device_env(RADEON, "win-vulkan") == {} and device_env(ARC, "win-vulkan") == {}


def _adapter(index, name, vendor, dedicated, shared=16 * GB):
    return gpus.Adapter(index, name, vendor, dedicated, shared, 0, f"luid_0x00000000_0x0000000{index}")


def test_windows_discovery(monkeypatch):
    adapters = [
        _adapter(0, "NVIDIA GeForce RTX 4090", gpus.NVIDIA, 24 * GB),
        _adapter(1, "AMD Radeon(TM) Graphics", gpus.AMD, 512 * 2**20),  # the APU, listed first by DXGI
        _adapter(2, "AMD Radeon RX 7900 XTX", gpus.AMD, 24 * GB),
        _adapter(3, "Intel(R) Arc(TM) B580 Graphics", gpus.INTEL, 12 * GB),
        _adapter(4, "AMD Radeon RX 580", gpus.AMD, 8 * GB),
        _adapter(5, "Microsoft Hyper-V Video", gpus.MICROSOFT, 64 * 2**20),  # a virtual machine's display
    ]
    monkeypatch.setattr(gpus.platform, "system", lambda: "Windows")  # DirectML is the fallback there
    monkeypatch.setattr(gpus, "dxgi_adapters", lambda: adapters)
    # OpenCL (HIP's order) lists the discrete card first; the RX 580 (gfx803) is no ROCm target.
    monkeypatch.setattr(gpus, "opencl_amd_gpus", lambda: [("gfx1100", "AMD Radeon RX 7900 XTX"), ("gfx1036", "AMD Radeon(TM) Graphics"), ("gfx803", "AMD Radeon RX 580")])
    monkeypatch.setattr(gpus, "vulkan_available", lambda: True)
    monkeypatch.setattr(memory, "adapter_usage", lambda: {"luid_0x00000000_0x00000002": 4 * GB})
    monkeypatch.setattr(rocm, "targets", lambda: {"gfx1100": False, "gfx1036": True})
    found = {d.name: d for d in gpus._windows()}
    assert "NVIDIA GeForce RTX 4090" not in found  # nvidia-smi reports it
    assert "Microsoft Hyper-V Video" not in found
    xtx, apu, arc, old = (found[n] for n in ("AMD Radeon RX 7900 XTX", "AMD Radeon(TM) Graphics", "Intel(R) Arc(TM) B580 Graphics", "AMD Radeon RX 580"))
    assert (xtx.backend, xtx.index, xtx.target, xtx.adapter, xtx.memory_free) == (Backend.ROCM, 0, "gfx1100", 2, 20 * GB)
    assert (apu.backend, apu.index, apu.device_type) == (Backend.ROCM, 1, DeviceType.INTEGRATED_GPU)
    assert apu.memory_total == 512 * 2**20 + 16 * GB  # an integrated GPU works in shared memory
    assert (arc.backend, arc.vendor, arc.index, arc.device_type) == (Backend.VULKAN, "intel", 0, DeviceType.GPU)
    assert (old.backend, old.target) == (Backend.VULKAN, "gfx803")
    # Without a Vulkan loader, what ROCm cannot run is left to DirectML.
    monkeypatch.setattr(gpus, "vulkan_available", lambda: False)
    assert {d.name: d.backend for d in gpus._windows()}["Intel(R) Arc(TM) B580 Graphics"] is Backend.DIRECTML


def test_targets_match_dxgi_adapters_by_board_name_then_by_order():
    a, b = _adapter(1, "Radeon A", gpus.AMD, GB), _adapter(2, "Radeon B", gpus.AMD, GB)
    assert gpus._match_targets([a, b], [("gfx1100", "Radeon B"), ("gfx1030", "")]) == {2: "gfx1100", 1: "gfx1030"}


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_linux_discovery(monkeypatch, tmp_path):
    drm, kfd = tmp_path / "drm", tmp_path / "kfd"
    cards = (("card0", "0x10de", 128), ("card1", "0x1002", 129), ("card2", "0x8086", 130), ("card3", "0x1234", 131))  # card3: QEMU's display
    for card, vendor, minor in cards:
        _write(drm / card / "device" / "vendor", vendor)
        (drm / card / "device" / "drm" / f"renderD{minor}").mkdir(parents=True)
    _write(drm / "card1" / "device" / "mem_info_vram_total", str(16 * GB))
    _write(drm / "card1" / "device" / "mem_info_vram_used", str(GB))
    (drm / "card1-DP-1").mkdir()  # a connector, not a card
    _write(kfd / "1" / "properties", "simd_count 192\ngfx_target_version 110000\ndrm_render_minor 129\n")
    _write(kfd / "0" / "properties", "simd_count 0\ngfx_target_version 0\n")  # the CPU node
    monkeypatch.setattr(gpus, "DRM", drm)
    monkeypatch.setattr(gpus, "KFD_NODES", kfd)
    monkeypatch.setattr(gpus, "vulkan_available", lambda: True)
    monkeypatch.setattr(rocm, "targets", lambda: {"gfx1100": False})
    found = gpus._linux()
    assert [(d.vendor, d.backend, d.target, d.index) for d in found] == [("amd", Backend.ROCM, "gfx1100", 0), ("intel", Backend.VULKAN, "", 0)]
    assert (found[0].memory_total, found[0].memory_free) == (16 * GB, 15 * GB)


def _index(pages: dict[str, str]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = next((text for path, text in pages.items() if request.url.path.endswith(path)), None)
        return httpx.Response(200, text=body) if body else httpx.Response(404)

    return httpx.MockTransport(handler)


def test_pytorch_follows_the_gpu(monkeypatch):
    monkeypatch.setattr(gpu_setup.platform, "system", lambda: "Windows")
    wheel = f'<a href="torch-2.14.0%2Bx{gpu_setup.wheel_suffix().replace("+", "%2B")}">torch</a>'
    transport = _index({"/rocm/whl-next/torch/": wheel, "/whl/xpu/torch/": wheel})
    amd = gpu_setup.choose_build(None, transport, RADEON)
    assert (amd.tag, amd.url, amd.label, amd.onnx_extra) == ("rocm-gfx1100", rocm.WHEEL_INDEX, "AMD ROCm (gfx1100)", "onnx-directml")
    assert amd.packages[:2] == ["torch[device-gfx1100]", "torchvision[device-gfx1100]"]
    assert gpu_setup.choose_build(None, transport, ARC).tag == "xpu"
    # Intel graphics older than Arc have no XPU support: CPU PyTorch, DirectML ONNX.
    iris = gpu_setup.choose_build(None, transport, IRIS)
    assert (iris.tag, iris.onnx_extra) == ("cpu", "onnx-directml")
    assert gpu_setup.choose_build(None, transport, None).onnx_extra == "onnx"
    assert gpu_setup.choose_build(None, _index({}), RADEON).tag == "cpu"  # no wheel for this Python


def test_the_rocm_plan_allows_amds_pre_release_torchvision():
    build = gpu_setup.TorchBuild("rocm-gfx1100", rocm.WHEEL_INDEX, directml=True)
    torch_step, libraries = gpu_setup.plan(build, uv="uv")
    assert torch_step.command[5:] == ["torch[device-gfx1100]", "torchvision[device-gfx1100]", "--prerelease", "allow", "--index-url", rocm.WHEEL_INDEX]
    assert any(arg.startswith("onnxruntime-directml") for arg in libraries.command)


def test_onnx_reaches_other_gpus_through_directml(monkeypatch):
    monkeypatch.setattr("urakatsu.runtimes.onnx.platform.system", lambda: "Windows")
    runtime = OnnxRuntime.__new__(OnnxRuntime)
    runtime._ort = types.SimpleNamespace(get_available_providers=lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
    for device in (RADEON, ARC):
        runtime.device, runtime._fallback_reason = device, None
        assert runtime._select_providers()[0] == ("DmlExecutionProvider", {"device_id": device.adapter})
    runtime._ort = types.SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
    runtime.device, runtime._fallback_reason = ARC, None
    assert runtime._select_providers() == ["CPUExecutionProvider"] and runtime._fallback_reason


def test_other_gpus_report_memory_from_the_windows_counters(monkeypatch):
    luid = "luid_0x00000000_0x00000002"
    xtx = RADEON.model_copy(update={"locator": luid})
    dedicated = {f"pid_10_{luid}_phys_0": 3 * GB, f"pid_99_{luid}_phys_0": 5 * GB}
    monkeypatch.setattr(memory, "_sample", lambda: (dedicated, {f"pid_10_{luid}_phys_0": GB}, {f"{luid}_phys_0": 8 * GB}))
    pool = memory._windows_adapter(xtx, {10})
    assert (pool.total, pool.used, pool.app, pool.spilled) == (24 * GB, 8 * GB, 3 * GB, GB)


def test_macs_get_the_build_for_their_processor(monkeypatch):
    import platform

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    assert preferred_variants([Device.cpu()]) == ["macos-arm64"]
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    assert preferred_variants([Device.cpu()]) == ["macos-x64"]
