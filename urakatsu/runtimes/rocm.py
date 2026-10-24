"""AMD ROCm: which GPUs the ROCm builds run on, and where the ROCm libraries come from.

The targets are the ones the official llama.cpp and stable-diffusion.cpp release builds
compile for (``gpu_targets`` in their release workflows) that AMD also publishes PyTorch
device wheels for, as Koharu's ``hardware/hip/targets.rs`` lists them. An AMD GPU outside the
list runs on Vulkan instead.

The Windows release archives bundle the HIP runtime (``amdhip64_7.dll``) but load hipBLAS and
rocBLAS from the library path. AMD's PyTorch wheels install those libraries into the
environment (``rocm[libraries]`` plus one ``rocm-sdk-device-<target>`` wheel), so the native
servers take them from there; an installed HIP SDK (``HIP_PATH``, ``/opt/rocm``) also works.
"""

from __future__ import annotations

import importlib.util
import os
import platform
from pathlib import Path

# AMD's wheel index for PyTorch and the ROCm SDK (TheRock), Windows and Linux.
WHEEL_INDEX = "https://stable.repo.amd.com/rocm/whl-next"

# target -> integrated GPU (APU)
_COMMON = {
    "gfx1010": False, "gfx1011": False, "gfx1012": False,
    "gfx1030": False, "gfx1031": False, "gfx1032": False, "gfx1033": True,
    "gfx1034": False, "gfx1035": True, "gfx1036": True,
    "gfx1100": False, "gfx1101": False, "gfx1102": False, "gfx1103": True,
    "gfx1150": True, "gfx1151": True, "gfx1152": True,
    "gfx1200": False, "gfx1201": False,
}  # fmt: skip
_WINDOWS = {"gfx1153": True}
_LINUX = {"gfx908": False, "gfx90a": False, "gfx942": False, "gfx950": False}


def targets() -> dict[str, bool]:
    system = platform.system()
    extra = _WINDOWS if system == "Windows" else _LINUX if system == "Linux" else None
    return {} if extra is None else {**_COMMON, **extra}


def supported(target: str) -> bool:
    return target in targets()


def integrated(target: str) -> bool:
    return targets().get(target, False)


def target_from_version(version: int) -> str:
    """KFD's ``gfx_target_version`` (110000, 90010) as an ISA name (gfx1100, gfx90a)."""
    major, minor, stepping = version // 10000, version // 100 % 100, version % 100
    return f"gfx{major}{minor:x}{stepping:x}"


def library_dirs() -> list[Path]:
    """Folders holding hipBLAS, rocBLAS and the device kernels, best first."""
    sub = "bin" if platform.system() == "Windows" else "lib"
    found: list[Path] = []
    for package in ("_rocm_sdk_core", "_rocm_sdk_libraries"):
        try:
            spec = importlib.util.find_spec(package)
        except (ImportError, ValueError):
            spec = None
        for location in (spec.submodule_search_locations or []) if spec else []:
            if (Path(location) / sub).is_dir():
                found.append(Path(location) / sub)
    if platform.system() == "Windows":
        sdk = os.environ.get("HIP_PATH")
        if sdk and (Path(sdk) / "bin" / "hipblas.dll").is_file():
            found.append(Path(sdk) / "bin")
    elif Path("/opt/rocm/lib").is_dir():
        found.append(Path("/opt/rocm/lib"))
    return found


def available() -> bool:
    """Whether a ROCm build of llama.cpp or stable-diffusion.cpp can find the libraries it loads."""
    name = "hipblas.dll" if platform.system() == "Windows" else "libhipblas.so"
    return any(any(path.glob(name + "*")) for path in library_dirs())


def server_env(index: int) -> dict[str, str]:
    """Environment for a ROCm ``llama-server`` or ``sd-server``: the libraries, and only this GPU."""
    key = "PATH" if platform.system() == "Windows" else "LD_LIBRARY_PATH"
    dirs = [str(path) for path in library_dirs()]
    current = os.environ.get(key)
    return {key: os.pathsep.join(dirs + ([current] if current else [])), "HIP_VISIBLE_DEVICES": str(index)}
