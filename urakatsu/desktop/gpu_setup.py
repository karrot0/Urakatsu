"""First-run GPU setup for the desktop app: the newest PyTorch build this machine's GPU runs.

The launcher installs Urakatsu's core dependencies; PyTorch is installed here, on first start,
because the right build depends on the GPU:

- NVIDIA: the driver reports the newest CUDA version it supports (``nvidia-smi``). Starting
  there and counting down, the first PyTorch wheel index that has a wheel for this Python wins,
  so a new driver gets a new CUDA build without changes here (AGENTS.md: always the latest
  runtimes).
- AMD, on a GPU the ROCm builds support: AMD's PyTorch wheels with the device libraries for its
  ISA (``torch[device-gfx1100]``). They also bring the ROCm libraries llama.cpp and
  stable-diffusion.cpp load (``runtimes/rocm.py``).
- Intel Arc on Windows (discrete, or the Arc graphics of Core Ultra): PyTorch's XPU build.
- Anything else: the CPU build.

ONNX Runtime follows: the CUDA build for NVIDIA, DirectML for any other GPU on Windows.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import httpx

from urakatsu.core.config import home
from urakatsu.runtimes import rocm
from urakatsu.runtimes.device import Backend, Device

log = logging.getLogger(__name__)

TORCH_INDEX = "https://download.pytorch.org/whl"
CUDA_HEADER = re.compile(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)")
OLDEST_CUDA_MAJOR = 12  # PyTorch dropped CUDA 11 builds
NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW: no console flashes from pythonw


@dataclass(frozen=True)
class TorchBuild:
    tag: str  # cu132 | rocm-gfx1100 | xpu | cpu
    url: str
    directml: bool = False  # ONNX Runtime through DirectML (a non-NVIDIA GPU on Windows)

    @property
    def label(self) -> str:
        if self.tag == "cpu":
            return "CPU"
        if self.tag == "xpu":
            return "Intel Arc (XPU)"
        if self.tag.startswith("rocm-"):
            return f"AMD ROCm ({self.tag[5:]})"
        digits = self.tag[2:]
        return f"CUDA {digits[:-1]}.{digits[-1]}"

    @property
    def packages(self) -> list[str]:
        if self.tag.startswith("rocm-"):
            extra = f"[device-{self.tag[5:]}]"
            # AMD labels the torchvision built for the newest torch a pre-release (0.29.0a0);
            # without pre-releases uv pairs torch with the previous torchvision, built for an older torch.
            return [f"torch{extra}", f"torchvision{extra}", "--prerelease", "allow"]
        return ["torch", "torchvision"]

    @property
    def onnx_extra(self) -> str:
        if self.tag.startswith("cu"):
            return "onnx-gpu"
        return "onnx-directml" if self.directml else "onnx"


def needs_setup() -> bool:
    return importlib.util.find_spec("torch") is None


def driver_cuda() -> Optional[tuple[int, int]]:
    """The newest CUDA version the installed NVIDIA driver supports, or None without one."""
    try:
        output = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20, creationflags=NO_WINDOW).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = CUDA_HEADER.search(output)
    return (int(match.group(1)), int(match.group(2))) if match else None


def candidate_tags(cuda: tuple[int, int]) -> Iterator[str]:
    """cuXY tags from the driver's CUDA version down to the oldest PyTorch still builds."""
    major, minor = cuda
    for m in range(major, OLDEST_CUDA_MAJOR - 1, -1):
        for n in range(minor if m == major else 9, -1, -1):
            yield f"cu{m}{n}"


def wheel_suffix() -> str:
    python = f"cp{sys.version_info.major}{sys.version_info.minor}"
    machine = {"Windows": "win_amd64", "Linux": "linux_x86_64"}.get(platform.system(), "")
    return f"-{python}-{python}-{machine}.whl"


def primary_device() -> Device:
    """The GPU the pipeline will pick (discovery loads no GPU runtime), or the CPU."""
    from urakatsu.runtimes.hardware import discover

    return discover().candidates()[0]


def choose_build(
    cuda: Optional[tuple[int, int]],
    transport: Optional[httpx.BaseTransport] = None,
    device: Optional[Device] = None,
) -> TorchBuild:
    """The PyTorch build for an NVIDIA driver's CUDA version, else for ``device`` (AMD, Intel)."""
    directml = platform.system() == "Windows" and device is not None and device.is_accelerator and device.vendor != "nvidia"
    cpu = TorchBuild("cpu", f"{TORCH_INDEX}/cpu", directml)
    suffix = wheel_suffix()
    with httpx.Client(timeout=30.0, follow_redirects=True, transport=transport) as client:

        def has_wheel(index: str) -> bool:
            response = client.get(f"{index}/torch/")
            return response.status_code == 200 and suffix in response.text.replace("%2B", "+")

        if cuda is not None:
            for tag in candidate_tags(cuda):
                if has_wheel(f"{TORCH_INDEX}/{tag}"):
                    return TorchBuild(tag, f"{TORCH_INDEX}/{tag}")
            log.warning("no PyTorch CUDA build found for CUDA %s.%s; using the CPU build", *cuda)
            return cpu
        if device is None or not device.is_accelerator:
            return cpu
        if device.backend is Backend.ROCM:
            if has_wheel(rocm.WHEEL_INDEX):
                return TorchBuild(f"rocm-{device.target}", rocm.WHEEL_INDEX, directml)
            log.warning("AMD publishes no PyTorch ROCm build for this Python; using the CPU build")
        elif device.vendor == "intel" and platform.system() == "Windows" and "arc" in device.name.lower():
            if has_wheel(f"{TORCH_INDEX}/xpu"):
                return TorchBuild("xpu", f"{TORCH_INDEX}/xpu", directml)
            log.warning("no PyTorch XPU build for this Python; using the CPU build")
    return cpu


def extra_requirements(*extras: str, exclude: tuple[str, ...] = ()) -> list[str]:
    """Requirements of Urakatsu's optional extras, read from the installed package metadata."""
    found = []
    for requirement in importlib.metadata.requires("urakatsu") or []:
        spec, _, marker = requirement.partition(";")
        marker = marker.replace("'", '"')
        name = re.split(r"[<>=!~\[ ]", spec.strip(), maxsplit=1)[0].lower()
        if any(f'extra == "{extra}"' in marker for extra in extras) and name not in exclude:
            found.append(spec.strip())
    return found


def uv_executable() -> str:
    uv = os.environ.get("URAKATSU_UV") or shutil.which("uv")
    if not uv:
        raise RuntimeError("uv was not found; reinstall Urakatsu")
    return uv


@dataclass
class Step:
    id: str
    title: str
    command: list[str]


def plan(build: TorchBuild, uv: Optional[str] = None) -> list[Step]:
    uv = uv or uv_executable()
    base = [uv, "pip", "install", "--python", sys.executable]
    libraries = extra_requirements("torch", exclude=("torch", "torchvision")) + extra_requirements(build.onnx_extra)
    return [
        Step("torch", f"PyTorch ({build.label})", base + [*build.packages, "--index-url", build.url]),
        Step("libraries", "Model libraries", base + libraries),
    ]


def run_step(step: Step, on_line: Callable[[str], None]) -> None:
    """Runs one uv command, passing each output line on; raises with the tail of the output on failure."""
    log.info("setup step %s: %s", step.id, " ".join(step.command))
    process = subprocess.Popen(
        step.command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=NO_WINDOW,
        env={**os.environ, "NO_COLOR": "1"},
    )
    tail: list[str] = []
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.rstrip()
        if not line:
            continue
        log.info("uv: %s", line)
        tail = (tail + [line])[-8:]
        on_line(line)
    if process.wait() != 0:
        raise RuntimeError("\n".join(tail) or f"uv exited with {process.returncode}")


def record(build: TorchBuild) -> None:
    path = home() / "app" / "gpu-setup.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        version = None
    path.write_text(json.dumps({"build": build.tag, "index": build.url, "torch": version, "at": time.time()}), encoding="utf-8")


def default_log_path() -> Path:
    return home() / "logs" / "desktop.log"
