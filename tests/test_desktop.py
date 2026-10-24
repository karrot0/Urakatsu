"""Desktop app: choosing the PyTorch build for the driver, and the setup plan."""

import sys

import httpx

from urakatsu.desktop import gpu_setup


def test_the_driver_cuda_version_is_read_from_either_header():
    assert gpu_setup.CUDA_HEADER.search("| NVIDIA-SMI 610.74   KMD Version: 610.74   CUDA UMD Version: 13.3 |").groups() == ("13", "3")
    assert gpu_setup.CUDA_HEADER.search("| NVIDIA-SMI 560.94   Driver Version: 560.94   CUDA Version: 12.6 |").groups() == ("12", "6")


def test_candidates_count_down_from_the_driver():
    tags = list(gpu_setup.candidate_tags((13, 3)))
    assert tags[:4] == ["cu133", "cu132", "cu131", "cu130"]
    assert tags[4] == "cu129" and tags[-1] == "cu120"


def test_the_newest_index_with_a_wheel_for_this_python_wins():
    suffix = gpu_setup.wheel_suffix()
    pages = {
        "/whl/cu133/torch/": None,  # the driver is newer than any PyTorch build
        "/whl/cu132/torch/": f'<a href="torch-2.14.1%2Bcu132{suffix.replace("+", "%2B")}">torch</a>',
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = pages.get(request.url.path)
        return httpx.Response(200, text=body) if body else httpx.Response(404)

    build = gpu_setup.choose_build((13, 3), transport=httpx.MockTransport(handler))
    assert build.tag == "cu132" and build.url.endswith("/whl/cu132") and build.label == "CUDA 13.2"
    assert gpu_setup.choose_build(None).tag == "cpu"


def test_an_index_without_this_pythons_wheel_is_skipped():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/whl/cu132/torch/":
            return httpx.Response(200, text='<a href="torch-2.14.1+cu132-cp39-cp39-win_amd64.whl">old</a>')
        return httpx.Response(404)

    assert gpu_setup.choose_build((13, 2), transport=httpx.MockTransport(handler)).tag == "cpu"


def test_the_plan_installs_torch_from_its_index_then_the_libraries():
    build = gpu_setup.TorchBuild("cu132", "https://download.pytorch.org/whl/cu132")
    torch_step, libraries = gpu_setup.plan(build, uv="uv")
    assert torch_step.command[:5] == ["uv", "pip", "install", "--python", sys.executable]
    assert torch_step.command[5:7] == ["torch", "torchvision"] and torch_step.command[-2:] == ["--index-url", build.url]
    names = " ".join(libraries.command)
    assert "transformers" in names and "onnxruntime-gpu" in names
    assert " torch>=" not in names  # already installed from the CUDA index
    cpu = gpu_setup.plan(gpu_setup.TorchBuild("cpu", "https://download.pytorch.org/whl/cpu"), uv="uv")[1]
    assert any(arg.startswith("onnxruntime>=") for arg in cpu.command)
