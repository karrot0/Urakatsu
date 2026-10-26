"""Runtime packages: newest-release selection, checksum enforcement, offline use, GGUF metadata."""

import hashlib
import io
import json
import os
import struct
import zipfile

import httpx
import pytest

from urakatsu.runtimes.device import Backend, Device, DeviceType
from urakatsu.runtimes.gguf_metadata import read_metadata, special_token_text
from urakatsu.runtimes.packages import BUN, LLAMA_CPP, SUWAYOMI, PackageManager, preferred_variants

# The server binary as the system names it (find_executable looks for "<name>.exe" on Windows).
EXE = "llama-server.exe" if os.name == "nt" else "llama-server"


def build_zip(name: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr(name, b"binary")
    return buffer.getvalue()


def fake_github(releases: list[dict], blobs: dict[str, bytes]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/releases"):
            return httpx.Response(200, json=releases)
        name = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, content=blobs[name]) if name in blobs else httpx.Response(404)

    return httpx.MockTransport(handler)


def release(tag: str, published: str, assets: dict[str, bytes], tamper: bool = False) -> dict:
    return {
        "tag_name": tag,
        "published_at": published,
        "draft": False,
        "assets": [
            {
                "name": name,
                "browser_download_url": f"https://example.invalid/{tag}/{name}",
                "size": len(data),
                "digest": "sha256:" + hashlib.sha256(b"tampered" if tamper else data).hexdigest(),
            }
            for name, data in assets.items()
        ],
    }


CUDA = [Device.gpu(Backend.CUDA, 0, "RTX", driver_version="610.74"), Device.cpu()]


def test_variants_follow_the_driver():
    assert preferred_variants(CUDA)[:2] == ["win-cuda-13", "win-cuda-12"] or preferred_variants(CUDA)[0].startswith(("linux", "macos"))
    old = [Device.gpu(Backend.CUDA, 0, "RTX", driver_version="550.10")]
    assert not any(v.endswith("cuda-13") for v in preferred_variants(old))
    # macOS has one build per processor (Metal on Apple silicon, falling back to the CPU).
    assert preferred_variants([Device.cpu()])[-1].endswith(("cpu", "macos-arm64", "macos-x64"))


@pytest.mark.parametrize("package", [BUN, SUWAYOMI], ids=lambda package: package.id)
def test_on_demand_runtimes_install_on_both_mac_processors(package, monkeypatch):
    """Bun (Mana) and Suwayomi Server publish Apple silicon and Intel builds; both are picked up."""
    import platform

    assets, picked = {
        "bun": (
            ["bun-darwin-aarch64.zip", "bun-darwin-x64.zip", "bun-darwin-x64-baseline.zip", "bun-linux-x64-baseline.zip"],
            {"arm64": "bun-darwin-aarch64.zip", "x86_64": "bun-darwin-x64-baseline.zip"},  # baseline runs on any x64 CPU
        ),
        "suwayomi-server": (
            ["Suwayomi-Server-v2.4.2366-macOS-arm64.tar.gz", "Suwayomi-Server-v2.4.2366-macOS-x64.tar.gz"],
            {"arm64": "Suwayomi-Server-v2.4.2366-macOS-arm64.tar.gz", "x86_64": "Suwayomi-Server-v2.4.2366-macOS-x64.tar.gz"},
        ),
    }[package.id]
    manager = PackageManager(offline=False, transport=fake_github([release("v1", "2026-10-01T00:00:00Z", {name: b"x" for name in assets})], {}))
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    for machine, name in picked.items():
        monkeypatch.setattr(platform, "machine", lambda: machine)
        assert [asset.name for asset in manager.latest(package, preferred_variants([Device.cpu()])).assets] == [name]


def test_installs_the_newest_release_with_the_best_build(tmp_path, monkeypatch):
    monkeypatch.setattr("urakatsu.runtimes.packages.preferred_variants", lambda devices: ["win-cuda-13", "win-cpu"])
    server = build_zip(EXE)
    old = release("b100", "2026-10-01T00:00:00Z", {"llama-b100-bin-win-cuda-13.4-x64.zip": server, "cudart-llama-bin-win-cuda-13.4-x64.zip": build_zip("cudart.dll")})
    # The newest release only has a CPU build (its CUDA files are not uploaded yet)...
    newest = release("b102", "2026-10-03T00:00:00Z", {"llama-b102-bin-win-cpu-x64.zip": server})
    # ...so the newest release with the full CUDA build wins.
    newer = release("b101", "2026-10-02T00:00:00Z", {"llama-b101-bin-win-cuda-13.4-x64.zip": server, "cudart-llama-bin-win-cuda-13.4-x64.zip": build_zip("cudart.dll")})
    releases = [newest, old, newer]
    blobs = {a["name"]: build_zip(EXE) if a["name"].startswith("llama-b") else build_zip("cudart.dll") for r in releases for a in r["assets"]}
    for r in releases:
        for a in r["assets"]:
            a["digest"] = "sha256:" + hashlib.sha256(blobs[a["name"]]).hexdigest()
    manager = PackageManager(root=tmp_path, offline=False, transport=fake_github(releases, blobs))
    installed = manager.ensure(LLAMA_CPP, CUDA)
    assert (installed.tag, installed.variant) == ("b101", "win-cuda-13")  # best build, from its newest release
    assert installed.executable.name == EXE

    newest_cuda = release("b103", "2026-10-04T00:00:00Z", {"llama-b103-bin-win-cuda-13.4-x64.zip": server, "cudart-llama-bin-win-cuda-13.4-x64.zip": build_zip("cudart.dll")})
    for a in newest_cuda["assets"]:
        blobs[a["name"]] = build_zip(EXE) if a["name"].startswith("llama-b") else build_zip("cudart.dll")
        a["digest"] = "sha256:" + hashlib.sha256(blobs[a["name"]]).hexdigest()
    releases.append(newest_cuda)
    installed = manager.ensure(LLAMA_CPP, CUDA, force=True)
    assert (installed.tag, installed.variant) == ("b103", "win-cuda-13")
    assert (installed.path / "cudart.dll").exists()


def test_a_checksum_mismatch_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("urakatsu.runtimes.packages.preferred_variants", lambda devices: ["win-cpu"])
    data = build_zip(EXE)
    bad = release("b200", "2026-10-05T00:00:00Z", {"llama-b200-bin-win-cpu-x64.zip": data}, tamper=True)
    manager = PackageManager(root=tmp_path, offline=False, transport=fake_github([bad], {"llama-b200-bin-win-cpu-x64.zip": data}))
    with pytest.raises(RuntimeError, match="SHA-256"):
        manager.ensure(LLAMA_CPP, CUDA)
    assert manager.installed(LLAMA_CPP) is None


def test_offline_uses_the_installed_build_and_never_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr("urakatsu.runtimes.packages.preferred_variants", lambda devices: ["win-cpu"])
    data = build_zip(EXE)
    good = release("b300", "2026-10-06T00:00:00Z", {"llama-b300-bin-win-cpu-x64.zip": data})
    online = PackageManager(root=tmp_path, offline=False, transport=fake_github([good], {"llama-b300-bin-win-cpu-x64.zip": data}))
    online.ensure(LLAMA_CPP, CUDA)

    def refuse(request):
        raise AssertionError("offline manager touched the network")

    offline = PackageManager(root=tmp_path, offline=True, transport=httpx.MockTransport(refuse))
    assert offline.ensure(LLAMA_CPP, CUDA, force=True).tag == "b300"
    with pytest.raises(RuntimeError, match="offline"):
        PackageManager(root=tmp_path / "empty", offline=True).ensure(LLAMA_CPP, CUDA)


def gguf_bytes(kv: dict) -> bytes:
    def string(text: str) -> bytes:
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kv))
    for key, value in kv.items():
        out += string(key)
        if isinstance(value, str):
            out += struct.pack("<I", 8) + string(value)
        elif isinstance(value, bool):
            out += struct.pack("<I", 7) + struct.pack("<?", value)
        elif isinstance(value, int):
            out += struct.pack("<I", 4) + struct.pack("<I", value)
        else:
            out += struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(value)) + b"".join(string(v) for v in value)
    return out


def test_gguf_metadata_reads_the_template_and_special_tokens(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(
        gguf_bytes(
            {
                "tokenizer.chat_template": "{{ bos_token }}{{ messages[0].content }}",
                "tokenizer.ggml.add_bos_token": True,
                "tokenizer.ggml.tokens": ["<pad>", "<bos>", "<eos>", "a"],
                "tokenizer.ggml.bos_token_id": 1,
                "tokenizer.ggml.eos_token_id": 2,
            }
        )
    )
    meta = read_metadata(path, {"tokenizer.chat_template", "tokenizer.ggml.add_bos_token"})
    assert meta == {"tokenizer.chat_template": "{{ bos_token }}{{ messages[0].content }}", "tokenizer.ggml.add_bos_token": True}
    assert special_token_text(path, "bos") == "<bos>" and special_token_text(path, "eos") == "<eos>"
    (tmp_path / "bad.gguf").write_bytes(b"NOPE")
    with pytest.raises(ValueError):
        read_metadata(tmp_path / "bad.gguf")


def test_a_cpu_build_on_a_gpu_machine_is_rechecked_at_once(tmp_path, monkeypatch):
    # A GPU missed at startup (nvidia-smi timing out) must not leave a CPU build for six hours.
    cpu = release("b400", "2026-10-07T00:00:00Z", {"llama-b400-bin-win-cpu-x64.zip": build_zip(EXE)})
    cuda_assets = {"llama-b400-bin-win-cuda-13.4-x64.zip": build_zip(EXE), "cudart-llama-bin-win-cuda-13.4-x64.zip": build_zip("cudart.dll")}
    both = release("b400", "2026-10-07T00:00:00Z", {**{a["name"]: build_zip(EXE) for a in cpu["assets"]}, **cuda_assets})
    blobs = {a["name"]: build_zip(EXE) if a["name"].startswith("llama-b") else build_zip("cudart.dll") for a in both["assets"]}
    for a in both["assets"]:
        a["digest"] = "sha256:" + hashlib.sha256(blobs[a["name"]]).hexdigest()
    manager = PackageManager(root=tmp_path, offline=False, transport=fake_github([both], blobs))
    monkeypatch.setattr("urakatsu.runtimes.packages.preferred_variants", lambda devices: ["win-cpu"])
    assert manager.ensure(LLAMA_CPP, [Device.cpu()]).variant == "win-cpu"
    monkeypatch.setattr("urakatsu.runtimes.packages.preferred_variants", lambda devices: ["win-cuda-13", "win-cpu"])
    assert manager.ensure(LLAMA_CPP, CUDA).variant == "win-cuda-13"  # not force, checked seconds ago


def build_tar(files: dict[str, bytes]) -> bytes:
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def github_repos(repos: dict[str, list[dict]], blobs: dict[str, bytes]) -> httpx.MockTransport:
    """GitHub with several repositories' releases: /repos/<owner>/<repo>/releases[/tags/<tag>]."""

    def handler(request: httpx.Request) -> httpx.Response:
        parts = request.url.path.strip("/").split("/")
        if parts[0] == "repos":
            releases = repos.get(f"{parts[1]}/{parts[2]}")
            if releases is None:
                return httpx.Response(404)
            if parts[3:] == ["releases"]:
                return httpx.Response(200, json=releases)
            tag = parts[5]
            found = next((r for r in releases if r["tag_name"] == tag), None)
            return httpx.Response(200, json=found) if found else httpx.Response(404)
        name = parts[-1]
        return httpx.Response(200, content=blobs[name]) if name in blobs else httpx.Response(404)

    return httpx.MockTransport(handler)


def sd_repos(build_repo: bool = True):
    """Upstream's releases (Vulkan only on Linux) and the CUDA build repository's, which lags a release."""
    server = "sd-server.exe" if os.name == "nt" else "sd-server"
    blobs: dict[str, bytes] = {}

    def make(tag: str, published: str, names: list[str]) -> dict:
        assets = {}
        for name in names:
            blobs[name] = build_tar({server: b"binary"}) if name.startswith("sd-") else build_tar({"libcublas.so.13": b"lib"})
            assets[name] = blobs[name]
        return release(tag, published, assets)

    upstream = [
        make("master-12-bbb", "2026-10-25T12:00:00Z", ["sd-master-bbb-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip"]),
        make("master-11-aaa", "2026-10-25T06:00:00Z", ["sd-master-aaa-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip"]),
    ]
    for r in upstream:  # zips, as upstream ships them
        for a in r["assets"]:
            blobs[a["name"]] = build_zip(server)
            a["digest"] = "sha256:" + hashlib.sha256(blobs[a["name"]]).hexdigest()
    repos = {"leejet/stable-diffusion.cpp": upstream}
    if build_repo:
        repos["karrot0/urakatsu-diffusion"] = [
            make("master-11-aaa", "2026-10-25T08:00:00Z", ["sd-master-11-aaa-bin-linux-x64-cuda-13.tar.gz", "cudart-sd-master-11-aaa-bin-linux-x64-cuda-13.tar.gz"]),
        ]
    return github_repos(repos, blobs)


LINUX_CUDA = ["linux-cuda-13", "linux-cuda-12", "linux-vulkan", "linux-cpu"]


def test_linux_cuda_comes_from_the_build_repository(tmp_path, monkeypatch):
    from urakatsu.runtimes.packages import STABLE_DIFFUSION_CPP

    monkeypatch.setattr("urakatsu.runtimes.packages.preferred_variants", lambda devices: LINUX_CUDA)
    manager = PackageManager(root=tmp_path, offline=False, transport=sd_repos())
    installed = manager.ensure(STABLE_DIFFUSION_CPP, CUDA)
    # The newest release built with CUDA, over a newer one that only has Vulkan.
    assert (installed.tag, installed.variant) == ("master-11-aaa", "linux-cuda-13")
    assert (installed.path / "libcublas.so.13").exists()  # its CUDA libraries, next to sd-server
    assert installed.executable.name.startswith("sd-server")

    listed = {r["tag"]: r["variant"] for r in manager.releases(STABLE_DIFFUSION_CPP, LINUX_CUDA)}
    assert listed == {"master-12-bbb": "linux-vulkan", "master-11-aaa": "linux-cuda-13"}
    # A pinned release takes the CUDA build where there is one, else upstream's.
    assert manager.release(STABLE_DIFFUSION_CPP, "master-11-aaa", LINUX_CUDA).variant == "linux-cuda-13"
    assert manager.release(STABLE_DIFFUSION_CPP, "master-12-bbb", LINUX_CUDA).variant == "linux-vulkan"


def test_without_the_build_repository_linux_keeps_upstreams_builds(tmp_path, monkeypatch):
    from urakatsu.runtimes.packages import STABLE_DIFFUSION_CPP

    monkeypatch.setattr("urakatsu.runtimes.packages.preferred_variants", lambda devices: LINUX_CUDA)
    manager = PackageManager(root=tmp_path, offline=False, transport=sd_repos(build_repo=False))
    installed = manager.ensure(STABLE_DIFFUSION_CPP, CUDA)
    assert (installed.tag, installed.variant) == ("master-12-bbb", "linux-vulkan")
