"""Model catalog: GGUF headers, hardware fit, runtime support, installed models, runtime pins."""

import hashlib
import json
import time

import httpx
import pytest

from tests.test_runtime_packages import EXE, build_zip, gguf_bytes, release
from urakatsu.catalog import fit as fitting
from urakatsu.catalog.facts import ModelFacts
from urakatsu.catalog.hf import _NeedMore, parse_gguf_header
from urakatsu.catalog.installed import InstalledFile, InstalledModel, InstalledModels
from urakatsu.catalog.support import match_family, parse_architectures, parse_families
from urakatsu.runtimes.device import Backend, Device
from urakatsu.runtimes.hardware import SystemProfile
from urakatsu.runtimes.packages import LLAMA_CPP, PackageManager
from urakatsu.runtimes.store import ModelStore

GB = 1 << 30
GPU12 = [Device.gpu(Backend.CUDA, 0, "RTX", memory_total=12 * GB, driver_version="610.74"), Device.cpu()]
RAM32 = SystemProfile(ram_total=32 * GB, ram_available=16 * GB, cpu_threads=6, cpu_features=["avx", "avx2"])


def test_gguf_header_reads_wanted_keys_and_asks_for_more_bytes():
    data = gguf_bytes({"general.architecture": "gemma4", "gemma4.block_count": 48, "gemma4.context_length": 131072, "tokenizer.chat_template": "{{ x }}"})
    assert parse_gguf_header(data, {"general.architecture", "gemma4.*"}) == {"general.architecture": "gemma4", "gemma4.block_count": 48, "gemma4.context_length": 131072}
    with pytest.raises(_NeedMore):
        parse_gguf_header(data[:40], {"tokenizer.chat_template"})
    assert parse_gguf_header(data[:40], {"tokenizer.chat_template"}, partial=True) == {}


def test_support_comes_from_the_runtime_sources():
    arch = 'static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {\n    { LLM_ARCH_LLAMA, "llama" },\n    { LLM_ARCH_GEMMA4, "gemma4" },\n};'
    assert parse_architectures(arch) == {"llama", "gemma4"}
    readme = "## Supported models\n- [FLUX.2-dev / FLUX.2-klein](./docs/flux2.md)\n- [SDXL](./docs/sdxl.md)\n- [Image edit](./docs/edit.md)\n## Supported backends\n- [Nope](./docs/nope.md)"
    families = parse_families(readme)
    assert [f.doc for f in families] == ["flux2", "sdxl"]
    assert match_family(families, ["black-forest-labs/FLUX.2-klein-4B"]).doc == "flux2"
    assert match_family(families, ["stabilityai/stable-diffusion-xl-base-1.0"]).doc == "sdxl"
    assert match_family(families, ["someone/whisper-large"]) is None


def test_fit_levels_and_the_recommended_quantization():
    assert fitting.fit_llm(6 * GB, GB // 2, GPU12, RAM32).level == "gpu"
    partial = fitting.fit_llm(14 * GB, GB // 2, GPU12, RAM32)
    assert partial.level == "partial" and 0.5 < partial.gpu_share < 1
    assert fitting.fit_llm(20 * GB, 0, [Device.cpu()], RAM32).level == "too_large"
    assert fitting.fit_llm(8 * GB, 0, [Device.cpu()], RAM32).level == "cpu"
    assert fitting.fit_diffusion([9 * GB, 4 * GB], GPU12, RAM32).level == "offload"

    files = [
        ("Model-Q4_K_M.gguf", 5 * GB),
        ("Model-Q8_0.gguf", 9 * GB),
        ("Model-BF16.gguf", 10 * GB),
        ("BF16/Model-BF16-00001-of-00002.gguf", 8 * GB),
        ("BF16/Model-BF16-00002-of-00002.gguf", 8 * GB),
        ("mmproj-F16.gguf", GB),
        ("mtp/Model-MTP-Q8_0.gguf", GB),
    ]
    groups = fitting.group_quantizations(files)
    assert set(groups) == {"Q4_K_M", "Q8_0", "BF16"}  # projector and MTP draft are not quantizations
    assert fitting.first_shard(groups["BF16"]) in ("Model-BF16.gguf", "BF16/Model-BF16-00001-of-00002.gguf")
    sizes = {"Q4_K_M": 5 * GB, "Q8_0": 9 * GB, "BF16": 10 * GB}
    fits = {q: fitting.fit_llm(s, 0, GPU12, RAM32) for q, s in sizes.items()}
    assert fits["BF16"].level == "gpu"
    assert fitting.best_quantization(fits, sizes) == "Q8_0"  # unquantized only when nothing else fits as well


def write_llama_install(root, tag):
    folder = root / "llama.cpp" / f"{tag}-win-cpu"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / EXE).write_bytes(b"binary")
    (root / "llama.cpp" / "current.json").write_text(
        json.dumps({"tag": tag, "variant": "win-cpu", "path": folder.name, "installed_at": 0, "checked_at": time.time()}), encoding="utf-8"
    )


def write_support(root, tag, architectures):
    path = root / "llama.cpp" / "support" / f"{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"architectures": architectures, "families": []}), encoding="utf-8")


@pytest.fixture
def offline_home(tmp_path, monkeypatch):
    monkeypatch.setenv("URAKATSU_HOME", str(tmp_path))
    monkeypatch.setenv("URAKATSU_OFFLINE", "1")
    return tmp_path


def test_an_older_runtime_warns_instead_of_failing(offline_home):
    from urakatsu.catalog.service import Catalog

    runtimes = offline_home / "runtimes"
    write_llama_install(runtimes, "b100")
    write_support(runtimes, "b100", ["llama", "gemma3"])
    write_support(runtimes, "b200", ["llama", "gemma3", "gemma4"])
    catalog = Catalog()
    catalog._latest[LLAMA_CPP.id] = ("b200", time.time())
    gemma4 = ModelFacts(runtime="llama", repo="x/gemma-4-GGUF", architecture="gemma4", chat_template=True)
    gemma3 = ModelFacts(runtime="llama", repo="x/gemma-3-GGUF", architecture="gemma3", chat_template=True)
    novel = ModelFacts(runtime="llama", repo="x/new-GGUF", architecture="brandnew", chat_template=True)

    assert catalog.compatibility(gemma3)["status"] == "supported"
    needs = catalog.compatibility(gemma4)
    assert needs["status"] == "needs_update" and "b200" in needs["message"] and "Update it" in needs["message"]
    assert catalog.compatibility(novel)["status"] == "unsupported"

    catalog.packages.set_pin(LLAMA_CPP, "b100")
    assert "Unpin" in catalog.compatibility(gemma4)["message"]
    assert [u["provider"] for u in catalog.usable_by(gemma4)] == ["local"]
    pickle_only = ModelFacts(runtime="torch", repo="x/ocr", model_type="vision-encoder-decoder", files=["pytorch_model.bin", "vocab.txt", "preprocessor_config.json"])
    assert catalog.usable_by(pickle_only) == []
    assert catalog.compatibility(pickle_only)["status"] == "unsupported"


def test_installed_models_swap_into_providers_and_remove_cleanly(offline_home):
    from urakatsu.core.registry import registry

    store = ModelStore()
    models = InstalledModels(store)
    common = dict(repo="someone/flux2-klein-4b-gguf", revision="abc", runtime="diffusion", family="flux2")
    q4 = InstalledModel(id="someone/flux2-klein-4b-gguf:Q4_0", variant="Q4_0", files=[InstalledFile(path="klein-Q4_0.gguf", role="diffusion_model", size=4)], **common)
    q8 = InstalledModel(id="someone/flux2-klein-4b-gguf:Q8_0", variant="Q8_0", files=[InstalledFile(path="klein-Q8_0.gguf", role="diffusion_model", size=8)], **common)
    for model in (q4, q8):
        path = model.source(model.files[0].path).local_path(store)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        models.add(model)
    assert models.complete(q4) and len(models.all()) == 2

    cls = registry.get("inpainting", "flux2-klein")
    provider = cls(settings={"model": q4.id}, runtime="diffusion")
    defaults = cls.weight_sources("diffusion")
    chosen = provider.selected_sources(defaults)
    assert chosen[0].filename == "klein-Q4_0.gguf" and chosen[0].repository == q4.repo
    assert chosen[1:] == defaults[1:]  # text encoder and VAE stay the defaults

    models.remove(q4.id)
    assert not q4.source("klein-Q4_0.gguf").local_path(store).exists()
    assert q8.source("klein-Q8_0.gguf").local_path(store).exists()
    from urakatsu.core.errors import ProviderUnavailable

    with pytest.raises(ProviderUnavailable, match="removed"):
        provider.selected_sources(defaults)


def test_a_pinned_runtime_stays_put_and_unpinning_reuses_the_kept_build(tmp_path, monkeypatch):
    monkeypatch.setattr("urakatsu.runtimes.packages.preferred_variants", lambda devices: ["win-cpu"])
    releases, blobs = [], {}
    for tag, day in (("b100", 1), ("b101", 2)):
        name = f"llama-{tag}-bin-win-cpu-x64.zip"
        blobs[name] = build_zip(EXE)
        releases.append(release(tag, f"2026-10-0{day}T00:00:00Z", {name: blobs[name]}))
    downloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/releases"):
            return httpx.Response(200, json=releases)
        if "/releases/tags/" in path:
            tag = path.rsplit("/", 1)[-1]
            found = next((r for r in releases if r["tag_name"] == tag), None)
            return httpx.Response(200, json=found) if found else httpx.Response(404)
        name = path.rsplit("/", 1)[-1]
        downloads.append(name)
        return httpx.Response(200, content=blobs[name])

    manager = PackageManager(root=tmp_path, offline=False, transport=httpx.MockTransport(handler))
    assert manager.ensure(LLAMA_CPP, GPU12).tag == "b101"
    write_support(tmp_path, "b101", ["llama"])  # a cache folder beside the installs, newer than them
    manager.set_pin(LLAMA_CPP, "b100")
    assert manager.ensure(LLAMA_CPP, GPU12, force=True).tag == "b100"  # rolled back
    assert manager.ensure(LLAMA_CPP, GPU12, force=True).tag == "b100"  # and never auto-updated
    assert [r["tag"] for r in manager.releases(LLAMA_CPP, ["win-cpu"])] == ["b101", "b100"]
    assert {t for t, _, _ in manager.local_installs(LLAMA_CPP)} == {"b100", "b101"}  # the previous build is kept
    assert (tmp_path / "llama.cpp" / "support" / "b101.json").exists()

    manager.set_pin(LLAMA_CPP, None)
    before = len(downloads)
    assert manager.install_tag(LLAMA_CPP, "b101", GPU12).tag == "b101"
    assert len(downloads) == before  # the kept build was reused, nothing downloaded
    assert hashlib.sha256(blobs["llama-b101-bin-win-cpu-x64.zip"]).hexdigest()


def test_gguf_header_stops_at_the_tokenizer():
    """Only general.* and architecture keys are wanted; the tokenizer (most of a real header) is never read."""
    data = gguf_bytes({"general.architecture": "qwen3", "qwen3.block_count": 36, "tokenizer.ggml.tokens": ["a"] * 1000, "qwen3.late": 1})
    start = data.index(b"tokenizer.ggml.tokens")
    keys = {"general.architecture", "qwen3.block_count", "qwen3.context_length"}  # context_length is absent
    assert parse_gguf_header(data[: start + 30], keys) == {"general.architecture": "qwen3", "qwen3.block_count": 36}
    assert parse_gguf_header(data, {"tokenizer.ggml.tokens"})["tokenizer.ggml.tokens"][:2] == ["a", "a"]  # asked for: read


def test_hub_answers_survive_a_restart_and_refresh_in_the_background(tmp_path, monkeypatch):
    from urakatsu.catalog import hf
    from urakatsu.catalog.diskcache import DiskCache

    served = []

    def handler(request):
        served.append(str(request.url))
        return httpx.Response(200, json=[{"id": "a/b", "downloads": len(served)}])

    first = hf.HubClient(transport=httpx.MockTransport(handler), cache_dir=tmp_path)
    assert first.search("llama")[0].downloads == 1
    first._refresher.shutdown(wait=True)  # the answer is saved off the request's time
    restarted = hf.HubClient(transport=httpx.MockTransport(handler), cache_dir=tmp_path)
    assert restarted.search("llama")[0].downloads == 1 and len(served) == 1  # from disk, no request
    later = time.time() + hf.CACHE_SECONDS + 5
    monkeypatch.setattr(time, "time", lambda: later)  # the saved answer is now stale
    again = hf.HubClient(transport=httpx.MockTransport(handler), cache_dir=tmp_path)
    assert again.search("llama")[0].downloads == 1  # shown at once...
    again._refresher.shutdown(wait=True)
    assert len(served) == 2  # ...and refreshed behind it
    assert DiskCache(tmp_path).get("missing") is None
