"""stable-diffusion.cpp recipes: matching repos, finding companions, the generic inpainter."""

import numpy as np
import pytest
from PIL import Image

from urakatsu.catalog import fit as fitting
from urakatsu.catalog.hf import RepoFile, RepoInfo
from urakatsu.catalog.recipes import match_recipe, parse_recipe, parse_sources, pick, slot_candidates
from urakatsu.catalog.support import Family, RuntimeSupport
from urakatsu.providers.inpainting.diffusion_recipe import DiffusionRecipeSettings, RecipeProcessor, flag_for, sample_params, working_size
from urakatsu.runtimes.device import Backend, Device
from urakatsu.runtimes.hardware import SystemProfile

GB = 1 << 30

QWEN_21 = parse_recipe(
    {
        "schema_version": 1,
        "id": "qwen-image-21",
        "name": "Qwen Image 2.1",
        "model_url": "https://huggingface.co/Qwen/Qwen-Image-2.1",
        "slots": [
            {"id": "diffusion_model", "required": True},
            {"id": "llm", "required": True, "label": "Qwen 3 VL 8B Text Encoder"},
            {"id": "llm_vision", "required": False},
            {"id": "vae", "required": False},
            {"id": "tae", "required": False},
        ],
        "required_any": [["vae", "tae"]],
        "defaults": {"sampling_method": "euler", "cfg_scale": 1, "steps": 20, "width": 2048, "height": 2048},
        "features": ["flow_shift", "ref_images"],
    }
)
MOSS_BASE = parse_recipe({"id": "moss-image-base", "name": "MOSS Image Base", "model_url": "https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base", "slots": [{"id": "diffusion_model", "required": True}, {"id": "llm", "required": True}, {"id": "tokenizer", "required": True}, {"id": "vae", "required": False}], "required_any": [["vae", "tae"]]})
MOSS_SFT = parse_recipe({"id": "moss-image-sft", "name": "MOSS Image SFT", "model_url": "https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT", "slots": [{"id": "diffusion_model", "required": True}]})
WAN = parse_recipe({"id": "wan21-t2v", "name": "Wan 2.1 T2V", "slots": [{"id": "diffusion_model", "required": True}, {"id": "t5xxl", "required": True}], "disabled_features": ["img2img", "mask_image"]})
CONTROL = parse_recipe({"id": "sdxl-controlnet", "name": "SDXL ControlNet", "slots": [{"id": "model", "required": True}, {"id": "control_net", "required": True}]})
RECIPES = (QWEN_21, MOSS_BASE, MOSS_SFT, WAN, CONTROL)

QWEN_DOC = """# How to Use

## Download weights

- Download Qwen Image 2.1
    - safetensors: https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/diffusion_models
    - gguf: https://huggingface.co/leejet/Qwen-Image-2.1-GGUF/tree/main
- Download vae
    - safetensors: https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/vae
- Download Qwen3-VL-8B-Instruct
    - gguf: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/tree/main

## Examples
"""


def test_recipes_say_what_a_model_needs_and_whether_it_can_clean_a_page():
    assert [s.id for s in QWEN_21.needed()] == ["diffusion_model", "llm", "vae"]  # the full VAE of vae/tae
    assert QWEN_21.edits and QWEN_21.cleans_pages() and QWEN_21.default("steps") == 20
    assert not WAN.cleans_pages()  # video: no masked image-to-image
    assert not CONTROL.cleans_pages()  # needs a control image Urakatsu cannot supply
    assert CONTROL.main_slot == "model" and QWEN_21.main_slot == "diffusion_model"


def test_repos_match_their_recipe():
    assert match_recipe(RECIPES, "abenzerps/Qwen-Image-2.1-Uncensored-Quantized", ["Qwen/Qwen-Image-2.1"]) is QWEN_21
    # The repo itself wins over its base model (SFT is fine-tuned from Base).
    assert match_recipe(RECIPES, "OpenMOSS-Team/MOSS-Image-SFT", ["OpenMOSS-Team/MOSS-Image-Base"]) is MOSS_SFT
    assert match_recipe(RECIPES, "leejet/Qwen-Image-2.1-GGUF", []) is QWEN_21  # by name
    assert match_recipe(RECIPES, "Comfy-Org/MOSS-Image", []) in (MOSS_BASE, MOSS_SFT)  # a repack of the family
    assert match_recipe(RECIPES, "someone/llama-3-8b", []) is None


def test_doc_download_bullets_become_companion_sources():
    sources = parse_sources(QWEN_DOC, QWEN_21)
    assert [(s.repo, s.folder) for s in sources["vae"]] == [("Comfy-Org/Qwen-Image-2.1", "vae")]
    assert [s.repo for s in sources["llm"]] == ["Qwen/Qwen3-VL-8B-Instruct-GGUF"]
    assert sources["diffusion_model"][0].folder == "diffusion_models"


KLEIN_4B = parse_recipe({"id": "flux2-klein-4b", "name": "FLUX.2 Klein 4B", "model_url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B", "slots": [{"id": "diffusion_model", "required": True}, {"id": "llm", "required": True}, {"id": "vae", "required": False}], "required_any": [["vae", "tae"]], "defaults": {"steps": 4, "cfg_scale": 1}, "features": ["ref_images"]})
KLEIN_9B = parse_recipe({"id": "flux2-klein-9b", "name": "FLUX.2 Klein 9B", "model_url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-9B", "slots": [{"id": "diffusion_model", "required": True}, {"id": "llm", "required": True}, {"id": "vae", "required": False}], "required_any": [["vae", "tae"]], "defaults": {"steps": 4, "cfg_scale": 1}, "features": ["ref_images"]})
FLUX2_DEV = parse_recipe({"id": "flux2-dev", "name": "FLUX.2 dev", "model_url": "https://huggingface.co/black-forest-labs/FLUX.2-dev", "slots": [{"id": "diffusion_model", "required": True}, {"id": "llm", "required": True}, {"id": "vae", "required": False}], "required_any": [["vae", "tae"]], "defaults": {"steps": 20, "cfg_scale": 1, "distilled_guidance": 4}, "features": ["ref_images"]})

# stable-diffusion.cpp's docs/flux2.md: one "Download weights" section per model.
FLUX2_DOC = """# How to Use

## Flux.2-dev

### Download weights

- Download FLUX.2-dev
    - gguf: https://huggingface.co/city96/FLUX.2-dev-gguf/tree/main
- Download vae
    - safetensors: https://huggingface.co/black-forest-labs/FLUX.2-dev/tree/main
- Download Mistral-Small-3.2-24B-Instruct-2506-GGUF
    - gguf: https://huggingface.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF/tree/main

### Examples

## Flux.2 klein 9B / Flux.2 klein base 9B

### Download weights

- Download FLUX.2-klein-9B
    - gguf: https://huggingface.co/leejet/FLUX.2-klein-9B-GGUF/tree/main
- Download FLUX.2-klein-base-9B
    - gguf: https://huggingface.co/leejet/FLUX.2-klein-base-9B-GGUF/tree/main
- Download vae
    - safetensors: https://huggingface.co/black-forest-labs/FLUX.2-dev/tree/main
- Download Qwen3 8B
    - gguf: https://huggingface.co/unsloth/Qwen3-8B-GGUF/tree/main

### Examples
"""


def test_a_doc_covering_several_models_gives_each_recipe_its_own_companions():
    nine = parse_sources(FLUX2_DOC, KLEIN_9B)
    assert [s.repo for s in nine["llm"]] == ["unsloth/Qwen3-8B-GGUF"]  # not dev's Mistral, nor the base model
    assert [s.repo for s in nine["diffusion_model"]] == ["leejet/FLUX.2-klein-9B-GGUF", "leejet/FLUX.2-klein-base-9B-GGUF"]
    assert [s.repo for s in parse_sources(FLUX2_DOC, FLUX2_DEV)["llm"]] == ["unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF"]
    assert [s.repo for s in parse_sources(FLUX2_DOC, KLEIN_4B)["llm"]] == ["unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF"]  # no section of its own: the first


def test_companions_are_picked_sensibly():
    comfy = [
        ("text_encoders/qwen3vl_8b_bf16.safetensors", 17 * GB),
        ("text_encoders/qwen3vl_8b_int8_convrot.safetensors", 9 * GB),
        ("text_encoders/qwen3vl_8b_w4a8_gptq.safetensors", 5 * GB),  # a format sd.cpp does not read
        ("text_encoders/qwen3vl_8b_fp4mixed.safetensors", 4 * GB),  # nor is this one (NVFP4)
        ("vae/flux2-vae.safetensors", 300 << 20),
        ("vae/flux2-vae_small.safetensors", 100 << 20),
        ("qwen-image-Q4_K_M.gguf", 4 * GB),
    ]
    assert pick("llm", slot_candidates("llm", "r", comfy)).path.endswith("int8_convrot.safetensors")
    assert pick("vae", slot_candidates("vae", "r", comfy)).path == "vae/flux2-vae.safetensors"
    dev = [("ae.safetensors", 320 << 20), ("flux2-dev.safetensors", 64 * GB), ("vae/diffusion_pytorch_model.safetensors", 320 << 20)]
    assert pick("vae", slot_candidates("vae", "b", dev, anywhere=True)).path == "ae.safetensors"  # not the model beside it
    gguf_repo = [("Qwen3VL-8B-Instruct-Q8_0.gguf", 8 * GB), ("Qwen3VL-8B-Instruct-Q4_K_M.gguf", 5 * GB), ("mmproj-Qwen3VL-8B-Instruct-F16.gguf", GB), ("mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf", GB // 2)]
    assert pick("llm", slot_candidates("llm", "q", gguf_repo, anywhere=True)).path == "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
    assert pick("llm_vision", slot_candidates("llm_vision", "q", gguf_repo)).path.startswith("mmproj-Qwen3VL-8B-Instruct-F16")


def test_diffusion_fit_streams_parts_and_keeps_the_diffusion_model_off_disk():
    gpu = [Device.gpu(Backend.CUDA, 0, "RTX", memory_total=12 * GB), Device.cpu()]
    system = SystemProfile(ram_total=32 * GB, ram_available=8 * GB)
    assert fitting.fit_diffusion([4 * GB, 9 * GB, GB], gpu, system).level == "offload"  # runs one part at a time
    assert fitting.fit_diffusion([3 * GB, 2 * GB, GB], gpu, system).level == "gpu"
    assert fitting.fit_diffusion([14 * GB, 9 * GB], gpu, system).level == "offload"  # streamed in segments
    # Qwen Image Q4_0 with its 8.7 GB text encoder on an 8 GB GPU and 16 GB of RAM.
    small = [Device.gpu(Backend.CUDA, 0, "RTX 3060 Ti", memory_total=8 * GB), Device.cpu()]
    laptop = SystemProfile(ram_total=16 * GB, ram_available=4 * GB)
    assert fitting.fit_diffusion([int(3.9 * GB), int(8.7 * GB), int(0.6 * GB)], small, laptop).level == "offload"
    big = fitting.fit_diffusion([12 * GB, 16 * GB], small, laptop)  # the text encoder spills to disk
    assert big.level == "offload" and len(big.notes) == 2
    assert fitting.fit_diffusion([24 * GB, GB], small, laptop).level == "too_large"  # the model itself would stream from disk


# ---- the catalog with a fake Hub -----------------------------------------------------------------


def repo(repo_id: str, files: list[tuple[str, int]], base: list[str] = ()) -> RepoInfo:
    return RepoInfo(repo_id, f"sha-{repo_id}", repo_id.split("/")[0], "text-to-image", None, [], 0, 0, "", False, list(base), {}, {}, [RepoFile(p, s) for p, s in files])


class FakeHub:
    def __init__(self, *repos: RepoInfo):
        self.repos = {r.id: r for r in repos}

    def model(self, repo_id, revision=None):
        return self.repos[repo_id]


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("URAKATSU_HOME", str(tmp_path))
    monkeypatch.setenv("URAKATSU_OFFLINE", "1")
    from urakatsu.catalog.service import Catalog

    gguf_only = repo("leejet/Qwen-Image-2.1-GGUF", [("qwen_image_2.1-Q4_K.gguf", 4 * GB), ("qwen_image_2.1-Q8_0.gguf", 7 * GB)], ["Qwen/Qwen-Image-2.1"])
    comfy = repo("Comfy-Org/Qwen-Image-2.1", [("diffusion_models/qwen_image_2.1_bf16.safetensors", 13 * GB), ("vae/qwen_image_2.1_vae_bf16.safetensors", GB // 2), ("text_encoders/qwen3vl_8b_bf16.safetensors", 17 * GB)])
    encoders = repo("Qwen/Qwen3-VL-8B-Instruct-GGUF", [("Qwen3VL-8B-Instruct-Q4_K_M.gguf", 5 * GB), ("mmproj-Qwen3VL-8B-Instruct-F16.gguf", GB)])
    c = Catalog(hub=FakeHub(gguf_only, comfy, encoders))
    support = RuntimeSupport("stable-diffusion.cpp", "master-1", families=(Family("Qwen Image 2.1", "qwen_image_2.1", ("qwenimage21",)),), recipes=RECIPES)
    monkeypatch.setattr(c, "_latest_support", lambda runtime: support)
    monkeypatch.setattr(c, "latest_tag", lambda package: "master-1")
    monkeypatch.setattr(c.support, "doc", lambda package, tag, doc: QWEN_DOC)
    return c


def test_a_gguf_only_repo_gets_its_companions_from_the_doc(catalog):
    info = catalog.hub.model("leejet/Qwen-Image-2.1-GGUF")
    facts = catalog.facts("diffusion", info)
    assert facts.recipe is QWEN_21 and [u["provider"] for u in catalog.usable_by(facts)] == ["diffusion-model"]
    plan = catalog.components(info, facts)
    assert not plan.missing
    assert (plan.files["vae"].repo, plan.files["llm"].repo) == ("Comfy-Org/Qwen-Image-2.1", "Qwen/Qwen3-VL-8B-Instruct-GGUF")
    assert plan.files["llm_vision"].path.startswith("mmproj")  # GGUF text encoder + editing -> projector

    started = []
    monkeypatch_start = lambda model: started.append(model) or type("D", (), {"model_dump": lambda self, mode=None: {}})()  # noqa: E731
    catalog.downloads.start = monkeypatch_start
    catalog.install("diffusion", "leejet/Qwen-Image-2.1-GGUF", "Q4_K")
    model = started[0]
    roles = {f.role: f for f in model.files}
    assert model.recipe == "qwen-image-21" and set(roles) == {"diffusion_model", "llm", "llm_vision", "vae"}
    assert roles["diffusion_model"].repo is None  # from the repo itself
    assert roles["vae"].repo == "Comfy-Org/Qwen-Image-2.1" and roles["vae"].revision == "sha-Comfy-Org/Qwen-Image-2.1"
    assert model.source(roles["vae"].path).repository == "Comfy-Org/Qwen-Image-2.1"


def test_any_flux2_model_runs_on_flux2_klein_with_its_own_text_encoder(catalog, monkeypatch):
    nine = repo("unsloth/FLUX.2-klein-9B-GGUF", [("flux-2-klein-9b-Q4_K_M.gguf", 6 * GB), ("flux-2-klein-9b-Q8_0.gguf", 10 * GB)], ["black-forest-labs/FLUX.2-klein-9B"])
    qwen = repo("unsloth/Qwen3-8B-GGUF", [("Qwen3-8B-Q4_K_M.gguf", 5 * GB), ("Qwen3-8B-Q8_0.gguf", 9 * GB)])
    vae = repo("black-forest-labs/FLUX.2-dev", [("ae.safetensors", GB // 3), ("flux2-dev.safetensors", 64 * GB)])
    catalog.hub.repos.update({r.id: r for r in (nine, qwen, vae)})
    support = RuntimeSupport("stable-diffusion.cpp", "master-1", families=(Family("FLUX.2-dev/FLUX.2-klein", "flux2", ("flux2dev", "flux2klein")),), recipes=(KLEIN_4B, KLEIN_9B, FLUX2_DEV))
    monkeypatch.setattr(catalog, "_latest_support", lambda runtime: support)
    monkeypatch.setattr(catalog.support, "doc", lambda package, tag, doc: FLUX2_DOC)
    facts = catalog.facts("diffusion", nine)
    assert facts.recipe is KLEIN_9B and [u["provider"] for u in catalog.usable_by(facts)] == ["flux2-klein", "diffusion-model"]
    plan = catalog.components(nine, facts)
    assert plan.files["llm"].repo == "unsloth/Qwen3-8B-GGUF" and plan.files["llm"].path == "Qwen3-8B-Q4_K_M.gguf"


def test_flux2_klein_runs_its_defaults_or_an_installed_flux2_model(tmp_path, monkeypatch):
    from urakatsu.catalog.installed import InstalledFile, InstalledModel, InstalledModels
    from urakatsu.core.errors import ProviderUnavailable
    from urakatsu.providers.inpainting.flux2_klein import DEFAULT_RECIPE, Flux2KleinProvider
    from urakatsu.runtimes import models_folder
    from urakatsu.runtimes.store import ModelStore

    monkeypatch.setattr(models_folder, "_root", tmp_path / "models")  # resolved once per process

    defaults = Flux2KleinProvider(None, None, "diffusion")._components()
    assert defaults[1] == DEFAULT_RECIPE and [slot for slot, _ in defaults[2]] == ["diffusion_model", "llm", "vae"]
    assert [source for _, source in defaults[2]] == list(Flux2KleinProvider.weight_sources())

    models = InstalledModels(ModelStore())
    files = [InstalledFile(path="flux-2-klein-9b-Q4_K_M.gguf", role="diffusion_model"), InstalledFile(path="Qwen3-8B-Q4_K_M.gguf", role="llm", repo="unsloth/Qwen3-8B-GGUF", revision="r")]
    nine = InstalledModel(id="unsloth/FLUX.2-klein-9B-GGUF:Q4_K_M", repo="unsloth/FLUX.2-klein-9B-GGUF", revision="s", runtime="diffusion", family="flux2", recipe="flux2-klein-9b", files=files)
    qwen = InstalledModel(id="leejet/Qwen-Image-2.1-GGUF:Q4_K", repo="leejet/Qwen-Image-2.1-GGUF", revision="s", runtime="diffusion", family="qwen_image", recipe="qwen-image-21", files=files[:1])
    models.add(nine)
    models.add(qwen)
    name, recipe, components = Flux2KleinProvider({"model": nine.id}, None, "diffusion")._components()
    assert (name, recipe) == (nine.id, "flux2-klein-9b")
    assert [(slot, source.repository) for slot, source in components] == [("diffusion_model", nine.repo), ("llm", "unsloth/Qwen3-8B-GGUF")]
    with pytest.raises(ProviderUnavailable, match="not a FLUX.2 model"):
        Flux2KleinProvider({"model": qwen.id}, None, "diffusion")._components()


def test_flux2_klein_takes_steps_and_guidance_from_the_recipe_unless_set():
    from urakatsu.providers.inpainting.flux2_klein import Flux2KleinProvider

    dev = Flux2KleinProvider(None, None, "diffusion")._processor_for(FakeServer(), FLUX2_DEV)
    assert (dev.steps, dev.cfg_scale, dev.distilled_guidance) == (20, 1.0, 4.0)
    chosen = Flux2KleinProvider({"steps": 6, "cfg_scale": 2}, None, "diffusion")._processor_for(FakeServer(), FLUX2_DEV)
    assert (chosen.steps, chosen.cfg_scale) == (6, 2.0)
    koharu = Flux2KleinProvider(None, None, "diffusion")._processor_for(FakeServer())  # an attached server: no recipe
    assert (koharu.steps, koharu.cfg_scale, koharu.distilled_guidance) == (4, 1.0, 3.5)


def test_a_text_encoder_named_after_an_image_model_is_not_installed_as_one(catalog, monkeypatch):
    """AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF holds only Qwen3-VL GGUFs (a --llm for
    stable-diffusion.cpp); its name matched the Qwen Image recipe and it was installed as the
    diffusion model, which sd-server cannot load ("get sd version from file failed")."""
    images = catalog._latest_support("diffusion")
    llama = RuntimeSupport("llama.cpp", "b1", architectures=frozenset({"qwen3vl", "t5encoder"}))
    monkeypatch.setattr(catalog, "_latest_support", lambda runtime: llama if runtime == "llama" else images)
    encoder = repo("AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF", [("Qwen-Image-2.1-Turbo-Uncensored-AD-Q4_K.gguf", 5 * GB)], ["Qwen/Qwen3-VL-8B-Instruct"])
    encoder.gguf = {"architecture": "qwen3vl"}
    facts = catalog.facts("diffusion", encoder)
    assert facts.text_encoder and facts.recipe is None and catalog.usable_by(facts) == []
    report = catalog.compatibility(facts)
    assert report["status"] == "unsupported" and "text encoder" in report["message"]
    denoiser = catalog.hub.model("leejet/Qwen-Image-2.1-GGUF")
    denoiser.gguf = {"architecture": "qwen_image21"}  # diffusion GGUFs name their own architecture
    assert catalog.facts("diffusion", denoiser).recipe is QWEN_21


# ---- the generic inpainter -----------------------------------------------------------------------


class FakeServer:
    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return [Image.new("RGB", (request["width"], request["height"]), (255, 0, 0))]


def test_requests_follow_the_recipe_and_editing_models_get_a_reference():
    params = sample_params({"sampling_method": "euler", "steps": 20, "cfg_scale": 1, "flow_shift": 3}, DiffusionRecipeSettings(steps=8))
    assert params == {"sample_method": "euler", "sample_steps": 8, "flow_shift": 3.0, "guidance": {"txt_cfg": 1.0}}
    server = FakeServer()
    editing = RecipeProcessor(server, True, dict(QWEN_21.defaults), DiffusionRecipeSettings())
    request = editing.request(Image.new("RGB", (64, 64)), Image.new("L", (64, 64)))
    assert request["ref_images"] and "Remove all the text" in request["prompt"] and "negative_prompt" not in request
    filling = RecipeProcessor(server, False, {"cfg_scale": 5}, DiffusionRecipeSettings())
    request = filling.request(Image.new("RGB", (64, 64)), Image.new("L", (64, 64)))
    assert "ref_images" not in request and "text" in request["negative_prompt"]


def test_inpainting_keeps_every_pixel_outside_the_mask():
    server = FakeServer()
    processor = RecipeProcessor(server, True, dict(QWEN_21.defaults), DiffusionRecipeSettings(megapixels=0.1))
    image = Image.new("RGB", (120, 80), (10, 200, 30))
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[20:60, 30:90] = 255
    out = np.asarray(processor.inference(image, mask))
    assert out.shape == (80, 120, 3)
    assert tuple(out[5, 5]) == (10, 200, 30)  # untouched
    assert out[40, 60, 0] > 200  # repainted
    width, height = server.requests[0]["width"], server.requests[0]["height"]
    assert width % 64 == 0 and height % 64 == 0 and (width, height) == working_size(120, 80, 0.1)
    assert processor.inference(image, np.zeros((80, 120), dtype=np.uint8)) is image  # nothing to paint
    assert len(server.requests) == 1


def test_slots_map_to_the_servers_own_flags():
    flags = {"diffusion_model": "--diffusion-model", "llm": "--llm", "llm_vision": "--llm_vision", "taesd": "--taesd"}
    assert flag_for("diffusion_model", flags) == "--diffusion-model"
    assert flag_for("tae", flags) == "--taesd"
    from urakatsu.core.errors import ProviderUnavailable

    with pytest.raises(ProviderUnavailable):
        flag_for("audio_vae", flags)
