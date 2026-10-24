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


def test_companions_are_picked_sensibly():
    comfy = [
        ("text_encoders/qwen3vl_8b_bf16.safetensors", 17 * GB),
        ("text_encoders/qwen3vl_8b_int8_convrot.safetensors", 9 * GB),
        ("text_encoders/qwen3vl_8b_w4a8_gptq.safetensors", 5 * GB),  # a format sd.cpp does not read
        ("vae/flux2-vae.safetensors", 300 << 20),
        ("vae/flux2-vae_small.safetensors", 100 << 20),
        ("qwen-image-Q4_K_M.gguf", 4 * GB),
    ]
    assert pick("llm", slot_candidates("llm", "r", comfy)).path.endswith("int8_convrot.safetensors")
    assert pick("vae", slot_candidates("vae", "r", comfy)).path == "vae/flux2-vae.safetensors"
    gguf_repo = [("Qwen3VL-8B-Instruct-Q8_0.gguf", 8 * GB), ("Qwen3VL-8B-Instruct-Q4_K_M.gguf", 5 * GB), ("mmproj-Qwen3VL-8B-Instruct-F16.gguf", GB), ("mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf", GB // 2)]
    assert pick("llm", slot_candidates("llm", "q", gguf_repo, anywhere=True)).path == "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
    assert pick("llm_vision", slot_candidates("llm_vision", "q", gguf_repo)).path.startswith("mmproj-Qwen3VL-8B-Instruct-F16")


def test_diffusion_fit_counts_the_largest_part_on_the_gpu_and_the_total_in_ram():
    gpu = [Device.gpu(Backend.CUDA, 0, "RTX", memory_total=12 * GB), Device.cpu()]
    system = SystemProfile(ram_total=32 * GB, ram_available=8 * GB)
    assert fitting.fit_diffusion([4 * GB, 9 * GB, GB], gpu, system).level == "offload"  # runs one part at a time
    assert fitting.fit_diffusion([3 * GB, 2 * GB, GB], gpu, system).level == "gpu"
    assert fitting.fit_diffusion([14 * GB, 9 * GB], gpu, system).level == "cpu"


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
