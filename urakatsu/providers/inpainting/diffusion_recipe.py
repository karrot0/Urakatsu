"""Any stable-diffusion.cpp image model as an inpainter, driven by ``sd-server``'s recipes.

- the component slots, passed to ``sd-server`` as the matching command-line flags (read
  from ``sd-server --help``, e.g. ``llm`` -> ``--llm``, ``diffusion_model`` -> ``--diffusion-model``);
- the sampler defaults (method, steps, CFG, flow shift, scheduler), sent with each request;
- whether the model edits natively (``ref_images``): such models get the crop as a reference
  image and an instruction to remove the text; the others repaint the masked area from a
  description of a clean page.

Each crop is scaled to about the requested megapixels (diffusion models work at their
training size, not at the size of a speech balloon), generated, scaled back by the stage,
and blended so pixels outside the mask stay exactly as they were.
"""

from __future__ import annotations

import functools
import logging
import math
import re
import subprocess
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np
from PIL import Image
from pydantic import Field

from urakatsu.catalog.recipes import Recipe
from urakatsu.core.errors import ProviderUnavailable
from urakatsu.core.registry import register
from urakatsu.providers.base import InpaintingProvider, ProviderSettings
from urakatsu.runtimes.device import Backend
from urakatsu.runtimes.hardware import system_profile
from urakatsu.runtimes.diffusion import SdServer, image_b64
from urakatsu.runtimes.store import WeightSource

from .common import LANCZOS3, apply_overlay, resize_pil

log = logging.getLogger(__name__)

GRID = 64  # a multiple of every model's VAE and patch size
MAX_UPSCALE = 4.0
EDIT_PROMPT = "Remove all the text and lettering. Fill those areas with the background and artwork around them, matching its style."
FILL_PROMPT = "clean manga artwork, seamless background, empty speech bubble, no text"
FILL_NEGATIVE = "text, letters, words, writing, watermark, signature"


class DiffusionRecipeSettings(ProviderSettings):
    prompt: Optional[str] = Field(default=None, description="Empty uses an instruction for editing models, a description of a clean page for the others.")
    negative_prompt: Optional[str] = Field(default=None, description="Empty uses a text-free negative prompt for models that use CFG.")
    steps: Optional[int] = Field(default=None, description="Sampling steps; empty uses the model's recipe.")
    cfg_scale: Optional[float] = Field(default=None, description="Guidance scale; empty uses the model's recipe.")
    strength: Optional[float] = Field(default=None, description="How much of the masked area is regenerated (0 to 1); empty uses the recipe, else 1.")
    megapixels: float = Field(default=1.0, description="Working size of each crop, in megapixels.")
    reference: bool = Field(default=True, description="Give editing models the crop as a reference image.")


@functools.lru_cache(maxsize=4)
def slot_flags(executable: str) -> dict[str, str]:
    """``sd-server``'s flag for each recipe slot, read from its ``--help``."""
    try:
        text = subprocess.run([executable, "--help"], capture_output=True, text=True, timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except (OSError, subprocess.SubprocessError):
        text = ""
    flags = set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9_-]*)", text))
    return {flag[2:].replace("-", "_"): flag for flag in flags}


def flag_for(slot: str, flags: dict[str, str]) -> str:
    if slot in flags:
        return flags[slot]
    if slot == "tae" and "taesd" in flags:
        return flags["taesd"]  # the legacy name of the tiny autoencoder flag
    raise ProviderUnavailable(f"this stable-diffusion.cpp release has no option for the {slot} component; update it in Settings, Runtimes")


def working_size(width: int, height: int, megapixels: float) -> tuple[int, int]:
    """The crop's size scaled to about ``megapixels``, on the model grid."""
    scale = math.sqrt(megapixels * 1_000_000 / max(width * height, 1))
    scale = min(scale, MAX_UPSCALE)
    return (max(GRID, round(width * scale / GRID) * GRID), max(GRID, round(height * scale / GRID) * GRID))


def sample_params(defaults: dict[str, Any], settings: DiffusionRecipeSettings) -> dict[str, Any]:
    """``sd-server``'s ``sample_params`` from the recipe defaults and the user's overrides."""
    params: dict[str, Any] = {}
    guidance: dict[str, Any] = {}
    if defaults.get("sampling_method"):
        params["sample_method"] = defaults["sampling_method"]
    if defaults.get("scheduler"):
        params["scheduler"] = defaults["scheduler"]
    steps = settings.steps or defaults.get("steps")
    if steps:
        params["sample_steps"] = int(steps)
    if defaults.get("flow_shift") is not None:
        params["flow_shift"] = float(defaults["flow_shift"])
    cfg = settings.cfg_scale if settings.cfg_scale is not None else defaults.get("cfg_scale")
    if cfg is not None:
        guidance["txt_cfg"] = float(cfg)
    if defaults.get("img_cfg_scale") is not None:
        guidance["img_cfg"] = float(defaults["img_cfg_scale"])
    if defaults.get("distilled_guidance") is not None:
        guidance["distilled_guidance"] = float(defaults["distilled_guidance"])
    if guidance:
        params["guidance"] = guidance
    return params


class RecipeProcessor:
    def __init__(self, server: SdServer, edits: bool, defaults: dict[str, Any], settings: DiffusionRecipeSettings):
        self.server = server
        self.edits = edits and settings.reference
        self.defaults = defaults
        self.settings = settings

    def request(self, init_image: Image.Image, mask: Image.Image, prompt: Optional[str] = None) -> dict[str, Any]:
        settings = self.settings
        prompt = prompt or settings.prompt or (EDIT_PROMPT if self.edits else FILL_PROMPT)
        cfg = settings.cfg_scale if settings.cfg_scale is not None else self.defaults.get("cfg_scale", 1)
        negative = settings.negative_prompt if settings.negative_prompt is not None else ("" if self.edits or float(cfg) <= 1 else FILL_NEGATIVE)
        strength = settings.strength if settings.strength is not None else self.defaults.get("strength", 1.0)
        body: dict[str, Any] = {
            "prompt": prompt,
            "width": init_image.width,
            "height": init_image.height,
            "strength": float(strength),
            "seed": -1,
            "batch_count": 1,
            "init_image": image_b64(init_image),
            "mask_image": image_b64(mask),
            "sample_params": sample_params(self.defaults, settings),
            "output_format": "png",
        }
        if negative:
            body["negative_prompt"] = negative
        if self.edits:
            body["ref_images"] = [image_b64(init_image)]
        return body

    def inference(self, image: Image.Image, mask: np.ndarray, prompt: Optional[str] = None) -> Image.Image:
        if image.size != (mask.shape[1], mask.shape[0]):
            raise ValueError(f"image/mask dimensions differ: image={image.size}, mask={mask.shape[1]}x{mask.shape[0]}")
        if not (mask >= 128).any():
            return image
        width, height = working_size(image.width, image.height, self.settings.megapixels)
        init_image = resize_pil(image.convert("RGB"), width, height, LANCZOS3)
        soft_mask = np.asarray(resize_pil(Image.fromarray(mask.astype(np.uint8), "L"), width, height, LANCZOS3))
        native_mask = Image.fromarray(np.where(soft_mask < 128, 0, 255).astype(np.uint8), "L")
        generated = self.server.generate(self.request(init_image, native_mask, prompt))
        if not generated:
            raise RuntimeError("the diffusion model returned no image")
        result = generated[0].convert("RGB")
        if result.size != (width, height):
            result = resize_pil(result, width, height, LANCZOS3)
        blended = Image.fromarray(apply_overlay(soft_mask, np.asarray(init_image), np.asarray(result)), "RGB")
        return resize_pil(blended, image.width, image.height, LANCZOS3)


def _is_text_encoder(installed) -> bool:
    """Installs from before the catalog told text encoders apart (a Qwen3-VL GGUF named after Qwen-Image)."""
    from urakatsu.catalog.support import SupportIndex
    from urakatsu.runtimes.packages import LLAMA_CPP, PackageManager

    if not installed.architecture:
        return False
    support = SupportIndex(PackageManager()).installed(LLAMA_CPP)
    return support is not None and installed.architecture in support.architectures


def _release_torch_cache() -> None:
    """Hands memory PyTorch cached for earlier stages (detection) back to the GPU for sd-server."""
    from urakatsu.runtimes.torch_runtime import release_cache

    release_cache()


@register
class DiffusionRecipeProvider(InpaintingProvider):
    id: ClassVar[str] = "diffusion-model"
    name: ClassVar[str] = "Diffusion model"
    description: ClassVar[str] = "Any image model stable-diffusion.cpp has a recipe for, installed from the Models page."
    runtimes: ClassVar[tuple[str, ...]] = ("diffusion",)
    prompted: ClassVar[bool] = True  # instructed per model: the installed model's recipe ``edits``
    Settings: ClassVar[type[ProviderSettings]] = DiffusionRecipeSettings

    @classmethod
    def accepts(cls, facts) -> Optional[str]:
        """Image models whose stable-diffusion.cpp recipe allows masked image-to-image."""
        if facts.runtime == "diffusion" and facts.recipe is not None and facts.recipe.cleans_pages():
            return "diffusion"
        return None

    def __init__(self, settings=None, runtimes=None, runtime=None):
        super().__init__(settings, runtimes, runtime)
        self._processor: RecipeProcessor | None = None

    def _installed(self):
        """The installed model chosen in settings."""
        from urakatsu.catalog.installed import InstalledModels

        model_id = self.settings.model
        if not model_id:
            raise ProviderUnavailable(f"{self.name}: pick an installed model in Settings, Pipeline (install one from the Models page)")
        installed = InstalledModels(self.store()).get(model_id)
        if installed is None or not installed.recipe:
            raise ProviderUnavailable(f"{self.name}: {model_id} is not an installed stable-diffusion.cpp model")
        if _is_text_encoder(installed):
            raise ProviderUnavailable(
                f"{self.name}: {installed.repo} is a text encoder ({installed.architecture}), not an image model, so it cannot clean pages. "
                "Remove it on the Models page and install a denoiser for its family instead."
            )
        return installed

    def _components(self) -> tuple[str, str, list[tuple[str, WeightSource]]]:
        """The server's name, the recipe id and each component as (slot, weights)."""
        installed = self._installed()
        components = [(f.role, installed.source(f.path)) for f in installed.files if f.role not in ("shard", "file")]
        return installed.id, installed.recipe, components

    def _recipe(self, tag: str, recipe_id: str) -> Recipe:
        from urakatsu.catalog.support import SupportIndex
        from urakatsu.runtimes.packages import STABLE_DIFFUSION_CPP, PackageManager

        support = SupportIndex(PackageManager()).for_tag(STABLE_DIFFUSION_CPP, tag)
        recipe = support.recipe(recipe_id) if support else None
        if recipe is None:
            raise ProviderUnavailable(f"stable-diffusion.cpp {tag} has no recipe {recipe_id}; update it in Settings, Runtimes")
        return recipe

    def _server_args(self, device) -> list[str]:
        """Flags beyond the components, placement and memory mapping."""
        return ["--flash-attn", "on" if device.backend is Backend.CUDA else "off"]

    def _processor_for(self, server: SdServer, recipe: Recipe):
        return RecipeProcessor(server, recipe.edits, dict(recipe.defaults), self.settings)

    def _load(self) -> None:
        name, recipe_id, components = self._components()
        runtime = self.runtime_instance()
        package = runtime.package()  # installs or updates stable-diffusion.cpp first
        recipe = self._recipe(package.tag, recipe_id)

        flags = slot_flags(str(package.executable))
        args: list[str] = []
        total = 0
        for slot, source in components:
            path: Path = self.resolve(source)
            total += path.stat().st_size
            args += [flag_for(slot, flags), str(path)]
        device = runtime.device
        args += self._server_args(device)
        # When the components do not fit the GPU together, keep them in RAM and move each one
        # in only while it runs (text encoder, then diffusion model, then VAE), as the model
        # docs do: the GPU is shared with the other stages' models and other programs, and
        # placement alone leaves no room for the compute buffers. ``--offload-to-cpu`` pins every
        # weight in RAM and turns ``--auto-fit`` off, so when RAM cannot hold them either (Qwen
        # Image on 16 GB) leave placement to ``--auto-fit``: GPU, then RAM, then disk.
        ram_available = system_profile().ram_available
        offload = device.is_accelerator and total > 0.8 * (device.memory_free or device.memory_total) and total <= 0.9 * ram_available
        if offload:
            args.append("--offload-to-cpu")
        # Memory-mapping only speeds up loading. With offload the weights then live in mapped
        # pages that Windows evicts and re-reads from disk for every image: on a hard drive that
        # was ~5 MB/s and minutes per crop. Read them into RAM once when it fits.
        if not offload or ram_available < 1.2 * total:
            args.append("--mmap")
        _release_torch_cache()
        server = runtime.launch(re.sub(r"[^A-Za-z0-9_.-]+", "-", name)[:60], args)
        self._processor = self._processor_for(server, recipe)
        log.info("loaded %s (recipe %s) on %s", name, recipe.id, runtime.device.label())

    def _unload(self) -> None:
        if self._processor is not None:
            self._processor.server.close()
        self._processor = None

    def _inpaint(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        if self._processor is None:
            raise RuntimeError("the diffusion model is not loaded")
        return self._processor.inference(image.convert("RGB"), mask)

    def _fill(self, image: Image.Image, mask: np.ndarray, prompt: Optional[str]) -> Image.Image:
        if self._processor is None:
            raise RuntimeError("the diffusion model is not loaded")
        return self._processor.inference(image.convert("RGB"), mask, prompt)
