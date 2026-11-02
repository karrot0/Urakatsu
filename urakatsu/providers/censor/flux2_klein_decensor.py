"""FLUX.2 Klein with a censor bar removal LoRA. Not in Koharu.

Runs the same stable-diffusion.cpp path as ``flux2-klein`` with settings of its own, so text
cleanup never gets its prompt or LoRA. Its default is Klein 9B, not the 4B text cleanup uses: the
bar LoRAs are trained on 9B. 9B is under a non-commercial license, so it is only downloaded once
the user has turned censorship on, and Settings says so there. sd-server applies the LoRA per
request (``lora`` of ``/sdcpp/v1/img_gen``).

The Koharu fork pairs 9B with ``ponpoke/flux2-klein-9b-uncensored-text-encoder``. That repository
is gated, so it cannot download on its own; this uses the public Qwen 8B encoder instead.

Removing a bar needs more than removing text: the Koharu fork at
https://github.com/spgsroot/koharu found bars still visible after 4 steps, and that partial
strength keeps them from the initial image. So this runs at full strength for 8 steps, and the
model works from the crop as its reference.

On a worker (Colab, Kaggle) the worker downloads the pinned LoRA itself; one the user picked from
disk is sent to it once (``remote_files``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np
from PIL import Image
from pydantic import Field

from urakatsu.catalog.recipes import Recipe
from urakatsu.core.errors import ProviderUnavailable
from urakatsu.core.registry import register
from urakatsu.providers.base import DecensorProvider, ProviderSettings
from urakatsu.providers.inpainting.flux2_klein import DISTILLED_GUIDANCE, TEXT_CFG, Flux2KleinProcessor, Flux2KleinProvider
from urakatsu.runtimes.diffusion import SdServer
from urakatsu.runtimes.store import HuggingFaceFile

# Norian's bar LoRA for Klein 9B, version 1 (``uncensorMangaKlein_Monochrome``, the file the Koharu
# fork documents). Version 2's only source, Civitai, no longer serves it; this is a public mirror.
LORA = HuggingFaceFile("ComfyCloudModel/Flux2Klein-Loras", "loras/Flux2Klein9b_MangaUncensored.safetensors", "4e66a3f426f63c3f0a790059f2353b8c27b4c6ae")
LORA_SHA256 = "913613a95e72da845f8f113d521af66cc2a7d4f3ad3caac4e207d5bc01a45d51"
TRANSFORMER = HuggingFaceFile("unsloth/FLUX.2-klein-9B-GGUF", "flux-2-klein-9b-Q8_0.gguf", "6db9b2d0479c98795589c94c9beef5cc818c6966")
TEXT_ENCODER = HuggingFaceFile("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf", "a6adef130ffb23ddaf1a62fec9dced968c9bc482")
VAE = HuggingFaceFile("Comfy-Org/Flux-2-Klein-9B", "split_files/vae/flux2-vae.safetensors", "1138ce884c8e977cdacacbe779784fec4426bb3f")
RECIPE = "flux2-klein-9b"
STEPS = 8
BLACK_PROMPT = "Remove the black censor bars."
WHITE_PROMPT = "Remove the white censor bars."
# Sizes the bar LoRAs name in their metadata, told apart from the model's repository name.
SIZES = ("4b", "9b")


class Flux2KleinDecensorSettings(ProviderSettings):
    """Not ``Flux2KleinSettings``: this has no ``server_url``, so pages go nowhere but this computer
    and a worker the user picks in the runtime menu."""

    prompt: str = Field(default=BLACK_PROMPT, description="Instruction for black bars")
    white_prompt: str = Field(default=WHITE_PROMPT, description="Instruction for white bars")
    steps: Optional[int] = Field(default=STEPS, ge=1, description="Sampling steps; fewer can leave bars visible.")
    cfg_scale: Optional[float] = Field(default=None, ge=0, description="Text guidance scale; empty uses the model's recipe. Above 1 about doubles the time.")
    strength: float = Field(default=1.0, gt=0.0, le=1.0, description="How much of the crop is regenerated; below 1 keeps some of the bars.")
    lora: Optional[str] = Field(default=None, description="Pick one from the models folder's loras directory or type a path to a .safetensors file; empty uses the built-in bar LoRA, for Klein 9B.")
    lora_strength: float = Field(default=1.0, ge=0.0, le=2.0, description="How strongly the LoRA applies; 0 runs without one")


def lora_metadata(path: Path) -> dict[str, str]:
    """The ``__metadata__`` of a safetensors file, from its header alone."""
    try:
        with path.open("rb") as file:
            length = int.from_bytes(file.read(8), "little")
            header = json.loads(file.read(length))
    except (OSError, ValueError) as error:
        raise ProviderUnavailable(f"{path} is not a safetensors file: {error}") from error
    return header.get("__metadata__") or {}


def check_lora(path: Path, model: str) -> None:
    """Refuses a LoRA whose metadata names another size of FLUX.2 Klein than ``model``.
    Metadata is optional, so a LoRA that names none runs."""
    named = " ".join(lora_metadata(path).values()).lower().replace(".", "").replace("-", "").replace("_", "")
    wanted = model.lower().replace("-", "").replace("_", "")
    for size, other in (SIZES, SIZES[::-1]):
        if f"klein{size}" in named and f"klein{size}" not in wanted and f"klein{other}" in wanted:
            raise ProviderUnavailable(f"{path.name} is for FLUX.2 Klein {size.upper()}, not {model}")


@register
class Flux2KleinDecensorProvider(DecensorProvider, Flux2KleinProvider):
    id: ClassVar[str] = "flux2-klein-decensor"
    name: ClassVar[str] = "FLUX.2 Klein 9B"
    description: ClassVar[str] = "Reconstructs the art under censor bars with Klein 9B and a bar LoRA, or any installed FLUX.2 model."
    Settings: ClassVar[type[ProviderSettings]] = Flux2KleinDecensorSettings
    _checked_lora: Optional[Path] = None

    def _lora(self) -> Optional[Path]:
        if not self.settings.lora_strength:
            return None
        if not self.settings.lora:
            return self._pinned_lora()
        path = Path(self.settings.lora).expanduser()
        if not path.is_file():
            raise ProviderUnavailable(f"{self.name}: the LoRA {path} was not found")
        return path

    def remote_files(self) -> dict[str, Path]:
        """A LoRA the user picked goes to the worker; the pinned one the worker downloads itself."""
        if not self.settings.lora or not self.settings.lora_strength:
            return {}
        return {"lora": self._lora()}

    def _pinned_lora(self) -> Optional[Path]:
        """The bar LoRA, on the only size it was trained for; checked once per load."""
        if "9b" not in (self.settings.model or RECIPE).lower():
            return None
        path = self.resolve(LORA)
        if self._checked_lora != path:
            if hashlib.sha256(path.read_bytes()).hexdigest() != LORA_SHA256:
                raise ProviderUnavailable(f"{path} is not the LoRA this version pins")
            self._checked_lora = path
        return path

    @classmethod
    def weight_sources(cls, runtime=None):
        return (TRANSFORMER, TEXT_ENCODER, VAE)

    @classmethod
    def pending_downloads(cls, settings, cached):
        model = settings.get("model")
        found = []
        if not model and not all(cached(source) for source in cls.weight_sources()):
            found.append({"label": "FLUX.2 Klein 9B (about 15.6 GB, under a non-commercial license)", "model": True})
        pinned = settings.get("lora_strength", 1.0) and not settings.get("lora") and "9b" in (model or RECIPE).lower()
        if pinned and not cached(LORA):
            found.append({"label": "the bar LoRA (166 MB)", "model": False})
        return found

    def _components(self):
        if not self.settings.model:
            return self.id, RECIPE, list(zip(self.weight_roles["diffusion"], self.weight_sources()))
        components = super()._components()
        lora = self._lora()
        if lora is not None:
            check_lora(lora, self.settings.model)
        return components

    def _server_args(self, device) -> list[str]:
        args = super()._server_args(device)
        lora = self._lora()
        return [*args, "--lora-model-dir", str(lora.parent)] if lora is not None else args

    def _processor_for(self, server: SdServer, recipe: Optional[Recipe] = None) -> Flux2KleinProcessor:
        defaults = dict(recipe.defaults) if recipe is not None else {}
        settings = self.settings
        cfg_scale = settings.cfg_scale if settings.cfg_scale is not None else float(defaults.get("cfg_scale", TEXT_CFG))
        lora = self._lora()
        loras: list[dict[str, Any]] = [{"path": lora.name, "multiplier": float(settings.lora_strength)}] if lora is not None else []
        return Flux2KleinProcessor(server, settings.prompt, settings.steps or STEPS, cfg_scale, float(defaults.get("distilled_guidance", DISTILLED_GUIDANCE)), settings.strength, loras)

    def _reconstruct(self, image: Image.Image, mask: np.ndarray, white: bool) -> Image.Image:
        return self._fill(image, mask, self.settings.white_prompt if white else self.settings.prompt)
