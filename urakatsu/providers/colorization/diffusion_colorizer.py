"""Any stable-diffusion.cpp image editor as a colorizer. Not in Koharu.

The model and its LoRA are the user's: an installed model that edits from reference images (a
recipe with ``ref_images``, or the editing recipe of its weights, as ``anima-edit`` is for Anima),
with any LoRA from the models folder. Nothing here names one. The page goes in as the first
reference image and an optional color reference as the second, the order reference LoRAs expect.
stable-diffusion.cpp picks how references reach each model (``--ref-image-args`` presets).
On a worker (Colab, Kaggle) the LoRA is sent to it once (``remote_files``), and the reference
goes along with each page.

The model redraws the page rather than painting on it, at about ``megapixels``: its lines are
softer than the page's and land several pixels away, more in some places than others. So its
drawing is fitted to the page by optical flow, then solid ink comes back from the page itself.
Restoring ink without the fit doubles every line, and taking only hue from the model would not
do: white paper has no room for color, so skin and clothes would stay white.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Optional

import cv2
import numpy as np
from PIL import Image
from pydantic import Field

from urakatsu.catalog.recipes import Recipe, edit_recipe
from urakatsu.core.errors import ProviderUnavailable
from urakatsu.core.registry import register
from urakatsu.providers.base import ColorizationProvider, ProviderSettings
from urakatsu.providers.inpainting.common import LANCZOS3, resize_pil
from urakatsu.providers.inpainting.diffusion_recipe import DiffusionRecipeProvider, sample_params, working_size
from urakatsu.runtimes.diffusion import SdServer, image_b64

PROMPT = "Colorize this manga page."
# Page gray levels: at or below ``INK_SOLID`` the page's pixel is kept, above ``INK_SOFT`` the
# model's. Screentone and gray shading sit above, so they take the model's color.
INK_SOLID = 48
INK_SOFT = 112
# cv2.calcOpticalFlowFarneback: pyramid scale, levels, window, iterations, poly_n, poly_sigma,
# flags. The window is wider than a line, so lines follow what is around them.
FLOW = (0.5, 4, 31, 4, 7, 1.5, 0)


class DiffusionColorizerSettings(ProviderSettings):
    model: Optional[str] = Field(default=None, description="Installed model that edits images (Models page).")
    prompt: Optional[str] = Field(default=None, description="Instruction, with the LoRA's trigger word if it has one. Empty asks for the page in color.")
    negative_prompt: Optional[str] = Field(default=None, description="What to avoid, for models that use CFG.")
    steps: Optional[int] = Field(default=None, ge=1, description="Sampling steps; empty uses the model's recipe.")
    cfg_scale: Optional[float] = Field(default=None, ge=0, description="Guidance scale; empty uses the model's recipe.")
    megapixels: float = Field(default=1.5, gt=0, description="Working size of the page, in megapixels.")
    lora: Optional[str] = Field(default=None, description="Pick one from the models folder's loras directory or type a path to a .safetensors file.")
    lora_strength: float = Field(default=1.0, ge=0.0, le=2.0, description="How strongly the LoRA applies; 0 runs without one")
    reference: Optional[str] = Field(default=None, description="Path to a colored image (a cover or character sheet), for models and LoRAs that color by reference.")
    keep_ink: bool = Field(default=True, description="Fit the model's drawing to the page and keep the page's solid ink. Turn off to keep the model's own lines, which move the art.")


def fit_to_page(working: Image.Image, generated: Image.Image, size: tuple[int, int]) -> Image.Image:
    """``generated`` at ``size``, its drawing moved onto the lines of ``working``, the page as the
    model saw it. The flow is found at the model's size, so its window suits any page size."""
    flow = cv2.calcOpticalFlowFarneback(np.asarray(working.convert("L")), np.asarray(generated.convert("L")), None, *FLOW)
    width, height = size
    scale = np.array([width / working.width, height / working.height], dtype=np.float32)
    flow = cv2.resize(flow, size, interpolation=cv2.INTER_LINEAR) * scale
    color = np.asarray(resize_pil(generated, width, height, LANCZOS3))
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    return Image.fromarray(cv2.remap(color, xs + flow[..., 0], ys + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), "RGB")


def keep_ink(page: Image.Image, color: Image.Image) -> Image.Image:
    """``color`` with the solid ink of ``page`` restored; both are the same size."""
    gray = np.asarray(page.convert("L"), dtype=np.float32)
    ink = np.clip((INK_SOFT - gray) / (INK_SOFT - INK_SOLID), 0.0, 1.0)[..., None]
    pixels = np.asarray(page.convert("RGB"), dtype=np.float32) * ink + np.asarray(color.convert("RGB"), dtype=np.float32) * (1.0 - ink)
    return Image.fromarray(np.rint(pixels).astype(np.uint8), "RGB")


class ColorizeProcessor:
    edits = True  # the page always goes in as ``ref_images``

    def __init__(self, server: SdServer, defaults: dict[str, Any], settings: DiffusionColorizerSettings, lora: Optional[Path] = None):
        self.server = server
        self.defaults = defaults
        self.settings = settings
        self.lora = lora

    def request(self, page: Image.Image, reference: Optional[Image.Image] = None) -> dict[str, Any]:
        settings = self.settings
        references = [page] if reference is None else [page, reference]
        body: dict[str, Any] = {
            "prompt": settings.prompt or PROMPT,
            "width": page.width,
            "height": page.height,
            "seed": -1,
            "batch_count": 1,
            "ref_images": [image_b64(image) for image in references],
            "sample_params": sample_params(self.defaults, settings),
            "output_format": "png",
        }
        if settings.negative_prompt:
            body["negative_prompt"] = settings.negative_prompt
        if self.lora is not None:
            body["lora"] = [{"path": self.lora.name, "multiplier": float(settings.lora_strength)}]
        return body

    def inference(self, image: Image.Image, reference: Optional[Image.Image] = None) -> Image.Image:
        width, height = working_size(image.width, image.height, self.settings.megapixels)
        working = resize_pil(image, width, height, LANCZOS3)
        generated = self.server.generate(self.request(working, reference))
        if not generated:
            raise RuntimeError("the diffusion model returned no image")
        result = resize_pil(generated[0].convert("RGB"), width, height, LANCZOS3)
        if not self.settings.keep_ink:
            return resize_pil(result, image.width, image.height, LANCZOS3)
        return keep_ink(image, fit_to_page(working, result, image.size))


@register
class DiffusionColorizerProvider(ColorizationProvider, DiffusionRecipeProvider):
    id: ClassVar[str] = "diffusion-colorizer"
    name: ClassVar[str] = "Colorizer"
    description: ClassVar[str] = "Any image model stable-diffusion.cpp can edit with, installed from the Models page, with an optional LoRA."
    Settings: ClassVar[type[ProviderSettings]] = DiffusionColorizerSettings

    @classmethod
    def accepts(cls, facts) -> Optional[str]:
        """Image models with a recipe that edits from reference images."""
        if facts.runtime == "diffusion" and facts.edit_recipe is not None and facts.edit_recipe.cleans_pages():
            return "diffusion"
        return None

    def _lora(self) -> Optional[Path]:
        if not self.settings.lora or not self.settings.lora_strength:
            return None
        path = Path(self.settings.lora).expanduser()
        if not path.is_file():
            raise ProviderUnavailable(f"{self.name}: the LoRA {path} was not found")
        return path

    def remote_files(self) -> dict[str, Path]:
        lora = self._lora()
        return {"lora": lora} if lora is not None else {}

    def reference_image(self) -> Optional[Image.Image]:
        if not self.settings.reference:
            return None
        path = Path(self.settings.reference).expanduser()
        if not path.is_file():
            raise ProviderUnavailable(f"{self.name}: the color reference {path} was not found")
        with Image.open(path) as image:
            return image.convert("RGB")

    def _recipe(self, tag: str, recipe_id: str) -> Recipe:
        from urakatsu.catalog.support import SupportIndex
        from urakatsu.runtimes.packages import STABLE_DIFFUSION_CPP, PackageManager

        recipe = super()._recipe(tag, recipe_id)
        support = SupportIndex(PackageManager()).for_tag(STABLE_DIFFUSION_CPP, tag)
        editor = edit_recipe(support.recipes, recipe) if support else None
        if editor is None:
            raise ProviderUnavailable(f"{self.name}: {recipe.name} cannot edit from reference images; pick another model")
        return editor

    def _server_args(self, device) -> list[str]:
        args = super()._server_args(device)
        lora = self._lora()
        return [*args, "--lora-model-dir", str(lora.parent)] if lora is not None else args

    def _processor_for(self, server: SdServer, recipe: Recipe) -> ColorizeProcessor:
        return ColorizeProcessor(server, dict(recipe.defaults), self.settings, self._lora())

    def _colorize(self, image: Image.Image, reference: Optional[Image.Image] = None) -> Image.Image:
        if self._processor is None:
            raise RuntimeError("the diffusion model is not loaded")
        return self._processor.inference(image.convert("RGB"), reference)
