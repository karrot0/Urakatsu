"""RORem mixed-resolution SDXL inpainting provider.

Runs on the ``diffusion`` runtime (stable-diffusion.cpp), with the upstream component layout
from ``rorem_mixed/mod.rs`` and ``model.rs``. Preprocessing and compositing follow
``rorem_mixed/processor.rs`` and the reference script
https://github.com/leeruibin/RORem/blob/891ab8a17bbcfb16a773c078cb256aae8cb30468/inference_RORem.py
"""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

import numpy as np
from PIL import Image

from urakatsu.core.registry import register
from urakatsu.providers.base import InpaintingProvider, ProviderSettings
from urakatsu.runtimes.device import Backend
from urakatsu.runtimes.diffusion import SdServer, image_b64
from urakatsu.runtimes.store import HuggingFaceFile

from .common import LANCZOS3, NEAREST, resize_pil

log = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "clean manga illustration, crisp black line art, flat colors, seamless original background, "
    "clean white speech bubble, no text"
)
DEFAULT_NEGATIVE_PROMPT = (
    "text, letters, words, symbols, watermark, signature, blurry, smudged, dirty, gray artifacts, "
    "extra objects, photorealistic"
)

DIFFUSION_MODEL = HuggingFaceFile(
    "mayocream/RORem-mixed-GGUF", "rorem-mixed-unet-q4_K.gguf", "62c75b3e6f078a19e2698b0f677e8a4aa4c9ea56"
)
SDXL_VERSION_MARKER = HuggingFaceFile(
    "mayocream/RORem-mixed-GGUF", "sdxl-version-marker.safetensors", "62c75b3e6f078a19e2698b0f677e8a4aa4c9ea56"
)
VAE_MODEL = HuggingFaceFile(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "115134f363124c53c7d878647567d04daf26e41e",
)
CLIP_L_MODEL = HuggingFaceFile(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "text_encoder/model.fp16.safetensors",
    "115134f363124c53c7d878647567d04daf26e41e",
)
CLIP_G_MODEL = HuggingFaceFile(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "text_encoder_2/model.fp16.safetensors",
    "115134f363124c53c7d878647567d04daf26e41e",
)

# RoremMixedOptions::default()
RESOLUTION = 512
MASK_DILATION = 0
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 8.0
STRENGTH = 0.999
SEED = -1


class RoremMixedSettings(ProviderSettings):
    """``RoremMixedConfig``."""

    prompt: str = DEFAULT_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT


def resize_mask(mask: np.ndarray, resolution: int) -> np.ndarray:
    """Binarize (below 128 is 0), then nearest-resize to the square model resolution."""
    binary = np.where(mask < 128, 0, 255).astype(np.uint8)
    if binary.shape == (resolution, resolution):
        return binary
    image = resize_pil(Image.fromarray(binary, "L"), resolution, resolution, NEAREST)
    return np.asarray(image)


def composite(original: Image.Image, native_mask: np.ndarray, generated: Image.Image) -> Image.Image:
    """Restores the caller's size and exact unmasked pixels (``Processor::composite``)."""
    width, height = original.size
    generated = resize_pil(generated, width, height, LANCZOS3)
    mask = np.asarray(resize_pil(Image.fromarray(native_mask, "L"), width, height, NEAREST))
    output = np.asarray(generated.convert("RGB")).copy()
    source = np.asarray(original.convert("RGB"))
    keep = mask == 0
    output[keep] = source[keep]
    return Image.fromarray(output, "RGB")


class RoremMixedProcessor:
    def __init__(self, server: SdServer, prompt: str, negative_prompt: str):
        self.server = server
        self.prompt = prompt
        self.negative_prompt = negative_prompt

    def inference(self, image: Image.Image, mask: np.ndarray, prompt: Optional[str] = None) -> Image.Image:
        if image.width <= 0 or image.height <= 0:
            raise ValueError("image dimensions must be non-zero")
        if image.size != (mask.shape[1], mask.shape[0]):
            raise ValueError(f"image and mask dimensions differ: image={image.size}, mask={mask.shape[1]}x{mask.shape[0]}")

        original = image.convert("RGB")
        init_image = resize_pil(original, RESOLUTION, RESOLUTION, LANCZOS3)
        native_mask = resize_mask(mask, RESOLUTION)
        if not native_mask.any():
            return original

        generated = self.server.generate(
            {
                "prompt": prompt or self.prompt,
                "negative_prompt": self.negative_prompt,
                "width": RESOLUTION,
                "height": RESOLUTION,
                "strength": STRENGTH,
                "seed": SEED,
                "batch_count": 1,
                "init_image": image_b64(init_image),
                "mask_image": image_b64(Image.fromarray(native_mask, "L")),
                "sample_params": {
                    "scheduler": "discrete",
                    "sample_method": "euler",
                    "sample_steps": NUM_INFERENCE_STEPS,
                    "guidance": {"txt_cfg": GUIDANCE_SCALE, "distilled_guidance": 3.5},
                },
                "output_format": "png",
            }
        )
        if not generated:
            raise RuntimeError("RORem mixed returned no inpainted image")
        result = generated[0]
        if result.size != (RESOLUTION, RESOLUTION):
            raise RuntimeError(f"RORem mixed returned an unexpected image size: {result.size}")
        return composite(original, native_mask, result)


@register
class RoremMixedProvider(InpaintingProvider):
    id: ClassVar[str] = "rorem-mixed"
    name: ClassVar[str] = "RORem mixed"
    description: ClassVar[str] = "RORem mixed-resolution SDXL inpainting (stable-diffusion.cpp, GGUF)."
    runtimes: ClassVar[tuple[str, ...]] = ("diffusion",)
    prompted: ClassVar[bool] = True
    Settings: ClassVar[type[ProviderSettings]] = RoremMixedSettings

    def __init__(self, settings=None, runtimes=None, runtime=None):
        super().__init__(settings, runtimes, runtime)
        self._processor: RoremMixedProcessor | None = None

    @classmethod
    def weight_sources(cls, runtime=None):
        return (DIFFUSION_MODEL, SDXL_VERSION_MARKER, VAE_MODEL, CLIP_L_MODEL, CLIP_G_MODEL)

    def _load(self) -> None:
        if "\0" in self.settings.prompt or "\0" in self.settings.negative_prompt:
            raise ValueError("RORem prompt contains NUL")
        runtime = self.runtime_instance()
        marker, diffusion_model, vae, clip_l, clip_g = (
            self.resolve(source) for source in (SDXL_VERSION_MARKER, DIFFUSION_MODEL, VAE_MODEL, CLIP_L_MODEL, CLIP_G_MODEL)
        )
        use_cuda = runtime.device.backend is Backend.CUDA
        args = ["--model", str(marker), "--diffusion-model", str(diffusion_model), "--vae", str(vae)]
        args += ["--clip_l", str(clip_l), "--clip_g", str(clip_g), "--mmap", "--flash-attn", "on" if use_cuda else "off"]
        if use_cuda:
            args += ["--conv-direct", "auto,diffusion=on,vae=on"]
        server = runtime.launch("rorem-mixed", args)
        self._processor = RoremMixedProcessor(server, self.settings.prompt, self.settings.negative_prompt)
        log.info("loaded RORem mixed on %s", runtime.device.label())

    def _unload(self) -> None:
        if self._processor is not None:
            self._processor.server.close()
        self._processor = None

    def _inpaint(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        if self._processor is None:
            raise RuntimeError("RORem mixed is not loaded")
        return self._processor.inference(image.convert("RGB"), mask)

    def _fill(self, image: Image.Image, mask: np.ndarray, prompt: Optional[str]) -> Image.Image:
        if self._processor is None:
            raise RuntimeError("RORem mixed is not loaded")
        return self._processor.inference(image.convert("RGB"), mask, prompt)
