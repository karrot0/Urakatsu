"""FLUX.2 Klein inpainting provider.

Runs on the ``diffusion`` runtime (the newest stable-diffusion.cpp release as ``sd-server``), with the GGUF weights
and component layout from ``flux2_klein/mod.rs`` and ``model.rs``. The preprocessing,
strength and overlay logic follow ``processor.rs`` and diffusers' FLUX.2 inpaint
pipeline (https://github.com/huggingface/diffusers/blob/a37f6f8394ac2a7ee8360c3abea811efe54512b1/src/diffusers/pipelines/flux2/pipeline_flux2_klein_inpaint.py).

The FLUX.2 scheduler (``Scheduler::Flux2``) is computed here and passed to
stable-diffusion.cpp as custom sigmas, which also pins the schedule to exactly
values. They are identical to upstream's ``Flux2Scheduler`` (``black-forest-labs/flux2
sampling.py``, as ported in stable-diffusion.cpp).
"""

from __future__ import annotations

import logging
import math
from typing import Optional, ClassVar

import numpy as np
from PIL import Image

from urakatsu.core.registry import register
from urakatsu.providers.base import InpaintingProvider, ProviderSettings, server_fields
from urakatsu.runtimes.device import Backend
from urakatsu.runtimes.diffusion import SdServer, image_b64
from urakatsu.runtimes.store import HuggingFaceFile

from .common import LANCZOS3, resize_pil

log = logging.getLogger(__name__)

TRANSFORMER_WEIGHTS = HuggingFaceFile(
    "unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q4_K_M.gguf", "0084d1df98e2e2137fe776d55170bc4792ec1d66"
)
TEXT_ENCODER_WEIGHTS = HuggingFaceFile(
    "unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf", "22c9fc8a8c7700b76a1789366280a6a5a1ad1120"
)
VAE_WEIGHTS = HuggingFaceFile(
    "black-forest-labs/FLUX.2-small-decoder", "full_encoder_small_decoder.safetensors", "a3efc24f613ef42d9428af62fdbd6f5fd8856c4a"
)

# Flux2KleinInpaintOptions::default()
INPAINT_STRENGTH = 0.8
INPAINT_STEPS = 4
# Inpaint pre-resize limit (Flux2ImageProcessor) and the 16-pixel grid of the VAE and patchifier.
TARGET_AREA = 1024 * 1024
GRID = 16
# GuidanceParams { text_cfg: 1.0, ..default } and the distilled guidance default.
TEXT_CFG = 1.0
DISTILLED_GUIDANCE = 3.5


_SERVER_URL, _SERVER_KEY = server_fields("diffusion")


class Flux2KleinSettings(ProviderSettings):
    """``Flux2KleinConfig``."""

    prompt: str = "Remove the text and reconstruct the background."
    server_url: Optional[str] = _SERVER_URL
    server_api_key: Optional[str] = _SERVER_KEY


def flux2_sigmas(image_seq_len: int, num_steps: int) -> list[float]:
    """``Flux2Scheduler::get_sigmas`` (upstream ``Flux2Scheduler``)."""
    f32 = np.float32
    a1, b1 = f32(8.73809524e-05), f32(1.89833333)
    a2, b2 = f32(0.00016927), f32(0.45666666)
    if image_seq_len > 4300:
        mu = a2 * f32(image_seq_len) + b2
    else:
        m_200 = a2 * f32(image_seq_len) + b2
        m_10 = a1 * f32(image_seq_len) + b1
        a = (m_200 - m_10) / f32(190.0)
        b = m_200 - f32(200.0) * a
        mu = a * f32(num_steps) + b
    if num_steps == 0:
        return [1.0]
    exp_mu = np.exp(mu).astype(np.float32)
    sigmas: list[float] = []
    for i in range(num_steps + 1):
        t = f32(1.0) - f32(i) / f32(num_steps)
        if t <= 0.0:
            sigmas.append(0.0)
        elif t >= 1.0:
            sigmas.append(1.0)
        else:
            sigmas.append(float(exp_mu / (exp_mu + np.power(f32(1.0) / t - f32(1.0), f32(1.0)))))
    sigmas[num_steps] = 0.0
    return sigmas


def inpaint_strength(num_inference_steps: int, strength: float) -> np.float32:
    """``Flux2KleinImpaint::inference``: the largest f32 below the effective step boundary."""
    if strength >= 1.0:
        return np.float32(1.0)
    effective_steps = num_inference_steps - int(math.floor(num_inference_steps * (1.0 - strength)))
    boundary = np.float32(effective_steps) / np.float32(num_inference_steps)
    return np.nextafter(boundary, np.float32(0.0))


def resize_to_target_area(image: Image.Image, target_area: int) -> Image.Image:
    scale = math.sqrt(target_area / (image.width * image.height))
    width = max(math.floor(image.width * scale), 1)
    height = max(math.floor(image.height * scale), 1)
    return resize_pil(image, width, height, LANCZOS3)


def _binarize(mask: np.ndarray) -> np.ndarray:
    return np.where(mask < 128, 0, 255).astype(np.uint8)


def apply_overlay(mask: np.ndarray, init_image: np.ndarray, generated: np.ndarray) -> np.ndarray:
    """``Flux2ImageProcessor::apply_overlay`` without a crop region: a soft blend by the (unbinarized) mask."""
    amount = (mask.astype(np.float32) / np.float32(255.0))[..., None]
    original = init_image.astype(np.float32)
    blended = generated.astype(np.float32) * amount + original * (np.float32(1.0) - amount)
    return np.floor(blended + np.float32(0.5)).clip(0.0, 255.0).astype(np.uint8)


class Flux2KleinProcessor:
    """``Flux2KleinInpaint::inference`` with the default options (no padding crop, no reference)."""

    def __init__(self, server: SdServer, prompt: str):
        self.server = server
        self.prompt = prompt

    def inference(self, image: Image.Image, mask: np.ndarray, prompt: Optional[str] = None) -> Image.Image:
        if image.size != (mask.shape[1], mask.shape[0]):
            raise ValueError(f"image/mask dimensions differ: image={image.size}, mask={mask.shape[1]}x{mask.shape[0]}")
        if image.width * image.height > TARGET_AREA:
            image = resize_to_target_area(image, TARGET_AREA)
        width = (image.width // GRID) * GRID
        height = (image.height // GRID) * GRID
        if width == 0 or height == 0:
            raise ValueError("FLUX.2 inpaint image is smaller than one 16-pixel grid cell")

        init_image = resize_pil(image.convert("RGB"), width, height, LANCZOS3)
        mask_image = resize_pil(Image.fromarray(mask.astype(np.uint8), "L"), width, height, LANCZOS3)
        mask_array = np.asarray(mask_image)  # kept unbinarized for the overlay
        native_mask = Image.fromarray(_binarize(mask_array), "L")

        strength = inpaint_strength(INPAINT_STEPS, INPAINT_STRENGTH)
        sigmas = flux2_sigmas((height // GRID) * (width // GRID), INPAINT_STEPS)
        generated = self.server.generate(
            {
                "prompt": prompt or self.prompt,
                "width": width,
                "height": height,
                "strength": float(strength),
                "seed": -1,
                "batch_count": 1,
                "init_image": image_b64(init_image),
                "ref_images": [image_b64(init_image)],
                "ref_image_args": "resize_before_vae=false",
                "mask_image": image_b64(native_mask),
                "sample_params": {
                    "sample_method": "euler",
                    "sample_steps": INPAINT_STEPS,
                    "custom_sigmas": [float(s) for s in sigmas],
                    "guidance": {"txt_cfg": TEXT_CFG, "distilled_guidance": DISTILLED_GUIDANCE},
                },
                "output_format": "png",
            }
        )
        if not generated:
            raise RuntimeError("FLUX.2 Klein returned no inpainted image")
        result = np.asarray(generated[0].convert("RGB"))
        if result.shape[:2] != (height, width):
            raise RuntimeError(f"FLUX.2 Klein returned an unexpected size: {result.shape[1]}x{result.shape[0]}")
        return Image.fromarray(apply_overlay(mask_array, np.asarray(init_image), result), "RGB")


@register
class Flux2KleinProvider(InpaintingProvider):
    id: ClassVar[str] = "flux2-klein"
    name: ClassVar[str] = "FLUX.2 Klein"
    description: ClassVar[str] = "FLUX.2 Klein 4B diffusion inpainting (stable-diffusion.cpp, GGUF Q4_K_M)."
    runtimes: ClassVar[tuple[str, ...]] = ("diffusion",)
    prompted: ClassVar[bool] = True
    instructed: ClassVar[bool] = True  # edits the crop it gets as a reference image

    weight_roles = {"diffusion": ("diffusion_model", "llm", "vae")}

    catalog_queries = ('flux.2 klein 4b',)

    @classmethod
    def accepts(cls, facts) -> Optional[str]:
        """FLUX.2 Klein 4B weights (any quantization); the Qwen3-4B text encoder and VAE stay the defaults."""
        if facts.runtime != "diffusion" or facts.family is None or facts.family.doc != "flux2":
            return None
        names = " ".join(facts.names()).replace("-", "").replace("_", "").replace(".", "")
        return "diffusion" if "klein4b" in names else None
    Settings: ClassVar[type[ProviderSettings]] = Flux2KleinSettings

    def __init__(self, settings=None, runtimes=None, runtime=None):
        super().__init__(settings, runtimes, runtime)
        self._processor: Flux2KleinProcessor | None = None

    @classmethod
    def weight_sources(cls, runtime=None):
        return (TRANSFORMER_WEIGHTS, TEXT_ENCODER_WEIGHTS, VAE_WEIGHTS)

    def _load(self) -> None:
        prompt = self.settings.prompt
        if "\0" in prompt:
            raise ValueError("FLUX.2 prompt contains NUL")
        if self.attached():  # an sd-server started elsewhere with the Klein weights
            self._processor = Flux2KleinProcessor(SdServer.attach(self.settings.server_url, self.settings.server_api_key), prompt)
            log.info("attached FLUX.2 Klein to %s", self.settings.server_url)
            return
        runtime = self.runtime_instance()
        transformer, text_encoder, vae = (self.resolve(source) for source in self.weight_sources(self.runtime_name))
        device = runtime.device
        use_accelerator = device.is_accelerator
        use_cuda = device.backend is Backend.CUDA
        # Parameters stay resident only on accelerators with at least 20 GiB free;
        # otherwise the parameters stay in host memory ("*=cpu" params_backend).
        keep_resident = use_accelerator and device.memory_free >= 20 * 1024**3
        args = ["--diffusion-model", str(transformer), "--llm", str(text_encoder), "--vae", str(vae), "--mmap"]
        args += ["--flash-attn", "on" if use_cuda else "off"]
        if use_cuda:
            args += ["--conv-direct", "auto,diffusion=on,vae=on"]
        if use_accelerator and not keep_resident:
            args.append("--offload-to-cpu")
        self._processor = Flux2KleinProcessor(runtime.launch("flux2-klein", args), prompt)
        log.info("loaded FLUX.2 Klein on %s", device.label())

    def _unload(self) -> None:
        if self._processor is not None:
            self._processor.server.close()
        self._processor = None

    def _inpaint(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        if self._processor is None:
            raise RuntimeError("FLUX.2 Klein is not loaded")
        return self._processor.inference(image.convert("RGB"), mask)

    def _fill(self, image: Image.Image, mask: np.ndarray, prompt: Optional[str]) -> Image.Image:
        if self._processor is None:
            raise RuntimeError("FLUX.2 Klein is not loaded")
        return self._processor.inference(image.convert("RGB"), mask, prompt)
