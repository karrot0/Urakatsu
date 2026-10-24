"""AOT inpainting provider.

- ``aot_net.py``: the AOT generator (``aot_inpainting/model.rs``).
- This module: BallonsTranslator-compatible processor (``aot_inpainting/processor.rs``)
  and the provider lifecycle (``aot_inpainting/mod.rs``).

https://github.com/dmMaze/BallonsTranslator/blob/4bcc635c19f6c63a902872cf77b3d554e14ed1b7/ballontranslator/modules/inpaint/inpaint_default.py#L124-L176
https://github.com/dmMaze/BallonsTranslator/blob/4bcc635c19f6c63a902872cf77b3d554e14ed1b7/ballontranslator/utils/imgproc_utils.py#L132-L164
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from urakatsu.core.registry import register
from urakatsu.providers.base import InpaintingProvider, ProviderSettings
from urakatsu.runtimes.store import HuggingFaceFile

from .aot_net import AOTGenerator
from .common import pad_symmetric, python_round, resize_bilinear_nonaliased

log = logging.getLogger(__name__)

WEIGHTS = HuggingFaceFile(
    "mayocream/aot-inpainting",
    "model.safetensors",
    "cffe2346ac2b5ebe1f2d61335d602d12cc144c6f",
)

MAX_SIDE = 2048  # AotInpainting::inference
MIN_PAD_SIDE = 128  # the network needs at least 128 pixels per side


class AotSettings(ProviderSettings):
    """No options: AOT has no processor profile."""


class AotProcessor:
    def __init__(self, generator: AOTGenerator, device: torch.device, dtype: torch.dtype):
        self.generator = generator
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def call(self, image: np.ndarray, mask: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
        if max_side <= 0:
            raise ValueError("AOT inpainting max side must be positive")
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"image and mask dimensions differ: image={image.shape[:2]}, mask={mask.shape[:2]}")
        original_height, original_width = mask.shape[:2]
        if original_width == 0 or original_height == 0:
            raise ValueError("image dimensions must be non-zero")

        original_mask = (mask >= 127).astype(np.uint8)  # compositing mask, binarized

        if max(original_width, original_height) > max_side:
            input_width, input_height = _resize_dimensions(original_width, original_height, max_side)
        else:
            input_width, input_height = original_width, original_height
        input_image = resize_bilinear_nonaliased(image, input_width, input_height)
        input_mask = resize_bilinear_nonaliased(mask, input_width, input_height)

        pad_bottom = max(MIN_PAD_SIDE - input_height, 0)
        pad_right = max(MIN_PAD_SIDE - input_width, 0)
        image_t = (
            torch.from_numpy(np.ascontiguousarray(input_image)).to(self.device).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
            / 127.5
            - 1.0
        )
        mask_t = torch.from_numpy(np.ascontiguousarray(input_mask)).to(self.device).unsqueeze(0).unsqueeze(0).to(torch.float32) / 255.0
        image_t = _pad_bottom_right(image_t, pad_bottom, pad_right)
        mask_t = (_pad_bottom_right(mask_t, pad_bottom, pad_right) >= 0.5).to(torch.float32)
        masked_image = image_t * (1.0 - mask_t)

        output = self.generator(masked_image.to(self.dtype), mask_t.to(self.dtype))
        output_height = output.shape[2] - pad_bottom
        output_width = output.shape[3] - pad_right
        if output_height <= 0 or output_width <= 0:
            raise RuntimeError(f"AOT output is too small after removing padding: {tuple(output.shape)}")
        output = output[:, :, :output_height, :output_width]
        rgb = _tensor_to_rgb(output)
        if rgb.shape[:2] != (original_height, original_width):
            rgb = resize_bilinear_nonaliased(rgb, original_width, original_height)

        result = rgb.copy()
        keep = original_mask == 0
        result[keep] = image[keep]
        return result


def _resize_dimensions(width: int, height: int, max_side: int) -> tuple[int, int]:
    ratio = max_side / max(width, height)
    return max(python_round(width * ratio), 1), max(python_round(height * ratio), 1)


def _pad_bottom_right(tensor: torch.Tensor, pad_bottom: int, pad_right: int) -> torch.Tensor:
    if pad_bottom == 0 and pad_right == 0:
        return tensor
    return pad_symmetric(tensor, tensor.shape[2] + pad_bottom, tensor.shape[3] + pad_right)


def _tensor_to_rgb(output: torch.Tensor) -> np.ndarray:
    pixels = ((output.to(torch.float32) + 1.0) * 127.5).round().clamp(0.0, 255.0).to(torch.uint8)
    return pixels.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()


@register
class AotInpaintingProvider(InpaintingProvider):
    id: ClassVar[str] = "aot-inpainting"
    name: ClassVar[str] = "AOT Inpainting"
    description: ClassVar[str] = "AOT GAN inpainting (BallonsTranslator)"
    runtimes: ClassVar[tuple[str, ...]] = ("torch",)
    Settings: ClassVar[type[ProviderSettings]] = AotSettings

    def __init__(self, settings=None, runtimes=None, runtime=None):
        super().__init__(settings, runtimes, runtime)
        self._processor: AotProcessor | None = None

    @classmethod
    def weight_sources(cls, runtime=None):
        return (WEIGHTS,)

    def _load(self) -> None:
        runtime = self.runtime_instance()
        weights = self.resolve(WEIGHTS)
        device, dtype = runtime.torch_device, runtime.dtype()
        generator = AOTGenerator()
        generator.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
        generator = generator.to(device=device, dtype=dtype).eval()
        for parameter in generator.parameters():
            parameter.requires_grad_(False)
        self._processor = AotProcessor(generator, device, dtype)
        log.info("loaded AOT inpainting on %s (%s)", device, dtype)

    def _unload(self) -> None:
        self._processor = None

    def _inpaint(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        if self._processor is None:
            raise RuntimeError("AOT inpainting is not loaded")
        rgb = np.array(image.convert("RGB"))
        return Image.fromarray(self._processor.call(rgb, mask.astype(np.uint8), MAX_SIDE), "RGB")
