"""LaMa inpainting provider.

- ``lama_net.py``: the FFC-ResNet generator (``lama/model.rs``, ``lama/config.rs``).
- This module: the IOPaint-compatible processor (``lama/processor.rs``) and the
  provider lifecycle (``lama/mod.rs``: weights, ``InpaintRequest::default()``).

Original implementations:
https://github.com/Sanster/IOPaint/blob/61a759fb3f332bacdce8b2813f4837495c9b86e0/iopaint/model/base.py#L57-L192
https://github.com/Sanster/IOPaint/blob/61a759fb3f332bacdce8b2813f4837495c9b86e0/iopaint/helper.py#L187-L267
"""

from __future__ import annotations

import logging
from typing import Optional, ClassVar

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from urakatsu.core.registry import register
from urakatsu.providers.base import InpaintingProvider, ProviderSettings
from urakatsu.runtimes.store import HuggingFaceFile

from .common import ceil_modulo, pad_symmetric
from .lama_net import FFCResNetGenerator, FFCResNetGeneratorConfig, generator_forward

log = logging.getLogger(__name__)

WEIGHTS = HuggingFaceFile(
    "mayocream/lama-manga",
    "lama-manga.safetensors",
    "f91c85b26913b3e83f9877867b4c336da3675238",
)

# InpaintRequest::default(): HDStrategy::Crop, trigger 800, margin 128, keep unmasked area.
HD_CROP_TRIGGER_SIZE = 800
HD_CROP_MARGIN = 128
SD_KEEP_UNMASKED_AREA = True
MODULO = 8


class LamaSettings(ProviderSettings):
    """No options: LaMa has no processor profile."""


class LamaProcessor:
    """IOPaint-compatible preprocessing, crop orchestration and postprocessing (``processor.rs``).

    Operates on RGB uint8 arrays (HxWx3) and binary masks (HxW, 0 or 255).
    """

    def __init__(self, generator: FFCResNetGenerator, device: torch.device, dtype: torch.dtype):
        self.generator = generator
        self.device = device
        self.dtype = dtype

    def call(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"image and mask dimensions differ: image={image.shape[:2]}, mask={mask.shape[:2]}")
        height, width = mask.shape[:2]
        if width == 0 or height == 0:
            raise ValueError("image dimensions must be non-zero")
        if max(width, height) > HD_CROP_TRIGGER_SIZE:
            result = image.copy()
            for left, top, right, bottom in boxes_from_mask(mask):
                crop_box = crop_box_for(width, height, (left, top, right, bottom), HD_CROP_MARGIN)
                c_left, c_top, c_right, c_bottom = crop_box
                crop_result = self.pad_forward(
                    image[c_top:c_bottom, c_left:c_right], mask[c_top:c_bottom, c_left:c_right]
                )
                result[c_top:c_bottom, c_left:c_right] = crop_result
            return result
        return self.pad_forward(image, mask)

    @torch.no_grad()
    def pad_forward(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        height, width = mask.shape
        image_t = torch.from_numpy(np.ascontiguousarray(image)).to(self.device).permute(2, 0, 1).unsqueeze(0)
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).to(self.device).unsqueeze(0).unsqueeze(0)

        model_image = _pad_to_modulo(image_t.to(torch.float32) / 255.0, MODULO)
        model_mask = _pad_to_modulo((mask_t > 0).to(torch.float32), MODULO)
        output = generator_forward(self.generator, model_image.to(self.dtype), model_mask.to(self.dtype))
        output = output[:, :, :height, :width].clamp(0.0, 1.0) * 255.0
        output = output.to(torch.uint8)
        if SD_KEEP_UNMASKED_AREA:
            alpha = mask_t.to(torch.float32) / 255.0
            output = (output.to(torch.float32) * alpha + image_t.to(torch.float32) * (1.0 - alpha)).to(torch.uint8)
        return output.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()


def _pad_to_modulo(tensor: torch.Tensor, modulo: int) -> torch.Tensor:
    height, width = tensor.shape[2], tensor.shape[3]
    return pad_symmetric(tensor, ceil_modulo(height, modulo), ceil_modulo(width, modulo))


def boxes_from_mask(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Boxes of the outermost mask contours, padded by one pixel (``boxes_from_mask``)."""
    rows = np.flatnonzero((mask > 127).any(axis=1))
    cols = np.flatnonzero((mask > 127).any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return []
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    cropped = mask[top:bottom, left:right]
    padded = np.zeros((cropped.shape[0] + 2, cropped.shape[1] + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = np.where(cropped > 127, 255, 0)

    contours, hierarchy = cv2.findContours(padded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    boxes = []
    for contour, node in zip(contours, hierarchy[0]):
        if node[3] != -1:  # only top-level outer borders (no enclosing contour)
            continue
        points = contour.reshape(-1, 2)
        c_left, c_top = int(points[:, 0].min()), int(points[:, 1].min())
        c_right, c_bottom = int(points[:, 0].max()), int(points[:, 1].max())
        box = (
            left + max(c_left - 1, 0),
            top + max(c_top - 1, 0),
            min(left + c_right, mask.shape[1]),
            min(top + c_bottom, mask.shape[0]),
        )
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append(box)
    return boxes


def crop_box_for(
    image_width: int, image_height: int, box: tuple[int, int, int, int], margin: int
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    crop_width = (right - left) + margin * 2
    crop_height = (bottom - top) + margin * 2
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2

    raw_left = center_x - crop_width // 2
    raw_right = center_x + crop_width // 2
    raw_top = center_y - crop_height // 2
    raw_bottom = center_y + crop_height // 2

    new_left = max(raw_left, 0)
    new_right = min(raw_right, image_width)
    new_top = max(raw_top, 0)
    new_bottom = min(raw_bottom, image_height)
    if raw_left < 0:
        new_right += -raw_left
    if raw_right > image_width:
        new_left -= raw_right - image_width
    if raw_top < 0:
        new_bottom += -raw_top
    if raw_bottom > image_height:
        new_top -= raw_bottom - image_height

    return (
        min(max(new_left, 0), image_width),
        min(max(new_top, 0), image_height),
        min(max(new_right, 0), image_width),
        min(max(new_bottom, 0), image_height),
    )


@register
class LamaProvider(InpaintingProvider):
    id: ClassVar[str] = "lama"
    name: ClassVar[str] = "LaMa"
    description: ClassVar[str] = "LaMa manga inpainting (lama-manga), IOPaint-compatible crop orchestration."
    runtimes: ClassVar[tuple[str, ...]] = ("torch",)

    catalog_queries = ('lama',)

    @classmethod
    def accepts(cls, facts) -> Optional[str]:
        """LaMa inpainting checkpoints in safetensors form."""
        if facts.runtime == "torch" and "lama" in (facts.model_type or "") and any(f.endswith(".safetensors") for f in facts.files):
            return "torch"
        return None
    Settings: ClassVar[type[ProviderSettings]] = LamaSettings

    def __init__(self, settings=None, runtimes=None, runtime=None):
        super().__init__(settings, runtimes, runtime)
        self._processor: LamaProcessor | None = None

    @classmethod
    def weight_sources(cls, runtime=None):
        return (WEIGHTS,)

    def _load(self) -> None:
        runtime = self.runtime_instance()
        weights = self.resolve(WEIGHTS)
        device, dtype = runtime.torch_device, runtime.dtype()
        generator = FFCResNetGenerator(FFCResNetGeneratorConfig())
        generator.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
        generator = generator.to(device=device, dtype=dtype).eval()
        for parameter in generator.parameters():
            parameter.requires_grad_(False)
        self._processor = LamaProcessor(generator, device, dtype)
        log.info("loaded LaMa on %s (%s)", device, dtype)

    def _unload(self) -> None:
        self._processor = None

    def _inpaint(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        if self._processor is None:
            raise RuntimeError("LaMa is not loaded")
        rgb = np.array(image.convert("RGB"))
        return Image.fromarray(self._processor.call(rgb, mask.astype(np.uint8)), "RGB")
