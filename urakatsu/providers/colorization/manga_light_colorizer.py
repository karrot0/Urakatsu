"""Manga Light Colorizer: a one-pass ONNX colorizer. Not in Koharu.

https://huggingface.co/sharky172/manga-light-colorizer at ``REVISION``: a FastViT encoder and a
UNet decoder guided by SAM 2.1 features of the page, as two ONNX graphs. The steps follow its
``inference.py``: the page in gray at the working size, the SAM encoder's two feature levels from
it at 1024 square, the generator with an empty WD14 tag embedding (the release leaves WD14 off), and
the result back at the size of the page.

The weights are CC BY-NC-SA 4.0: free to use, not for commercial work.

Divergence: upstream squeezes the page into a square of ``--infer-size``. Here the working size
keeps the page's shape at about the area of a ``size`` square, which drew cleaner color on long
pages, and in sides of a multiple of 32 as the encoder's five levels need. The page's solid ink
is restored over the result (``keep_ink``), as the model's lines are soft at its working size.
"""

from __future__ import annotations

import logging
import math
from typing import ClassVar, Optional

import cv2
import numpy as np
from PIL import Image
from pydantic import Field

from urakatsu.core.registry import register
from urakatsu.providers.base import ColorizationProvider, ProviderSettings
from urakatsu.runtimes.store import HuggingFaceFile, WeightSource

from .diffusion_colorizer import keep_ink

log = logging.getLogger(__name__)

REPOSITORY = "sharky172/manga-light-colorizer"
REVISION = "2fb022c4ce55632b7671a1df306f63984928e36a"
GENERATOR = HuggingFaceFile(REPOSITORY, "models/v6_generator.onnx", REVISION)
SAM_ENCODER = HuggingFaceFile(REPOSITORY, "models/v6_sam_encoder.onnx", REVISION)
SAM_SIZE = 1024
MULTIPLE = 32
WD14_SIZE = 1024


class MangaLightColorizerSettings(ProviderSettings):
    size: int = Field(default=512, ge=256, le=1536, multiple_of=MULTIPLE, description="Working size in pixels, as the side of a square of the same area; the model learned at 512, and colors drift as it grows.")
    keep_ink: bool = Field(default=True, description="Keep the page's solid ink over the color. Turn off to keep the model's own, softer lines.")


def working_size(width: int, height: int, size: int) -> tuple[int, int]:
    """The page's shape at about the area of a ``size`` square, in sides of a multiple of 32."""
    scale = size / math.sqrt(width * height)
    return (max(round(width * scale / MULTIPLE), 1) * MULTIPLE, max(round(height * scale / MULTIPLE), 1) * MULTIPLE)


@register
class MangaLightColorizerProvider(ColorizationProvider):
    id: ClassVar[str] = "manga-light-colorizer"
    name: ClassVar[str] = "Manga Light Colorizer"
    description: ClassVar[str] = "Colors a page in about a second, with no prompt to steer it (sharky172, CC BY-NC-SA 4.0: not for commercial work)."
    runtimes: ClassVar[tuple[str, ...]] = ("onnx",)
    Settings: ClassVar[type[ProviderSettings]] = MangaLightColorizerSettings

    def __init__(self, settings=None, runtimes=None, runtime=None):
        super().__init__(settings, runtimes, runtime)
        self._generator = None
        self._sam = None

    @classmethod
    def weight_sources(cls, runtime: Optional[str] = None) -> tuple[WeightSource, ...]:
        return (GENERATOR, SAM_ENCODER)

    def _load(self) -> None:
        runtime = self.runtime_instance()
        generator, sam = self.resolve_weights()
        self._generator = runtime.session(generator)
        self._sam = runtime.session(sam)
        log.info("loaded Manga Light Colorizer on %s", runtime.effective_device().label())

    def _unload(self) -> None:
        self._generator = None
        self._sam = None

    def _colorize(self, image: Image.Image, reference: Optional[Image.Image] = None) -> Image.Image:
        if self._generator is None:
            raise RuntimeError("Manga Light Colorizer is not loaded")
        page = image.convert("RGB")
        width, height = working_size(page.width, page.height, self.settings.size)
        gray = cv2.resize(np.asarray(page.convert("L")), (width, height), interpolation=cv2.INTER_AREA)
        light = gray.astype(np.float32) / 127.5 - 1.0
        square = cv2.resize((light + 1.0) / 2.0, (SAM_SIZE, SAM_SIZE), interpolation=cv2.INTER_LINEAR)
        level0, level1 = self._sam.run(None, {"rgb_input": np.stack([square] * 3)[None].astype(np.float32)})[:2]
        inputs = {
            "L_bw": light[None, None],
            "sam_level0": level0,
            "sam_level1": level1,
            "wd14_embedding": np.zeros((1, WD14_SIZE), dtype=np.float32),
        }
        rgb = self._generator.run(None, inputs)[0][0].transpose(1, 2, 0)
        rgb = np.clip((rgb + 1.0) * 127.5, 0, 255).astype(np.uint8)
        color = Image.fromarray(cv2.resize(rgb, page.size, interpolation=cv2.INTER_LANCZOS4), "RGB")
        return keep_ink(page, color) if self.settings.keep_ink else color
