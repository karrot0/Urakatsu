"""Manga Colorization v2 (qweasdd): a one-pass GAN colorizer. Not in Koharu.

https://github.com/qweasdd/manga-colorization-v2 at ``a0d0e44``: an FFDNet denoiser cleans the
page, then a SE-ResNeXt generator paints it. Its weights are on Google Drive, so they come from
Hugging Face here:

- The generator is the fp32 ONNX export in ``ifritraen/manga-colorization-v2-fp32``. On a test page
  it matched the repository's PyTorch generator (``generator.zip``) to within 0.0004.
- The denoiser is FFDNet's color model (``net_rgb.pth``, Tassano et al., "FFDNet: Toward a fast and
  flexible solution for CNN based image denoising", 2018), from ``vergil1000/manga-colorization-v2``,
  which holds the repository's files byte for byte as ``Kaiser41/manga-colorization-v2`` does.
  ``FFDNet`` below is written from the paper; its checkpoint loads into it unchanged.

The repository states no license. The steps follow its ``colorizator.py`` and ``inference.py``:
the denoiser at sigma 25 on the page shrunk to 1200 pixels at most, the result read back as 8-bit
BGR, its first channel as the gray input, the page resized to ``size`` across (or 1.5 x ``size``
tall, for a page wider than tall) and padded to a multiple of 32, and no color hints.

Divergence: the result comes back at the size of the page, not the working size, and the page's
solid ink is restored over it (``keep_ink``).
"""

from __future__ import annotations

import importlib.util
import logging
import math
from typing import Any, ClassVar, Optional

import cv2
import numpy as np
from PIL import Image
from pydantic import Field

from urakatsu.core.errors import ProviderUnavailable
from urakatsu.core.registry import register
from urakatsu.providers.base import ColorizationProvider, ProviderSettings
from urakatsu.runtimes.store import HuggingFaceFile, WeightSource

from .diffusion_colorizer import keep_ink

log = logging.getLogger(__name__)

GENERATOR = HuggingFaceFile("ifritraen/manga-colorization-v2-fp32", "manga_colorization_v2_fp32.onnx", "90500178f6bb00374afa95b031bef298cb11808d")
DENOISER = HuggingFaceFile("vergil1000/manga-colorization-v2", "net_rgb.pth", "73c8e239790b659e8c108c60ae32dd8295378295")
MULTIPLE = 32
HINT_CHANNELS = 4  # a color hint (3) and where it applies (1), empty here
DENOISE_MAX_SIDE = 1200  # FFDNetDenoiser.get_denoised_image
# FFDNet's color model: 12 convolutions of 96 features over the 2x2 subimages and the noise map.
FFDNET_LAYERS = 12
FFDNET_FEATURES = 96
CHECKPOINT_PREFIX = "module.intermediate_dncnn.itermediate_dncnn."


class MangaColorizationV2Settings(ProviderSettings):
    size: int = Field(default=576, ge=256, le=1536, multiple_of=MULTIPLE, description="Width the page is painted at, in pixels (height for a page wider than tall: 1.5 times this).")
    denoise: bool = Field(default=True, description="Clean scan noise first, as upstream does. Without it colors come out stronger and more orange. Needs PyTorch.")
    denoise_sigma: int = Field(default=25, ge=0, le=75, description="Strength of the denoiser, as noise on a 0-255 scale.")
    keep_ink: bool = Field(default=True, description="Keep the page's solid ink over the color. Turn off to keep the model's own lines.")


def ffdnet(torch_module: Any):
    """FFDNet for color images, as a module class of ``torch_module`` (imported lazily)."""
    nn = torch_module.nn
    F = torch_module.nn.functional

    class FFDNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = [nn.Conv2d(15, FFDNET_FEATURES, 3, padding=1, bias=False), nn.ReLU()]
            for _ in range(FFDNET_LAYERS - 2):
                layers += [nn.Conv2d(FFDNET_FEATURES, FFDNET_FEATURES, 3, padding=1, bias=False), nn.BatchNorm2d(FFDNET_FEATURES), nn.ReLU()]
            layers.append(nn.Conv2d(FFDNET_FEATURES, 12, 3, padding=1, bias=False))
            self.layers = nn.Sequential(*layers)

        def forward(self, image, sigma: float):
            """The noise in ``image`` (1x3xHxW, even sides, 0-1) at noise level ``sigma`` (0-1)."""
            subimages = F.pixel_unshuffle(image, 2)
            noise_map = torch_module.full((image.shape[0], 3, *subimages.shape[2:]), sigma, device=image.device, dtype=image.dtype)
            return F.pixel_shuffle(self.layers(torch_module.cat([noise_map, subimages], 1)), 2)

    return FFDNet


def resize_pad(gray: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """utils.resize_pad: ``gray`` at ``size`` across (1.5 x ``size`` tall when wider than tall),
    padded on the bottom or right with each line's brightest pixel, by 1 to 32 pixels: a side
    already a multiple of 32 gains 32, as upstream. Returns the padded image (0-1) and the
    (bottom, right) padding. A ``uint8`` image is resized before it is scaled, as upstream's
    denoised page is."""
    height, width = gray.shape
    if height < width:
        new_height = int(size * 1.5)
        new_width = int(math.ceil(width / (height / new_height)))
        pad = (0, MULTIPLE - new_width % MULTIPLE)
    else:
        new_width = size
        new_height = int(math.ceil(height / (width / size)))
        pad = (MULTIPLE - new_height % MULTIPLE, 0)
    resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
    if resized.dtype == np.uint8:
        resized = resized.astype(np.float32) / 255.0
    return np.pad(np.clip(resized, 0.0, 1.0), ((0, pad[0]), (0, pad[1])), "maximum"), pad


@register
class MangaColorizationV2Provider(ColorizationProvider):
    id: ClassVar[str] = "manga-colorization-v2"
    name: ClassVar[str] = "Manga Colorization v2"
    description: ClassVar[str] = "qweasdd's GAN colorizer: under a second a page, warm and light colors, no prompt to steer it."
    runtimes: ClassVar[tuple[str, ...]] = ("onnx",)
    Settings: ClassVar[type[ProviderSettings]] = MangaColorizationV2Settings

    def __init__(self, settings=None, runtimes=None, runtime=None):
        super().__init__(settings, runtimes, runtime)
        self._generator = None
        self._denoiser = None
        self._device = None

    @classmethod
    def weight_sources(cls, runtime: Optional[str] = None) -> tuple[WeightSource, ...]:
        return (GENERATOR, DENOISER)

    def _load(self) -> None:
        if self.settings.denoise and importlib.util.find_spec("torch") is None:
            raise ProviderUnavailable(f"{self.name}: the denoiser needs PyTorch (pip install 'urakatsu[torch]'); or turn Denoise off")
        runtime = self.runtime_instance()
        generator, denoiser = self.resolve_weights()
        self._generator = runtime.session(generator)
        if self.settings.denoise:
            self._load_denoiser(denoiser)
        log.info("loaded Manga Colorization v2 on %s", runtime.effective_device().label())

    def _load_denoiser(self, path) -> None:
        import torch

        state = torch.load(path, map_location="cpu", weights_only=True)
        net = ffdnet(torch)()
        net.load_state_dict({"layers." + key.removeprefix(CHECKPOINT_PREFIX): value for key, value in state.items()}, strict=True)
        self._device = self.runtimes.runtime("torch").torch_device
        self._denoiser = net.to(self._device).eval()
        for parameter in self._denoiser.parameters():
            parameter.requires_grad_(False)

    def _unload(self) -> None:
        self._generator = None
        if self._denoiser is not None:
            from urakatsu.runtimes.torch_runtime import release_cache

            self._denoiser = None
            release_cache(self._device)

    def denoised_gray(self, page: Image.Image) -> np.ndarray:
        """FFDNetDenoiser.get_denoised_image, then the first channel of its BGR result (0-255)."""
        import torch

        rgb = np.asarray(page, dtype=np.float32) / 255.0
        height, width = rgb.shape[:2]
        if max(height, width) > DENOISE_MAX_SIDE:
            ratio = max(height, width) / DENOISE_MAX_SIDE
            rgb = cv2.resize(rgb, (int(width / ratio), int(height / ratio)), interpolation=cv2.INTER_AREA)
        height, width = rgb.shape[:2]
        even = np.pad(rgb, ((0, height % 2), (0, width % 2), (0, 0)), "edge")  # odd sides repeat their last line
        with torch.inference_mode():
            image = torch.from_numpy(even.transpose(2, 0, 1).copy())[None].to(self._device)
            clean = (image - self._denoiser(image, self.settings.denoise_sigma / 255.0)).clamp(0.0, 1.0)
        clean = clean[0, :, :height, :width].cpu().numpy()
        return (clean[2] * 255.0).clip(0, 255).astype(np.uint8)  # BGR: blue comes first

    def _colorize(self, image: Image.Image, reference: Optional[Image.Image] = None) -> Image.Image:
        if self._generator is None:
            raise RuntimeError("Manga Colorization v2 is not loaded")
        page = image.convert("RGB")
        if self._denoiser is not None:
            gray = self.denoised_gray(page)
        else:
            gray = np.asarray(page, dtype=np.float32)[..., 0] / 255.0
        padded, (bottom, right) = resize_pad(gray, self.settings.size)
        inputs = np.concatenate([padded[None, None], np.zeros((1, HINT_CHANNELS, *padded.shape), np.float32)], 1)
        rgb = self._generator.run(None, {"input": inputs.astype(np.float32)})[0][0].transpose(1, 2, 0)
        rgb = rgb[: rgb.shape[0] - bottom, : rgb.shape[1] - right]
        rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        color = Image.fromarray(cv2.resize(rgb, page.size, interpolation=cv2.INTER_LANCZOS4), "RGB")
        return keep_ink(page, color) if self.settings.keep_ink else color
