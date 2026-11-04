"""Er0mangaSeg censor bar detector. Not in Koharu.

The checkpoint scores dark bars on the page (https://github.com/Er0manga/Er0mangaSeg, Apache-2.0).
Scores become a mask through ``bars``, and paper-white strokes, which the checkpoint does not
cover, come from the geometry in ``white_bars``. Both, and the preprocessing here, follow the
censor detection of the Koharu fork at https://github.com/spgsroot/koharu
(``crates/koharu-ml/src/eromanga_seg.rs``). The page is never inverted for the checkpoint: a white
background then reads as dark censorship.

The repository's mega.nz link is blocked, so the pinned checkpoint comes from the author's demo
image on Docker Hub (``er0manga/er0mangademo``, https://github.com/Er0manga/Er0mangaDemo), one
layer of which holds it: it downloads on first use, and nothing in the image runs. A ``.pth`` is a
pickle, and this one carries training metadata that only loads with arbitrary classes allowed.
``load_state`` keeps the tensors and replaces every other class with an inert stand-in, so loading
it runs no code from the file and needs neither mmengine nor mmsegmentation.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import types
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np
from PIL import Image, ImageFilter
from pydantic import Field

from urakatsu.core.errors import ProviderUnavailable
from urakatsu.core.registry import register
from urakatsu.providers.base import CensorDetectionProvider, ProviderSettings
from urakatsu.runtimes.store import ContainerLayerFile

log = logging.getLogger(__name__)

CHECKPOINT = "convnext_1024_iter_400.pth"
CHECKPOINT_SHA256 = "9c0e7ae74d043d6ff94e08408bf8a1c032158d2a86d6f7feb6738fb8843b31a6"
WEIGHTS = ContainerLayerFile(
    "er0manga/er0mangademo",
    "sha256:75cefe50d664b47b49a814678cdc629e69fd2f24e4030e271db6f9cbcf50f78a",
    f"home/user/Er0mangaSeg/pretrained/{CHECKPOINT}",
    CHECKPOINT_SHA256,
    666270749,
)
PREFIXES = ("backbone.", "decode_head.")
MODEL_SIZE = 1024
MEAN = (123.675, 116.28, 103.53)
STD = (58.395, 57.12, 57.375)

# What a checkpoint needs to hold tensors; storages are resolved by ``torch.load`` itself.
ALLOWED = {
    ("collections", "OrderedDict"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_parameter"),
}


class _Inert:
    """Stands in for any class the checkpoint names outside ``ALLOWED``, and does nothing."""

    def __new__(cls, *args: Any, **kwargs: Any) -> "_Inert":
        return super().__new__(cls)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> "_Inert":
        return _Inert()

    def __setstate__(self, state: Any) -> None:
        pass

    def __setitem__(self, key: Any, value: Any) -> None:
        pass

    def append(self, value: Any) -> None:
        pass

    def extend(self, values: Any) -> None:
        pass


class _Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        return super().find_class(module, name) if (module, name) in ALLOWED else _Inert


# ``torch.load`` subclasses ``pickle_module.Unpickler`` and only adds the storage types.
_restricted_pickle = types.ModuleType("restricted_pickle")
_restricted_pickle.Unpickler = _Unpickler
_restricted_pickle.load = lambda file, **kwargs: _Unpickler(file, **kwargs).load()


def load_state(path: Path) -> dict[str, Any]:
    """The checkpoint's weights, without running anything the file names."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_restricted_pickle)
    state = checkpoint["state_dict"]
    return {key: value for key, value in state.items() if key.startswith(PREFIXES)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def grow_and_blur(mask: np.ndarray, grow: int, blur: float) -> np.ndarray:
    """``mask`` widened by ``grow`` pixels on every side, with an edge softened by ``blur``."""
    if not mask.any():
        return mask
    if grow > 0:
        from scipy import ndimage  # here: the app imports every provider at startup, and scipy takes half a second

        mask = ndimage.maximum_filter(mask, size=2 * grow + 1)
    if blur > 0:
        mask = np.asarray(Image.fromarray(mask, "L").filter(ImageFilter.GaussianBlur(blur)))
    return mask


class Er0mangaSegSettings(ProviderSettings):
    weights: Optional[str] = Field(default=None, description=f"Path of a local {CHECKPOINT}; empty downloads it on first use.")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Lowest score that starts a dark bar")
    white_bars: bool = Field(default=True, description="Also look for paper-white strokes")
    mask_grow: int = Field(default=12, ge=0, le=255, description="Pixels added around the bars, to cover their edges")
    mask_blur: float = Field(default=4.0, ge=0.0, description="Softness of the mask's edge, in pixels")


@register
class Er0mangaSegProvider(CensorDetectionProvider):
    id: ClassVar[str] = "er0mangaseg"
    name: ClassVar[str] = "Er0mangaSeg"
    description: ClassVar[str] = "Finds dark censor bars with a ConvNeXt segmenter, and white strokes by their geometry."
    runtimes: ClassVar[tuple[str, ...]] = ("torch",)
    Settings: ClassVar[type[ProviderSettings]] = Er0mangaSegSettings

    def __init__(self, settings=None, runtimes=None, runtime=None):
        super().__init__(settings, runtimes, runtime)
        self._net = None
        self._device = None

    @classmethod
    def weight_sources(cls, runtime=None):
        return (WEIGHTS,)

    @classmethod
    def pending_downloads(cls, settings, cached):
        return [] if settings.get("weights") or cached(WEIGHTS) else [{"label": "the bar detector (about 670 MB)", "model": False}]

    def checkpoint(self) -> Path:
        if not self.settings.weights:
            return self.resolve(WEIGHTS)
        path = Path(self.settings.weights).expanduser()
        if not path.is_file():
            raise ProviderUnavailable(f"{self.name}: {path} was not found; clear the path to download {CHECKPOINT}")
        return path

    def _load(self) -> None:
        from .er0manga_net import Er0mangaSegNet

        path = self.checkpoint()
        if sha256(path) != CHECKPOINT_SHA256:
            raise ProviderUnavailable(f"{path} is not the Er0mangaSeg checkpoint this version pins")
        net = Er0mangaSegNet()
        net.load_state_dict(load_state(path), strict=True)
        runtime = self.runtime_instance()
        self._device = runtime.torch_device
        self._net = net.to(self._device).eval()
        for parameter in self._net.parameters():
            parameter.requires_grad_(False)
        log.info("loaded Er0mangaSeg on %s", self._device)

    def _unload(self) -> None:
        self._net = None

    def _input(self, resized: np.ndarray):
        """Normalized, then padded to the checkpoint's square input. Zero padding after
        normalization is the mean colour, as in the upstream test pipeline."""
        import torch
        from torch.nn import functional as F

        pixels = torch.from_numpy(resized).to(self._device).permute(2, 0, 1).float()
        mean = torch.tensor(MEAN, device=self._device).view(3, 1, 1)
        std = torch.tensor(STD, device=self._device).view(3, 1, 1)
        height, width = resized.shape[:2]
        return F.pad((pixels - mean) / std, (0, MODEL_SIZE - width, 0, MODEL_SIZE - height)).unsqueeze(0)

    def scores(self, rgb: np.ndarray) -> np.ndarray:
        """Each page pixel as 0, ``bars.SUPPORTED`` or ``bars.CONFIDENT`` (HxW uint8)."""
        import torch
        from torch.nn import functional as F

        from . import bars  # scipy: imported here, as the app imports every provider at startup

        height, width = rgb.shape[:2]
        scale = MODEL_SIZE / max(width, height)
        size = max(round(width * scale), 1), max(round(height * scale), 1)
        resized = np.array(Image.fromarray(rgb, "RGB").resize(size, Image.Resampling.BILINEAR))
        threshold = float(self.settings.threshold)
        with torch.inference_mode():
            probability = self._net(self._input(resized)).softmax(1)[:, 1:]
            probability = F.interpolate(probability, size=(MODEL_SIZE, MODEL_SIZE), mode="bilinear", align_corners=False)
            probability = probability[:, :, : size[1], : size[0]]
            probability = F.interpolate(probability, size=(height, width), mode="bilinear", align_corners=False)[0, 0]
            levels = (probability >= bars.support_threshold(threshold)).to(torch.uint8) + (probability >= threshold).to(torch.uint8)
        return levels.cpu().numpy()

    def _detect_censorship(self, image: Image.Image) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Er0mangaSeg is not loaded")
        from . import bars, white_bars

        rgb = np.array(image.convert("RGB"))
        found = bars.bar_mask(rgb, self.scores(rgb))
        mask = found if found is not None else np.zeros(rgb.shape[:2], dtype=np.uint8)
        if self.settings.white_bars:
            mask = np.maximum(mask, white_bars.white_bar_mask(rgb))
        return grow_and_blur(mask, self.settings.mask_grow, self.settings.mask_blur)
