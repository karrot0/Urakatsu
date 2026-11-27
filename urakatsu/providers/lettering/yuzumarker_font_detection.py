"""YuzuMarker.FontDetection: which of 6,150 CJK fonts set a crop of text (Torch).

A torchvision ResNet-50 whose last layer is 2048 to 6,162: one logit per font, then the text
direction and ten regression outputs, which are not used here. Koharu 0.100 runs it
(``crates/koharu-ml/src/yuzumarker_font_detection``), reading how likely each broad style of font
is: the fonts' probabilities summed by the kernel table published beside the weights.

Upstream: https://github.com/JeffersonQin/YuzuMarker.FontDetection at ``0a94e165`` (MIT).
Weights: ``mayocream/yuzumarker-fontdetection``, a tensor-only Safetensors conversion of the
checkpoint upstream's demo runs.

A crop is stretched to 512 x 512 without keeping its shape, scaled to 0 to 1 and not normalized,
as upstream's ``demo.py`` does. Splitting a long line or column into squarer pieces first was
tried and read no better: on rendered text in fonts of known style, the stretched crops were
right as often (the network is trained on stretched crops).

Difference from Koharu: the reading also carries the weight of the fonts named
(``font_names.weight_of`` over ``fonts.json``, weighed by probability). On rendered Kozuka Gothic
and Kozuka Mincho from ExtraLight to Heavy it read 217, 293, 368, 490, 689 and 889 for gothic and
209, 342, 436, 483, 676 and 896 for mincho, so it orders weights where a stroke width only splits
them in two. Those families are among the 6,150, as most text faces are.
"""

from __future__ import annotations

import json
import logging
from typing import Any, ClassVar, Optional

import numpy as np
from PIL import Image

from urakatsu.core.errors import ModelLoadError, ProcessingError
from urakatsu.core.registry import register
from urakatsu.core.styles import STYLES
from urakatsu.providers.base import FontReading, LetteringProvider, ProviderSettings
from urakatsu.runtimes.store import HuggingFaceFile, WeightSource

from .font_names import weight_of

log = logging.getLogger(__name__)

REPOSITORY = "mayocream/yuzumarker-fontdetection"
REVISION = "c58f09e7151904f5b6d3f1e6c5678558a16d5ea9"
WEIGHTS = HuggingFaceFile(REPOSITORY, "model.safetensors", REVISION)
KERNELS = HuggingFaceFile(REPOSITORY, "kernels.json", REVISION)
FONTS = HuggingFaceFile(REPOSITORY, "fonts.json", REVISION)

# config.FONT_COUNT: the first outputs, one per font.
FONT_COUNT = 6150
# config.FONT_COUNT + 12: the fonts, the direction and the regression outputs.
OUTPUTS = 6162
# demo.py's transforms.Resize((512, 512)).
SIZE = 512
BATCH = 8
# resnet50's blocks and planes per stage, and Bottleneck.expansion.
LAYERS = ((3, 64), (4, 128), (6, 256), (3, 512))
EXPANSION = 4
# The least share of a crop's probability on fonts whose names state a weight for the weight to
# be read from them.
MIN_WEIGHED_SHARE = 0.25


def resnet50(torch_module: Any):
    """torchvision's ResNet-50 (v1.5, the stride on each block's 3x3), with its parameter names,
    as a module class of ``torch_module`` (imported lazily)."""
    nn = torch_module.nn

    class Bottleneck(nn.Module):
        def __init__(self, inplanes: int, planes: int, stride: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, planes * EXPANSION, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * EXPANSION)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = None
            if stride != 1 or inplanes != planes * EXPANSION:
                self.downsample = nn.Sequential(
                    nn.Conv2d(inplanes, planes * EXPANSION, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * EXPANSION),
                )

        def forward(self, x):
            identity = x if self.downsample is None else self.downsample(x)
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.relu(self.bn2(self.conv2(out)))
            return self.relu(self.bn3(self.conv3(out)) + identity)

    class ResNet50(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            inplanes = 64
            for index, (blocks, planes) in enumerate(LAYERS):
                layer = []
                for block in range(blocks):
                    # _make_layer: the first stage keeps its size, each later one halves it in its first block.
                    layer.append(Bottleneck(inplanes, planes, 2 if block == 0 and index > 0 else 1))
                    inplanes = planes * EXPANSION
                setattr(self, f"layer{index + 1}", nn.Sequential(*layer))
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512 * EXPANSION, OUTPUTS)

        def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            return self.fc(torch_module.flatten(self.avgpool(x), 1))

    return ResNet50


def font_weights(fonts: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Each font's weight by its output, and whether its name states one."""
    weights = [weight_of(str(font.get("path", ""))) for font in fonts]
    known = np.array([weight is not None for weight in weights], dtype=np.float32)
    return np.array([weight or 0 for weight in weights], dtype=np.float32), known


@register
class YuzuMarkerFontDetection(LetteringProvider):
    id: ClassVar[str] = "yuzumarker-font-detection"
    name: ClassVar[str] = "YuzuMarker Font Detection"
    description: ClassVar[str] = "Reads what kind of typeface Japanese, Chinese or Korean lettering is in, and how heavy."
    runtimes: ClassVar[tuple[str, ...]] = ("torch",)
    requires: ClassVar[dict[str, tuple[str, ...]]] = {"torch": ("safetensors",)}
    Settings: ClassVar[type[ProviderSettings]] = ProviderSettings

    def __init__(self, settings=None, runtimes=None, runtime=None):
        super().__init__(settings, runtimes, runtime)
        self._model = None
        self._device = None
        self._dtype = None
        self._kernels = None  # fonts x styles, one 1 a row
        self._weights = None
        self._weighed = None

    @classmethod
    def weight_sources(cls, runtime: Optional[str] = None) -> tuple[WeightSource, ...]:
        return (WEIGHTS, KERNELS, FONTS)

    def _load(self) -> None:
        import torch
        from safetensors.torch import load_file

        runtime = self.runtime_instance()
        weights, kernels_path, fonts_path = self.resolve_weights()
        kernels = json.loads(kernels_path.read_text(encoding="utf-8"))
        fonts = json.loads(fonts_path.read_text(encoding="utf-8"))
        if len(kernels) != FONT_COUNT or len(fonts) != FONT_COUNT:
            raise ModelLoadError(f"{self.id}: the kernel table has {len(kernels)} fonts and the font list {len(fonts)}, but the network names {FONT_COUNT}")
        unknown = sorted(set(kernels) - set(STYLES))
        if unknown:
            raise ModelLoadError(f"{self.id}: the kernel table names styles this version does not know: {unknown}")
        device, dtype = runtime.torch_device, runtime.dtype()
        model = resnet50(torch)()
        model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
        model = model.to(device=device, dtype=dtype).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        table = torch.zeros(FONT_COUNT, len(STYLES))
        table[torch.arange(FONT_COUNT), torch.tensor([STYLES.index(kernel) for kernel in kernels])] = 1.0
        font_weight, weighed = font_weights(fonts)
        self._model, self._device, self._dtype = model, device, dtype
        self._kernels = table.to(device)
        self._weights = torch.from_numpy(font_weight).to(device)
        self._weighed = torch.from_numpy(weighed).to(device)
        log.info("loaded YuzuMarker.FontDetection on %s (%s)", device, dtype)

    def _unload(self) -> None:
        from urakatsu.runtimes.torch_runtime import release_cache

        self._model = self._kernels = self._weights = self._weighed = None
        release_cache(self._device)

    def _read_fonts(self, crops: list[Image.Image]) -> list[FontReading]:
        import torch

        if self._model is None:
            raise ProcessingError(f"{self.id} is not loaded")
        readings: list[FontReading] = []
        for start in range(0, len(crops), BATCH):
            batch = []
            for crop in crops[start : start + BATCH]:
                if crop.width <= 0 or crop.height <= 0:
                    raise ProcessingError("cannot read the font of an empty crop")
                # torchvision's Resize on a PIL image is Pillow's bilinear filter.
                batch.append(np.asarray(crop.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.uint8))
            with torch.inference_mode():
                pixels = torch.from_numpy(np.stack(batch)).to(self._device).permute(0, 3, 1, 2).to(self._dtype) / 255.0
                fonts = self._model(pixels)[:, :FONT_COUNT].float().softmax(1)
                kernels = (fonts @ self._kernels).cpu().numpy()
                weighed = (fonts * self._weighed).sum(1)
                weight = ((fonts * self._weights).sum(1) / weighed.clamp_min(1e-6)).cpu().numpy()
                weighed = weighed.cpu().numpy()
            for row, value, share in zip(kernels, weight, weighed):
                readings.append(
                    FontReading(
                        kernels={style: float(probability) for style, probability in zip(STYLES, row) if probability > 0.0},
                        weight=float(value) if share >= MIN_WEIGHED_SHARE else None,
                    )
                )
        return readings
