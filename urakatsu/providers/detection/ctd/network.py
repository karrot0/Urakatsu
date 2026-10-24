"""Comic text detector network: YOLOv5 block backbone, U-Net text segmenter and DBNet line head.

Port of Koharu's ``comic_text_detector/model.rs`` (BallonsTranslator ``ctd/basemodel.py`` and
``yolov5/yolo.py``). Attribute names are the checkpoint key prefixes, so each part loads with
``load_state_dict(strict=True)`` and a key mismatch fails loudly:

- ``blk_det`` (``yolo-v5.safetensors``): keys ``model.<i>...``
- ``text_seg`` (``unet.safetensors``): keys ``down_conv1``, ``upconv0`` .. ``upconv6``
- ``text_det`` (``dbnet.safetensors``): keys ``upconv3``, ``upconv4``, ``conv``, ``binarize``, ``thresh``

As upstream does at inference, the YOLO neck and head (``model.10`` to ``model.24``) are
constructed so the checkpoint loads strictly, but ``YoloV5.forward`` does not run them: the
block predictions are discarded before the segmenter. Torch is imported only in this module,
which the provider imports inside ``_load``.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

SILU = "silu"
LEAKY = "leaky"  # LeakyReLU(0.1), as ``Activation::Leaky(0.1)``
RELU = "relu"
LEAKY_SLOPE = 0.1


def _activate(x: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == SILU:
        return F.silu(x)
    if activation == LEAKY:
        return F.leaky_relu(x, LEAKY_SLOPE)
    return F.relu(x)


class ConvBnAct(nn.Module):
    """``ConvBnAct``: bias-free conv, batch norm (inference statistics), activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int, padding: int, eps: float, activation: str):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _activate(self.bn(self.conv(x)), self.activation)


class Bottleneck(nn.Module):
    def __init__(self, channels: int, shortcut: bool, eps: float, activation: str):
        super().__init__()
        self.cv1 = ConvBnAct(channels, channels, 1, 1, 0, eps, activation)
        self.cv2 = ConvBnAct(channels, channels, 3, 1, 1, eps, activation)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.cv2(self.cv1(x))
        return x + output if self.add else output


class C3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, repeats: int, shortcut: bool, eps: float, activation: str):
        super().__init__()
        hidden = out_channels // 2
        self.cv1 = ConvBnAct(in_channels, hidden, 1, 1, 0, eps, activation)
        self.cv2 = ConvBnAct(in_channels, hidden, 1, 1, 0, eps, activation)
        self.cv3 = ConvBnAct(hidden * 2, out_channels, 1, 1, 0, eps, activation)
        self.m = nn.ModuleList(Bottleneck(hidden, shortcut, eps, activation) for _ in range(repeats))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.cv1(x)
        for block in self.m:
            y1 = block(y1)
        y2 = self.cv2(x)
        return self.cv3(torch.cat([y1, y2], 1))


class Sppf(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int):
        super().__init__()
        hidden = in_channels // 2
        self.cv1 = ConvBnAct(in_channels, hidden, 1, 1, 0, 1e-3, SILU)
        self.cv2 = ConvBnAct(hidden * 4, out_channels, 1, 1, 0, 1e-3, SILU)
        self.kernel = kernel

    def _pooled(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, self.kernel, stride=1, padding=self.kernel // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.cv1(x)
        x2 = self._pooled(x1)
        x3 = self._pooled(x2)
        x4 = self._pooled(x3)
        return self.cv2(torch.cat([x1, x2, x3, x4], 1))


class YoloHead(nn.Module):
    """Detection head at ``model.24``: three 1x1 convs (21 outputs) and the anchor table.

    Built for checkpoint compatibility; never executed (see the module docstring).
    """

    def __init__(self):
        super().__init__()
        self.m = nn.ModuleList([nn.Conv2d(128, 21, 1, bias=True), nn.Conv2d(256, 21, 1, bias=True), nn.Conv2d(512, 21, 1, bias=True)])
        self.register_buffer("anchors", torch.zeros(3, 3, 2))


class YoloV5(nn.Module):
    """``YoloV5``: backbone up to the SPPF, returning the five feature maps used by the U-Net.

    Returns ``[x1, x3, x5, x7, x9]`` at strides 4, 8, 16, 32, 32 (320, 160, 80, 40, 40 px for a
    1280 px input).
    """

    def __init__(self):
        super().__init__()
        eps = 1e-3
        self.model = nn.ModuleDict(
            {
                "0": ConvBnAct(3, 32, 6, 2, 2, eps, SILU),
                "1": ConvBnAct(32, 64, 3, 2, 1, eps, SILU),
                "2": C3(64, 64, 1, True, eps, SILU),
                "3": ConvBnAct(64, 128, 3, 2, 1, eps, SILU),
                "4": C3(128, 128, 2, True, eps, SILU),
                "5": ConvBnAct(128, 256, 3, 2, 1, eps, SILU),
                "6": C3(256, 256, 3, True, eps, SILU),
                "7": ConvBnAct(256, 512, 3, 2, 1, eps, SILU),
                "8": C3(512, 512, 1, True, eps, SILU),
                "9": Sppf(512, 512, 5),
                # Neck and head: loaded for strict compatibility, not executed at inference.
                "10": ConvBnAct(512, 256, 1, 1, 0, eps, SILU),
                "13": C3(512, 256, 1, False, eps, SILU),
                "14": ConvBnAct(256, 128, 1, 1, 0, eps, SILU),
                "17": C3(256, 128, 1, False, eps, SILU),
                "18": ConvBnAct(128, 128, 3, 2, 1, eps, SILU),
                "20": C3(256, 256, 1, False, eps, SILU),
                "21": ConvBnAct(256, 256, 3, 2, 1, eps, SILU),
                "23": C3(512, 512, 1, False, eps, SILU),
                "24": YoloHead(),
            }
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        m = self.model
        x0 = m["0"](x)
        x1 = m["1"](x0)
        x2 = m["2"](x1)
        x3 = m["3"](x2)
        x4 = m["4"](x3)
        x5 = m["5"](x4)
        x6 = m["6"](x5)
        x7 = m["7"](x6)
        x8 = m["8"](x7)
        x9 = m["9"](x8)
        return [x1, x3, x5, x7, x9]


class DoubleConvC3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.down = stride > 1
        self.conv = C3(in_channels, out_channels, 1, True, 1e-5, LEAKY)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down:
            x = F.avg_pool2d(x, 2, 2, 0)
        return self.conv(x)


class DoubleConvUpC3(nn.Module):
    """C3, then a stride-2 transposed conv and batch norm, then ReLU (keys ``conv.0`` to ``conv.2``)."""

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            C3(in_channels, mid_channels, 1, True, 1e-5, LEAKY),
            nn.ConvTranspose2d(mid_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(x))


class UnetHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_conv1 = DoubleConvC3(512, 512, 2)
        self.upconv0 = DoubleConvUpC3(512, 512, 256)
        self.upconv2 = DoubleConvUpC3(768, 512, 256)
        self.upconv3 = DoubleConvUpC3(512, 512, 256)
        self.upconv4 = DoubleConvUpC3(384, 256, 128)
        self.upconv5 = DoubleConvUpC3(192, 128, 64)
        self.upconv6 = nn.Sequential(nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=False))

    def forward(self, f160, f80, f40, f20, f3):
        """Returns ``(mask, (f80, f40, u40))``; ``mask`` is the sigmoid text mask at input resolution."""
        d10 = self.down_conv1(f3)
        u20 = self.upconv0(d10)
        u40 = self.upconv2(torch.cat([f20, u20], 1))
        u80 = self.upconv3(torch.cat([f40, u40], 1))
        u160 = self.upconv4(torch.cat([f80, u80], 1))
        u320 = self.upconv5(torch.cat([f160, u160], 1))
        mask = torch.sigmoid(self.upconv6(u320))
        return mask, (f80, f40, u40)


class _ProbabilityHead(nn.Sequential):
    """``BinarizeHead`` / ``ThreshHead``: conv 3x3 to 16, two stride-2 transposed convs, sigmoid.

    Child indices match the checkpoint keys: 0 conv, 1 bn, 3 deconv, 4 bn, 6 deconv.
    """

    def __init__(self, in_channels: int, conv_bias: bool):
        super().__init__(
            nn.Conv2d(in_channels, 16, 3, stride=1, padding=1, bias=conv_bias),
            nn.BatchNorm2d(16, eps=1e-5),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 2, stride=2, padding=0, output_padding=0, bias=True),
            nn.BatchNorm2d(16, eps=1e-5),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 2, stride=2, padding=0, output_padding=0, bias=True),
            nn.Sigmoid(),
        )


class DBHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.upconv3 = DoubleConvUpC3(512, 512, 256)
        self.upconv4 = DoubleConvUpC3(384, 256, 128)
        self.conv = nn.Sequential(nn.Conv2d(128, 64, 1, stride=1, padding=0, bias=True), nn.BatchNorm2d(64, eps=1e-5), nn.ReLU())
        self.binarize = _ProbabilityHead(64, conv_bias=True)
        self.thresh = _ProbabilityHead(64, conv_bias=False)

    def forward(self, f80, f40, u40) -> torch.Tensor:
        """Returns ``(N, 2, H, W)``: channel 0 the shrink map, channel 1 the threshold map."""
        u80 = self.upconv3(torch.cat([f40, u40], 1))
        x = self.upconv4(torch.cat([f80, u80], 1))
        x = self.conv(x)
        return torch.cat([self.binarize(x), self.thresh(x)], 1)


class ComicTextDetectorNetwork(nn.Module):
    """The three parts of ``model.rs::Model``; ``forward`` is ``Model::forward``."""

    def __init__(self):
        super().__init__()
        self.blk_det = YoloV5()
        self.text_seg = UnetHead()
        self.text_det = DBHead()

    def load_checkpoints(self, yolo_path: Path, unet_path: Path, dbnet_path: Path) -> None:
        """Strict load of each checkpoint into its part (``Model::load``)."""
        for module, path in ((self.blk_det, yolo_path), (self.text_seg, unet_path), (self.text_det, dbnet_path)):
            module.load_state_dict(load_file(str(path)), strict=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(N, 3, H, W)`` input in [0, 1] to ``(mask (N, 1, H, W), line_maps (N, 2, H, W))``."""
        x = x.to(next(self.parameters()).dtype)
        features = self.blk_det(x)
        mask, db_features = self.text_seg(features[0], features[1], features[2], features[3], features[4])
        line_maps = self.text_det(db_features[0], db_features[1], db_features[2])
        return mask, line_maps
