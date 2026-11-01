"""The Er0mangaSeg network in plain PyTorch: ConvNeXt-T under a UPerNet head.

Er0mangaSeg (https://github.com/Er0manga/Er0mangaSeg, Apache-2.0) trains this on mmsegmentation,
which is not a dependency here. Module and buffer names follow mmpretrain's ``ConvNeXt`` and
mmsegmentation's ``UPerHead``, so the checkpoint's ``state_dict`` loads with ``strict=True``. Only
inference is kept: stochastic depth and the head's dropout do nothing in eval mode, and the
auxiliary head is used only in training.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

DEPTHS = (3, 3, 9, 3)  # ConvNeXt-T
CHANNELS = (96, 192, 384, 768)
HEAD_CHANNELS = 512
POOL_SCALES = (1, 2, 3, 6)
CLASSES = 2


class LayerNorm2d(nn.LayerNorm):
    """mmpretrain's ``LN2d``: a LayerNorm over the channels of an NCHW tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.pointwise_conv1 = nn.Linear(channels, 4 * channels)
        self.pointwise_conv2 = nn.Linear(4 * channels, channels)
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise_conv(x).permute(0, 2, 3, 1)
        y = self.pointwise_conv2(F.gelu(self.pointwise_conv1(self.norm(y)))).permute(0, 3, 1, 2)
        return x + y.mul(self.gamma.view(1, -1, 1, 1))


class ConvNeXt(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.downsample_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(3, CHANNELS[0], kernel_size=4, stride=4), LayerNorm2d(CHANNELS[0], eps=1e-6))])
        for before, after in zip(CHANNELS, CHANNELS[1:]):
            self.downsample_layers.append(nn.Sequential(LayerNorm2d(before, eps=1e-6), nn.Conv2d(before, after, kernel_size=2, stride=2)))
        self.stages = nn.ModuleList(nn.Sequential(*(ConvNeXtBlock(channels) for _ in range(depth))) for depth, channels in zip(DEPTHS, CHANNELS))
        for index, channels in enumerate(CHANNELS):
            self.add_module(f"norm{index}", LayerNorm2d(channels, eps=1e-6))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for index, stage in enumerate(self.stages):
            x = stage(self.downsample_layers[index](x))
            features.append(getattr(self, f"norm{index}")(x))
        return features


class ConvModule(nn.Module):
    """mmcv's ``ConvModule`` with SyncBN and ReLU: the checkpoint's ``conv`` and ``bn`` names."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


def _resize(x: torch.Tensor, size: torch.Size) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class UPerHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.psp_modules = nn.ModuleList(nn.Sequential(nn.AdaptiveAvgPool2d(scale), ConvModule(CHANNELS[-1], HEAD_CHANNELS, 1)) for scale in POOL_SCALES)
        self.bottleneck = ConvModule(CHANNELS[-1] + len(POOL_SCALES) * HEAD_CHANNELS, HEAD_CHANNELS, 3, padding=1)
        self.lateral_convs = nn.ModuleList(ConvModule(channels, HEAD_CHANNELS, 1) for channels in CHANNELS[:-1])
        self.fpn_convs = nn.ModuleList(ConvModule(HEAD_CHANNELS, HEAD_CHANNELS, 3, padding=1) for _ in CHANNELS[:-1])
        self.fpn_bottleneck = ConvModule(len(CHANNELS) * HEAD_CHANNELS, HEAD_CHANNELS, 3, padding=1)
        self.conv_seg = nn.Conv2d(HEAD_CHANNELS, CLASSES, kernel_size=1)

    def _pyramid(self, x: torch.Tensor) -> torch.Tensor:
        pooled = [_resize(module(x), x.shape[2:]) for module in self.psp_modules]
        return self.bottleneck(torch.cat([x, *pooled], dim=1))

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        laterals = [conv(feature) for conv, feature in zip(self.lateral_convs, features)]
        laterals.append(self._pyramid(features[-1]))
        for index in range(len(laterals) - 1, 0, -1):
            laterals[index - 1] = laterals[index - 1] + _resize(laterals[index], laterals[index - 1].shape[2:])
        outputs = [conv(lateral) for conv, lateral in zip(self.fpn_convs, laterals)]
        outputs.append(laterals[-1])
        size = outputs[0].shape[2:]
        merged = torch.cat([outputs[0], *(_resize(output, size) for output in outputs[1:])], dim=1)
        return self.conv_seg(self.fpn_bottleneck(merged))


class Er0mangaSegNet(nn.Module):
    """``logits = net(image)``: two classes at a quarter of the input size."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = ConvNeXt()
        self.decode_head = UPerHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode_head(self.backbone(x))
