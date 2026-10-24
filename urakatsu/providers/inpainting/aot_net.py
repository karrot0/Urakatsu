"""Inference-only AOT generator (BallonsTranslator AOTGenerator).

Port of ``aot_inpainting/model.rs``, which follows
https://github.com/dmMaze/BallonsTranslator/blob/4bcc635c19f6c63a902872cf77b3d554e14ed1b7/ballontranslator/modules/inpaint/aot.py#L21-L262

Scaled weight-standardized convolutions with gating, reflection-padded convolutions,
and the AOT residual blocks. Attribute names and indices mirror the Rust ``nn::Path``
components so ``aot-inpainting/model.safetensors`` loads with ``strict=True``
(``head.0``, ``head.2``, ``head.4``, ``body_conv.N``, ``tail.0`` ... ``tail.8``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

RELU_NF_SCALE = 1.7139588594436646
WEIGHT_STANDARDIZATION_EPS = 1e-4
GATE_SCALE = 1.8


def standardized_weight(weight: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    variance, mean = torch.var_mean(weight, dim=(1, 2, 3), keepdim=True, correction=1)
    fan_in = float(weight[0].numel())
    scale = torch.rsqrt(torch.clamp(variance * fan_in, min=WEIGHT_STANDARDIZATION_EPS)) * gain
    return weight * scale - mean * scale


def relu_nf(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x) * RELU_NF_SCALE


def my_layer_norm(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = torch.std(x, dim=(2, 3), keepdim=True, correction=1) + 1e-9
    return ((x - mean) * 2.0 / std - 1.0) * 5.0


class ScaledWSConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int, dilation: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel, kernel))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.gain = nn.Parameter(torch.ones(out_channels, 1, 1, 1))
        self.stride = stride
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            standardized_weight(self.weight, self.gain),
            self.bias,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
        )


class ScaledWSTransposeConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(in_channels, out_channels, kernel, kernel))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.gain = nn.Parameter(torch.ones(in_channels, 1, 1, 1))
        self.stride = stride
        self.padding = (kernel - 1) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv_transpose2d(
            x,
            standardized_weight(self.weight, self.gain),
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=0,
            groups=1,
            dilation=1,
        )


class GatedWSConvPadded(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int, dilation: int):
        super().__init__()
        self.conv = ScaledWSConv2d(in_channels, out_channels, kernel, stride, dilation)
        self.conv_gate = ScaledWSConv2d(in_channels, out_channels, kernel, stride, dilation)
        self.padding = ((kernel - 1) * dilation) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding:
            x = F.pad(x, [self.padding] * 4, mode="reflect")
        return self.conv(x) * torch.sigmoid(self.conv_gate(x)) * GATE_SCALE


class GatedWSTransposeConvPadded(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int):
        super().__init__()
        self.conv = ScaledWSTransposeConv2d(in_channels, out_channels, kernel, stride)
        self.conv_gate = ScaledWSTransposeConv2d(in_channels, out_channels, kernel, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) * torch.sigmoid(self.conv_gate(x)) * GATE_SCALE


class AOTBlock(nn.Module):
    """Four dilated reflection-padded branches fused and gated against the input."""

    DILATIONS = (2, 4, 8, 16)

    def __init__(self, channels: int):
        super().__init__()
        # Rust path ``block{index:02} / 1``: index 0 is the reflection pad, index 1 the conv.
        self.block_names = []
        for index, dilation in enumerate(self.DILATIONS):
            name = f"block{index:02d}"
            setattr(
                self,
                name,
                nn.Sequential(
                    nn.ReflectionPad2d(dilation),
                    nn.Conv2d(channels, channels // 4, 3, dilation=dilation),
                ),
            )
            self.block_names.append(name)
        self.fuse = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3))
        self.gate = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [F.relu(getattr(self, name)(x)) for name in self.block_names]
        fused = self.fuse(torch.cat(branches, dim=1))
        mask = torch.sigmoid(my_layer_norm(self.gate(x)))
        return x * (1.0 - mask) + fused * mask


class AOTGenerator(nn.Module):
    """``AOTGenerator``: 4-channel (mask, masked image) input, 3-channel output in [-1, 1]."""

    def __init__(self, channels: int = 128, body_blocks: int = 10):
        super().__init__()
        self.head = nn.ModuleDict(
            {
                "0": GatedWSConvPadded(4, 32, 3, 1, 1),
                "2": GatedWSConvPadded(32, 64, 4, 2, 1),
                "4": GatedWSConvPadded(64, 128, 4, 2, 1),
            }
        )
        self.body_conv = nn.ModuleList([AOTBlock(channels) for _ in range(body_blocks)])
        self.tail = nn.ModuleDict(
            {
                "0": GatedWSConvPadded(128, 128, 3, 1, 1),
                "2": GatedWSConvPadded(128, 128, 3, 1, 1),
                "4": GatedWSTransposeConvPadded(128, 64, 4, 2),
                "6": GatedWSTransposeConvPadded(64, 32, 4, 2),
                "8": GatedWSConvPadded(32, 3, 3, 1, 1),
            }
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``Model::forward``: image and mask are already in the model dtype."""
        dtype = next(self.parameters()).dtype
        image = image.to(dtype)
        mask = mask.to(dtype)
        x = torch.cat([mask, image], dim=1)
        x = relu_nf(self.head["0"](x))
        x = relu_nf(self.head["2"](x))
        x = self.head["4"](x)
        for block in self.body_conv:
            x = block(x)
        x = relu_nf(self.tail["0"](x))
        x = relu_nf(self.tail["2"](x))
        x = relu_nf(self.tail["4"](x))
        x = relu_nf(self.tail["6"](x))
        return self.tail["8"](x).clamp(-1.0, 1.0)
