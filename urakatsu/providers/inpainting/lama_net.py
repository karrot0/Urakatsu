"""Inference-only LaMa (Big-LaMa) FFC-ResNet generator.

Port of ``lama/model.rs`` (``FFCResNetGenerator`` and the
FFC building blocks) and ``config.rs`` (``FFCResNetGeneratorConfig``), which follow
https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/modules/ffc.py

Attribute and index names mirror the Rust ``nn::Path`` components, so
``lama-manga.safetensors`` (``model.<index>.*``) loads with ``strict=True``. Indices
follow the original ``nn.Sequential`` positions; positions without parameters (the
reflection pads, the concat tuple layer and the ReLUs) are applied in ``forward``.
Reflection padding that the Rust ``padding_mode: Reflect`` convolutions apply is done
with ``F.pad`` before each convolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FFCConfig:
    ratio_gin: float
    ratio_gout: float


@dataclass(frozen=True)
class FFCResNetGeneratorConfig:
    """``FFCResNetGeneratorConfig::default()`` (Big-LaMa)."""

    input_nc: int = 4
    output_nc: int = 3
    ngf: int = 64
    n_downsampling: int = 3
    n_blocks: int = 18
    max_features: int = 1024
    init_conv: FFCConfig = FFCConfig(0.0, 0.0)
    downsample_conv: FFCConfig = FFCConfig(0.0, 0.0)
    resnet_conv: FFCConfig = FFCConfig(0.75, 0.75)


def _pad(x: torch.Tensor, padding: int) -> torch.Tensor:
    return F.pad(x, [padding] * 4, mode="reflect") if padding else x


class FFCBlock(nn.Module):
    """``FFC``: the local/global split convolution (Rust ``FFC``)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int,
        padding: int,
        ratio_gin: float,
        ratio_gout: float,
    ):
        super().__init__()
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        self.padding = padding

        def conv(cin: int, cout: int) -> nn.Conv2d:
            # Padding is applied by reflection in forward (Rust padding_mode: Reflect).
            return nn.Conv2d(cin, cout, kernel, stride=stride, padding=0, bias=False)

        self.convl2l = conv(in_cl, out_cl) if in_cl != 0 and out_cl != 0 else None
        self.convl2g = conv(in_cl, out_cg) if in_cl != 0 and out_cg != 0 else None
        self.convg2l = conv(in_cg, out_cl) if in_cg != 0 and out_cl != 0 else None
        self.convg2g = (
            SpectralTransform(in_cg, out_cg, stride) if in_cg != 0 and out_cg != 0 else None
        )

    def forward(
        self, local: torch.Tensor, global_: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        local_from_local = self.convl2l(_pad(local, self.padding)) if self.convl2l is not None else None
        local_from_global = (
            self.convg2l(_pad(global_, self.padding))
            if self.convg2l is not None and global_ is not None
            else None
        )
        global_from_local = self.convl2g(_pad(local, self.padding)) if self.convl2g is not None else None
        global_from_global = (
            self.convg2g(global_)
            if self.convg2g is not None and global_ is not None
            else None
        )

        if local_from_local is not None and local_from_global is not None:
            out_local = local_from_local + local_from_global
        elif local_from_local is not None:
            out_local = local_from_local
        elif local_from_global is not None:
            out_local = local_from_global
        else:
            raise RuntimeError("FFC local output is empty")

        if global_from_local is not None and global_from_global is not None:
            out_global = global_from_local + global_from_global
        elif global_from_local is not None:
            out_global = global_from_local
        elif global_from_global is not None:
            out_global = global_from_global
        else:
            out_global = None
        return out_local, out_global


class SpectralTransform(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        hidden = out_channels // 2
        self.stride = stride
        # Rust conv1/0 and conv1/1: the 1x1 conv and its batch norm.
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
        )
        self.fu = FourierUnit(hidden, hidden)
        self.conv2 = nn.Conv2d(hidden, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 2:
            x = F.avg_pool2d(x, 2, 2, 0, ceil_mode=False, count_include_pad=True)
        x = F.relu(self.conv1[1](self.conv1[0](x)))
        output = self.fu(x)
        return self.conv2(x + output)


class FourierUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels * 2, out_channels * 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        model_dtype = x.dtype
        # The FFT boundary stays in FP32 (torch.fft does not take BF16); the learned
        # 1x1 convolution and batch norm run in the model dtype, as in the Rust port.
        ffted = torch.fft.rfftn(x.float(), dim=(-2, -1), norm="ortho")
        ffted = torch.stack([ffted.real, ffted.imag], dim=-1).permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.to(model_dtype)
        ffted = ffted.view(batch, -1, ffted.shape[3], ffted.shape[4])

        ffted = F.relu(self.bn(self.conv_layer(ffted)))
        ffted = ffted.view(batch, -1, 2, ffted.shape[2], ffted.shape[3])
        ffted = ffted.permute(0, 1, 3, 4, 2).contiguous().float()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        return torch.fft.irfftn(ffted, s=(height, width), dim=(-2, -1), norm="ortho").to(model_dtype)


class FFCBnAct(nn.Module):
    """``FFC_BN_ACT``: FFC followed by batch norm and ReLU per branch."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int,
        padding: int,
        ratio_gin: float,
        ratio_gout: float,
    ):
        super().__init__()
        global_channels = int(out_channels * ratio_gout)
        local_channels = out_channels - global_channels
        self.ffc = FFCBlock(in_channels, out_channels, kernel, stride, padding, ratio_gin, ratio_gout)
        self.bn_l = nn.BatchNorm2d(local_channels) if ratio_gout != 1.0 else None
        self.bn_g = nn.BatchNorm2d(global_channels) if ratio_gout != 0.0 else None
        self.act_l = ratio_gout != 1.0
        self.act_g = ratio_gout != 0.0

    def forward(
        self, local: torch.Tensor, global_: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        out_local, out_global = self.ffc(local, global_)
        if self.bn_l is not None:
            out_local = self.bn_l(out_local)
        if self.act_l:
            out_local = F.relu(out_local)
        if out_global is not None and self.bn_g is not None:
            out_global = self.bn_g(out_global)
            if self.act_g:
                out_global = F.relu(out_global)
        return out_local, out_global


class FFCResnetBlock(nn.Module):
    def __init__(self, channels: int, ratio_gin: float, ratio_gout: float):
        super().__init__()
        self.conv1 = FFCBnAct(channels, channels, 3, 1, 1, ratio_gin, ratio_gout)
        self.conv2 = FFCBnAct(channels, channels, 3, 1, 1, ratio_gin, ratio_gout)

    def forward(
        self, local: torch.Tensor, global_: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        id_local, id_global = local, global_
        out_local, out_global = self.conv2(*self.conv1(local, global_))
        out_local = out_local + id_local
        if out_global is not None and id_global is not None:
            out_global = out_global + id_global
        elif out_global is None and id_global is not None:
            out_global = id_global
        return out_local, out_global


class FFCResNetGenerator(nn.Module):
    """The generator; its state dict is LaMa's ``model.*`` (Rust ``FFCResNetGenerator``)."""

    def __init__(self, config: FFCResNetGeneratorConfig = FFCResNetGeneratorConfig()):
        super().__init__()
        self.config = config
        self.model = nn.ModuleDict()
        self.model["1"] = FFCBnAct(
            config.input_nc, config.ngf, 7, 1, 0, config.init_conv.ratio_gin, config.init_conv.ratio_gout
        )
        self.n_downsampling = config.n_downsampling
        for idx in range(config.n_downsampling):
            multiplier = 1 << idx
            ratio_gout = (
                config.resnet_conv.ratio_gin
                if idx + 1 == config.n_downsampling
                else config.downsample_conv.ratio_gout
            )
            self.model[str(idx + 2)] = FFCBnAct(
                min(config.max_features, config.ngf * multiplier),
                min(config.max_features, config.ngf * multiplier * 2),
                3,
                2,
                1,
                config.downsample_conv.ratio_gin,
                ratio_gout,
            )
        bottleneck = min(config.max_features, config.ngf * (1 << config.n_downsampling))
        for idx in range(config.n_blocks):
            self.model[str(2 + config.n_downsampling + idx)] = FFCResnetBlock(
                bottleneck, config.resnet_conv.ratio_gin, config.resnet_conv.ratio_gout
            )
        # Index 2 + n_downsampling + n_blocks is the concat-tuple layer (no parameters).
        first_upsample = 3 + config.n_downsampling + config.n_blocks
        for idx in range(config.n_downsampling):
            multiplier = 1 << (config.n_downsampling - idx)
            in_channels = min(config.max_features, config.ngf * multiplier)
            out_channels = min(config.max_features, config.ngf * multiplier // 2)
            index = first_upsample + idx * 3
            self.model[str(index)] = nn.ConvTranspose2d(
                in_channels, out_channels, 3, stride=2, padding=1, output_padding=1, bias=True
            )
            self.model[str(index + 1)] = nn.BatchNorm2d(out_channels)
            # index + 2 is the ReLU (no parameters).
        # index first_upsample + n_downsampling * 3 is the reflection pad (no parameters).
        final_index = first_upsample + config.n_downsampling * 3 + 1
        self.model[str(final_index)] = nn.Conv2d(config.ngf, config.output_nc, 7, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, [3, 3, 3, 3], mode="reflect")
        local, global_ = self.model["1"](x, None)
        for idx in range(self.n_downsampling):
            local, global_ = self.model[str(idx + 2)](local, global_)
        for idx in range(self.config.n_blocks):
            local, global_ = self.model[str(2 + self.n_downsampling + idx)](local, global_)
        if global_ is not None:
            x = torch.cat([local, global_], dim=1)
        else:
            x = local
        first_upsample = 3 + self.n_downsampling + self.config.n_blocks
        for idx in range(self.n_downsampling):
            index = first_upsample + idx * 3
            x = F.relu(self.model[str(index + 1)](self.model[str(index)](x)))
        x = F.pad(x, [3, 3, 3, 3], mode="reflect")
        return torch.sigmoid(self.model[str(first_upsample + self.n_downsampling * 3 + 1)](x))


def generator_forward(generator: FFCResNetGenerator, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``Model::forward``: composites the generator output with the known pixels."""
    image = image.to(next(generator.parameters()).dtype)
    mask = mask.to(image.dtype)
    inverse_mask = 1.0 - mask
    masked_image = image * inverse_mask
    predicted = generator(torch.cat([masked_image, mask], dim=1))
    return predicted * mask + inverse_mask * image
