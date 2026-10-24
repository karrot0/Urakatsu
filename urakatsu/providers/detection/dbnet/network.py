"""DBNet with a ResNet34 backbone: the network behind manga-image-translator's default text detector.

Checkpoint: ``detect-20241225.ckpt`` (release ``beta-0.3``), a plain state dict (older checkpoints
nest it under ``"model"``; both load) with 437 entries (BatchNorm running statistics and ``num_batches_tracked`` included). Parameter names
here match it one to one, so ``load_network`` loads strictly. Module groups:

  backbone.*                    torchvision resnet34: conv1, bn1, layer1..layer4 and fc (fc is present
                                in the checkpoint but unused for detection)
  down_conv1..3.conv.*          three stride-2 blocks on the 512-channel C5 map
  upconv1..7.conv.*             up blocks (conv, conv, ConvTranspose x2); each takes the previous output
                                concatenated with a backbone skip, except upconv1
  conv_db.binarize.*            DB shrink (probability) map head, full input resolution
  conv_db.thresh.*              DB threshold map head, full input resolution, sigmoid output
  conv_mask.*                   text segmentation head, half input resolution, sigmoid output

Input is NCHW float in [-1, 1] (see ``preprocess.py``) with height and width multiples of 256, the
total stride of the network. Torch is imported by this module, so import it only from provider code
that runs after listing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet34

from urakatsu.core.errors import ModelLoadError

from .preprocess import Prepared


def _bn_relu(channels: int) -> list[nn.Module]:
    return [nn.BatchNorm2d(channels), nn.ReLU(inplace=True)]


class DownBlock(nn.Module):
    """AvgPool 2x, then three 3x3 conv + BN + ReLU stages at constant width (512). Keys: ``conv.0..8``."""

    def __init__(self, channels: int = 512):
        super().__init__()
        self.pool = nn.AvgPool2d(2, stride=2)  # no parameters
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), *_bn_relu(channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), *_bn_relu(channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), *_bn_relu(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Conv3x3 (``in_channels`` -> ``mid``), BN, ReLU, conv3x3 (``mid`` -> ``mid``), BN, ReLU,
    ConvTranspose 4x4 stride 2 (``mid`` -> ``out``), BN, ReLU. Keys: ``conv.0..8``; ConvTranspose has no bias."""

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False), *_bn_relu(mid_channels),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False), *_bn_relu(mid_channels),
            nn.ConvTranspose2d(mid_channels, out_channels, 4, stride=2, padding=1, bias=False),
            *_bn_relu(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DBHead(nn.Module):
    """Two branches on the 64-channel full-resolution feature: a shrink (probability) logit map and a
    threshold map. Both are upsampled to input resolution by two ConvTranspose stages. Eval output is
    ``cat([shrink_logits, threshold_sigmoid], dim=1)``."""

    def __init__(self, in_channels: int = 64):
        super().__init__()
        inner = in_channels // 4
        self.binarize = nn.Sequential(
            nn.Conv2d(in_channels, inner, 3, padding=1),
            *_bn_relu(inner),
            nn.ConvTranspose2d(inner, inner, 4, stride=2, padding=1),
            *_bn_relu(inner),
            nn.ConvTranspose2d(inner, 1, 4, stride=2, padding=1),
        )
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner, 3, padding=1, bias=False),
            *_bn_relu(inner),
            nn.ConvTranspose2d(inner, inner, 4, stride=2, padding=1),
            *_bn_relu(inner),
            nn.ConvTranspose2d(inner, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.binarize(x), self.thresh(x)], dim=1)


class DBNetResNet34(nn.Module):
    """The full text detector. ``forward`` returns ``(db, mask)``:

    * ``db``: ``(N, 2, H, W)``, input resolution. Channel 0 is the shrink logits (apply sigmoid for the
      probability map), channel 1 the sigmoid threshold map.
    * ``mask``: ``(N, 1, H/2, W/2)`` sigmoid text segmentation at half resolution.
    """

    def __init__(self):
        super().__init__()
        self.backbone = resnet34(weights=None)
        self.down_conv1 = DownBlock(512)
        self.down_conv2 = DownBlock(512)
        self.down_conv3 = DownBlock(512)
        self.upconv1 = UpBlock(512, 512, 256)
        self.upconv2 = UpBlock(256 + 512, 512, 256)
        self.upconv3 = UpBlock(256 + 512, 512, 256)
        self.upconv4 = UpBlock(256 + 512, 512, 256)
        self.upconv5 = UpBlock(256 + 256, 256, 128)
        self.upconv6 = UpBlock(128 + 128, 128, 64)
        self.upconv7 = UpBlock(64 + 64, 64, 64)
        self.conv_db = DBHead(64)
        self.conv_mask = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        h4 = b.layer1(x)      # 64 @ H/4
        h8 = b.layer2(h4)     # 128 @ H/8
        h16 = b.layer3(h8)    # 256 @ H/16
        h32 = b.layer4(h16)   # 512 @ H/32
        h64 = self.down_conv1(h32)    # 512 @ H/64
        h128 = self.down_conv2(h64)   # 512 @ H/128
        h256 = self.down_conv3(h128)  # 512 @ H/256

        u = self.upconv1(h256)                          # 256 @ H/128
        u = self.upconv2(torch.cat([u, h128], dim=1))   # 256 @ H/64
        u = self.upconv3(torch.cat([u, h64], dim=1))    # 256 @ H/32
        u = self.upconv4(torch.cat([u, h32], dim=1))    # 256 @ H/16
        u = self.upconv5(torch.cat([u, h16], dim=1))    # 128 @ H/8
        u8 = self.upconv6(torch.cat([u, h8], dim=1))    # 64 @ H/4
        u4 = self.upconv7(torch.cat([u8, h4], dim=1))   # 64 @ H/2
        return self.conv_db(u8), self.conv_mask(u4)


def load_network(weights_path: Union[str, Path], device: Union[str, torch.device], dtype: torch.dtype = torch.float32) -> DBNetResNet34:
    """Builds the network, loads the checkpoint strictly (no missing or unexpected keys), and returns it
    in eval mode on ``device`` with ``dtype``. The checkpoint is read with ``weights_only=True``."""
    path = Path(weights_path)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ModelLoadError(f"{path} could not be loaded with weights_only=True: {error}") from error
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    network = DBNetResNet34()
    try:
        network.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ModelLoadError(f"{path} does not match the DBNet ResNet34 layout: {error}") from error
    return network.to(device=device, dtype=dtype).eval()


@dataclass(frozen=True)
class Raw:
    """Network output for one page, with the geometry needed to map it back to the page.

    * ``db``: float32 ``(padded_h, padded_w)``, sigmoid of DB channel 0 (the probability map). It is at
      input resolution, in the resized-and-padded frame: pixel ``(x, y)`` is the same location as
      input pixel ``(x, y)`` of ``prepared.tensor``. The region ``[:resized_h, :resized_w]`` is the
      image; the rest is padding.
    * ``mask``: float32 ``(orig_h, orig_w)`` in [0, 1]. The mask head is upscaled x2 (bilinear), the
      padding is cropped off, and the result is resized to the original page size (bilinear). Unlike
      ``db``, this is already in original-image pixels.
    """

    db: np.ndarray
    mask: np.ndarray
    prepared: Prepared

    def db_to_original(self, points_xy: np.ndarray) -> np.ndarray:
        """Maps ``(N, 2)`` db-map points (x, y) to original-image pixels: divide by the resize ratio.
        Points beyond ``resized_width/height`` lie in the padding and have no original-image location."""
        return np.asarray(points_xy, dtype=np.float64) / self.prepared.ratio


@torch.inference_mode()
def forward(network: DBNetResNet34, prepared: Prepared, device: Union[str, torch.device], dtype: torch.dtype = torch.float32) -> Raw:
    """Runs the network on a prepared page and returns the db map and text mask (see ``Raw``)."""
    x = prepared.tensor.to(device=device, dtype=dtype)
    db_logits, mask_head = network(x)
    db = torch.sigmoid(db_logits[:, 0:1].float())[0, 0]
    mask = F.interpolate(mask_head.float(), scale_factor=2, mode="bilinear", align_corners=False)
    mask = mask[..., : prepared.resized_height, : prepared.resized_width]
    mask = F.interpolate(mask, size=(prepared.orig_height, prepared.orig_width), mode="bilinear", align_corners=False)
    mask = mask[0, 0].clamp(0.0, 1.0)
    return Raw(
        db=db.cpu().numpy().astype(np.float32),
        mask=mask.cpu().numpy().astype(np.float32),
        prepared=prepared,
    )
