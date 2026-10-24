"""Helpers shared by the OCR providers.

Each provider keeps its own preprocessing, decoding and post-processing; only
the pieces that several model modules implement identically live here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
from PIL import Image

if TYPE_CHECKING:  # pragma: no cover
    import torch

# Rust ``char::is_whitespace`` (Unicode White_Space). Python's ``str.isspace`` also
# accepts U+001C..U+001F
_RUST_WHITESPACE = frozenset(
    "\t\n\x0b\x0c\r \x85\xa0   　"
    + "".join(chr(c) for c in range(0x2000, 0x200B))
)


def is_whitespace(character: str) -> bool:
    return character in _RUST_WHITESPACE


def read_lines(path: Path) -> list[str]:
    """Rust ``str::lines``: splits on ``\\n`` only, strips one trailing ``\\r``, no final empty line."""
    text = Path(path).read_text(encoding="utf-8")
    lines = [line[:-1] if line.endswith("\r") else line for line in text.split("\n")]
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def pillow_grayscale(image: Image.Image) -> Image.Image:
    """Pillow's ITU-R 601-2 luma (``(19595 R + 38470 G + 7471 B + 32768) >> 16``) replicated to RGB.

    Pillow's ``L`` conversion uses the same integer formula that ``pillow_grayscale`` in
    ``manga_ocr/processor.rs`` reproduces, so the result matches upstream byte for byte.
    """
    return image.convert("L").convert("RGB")


def to_rgb(image: Image.Image) -> Image.Image:
    return image if image.mode == "RGB" else image.convert("RGB")


def raw_chw(image: Image.Image, device: "torch.device") -> "torch.Tensor":
    """``[1, 3, H, W]`` float32 tensor holding the raw 0..255 RGB values."""
    import torch

    array = np.asarray(to_rgb(image), dtype=np.uint8)
    return torch.from_numpy(array.copy()).to(device).permute(2, 0, 1)[None].to(torch.float32)


def normalize(pixels: "torch.Tensor", mean: Sequence[float], std: Sequence[float]) -> "torch.Tensor":
    """``(x - mean) / std`` per channel, in the tensor's dtype (float32 in the processors)."""
    import torch

    device = pixels.device
    mean_t = torch.tensor(mean, dtype=pixels.dtype, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=pixels.dtype, device=device).view(1, 3, 1, 1)
    return (pixels - mean_t) / std_t


def first_weights_dir(paths: Sequence[Path]) -> Path:
    """The model store keeps one repository revision in one directory; every file shares it."""
    directories = {Path(p).parent for p in paths}
    if len(directories) != 1:
        raise ValueError(f"weights are not in one directory: {sorted(map(str, directories))}")
    return directories.pop()
