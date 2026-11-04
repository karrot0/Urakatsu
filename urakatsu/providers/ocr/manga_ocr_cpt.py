"""Manga OCR CPT: Manga OCR continued on new data, with a 6,817 character vocabulary. Not in Koharu.

Upstream sources (muscgab/manga-ocr-CPT @ 2f6ceef):
- ``manga_ocr_cpt/preprocess.py``: ``grid`` and ``preprocess``.
- ``manga_ocr_cpt/model.py``: ``MangaOcrNet``.
- ``manga_ocr_cpt/__init__.py``: batching and ``_text``.

The architecture is Manga OCR's (ViT-B/16 encoder, 2-layer BERT decoder), but the checkpoint uses
upstream's own tensor names and takes input ``manga_ocr.py`` cannot give it: with ``input.fit``
"aspect" a crop keeps its aspect ratio, in at most 196 patches, so the 14 x 14 position table is
resized to the patch grid and padded patches are masked out of attention. Decoding is greedy.

The catalog can swap in ``manga-ocr-base-CPT``: the same network with ``input.fit`` "square", the
original 224 x 224 stretch.

Divergences:
- Only upstream's plain PyTorch path runs, not its CUDA graphs. Their text matched on all 245 boxes
  of a test chapter; the graphs took 7 ms per crop against 13, but 1.7 GB of VRAM against 0.4.
- Half-width characters become full width, as in ``manga_ocr.py``: upstream returns ``!`` and ``(``
  as the model spells them.
- Upstream's decoder="fp16" option is not ported: the evaluated outputs are fp32.

The weights are CC BY-NC-SA 4.0 (trained on AnimeText-OCR), so commercial use needs separate
permission from their author. That is why the provider is local only (``remotable``). The upstream
code is Apache-2.0.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional

import numpy as np
from PIL import Image

from urakatsu.core.errors import ProcessingError
from urakatsu.core.registry import register
from urakatsu.providers.base import OcrProvider
from urakatsu.runtimes.store import HuggingFaceFile, WeightSource
from urakatsu.runtimes.torch_runtime import release_cache

from ._common import pillow_grayscale
from .manga_ocr import halfwidth_to_fullwidth

if TYPE_CHECKING:  # pragma: no cover
    from .manga_ocr_cpt_network import Network  # torch: imported on load, not when the app lists providers

REPOSITORY = "muscgab/manga-ocr-base-CPT-flex"
REVISION = "9cc15742f8dc37cb5222a6569839bc774dfc0392"
VOCABULARY = "nar_vocabulary_6817.json"

PATCH_BUDGET = 196
PATCH = 16
PATCH_FEATURES = 768
HIDDEN = 768
HEADS = 12
EPSILON = 1e-12
# Token ids: 0 pad, 1 unk, 2 start, 3 end, 4 mask, 5 + i = character i.
START = 2
END = 3
FIRST_CHARACTER = 5
# __init__.py PLAIN_BATCH.
BATCH = 16


def grid(height: int, width: int, budget: int = PATCH_BUDGET) -> tuple[float, int, int]:
    """preprocess.py ``grid``: the largest scale whose patch grid fits ``budget``, and that grid."""
    low, high = 1e-4, 1000.0
    for _ in range(60):
        middle = (low + high) / 2
        if math.ceil(height * middle / PATCH) * math.ceil(width * middle / PATCH) <= budget:
            low = middle
        else:
            high = middle
    return low, math.ceil(height * low / PATCH), math.ceil(width * low / PATCH)


def _patches(pixels: np.ndarray, grid_height: int, grid_width: int) -> np.ndarray:
    return pixels.reshape(grid_height, PATCH, grid_width, PATCH, 3).transpose(0, 2, 4, 1, 3).reshape(grid_height * grid_width, -1)


def preprocess(image: Image.Image, fit: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    """preprocess.py ``preprocess``: ``[196, 768]`` patches, which of them are real, and the grid."""
    image = pillow_grayscale(image)
    if fit == "square":
        small = np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR), np.uint8)
        return _patches((small.astype(np.float32) / 255.0 - 0.5) / 0.5, 14, 14).copy(), np.ones(PATCH_BUDGET, bool), 14, 14
    if fit != "aspect":
        raise ValueError(f"unsupported Manga OCR CPT input fit {fit!r}")
    scale, grid_height, grid_width = grid(image.height, image.width)
    width = max(1, min(grid_width * PATCH, round(image.width * scale)))
    height = max(1, min(grid_height * PATCH, round(image.height * scale)))
    small = np.asarray(image.resize((width, height), Image.Resampling.BILINEAR), np.uint8)
    # Placed at the top left, the rest filled with the median of the resized image's border.
    border = np.concatenate([small[0], small[-1], small[:, 0], small[:, -1]])
    canvas = np.empty((grid_height * PATCH, grid_width * PATCH, 3), np.uint8)
    canvas[:] = np.median(border, axis=0).astype(np.uint8)
    canvas[:height, :width] = small
    count = grid_height * grid_width
    patches = np.zeros((PATCH_BUDGET, PATCH_FEATURES), np.float32)
    patches[:count] = _patches((canvas.astype(np.float32) / 255.0 - 0.5) / 0.5, grid_height, grid_width)
    valid = np.zeros(PATCH_BUDGET, bool)
    valid[:count] = True
    return patches, valid, grid_height, grid_width


def decode(token_ids: list[int], characters: list[str]) -> str:
    """__init__.py ``_text``: characters up to the end token, in full width."""
    if END in token_ids:
        token_ids = token_ids[: token_ids.index(END)]
    return halfwidth_to_fullwidth("".join(characters[token - FIRST_CHARACTER] for token in token_ids if token >= FIRST_CHARACTER))


@register
class MangaOcrCptProvider(OcrProvider):
    id = "manga-ocr-cpt"
    name = "Manga OCR CPT"
    description = "Retrained Manga OCR that leaves furigana out of the text. Japanese only. Non-commercial weights."
    runtimes = ("torch",)
    requires = {"torch": ("safetensors",)}
    remotable = False

    catalog_queries: ClassVar[tuple[str, ...]] = ("manga-ocr-base-CPT",)

    @classmethod
    def accepts(cls, facts) -> Optional[str]:
        """The manga-ocr-base-CPT family: its weights beside its config and vocabulary."""
        repo = facts.repo.lower()
        if facts.runtime != "torch" or "manga-ocr" not in repo or "cpt" not in repo:
            return None
        return "torch" if all(facts.has_file(name) for name in ("config.json", VOCABULARY, "model.safetensors")) else None

    def __init__(self, settings: Any = None, runtimes=None, runtime: Optional[str] = None):
        super().__init__(settings, runtimes, runtime)
        self._network: Optional[Network] = None

    @classmethod
    def weight_sources(cls, runtime: Optional[str] = None) -> tuple[WeightSource, ...]:
        return tuple(HuggingFaceFile(REPOSITORY, name, REVISION) for name in ("config.json", VOCABULARY, "model.safetensors"))

    def _load(self) -> None:
        from safetensors.torch import load_file

        from .manga_ocr_cpt_network import Network

        config_path, vocabulary_path, weights_path = self.resolve_weights()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self._fit = config["input"]["fit"]
        self._max_tokens = int(config["max_new_tokens"])
        self._characters = json.loads(Path(vocabulary_path).read_text(encoding="utf-8"))["characters"]
        if len(self._characters) != int(config["vocab"]):
            raise ValueError(f"Manga OCR CPT vocabulary has {len(self._characters)} entries but the config says {config['vocab']}")
        self._device = self.runtime_instance().torch_device
        self._network = Network(load_file(str(weights_path)), self._device)

    def _unload(self) -> None:
        self._network = None
        release_cache(self._device)

    def _recognize(self, crops: list[Image.Image]) -> list[str]:
        if self._network is None:
            raise ProcessingError(f"{self.id} is not loaded")
        texts: list[str] = []
        for start in range(0, len(crops), BATCH):
            texts += self._read(crops[start : start + BATCH])
        return texts

    def _read(self, crops: list[Image.Image]) -> list[str]:
        """__init__.py ``ids``, the plain path: one batch of crops."""
        import torch

        items = [preprocess(crop, self._fit) for crop in crops]
        patches = torch.from_numpy(np.stack([item[0] for item in items])).to(self._device)
        valid = torch.from_numpy(np.stack([item[1] for item in items])).to(self._device)
        positions = torch.stack([self._network.positions(item[2], item[3]) for item in items])
        token_ids = self._network.greedy(patches, valid, positions, self._max_tokens)
        return [decode(ids, self._characters) for ids in token_ids]
