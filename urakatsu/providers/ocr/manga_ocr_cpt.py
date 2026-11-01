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
from typing import Any, ClassVar, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from urakatsu.core.errors import ProcessingError
from urakatsu.core.registry import register
from urakatsu.providers.base import OcrProvider
from urakatsu.runtimes.store import HuggingFaceFile, WeightSource
from urakatsu.runtimes.torch_runtime import release_cache

from ._common import pillow_grayscale
from .manga_ocr import halfwidth_to_fullwidth

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


class Network:
    """model.py ``MangaOcrNet``: the architecture written directly on the checkpoint's tensors.

    On CUDA the encoder's matrix products run in fp16 with LayerNorm and the residual stream in
    fp32, the autocast it was trained and evaluated with. Everything else is fp32."""

    def __init__(self, weights: dict[str, Tensor], device: torch.device):
        self.device = device
        self.encoder_dtype = torch.float16 if device.type == "cuda" else torch.float32

        def tensor(name: str, dtype: torch.dtype = torch.float32) -> Tensor:
            return weights[name].to(device, dtype)

        def linear(prefix: str, dtype: torch.dtype = torch.float32) -> tuple[Tensor, Tensor]:
            return tensor(f"{prefix}.weight", dtype), tensor(f"{prefix}.bias", dtype)

        encoder_dtype = self.encoder_dtype
        self.patch = linear("enc.patch", encoder_dtype)
        self.cls = tensor("enc.cls")
        self.position = tensor("enc.pos")
        self.encoder_layers = [
            {
                **{name: linear(f"enc.{index}.{name}") for name in ("ln1", "ln2")},
                **{name: linear(f"enc.{index}.{name}", encoder_dtype) for name in ("q", "k", "v", "o", "f1", "f2")},
            }
            for index in range(12)
        ]
        self.encoder_norm = linear("enc.ln")
        self.word = tensor("dec.word")
        self.word_position = tensor("dec.pos")
        self.token_type = tensor("dec.type0")
        self.embedding_norm = linear("dec.ln")
        self.decoder_layers = [
            {
                name: linear(f"dec.{index}.{name}")
                for name in ("q", "k", "v", "o", "cq", "ck", "cv", "co", "f1", "f2", "ln1", "ln2", "ln3")
            }
            for index in range(2)
        ]
        self.head = linear("dec.head")
        self.head_norm = linear("dec.head_ln")
        self.output_bias = tensor("dec.out_bias")  # the output layer is tied to ``word``
        self._positions: dict[tuple[int, int], Tensor] = {}

    @staticmethod
    def _linear(x: Tensor, weights: tuple[Tensor, Tensor], dtype: torch.dtype) -> Tensor:
        return F.linear(x.to(dtype), weights[0], weights[1])

    @staticmethod
    def _norm(x: Tensor, weights: tuple[Tensor, Tensor]) -> Tensor:
        return F.layer_norm(x.float(), (HIDDEN,), weights[0], weights[1], EPSILON)

    @staticmethod
    def _heads(x: Tensor, batch: int) -> Tensor:
        return x.view(batch, -1, HEADS, HIDDEN // HEADS).transpose(1, 2)

    def positions(self, grid_height: int, grid_width: int) -> Tensor:
        """``[196, 768]`` position rows for a grid, with zeros on padded patches."""
        key = (grid_height, grid_width)
        if key not in self._positions:
            table = self.position[1:].reshape(14, 14, HIDDEN).permute(2, 0, 1)[None]
            resized = table if key == (14, 14) else F.interpolate(table, size=key, mode="bicubic", align_corners=False)
            rows = torch.zeros(PATCH_BUDGET, HIDDEN, device=self.device, dtype=self.encoder_dtype)
            rows[: grid_height * grid_width] = resized[0].flatten(1).T.to(self.encoder_dtype)
            self._positions[key] = rows
        return self._positions[key]

    def encode(self, patches: Tensor, valid: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        """The ``[B, 197, 768]`` fp32 memory and its ``[B, 197]`` mask."""
        dtype = self.encoder_dtype
        batch = patches.shape[0]
        embedded = self._linear(patches, self.patch, dtype) + positions
        hidden = torch.cat([(self.cls + self.position[0]).expand(batch, 1, HIDDEN), embedded.float()], 1)
        mask = torch.cat([torch.ones_like(valid[:, :1]), valid], 1)
        attention_mask = mask[:, None, None, :]
        for layer in self.encoder_layers:
            normed = self._norm(hidden, layer["ln1"])
            query, key, value = (self._heads(self._linear(normed, layer[name], dtype), batch) for name in "qkv")
            attended = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
            hidden = hidden + self._linear(attended.transpose(1, 2).reshape(batch, -1, HIDDEN), layer["o"], dtype)
            expanded = F.gelu(self._linear(self._norm(hidden, layer["ln2"]), layer["f1"], dtype))
            hidden = hidden + self._linear(expanded, layer["f2"], dtype)
        return self._norm(hidden, self.encoder_norm), mask

    def _step(self, token: Tensor, position: Tensor, keys, values, cross, memory_mask: Tensor, live: Tensor) -> Tensor:
        """One decoder step: the ``[B]`` tokens being fed at ``position`` give the next ``[B]`` ids."""
        dtype = torch.float32
        batch = token.shape[0]
        x = self._norm(self.word[token][:, None] + self.word_position[position][None] + self.token_type, self.embedding_norm)
        for index, layer in enumerate(self.decoder_layers):
            query, key, value = (self._heads(self._linear(x, layer[name], dtype), batch) for name in "qkv")
            keys[index].index_copy_(2, position, key)
            values[index].index_copy_(2, position, value)
            attended = F.scaled_dot_product_attention(query, keys[index], values[index], attn_mask=live)
            x = self._norm(self._linear(attended.transpose(1, 2).reshape(batch, 1, HIDDEN), layer["o"], dtype) + x, layer["ln1"])
            query = self._heads(self._linear(x, layer["cq"], dtype), batch)
            attended = F.scaled_dot_product_attention(query, cross[index][0], cross[index][1], attn_mask=memory_mask)
            x = self._norm(self._linear(attended.transpose(1, 2).reshape(batch, 1, HIDDEN), layer["co"], dtype) + x, layer["ln2"])
            x = self._norm(self._linear(F.gelu(self._linear(x, layer["f1"], dtype)), layer["f2"], dtype) + x, layer["ln3"])
        head = self._norm(F.gelu(self._linear(x, self.head, dtype)), self.head_norm)
        return self._linear(head[:, 0], (self.word, self.output_bias), dtype).argmax(-1)

    @torch.inference_mode()
    def greedy(self, patches: Tensor, valid: Tensor, positions: Tensor, max_tokens: int) -> list[list[int]]:
        memory, mask = self.encode(patches, valid, positions)
        batch = patches.shape[0]
        cross = [
            tuple(self._heads(self._linear(memory, layer[name], torch.float32), batch) for name in ("ck", "cv"))
            for layer in self.decoder_layers
        ]

        def cache() -> list[Tensor]:
            return [torch.zeros(batch, HEADS, max_tokens + 1, HIDDEN // HEADS, device=self.device) for _ in self.decoder_layers]

        keys, values = cache(), cache()
        slots = torch.arange(max_tokens + 1, device=self.device)
        token = torch.full((batch,), START, dtype=torch.long, device=self.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=self.device)
        tokens = []
        for step in range(max_tokens):
            position = torch.tensor([step], device=self.device)
            token = self._step(token, position, keys, values, cross, mask[:, None, None, :], (slots <= position)[None, None, None, :])
            tokens.append(token)
            finished |= token == END
            if bool(finished.all()):
                break
        return torch.stack(tokens, 1).tolist()


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
        items = [preprocess(crop, self._fit) for crop in crops]
        patches = torch.from_numpy(np.stack([item[0] for item in items])).to(self._device)
        valid = torch.from_numpy(np.stack([item[1] for item in items])).to(self._device)
        positions = torch.stack([self._network.positions(item[2], item[3]) for item in items])
        token_ids = self._network.greedy(patches, valid, positions, self._max_tokens)
        return [decode(ids, self._characters) for ids in token_ids]
