"""Charspot OCR: finds and classifies each character, then orders them. Not in Koharu.

Upstream sources (muscgab/manga-ocr-charspot-small @ REVISION):
- ``manga_ocr_charspot_small/core.py``: ``load_vocab``, ``pad_grid`` and ``preprocess``.
- ``manga_ocr_charspot_small/backends/onnx.py``: ``read_one``.

There is no text decoder. ``model.onnx`` holds everything after preprocessing (dense network,
box decoding, NMS, furigana removal, classification, reading order) and returns ``boxes``,
``scores``, ``ids`` and ``order``, so one session call reads one crop.

The catalog can swap in ``manga-ocr-charspot-nano``: it differs from the small model only in its
cell budget and weights, and ``config.json`` carries the budget.

``model.onnx`` always runs on CPU, like upstream's ONNX backend. On CUDA, ONNX Runtime warns that
the graph's ScatterND is only correct without duplicate indices, and its text did differ from the
CPU's on 6 of 200 crops, without being faster. Upstream's GPU path is a separate PyTorch port of
the same model, which this module does not use.

Divergences:
- The per-character boxes and scores are dropped: the OCR stage only keeps text.

The weights are CC BY-NC-SA 4.0 (trained on AnimeText-OCR), so commercial use needs separate
permission from their author. The upstream code is Apache-2.0.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np
from PIL import Image

from urakatsu.core.errors import ProcessingError
from urakatsu.core.registry import register
from urakatsu.providers.base import OcrProvider
from urakatsu.runtimes.store import HuggingFaceFile, WeightSource

from ._common import to_rgb

REPOSITORY = "muscgab/manga-ocr-charspot-small"
REVISION = "72ca29202673880ca9c67b1c02c107535be4472b"

# core.py: the grid search runs on 14-px cells and is scaled to the model's cells afterwards.
SEARCH_CELL = 14
# core.py ``COMBOS``: the tate-chu-yoko ``!?`` glyphs, the outputs after the ``n_cls`` characters.
COMBOS = [a + b for a in "!?" for b in "!?"] + [a + b + c for a in "!?" for b in "!?" for c in "!?"]


def load_vocabulary(path: Path, classes: int) -> list[str]:
    """core.py ``load_vocab``: characters, empty padding up to ``classes``, then ``COMBOS``."""
    characters = json.loads(Path(path).read_text(encoding="utf-8"))["characters"]
    return characters + [""] * (classes - len(characters)) + COMBOS


def pad_grid(height: int, width: int, budget: int, cell: int) -> tuple[float, int, int]:
    """core.py ``pad_grid``: the largest scale whose grid fits ``budget`` cells, and that grid."""
    low, high = 1e-4, 1000.0
    for _ in range(60):
        middle = (low + high) / 2
        if math.ceil(height * middle / SEARCH_CELL) * math.ceil(width * middle / SEARCH_CELL) <= budget:
            low = middle
        else:
            high = middle
    return low * cell / SEARCH_CELL, math.ceil(height * low / SEARCH_CELL), math.ceil(width * low / SEARCH_CELL)


def preprocess(image: Image.Image, budget: int, cell: int) -> tuple[np.ndarray, tuple[float, float]]:
    """core.py ``preprocess``: bicubic resize, then the median border colour on the bottom and right.

    Returns the ``[3, H, W]`` pixels and the x and y factors from them back to ``image``."""
    scale, grid_height, grid_width = pad_grid(image.height, image.width, budget, cell)
    width = max(1, min(grid_width * cell, round(image.width * scale)))
    height = max(1, min(grid_height * cell, round(image.height * scale)))
    small = np.asarray(image.resize((width, height), Image.Resampling.BICUBIC), np.uint8)
    border = np.concatenate([small[0], small[-1], small[:, 0], small[:, -1]])
    pixels = np.empty((grid_height * cell, grid_width * cell, 3), np.uint8)
    pixels[:] = np.median(border, axis=0).astype(np.uint8)
    pixels[:height, :width] = small
    return pixels.transpose(2, 0, 1), (image.width / width, image.height / height)


@register
class CharspotOcrProvider(OcrProvider):
    id = "charspot-ocr"
    name = "Charspot OCR"
    description = "Fast Japanese manga OCR that reads each character on its own and skips furigana. Non-commercial weights."
    runtimes = ("onnx",)

    catalog_queries: ClassVar[tuple[str, ...]] = ("manga-ocr-charspot",)

    @classmethod
    def accepts(cls, facts) -> Optional[str]:
        """The manga-ocr-charspot family: one ONNX graph beside its config and vocabulary."""
        if facts.runtime != "onnx" or "charspot" not in facts.repo.lower():
            return None
        return "onnx" if all(facts.has_file(name) for name in ("config.json", "vocab.json", "model.onnx")) else None

    def __init__(self, settings: Any = None, runtimes=None, runtime: Optional[str] = None):
        super().__init__(settings, runtimes, runtime)
        self._session = None

    @classmethod
    def weight_sources(cls, runtime: Optional[str] = None) -> tuple[WeightSource, ...]:
        return tuple(HuggingFaceFile(REPOSITORY, name, REVISION) for name in ("config.json", "vocab.json", "model.onnx"))

    def _load(self) -> None:
        config_path, vocabulary_path, weights_path = self.resolve_weights()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self._cell = int(config["input"]["cell"])
        self._budget = int(config["input"]["budget_cells"])
        self._vocabulary = load_vocabulary(vocabulary_path, int(config["n_cls"]))
        # backends/onnx.py: at most 8 threads, one inter-op thread.
        self._session = self.runtime_instance().session(
            weights_path, cpu=True, intra_op_num_threads=min(8, os.cpu_count() or 1), inter_op_num_threads=1
        )

    def _unload(self) -> None:
        self._session = None

    def _recognize(self, crops: list[Image.Image]) -> list[str]:
        if self._session is None:
            raise ProcessingError(f"{self.id} is not loaded")
        return [self._read(crop) for crop in crops]

    def _read(self, image: Image.Image) -> str:
        """backends/onnx.py ``read_one``, keeping the text. A crop without characters gives ``""``."""
        pixels, _ = preprocess(to_rgb(image), self._budget, self._cell)
        ids, order = self._session.run(["ids", "order"], {"img": pixels[None].astype(np.float32)})
        return "".join(self._vocabulary[int(ids[index])] for index in order.tolist())
