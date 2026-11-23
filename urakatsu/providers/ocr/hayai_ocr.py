"""Hayai OCR v2.7 Supernova: a SigLIP2 NaFlex encoder and a GQA decoder, greedy.

Upstream: ``modeling_hayai.py`` and ``processing_hayai.py`` at REVISION, and Koharu's port of it
(``crates/koharu-ml/src/hayai_ocr``, pinned at the same commit), which this one follows:
- ``mod.rs``: load order and ``inference``.
- ``config.rs``: SigLIP2 and decoder constants and validation.
- ``processor.rs``: the per-crop patch budget (``max_patches="auto"``), the NaFlex size search,
  bilinear resize and patchify order, and the decode filter.
- ``model.rs``: ``generate`` (greedy, 128 tokens), the ``DSCProjector`` that merges 2x2 patch
  blocks and reads them in both writing directions, ``StaticDepthFusion`` of every encoder depth,
  2D visual RoPE, text RoPE from ``TEXT_ROPE_OFFSET``, and the block-causal prefill mask.

Upstream's decoder attribute names are kept so the checkpoint loads strictly. v2 and v2.5
checkpoints have another projector and no depth fusion, so they are refused (as in Koharu, which
replaced v2 with v2.7).

Divergences:
- The vision encoder is transformers' ``Siglip2VisionModel`` (default SigLIP2 base config, which
  the released NaFlex config only overrides by model type), driven layer by layer to keep each
  layer's input for depth fusion. Its pooled head is loaded but never run, as upstream.
- Each crop is read alone. Koharu and upstream pad a batch to its longest crop and mask the
  padding, which leaves every crop's result the same.
- Upstream's optional ``escalate`` pass (re-reading dense crops at a larger budget) is off by
  default there and not ported, as in Koharu. Beam search and the repetition penalty are ignored
  upstream (``generate`` decodes greedily), so they are not ported either.
- The IDS head ships in the checkpoint and loads, but never runs (the model card says so).
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Optional

from PIL import Image

from urakatsu.core.registry import register
from urakatsu.providers.base import OcrProvider
from urakatsu.runtimes.store import HuggingFaceFile, WeightSource
from urakatsu.runtimes.torch_runtime import inference_mode, release_cache

from ._common import normalize, raw_chw

if TYPE_CHECKING:  # pragma: no cover
    import torch  # imported where used, and the network on load: the app imports every provider at startup

REPOSITORY = "JustANormalTinkerer/hayai-ocr-v2.7-supernova"
REVISION = "b00317ed69fccee5f787187d913fcf8a96f37f65"  # Koharu's pin; main adds only a README edit

# config.rs: SigLIP2 NaFlex constants and the hardcoded decoder layout.
SIGLIP_HIDDEN = 768
SIGLIP_LAYERS = 12
PATCH_SIZE = 16
QUERY_HEADS = 8
KEY_VALUE_HEADS = 2
HEAD_DIM = 64
ROPE_THETA = 10_000.0
DOWNSCALE = 2  # DSCProjector: each 2x2 block of patches becomes one decoder token
FLOW_HEADS = 8  # VisualCausalFlow: half read the merged grid by rows, half by columns from the right
TEXT_ROPE_OFFSET = 64.0  # HayaiConfig default (config.json leaves it out): text positions start here
# model.rs
MAX_NEW_TOKENS = 128
RMS_NORM_EPS = 1e-6
RESCALE_FACTOR = 1.0 / 255.0
IMAGE_MEAN = [0.5, 0.5, 0.5]
IMAGE_STD = [0.5, 0.5, 0.5]
# processing_hayai.py / preprocessor_config.json
MIN_PATCHES = 64
MAX_SCALE = 4.0
AUTO_PATCH_TIERS = (128, 192, 256, 384, 512)


def _naflex_target_size(height: int, width: int, max_patches: int) -> tuple[int, int]:
    """``_naflex_target_size``: the largest scale up to MAX_SCALE whose patch-rounded size fits
    ``max_patches``; a crop left under MIN_PATCHES takes the largest that fits MIN_PATCHES."""
    eps = 1e-5

    def scaled(scale: float, size: int) -> int:
        return max(PATCH_SIZE, int(math.ceil(size * scale / PATCH_SIZE)) * PATCH_SIZE)

    def search(low: float, high: float, budget: int) -> float:
        while high - low >= eps:
            scale = 0.5 * (low + high)
            if (scaled(scale, height) // PATCH_SIZE) * (scaled(scale, width) // PATCH_SIZE) <= budget:
                low = scale
            else:
                high = scale
        return low

    scale = search(eps, MAX_SCALE, max_patches)
    if (scaled(scale, height) // PATCH_SIZE) * (scaled(scale, width) // PATCH_SIZE) < MIN_PATCHES:
        scale = search(scale, 100.0, min(MIN_PATCHES, max_patches))
    return scaled(scale, height), scaled(scale, width)


def _select_max_patches(height: int, width: int) -> int:
    """``_select_best_max_patches``: the patch budget of AUTO_PATCH_TIERS that suits a crop's size
    and shape, with upstream's thresholds. Upstream nests its 512 rule for square crops inside the
    non-square branch, so that rule never runs; it is left out here too."""
    crop_height, crop_width = max(1, height), max(1, width)
    tiers = {}
    for budget in AUTO_PATCH_TIERS:
        th, tw = _naflex_target_size(height, width, budget)
        tiers[budget] = ((th // PATCH_SIZE) * (tw // PATCH_SIZE), min(th / crop_height, tw / crop_width))
    (n128, s128), (n192, s192), (n256, s256), (n384, s384), (n512, _) = (tiers[b] for b in AUTO_PATCH_TIERS)
    if n128 == n512:
        return 128
    aspect = max(crop_height, crop_width) / min(crop_height, crop_width)
    wide = crop_width / crop_height
    square = 0.77 <= wide <= 1.30
    horizontal = wide > 1.30
    if not square:
        if not horizontal and s128 >= 1.65 and aspect < 3.8:
            return 128
        if horizontal and s128 >= 2.35 and aspect < 3.2:
            return 128
        if s384 < 1.05 or (aspect >= 5.5 and s384 < 1.55) or (aspect >= 7.5 and s384 < 2.00):
            return 512
    if n384 > n256 and (
        s256 < 1.28
        or (square and s256 < 2.20)
        or (horizontal and s256 < 1.45)
        or (aspect >= 3.8 and s256 < 1.75)
        or (aspect >= 5.0 and s256 < 2.20)
    ):
        return 384
    if n192 == n128:
        return 128
    if n256 == n192:
        return 192
    if not square:
        if not horizontal and s192 >= 1.55 and aspect < 3.6:
            return 192
        if horizontal and s192 >= 1.85 and aspect < 3.0:
            return 192
    return 256


class ImageProcessor:
    """processor.rs ``ImageProcessor``: per-crop budget, aspect-preserving NaFlex resize, patchify."""

    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype

    def preprocess(self, image: Image.Image):
        import torch

        if image.width == 0 or image.height == 0:
            raise ValueError("cannot recognize an empty image")
        height, width = _naflex_target_size(image.height, image.width, _select_max_patches(image.height, image.width))
        patch_height = height // PATCH_SIZE
        patch_width = width // PATCH_SIZE
        num_patches = patch_height * patch_width

        image = image.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.BILINEAR)
        pixels = raw_chw(image, self.device) * RESCALE_FACTOR
        pixels = normalize(pixels, IMAGE_MEAN, IMAGE_STD)
        # (1, C, H, W) -> (1, patches, p*p*C) with patch order (row, col, pixel row, pixel col, channel).
        patches = (
            pixels.reshape(1, 3, patch_height, PATCH_SIZE, patch_width, PATCH_SIZE)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(1, num_patches, -1)
        )
        spatial = torch.tensor([[patch_height, patch_width]], dtype=torch.long, device=self.device)
        return patches.to(self.dtype), spatial, (patch_height, patch_width)


@register
class HayaiOcrProvider(OcrProvider):
    id = "hayai-ocr"
    name = "Hayai OCR"
    description = "Fast crop-level OCR for Japanese, Chinese, Korean and English (Hayai v2.7 Supernova, SigLIP2 NaFlex, greedy)."
    runtimes = ("torch",)

    catalog_queries = ('hayai-ocr',)

    @classmethod
    def accepts(cls, facts) -> Optional[str]:
        """v2.7-layout Hayai checkpoints. The Hub's config summary leaves out ``downscale_factor``
        and ``n_vision_layers``, so the layout is told by the image processor v2.7 ships (v2 and
        v2.5 have none, and a different projector this port cannot load)."""
        if facts.runtime == "torch" and "HayaiModel" in facts.architectures and facts.has_file("preprocessor_config.json"):
            return "torch"
        return None
    requires = {"torch": ("transformers", "tokenizers", "safetensors")}

    @classmethod
    def weight_sources(cls, runtime: Optional[str] = None) -> tuple[WeightSource, ...]:
        return tuple(
            HuggingFaceFile(REPOSITORY, name, REVISION) for name in ("config.json", "model.safetensors", "tokenizer.json")
        )

    def _load(self) -> None:
        from safetensors.torch import load_file
        from tokenizers import Tokenizer
        from transformers import Siglip2VisionConfig, Siglip2VisionModel

        from .hayai_ocr_network import _Decoder

        runtime = self.runtime_instance()
        config_path, weights_path, tokenizer_path = self.resolve_weights()
        self._device = runtime.torch_device
        self._dtype = runtime.dtype()

        config = json.loads(config_path.read_text(encoding="utf-8"))
        d_model = int(config["d_model"])
        d_vision = int(config["d_vision"])
        vocab_size = int(config["vocab_size"])
        if vocab_size <= 0:
            raise ValueError("Hayai OCR vocabulary size must be positive")
        if d_vision != SIGLIP_HIDDEN:
            raise ValueError(f"Hayai OCR expects the SigLIP2 NaFlex {SIGLIP_HIDDEN}-dimensional vision encoder")
        if d_model != QUERY_HEADS * HEAD_DIM:
            raise ValueError("decoder model size does not match the hardcoded query head layout")
        if int(config["n_layers"]) <= 0:
            raise ValueError("decoder must have at least one layer")
        # v2 / v2.5 checkpoints lack both keys: their projector and missing depth fusion differ.
        if config.get("n_vision_layers") != SIGLIP_LAYERS or config.get("downscale_factor") != DOWNSCALE:
            raise ValueError(
                f"Hayai OCR loads the v2.7 layout ({DOWNSCALE}x{DOWNSCALE} projector fusing all {SIGLIP_LAYERS} "
                "SigLIP2 depths); this checkpoint is an older Hayai release"
            )

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._bos = self._special("<bos>")
        self._eos = self._special("<eos>")
        self._pad = self._special("<pad>")
        if self.tokenizer.get_vocab_size(with_added_tokens=True) != vocab_size:
            raise ValueError("Hayai OCR vocabulary size does not match the decoder outputs")

        vision = Siglip2VisionModel(Siglip2VisionConfig())
        decoder = _Decoder(
            vocab_size, d_model, d_vision, int(config["d_ffn"]), int(config["n_layers"]), SIGLIP_LAYERS, int(config["num_ids_classes"])
        )
        state = load_file(str(weights_path))
        # transformers' Siglip2VisionModel is the vision tower itself (no ``vision_model`` wrapper).
        vision_state = {
            k[len("vision_encoder.vision_model.") :]: v
            for k, v in state.items()
            if k.startswith("vision_encoder.vision_model.")
        }
        decoder_state = {k[len("decoder.") :]: v for k, v in state.items() if k.startswith("decoder.")}
        unexpected = [k for k in state if not k.startswith(("vision_encoder.", "decoder."))]
        if unexpected:
            raise ValueError(f"Hayai OCR checkpoint has unexpected tensors: {unexpected[:5]}")
        vision.load_state_dict(vision_state, strict=True)
        decoder.load_state_dict(decoder_state, strict=True)

        self._vision = vision.to(device=self._device, dtype=self._dtype).eval()
        self._decoder = decoder.to(device=self._device, dtype=self._dtype).eval()
        self._processor = ImageProcessor(self._device, self._dtype)
        self._text_cos, self._text_sin = self._text_rope(MAX_NEW_TOKENS + 1)

    def _special(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Hayai OCR tokenizer lacks a {token} token")
        return int(token_id)

    def _text_rope(self, positions: int) -> tuple[torch.Tensor, torch.Tensor]:
        import torch

        d_axis = HEAD_DIM // 2
        freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, d_axis, 2, device=self._device).float() / d_axis))
        t = torch.arange(positions, device=self._device, dtype=torch.float32) + TEXT_ROPE_OFFSET
        outer = torch.outer(t, freqs)
        angles = torch.cat([outer, outer], dim=-1)
        return torch.cos(angles), torch.sin(angles)

    def _unload(self) -> None:
        self._vision = None
        self._decoder = None
        self.tokenizer = None
        release_cache(self._device)

    def _recognize(self, crops: list[Image.Image]) -> list[str]:
        return [self._inference(crop) for crop in crops]


    @inference_mode
    def _inference(self, image: Image.Image) -> str:
        patches, spatial, (height, width) = self._processor.preprocess(image)
        token_ids = self._generate(patches, spatial, height, width)
        kept = [t for t in token_ids if t not in (self._eos, self._pad)]
        return self.tokenizer.decode(kept, skip_special_tokens=True)

    def _encode(self, patches, spatial) -> tuple[list, torch.Tensor]:
        """Upstream ``_forward_vision_train``: each encoder layer's input (the embeddings first),
        which depth fusion reads, and the post-layernorm output the projector reads."""
        vision = self._vision
        hidden = vision.embeddings(patches, spatial)
        states = []
        for layer in vision.encoder.layers:
            states.append(hidden)
            out = layer(hidden, None)
            hidden = out[0] if isinstance(out, tuple) else out
        return states, vision.post_layernorm(hidden)

    @inference_mode
    def _generate(self, patches, spatial, height: int, width: int) -> list[int]:
        """model.rs ``generate``: prefill [visual | BOS] with a block-causal mask and each layer's
        depth residual on the visual tokens, then greedy steps."""
        import torch

        from .hayai_ocr_network import _visual_rope

        device = self._device
        dtype = self._dtype
        decoder = self._decoder
        states, last = self._encode(patches, spatial)
        visual, (merged_height, merged_width), blocks = decoder.projector(last, height, width)
        depth = decoder.depth_fusion(states, blocks)  # (layers, visual tokens, d_model)
        m_vision = visual.shape[1]
        bos = torch.tensor([[self._bos]], dtype=torch.long, device=device)
        hidden = torch.cat([visual, decoder.token_embeddings(bos)], dim=1)

        total = m_vision + 1
        mask = torch.zeros((total, total), dtype=dtype, device=device)
        mask[:m_vision, m_vision:] = float("-inf")  # visual tokens never see the text token
        mask = mask.view(1, 1, total, total)

        cos_visual, sin_visual = _visual_rope(merged_height, merged_width, device)
        cos = torch.cat([cos_visual, self._text_cos[0:1]], dim=0).unsqueeze(0)
        sin = torch.cat([sin_visual, self._text_sin[0:1]], dim=0).unsqueeze(0)

        caches: list = []
        for index, layer in enumerate(decoder.layers):
            hidden = torch.cat([hidden[:, :m_vision] + depth[index].unsqueeze(0), hidden[:, m_vision:]], dim=1)
            hidden, present = layer(hidden, mask, cos, sin, None)
            caches.append(present)
        logits = decoder.logits(hidden[:, -1, :])
        first = int(logits.argmax(-1)[0].item())
        if first in (self._eos, self._pad):
            return []
        tokens = [first]

        for step in range(1, MAX_NEW_TOKENS):
            current = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
            hidden = decoder.token_embeddings(current)
            cos_step = self._text_cos[step : step + 1].unsqueeze(0)
            sin_step = self._text_sin[step : step + 1].unsqueeze(0)
            for index, layer in enumerate(decoder.layers):
                hidden, present = layer(hidden, None, cos_step, sin_step, caches[index])
                caches[index] = present
            logits = decoder.logits(hidden[:, -1, :])
            next_token = int(logits.argmax(-1)[0].item())
            if next_token in (self._eos, self._pad):
                break
            tokens.append(next_token)
        return tokens
