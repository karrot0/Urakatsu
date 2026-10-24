"""Hayai OCR: SigLIP2 NaFlex encoder and a GQA decoder, greedy.

Upstream sources (JustANormalTinkerer/hayai-ocr-v2 @ REVISION):
- ``hayai_ocr/mod.rs``: load order and ``inference``.
- ``hayai_ocr/config.rs``: SigLIP2 and decoder constants and validation.
- ``hayai_ocr/processor.rs``: the NaFlex size search, bilinear resize,
  patchify order, padding to 256 patches, and the decode filter.
- ``hayai_ocr/model.rs``: ``generate`` (greedy, 256 tokens), 2D visual RoPE,
  block-causal prefill mask and the decoder layers.

Upstream: ``modeling_hayai.py`` at the pinned revision. Its decoder attribute names are kept so
the checkpoint loads strictly.

Divergences:
- The vision encoder is transformers' ``Siglip2VisionModel`` (default SigLIP2 base config,
  which the released NaFlex config only overrides by model type). This port hand-builds the same
  network. The pooled head is loaded but unused, as upstream does (``last_hidden_state``).
- The ``REPETITION_PENALTY`` constant is 1.0, so the penalty is a no-op and is not ported.
- Beam search is not ported; the upstream default is ``num_beams=1``.
"""

from __future__ import annotations

import json
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from urakatsu.core.registry import register
from urakatsu.providers.base import OcrProvider
from urakatsu.runtimes.store import HuggingFaceFile, WeightSource
from urakatsu.runtimes.torch_runtime import release_cache

from ._common import normalize, raw_chw

REPOSITORY = "JustANormalTinkerer/hayai-ocr-v2"
REVISION = "4a4ce477c9a8841f208b94e1d9ed5c0938965e05"

# hayai_ocr/config.rs: SigLIP2 NaFlex constants and the hardcoded decoder layout.
SIGLIP_HIDDEN = 768
PATCH_SIZE = 16
MAX_NUM_PATCHES = 256
QUERY_HEADS = 8
KEY_VALUE_HEADS = 2
HEAD_DIM = 64
ROPE_THETA = 10_000.0
# model.rs
MAX_NEW_TOKENS = 256
RMS_NORM_EPS = 1e-6
RESCALE_FACTOR = 1.0 / 255.0
IMAGE_MEAN = [0.5, 0.5, 0.5]
IMAGE_STD = [0.5, 0.5, 0.5]


def _image_size_for_max_patches(height: int, width: int, patch: int, max_patches: int) -> tuple[int, int]:
    """processor.rs ``get_image_size_for_max_num_patches``: binary search over the scale."""
    eps = 1e-5

    def scaled(scale: float, size: int) -> int:
        return max(math.ceil((size * scale) / patch) * patch, patch)

    scale_min = eps / 10.0
    scale_max = 100.0
    while scale_max - scale_min >= eps:
        scale = (scale_min + scale_max) / 2.0
        target_height = scaled(scale, height)
        target_width = scaled(scale, width)
        if (target_height // patch) * (target_width // patch) <= max_patches:
            scale_min = scale
        else:
            scale_max = scale
    return scaled(scale_min, height), scaled(scale_min, width)


class ImageProcessor:
    """processor.rs ``ImageProcessor``: aspect-preserving NaFlex resize, patchify, pad."""

    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype

    def preprocess(self, image: Image.Image):
        if image.width == 0 or image.height == 0:
            raise ValueError("cannot recognize an empty image")
        height, width = _image_size_for_max_patches(image.height, image.width, PATCH_SIZE, MAX_NUM_PATCHES)
        patch_height = height // PATCH_SIZE
        patch_width = width // PATCH_SIZE
        num_patches = patch_height * patch_width
        if num_patches > MAX_NUM_PATCHES:
            raise ValueError(f"image resize produced {num_patches} patches above the {MAX_NUM_PATCHES} limit")

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
        attention = torch.ones((1, num_patches), dtype=torch.long, device=self.device)
        if num_patches < MAX_NUM_PATCHES:
            pad = MAX_NUM_PATCHES - num_patches
            patches = torch.cat([patches, torch.zeros((1, pad, patches.shape[2]), device=self.device)], dim=1)
            attention = torch.cat([attention, torch.zeros((1, pad), dtype=torch.long, device=self.device)], dim=1)
        spatial = torch.tensor([[patch_height, patch_width]], dtype=torch.long, device=self.device)
        return patches.to(self.dtype), attention, spatial, (patch_height, patch_width)


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``apply_rotary_emb_2d``: interleaved pairs (even, odd), computed in float32, returned in x's dtype."""
    orig = x.dtype
    x = x.float()
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    return torch.stack([rot1, rot2], dim=-1).flatten(-2).to(orig)


def _visual_rope(height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """upstream ``_get_2d_visual_freqs``: (h*w, 32) angles, first 16 from the row, last 16 from the column."""
    d_axis = HEAD_DIM // 2
    freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, d_axis, 2, device=device).float() / d_axis))
    grid_y = torch.arange(height, device=device, dtype=torch.float32)
    grid_x = torch.arange(width, device=device, dtype=torch.float32)
    freqs_y = torch.outer(grid_y, freqs)
    freqs_x = torch.outer(grid_x, freqs)
    y_ext = freqs_y.unsqueeze(1).expand(height, width, -1)
    x_ext = freqs_x.unsqueeze(0).expand(height, width, -1)
    angles = torch.cat([y_ext, x_ext], dim=-1).flatten(0, 1)
    return torch.cos(angles), torch.sin(angles)


class _RMSNorm(nn.Module):
    """model.rs ``RmsNorm``: normalise in float32, scale, cast back to the input dtype."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x32 = x.float()
        variance = x32.pow(2).mean(-1, keepdim=True)
        return (self.weight * x32 * torch.rsqrt(variance + RMS_NORM_EPS)).to(input_dtype)


class _SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_up = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class _GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.w_q = nn.Linear(d_model, QUERY_HEADS * HEAD_DIM, bias=False)
        self.w_k = nn.Linear(d_model, KEY_VALUE_HEADS * HEAD_DIM, bias=False)
        self.w_v = nn.Linear(d_model, KEY_VALUE_HEADS * HEAD_DIM, bias=False)
        self.w_o = nn.Linear(QUERY_HEADS * HEAD_DIM, d_model, bias=False)
        self.q_norm = _RMSNorm(HEAD_DIM)
        self.k_norm = _RMSNorm(HEAD_DIM)

    def forward(self, x, mask, cos, sin, cache):
        b, s, _ = x.shape
        q = self.w_q(x).view(b, s, QUERY_HEADS, HEAD_DIM)
        k = self.w_k(x).view(b, s, KEY_VALUE_HEADS, HEAD_DIM)
        v = self.w_v(x).view(b, s, KEY_VALUE_HEADS, HEAD_DIM)
        q = _rotate(self.q_norm(q), cos, sin)
        k = _rotate(self.k_norm(k), cos, sin)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=1)
            v = torch.cat([cache[1], v], dim=1)
        present = (k, v)
        groups = QUERY_HEADS // KEY_VALUE_HEADS
        k_exp = k.repeat_interleave(groups, dim=2).transpose(1, 2)
        v_exp = v.repeat_interleave(groups, dim=2).transpose(1, 2)
        context = F.scaled_dot_product_attention(q.transpose(1, 2), k_exp, v_exp, attn_mask=mask, is_causal=False)
        context = context.transpose(1, 2).contiguous().view(b, s, -1)
        return self.w_o(context), present


class _DecoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.attn_norm = _RMSNorm(d_model)
        self.attn = _GroupedQueryAttention(d_model)
        self.ffn_norm = _RMSNorm(d_model)
        self.ffn = _SwiGLU(d_model, d_ffn)
        self.attn_res_scale = nn.Parameter(torch.ones(d_model))
        self.ffn_res_scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x, mask, cos, sin, cache):
        attn, present = self.attn(self.attn_norm(x), mask, cos, sin, cache)
        x = x + self.attn_res_scale * attn
        x = x + self.ffn_res_scale * self.ffn(self.ffn_norm(x))
        return x, present


class _Projector(nn.Module):
    def __init__(self, d_vision: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_vision, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, x):
        return self.proj(x)


class _Decoder(nn.Module):
    """upstream ``VisualCausalOCRDecoder`` (names kept for the checkpoint)."""

    def __init__(self, vocab_size: int, d_model: int, d_vision: int, d_ffn: int, n_layers: int):
        super().__init__()
        self.projector = _Projector(d_vision, d_model)
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([_DecoderLayer(d_model, d_ffn) for _ in range(n_layers)])
        self.final_norm = _RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

    def logits(self, hidden_last: torch.Tensor) -> torch.Tensor:
        return self.output_head(self.final_norm(hidden_last))


@register
class HayaiOcrProvider(OcrProvider):
    id = "hayai-ocr"
    name = "Hayai OCR"
    description = "Fast crop-level OCR for Japanese, Chinese, Korean and English (SigLIP2 NaFlex, greedy)."
    runtimes = ("torch",)

    catalog_queries = ('hayai-ocr',)

    @classmethod
    def accepts(cls, facts) -> Optional[str]:
        return "torch" if facts.runtime == "torch" and "HayaiModel" in facts.architectures else None
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

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._bos = self._special("<bos>")
        self._eos = self._special("<eos>")
        self._pad = self._special("<pad>")
        if self.tokenizer.get_vocab_size(with_added_tokens=True) != vocab_size:
            raise ValueError("Hayai OCR vocabulary size does not match the decoder outputs")

        vision = Siglip2VisionModel(Siglip2VisionConfig())
        decoder = _Decoder(vocab_size, d_model, d_vision, int(config["d_ffn"]), int(config["n_layers"]))
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
        d_axis = HEAD_DIM // 2
        freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, d_axis, 2, device=self._device).float() / d_axis))
        t = torch.arange(positions, device=self._device, dtype=torch.float32)
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

    @torch.inference_mode()
    def _inference(self, image: Image.Image) -> str:
        patches, attention, spatial, (height, width) = self._processor.preprocess(image)
        token_ids = self._generate(patches, attention, spatial, height, width)
        kept = [t for t in token_ids if t not in (self._eos, self._pad)]
        return self.tokenizer.decode(kept, skip_special_tokens=True)

    @torch.inference_mode()
    def _generate(self, patches, attention, spatial, height: int, width: int) -> list[int]:
        """model.rs ``generate``: prefill [visual | BOS] with a block-causal mask, then greedy steps."""
        device = self._device
        dtype = self._dtype
        decoder = self._decoder
        vision_hidden = self._vision(
            pixel_values=patches, pixel_attention_mask=attention, spatial_shapes=spatial
        ).last_hidden_state
        visual = decoder.projector(vision_hidden.to(dtype))
        m_vision = visual.shape[1]
        bos = torch.tensor([[self._bos]], dtype=torch.long, device=device)
        hidden = torch.cat([visual, decoder.token_embeddings(bos)], dim=1)

        total = m_vision + 1
        mask = torch.zeros((total, total), dtype=dtype, device=device)
        mask[:m_vision, m_vision:] = float("-inf")  # visual tokens never see the text token
        mask = mask.view(1, 1, total, total)

        cos_visual, sin_visual = _visual_rope(height, width, device)
        covered = height * width
        pad_rows = m_vision - covered
        cos = torch.cat(
            [cos_visual, torch.ones((pad_rows, HEAD_DIM // 2), device=device), self._text_cos[0:1]], dim=0
        ).unsqueeze(0)
        sin = torch.cat(
            [sin_visual, torch.zeros((pad_rows, HEAD_DIM // 2), device=device), self._text_sin[0:1]], dim=0
        ).unsqueeze(0)

        caches: list = []
        for index, layer in enumerate(decoder.layers):
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
