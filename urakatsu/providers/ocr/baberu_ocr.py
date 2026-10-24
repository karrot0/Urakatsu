"""Baberu OCR: DINOv2 -> MLP projector -> GQA char decoder, greedy.

Upstream sources (genshiai-daichi/baberu-ocr @ REVISION, facebook/dinov2-base @ DINOV2_REVISION):
- ``baberu_ocr/mod.rs``: load order and the generation call (repetition
  penalty from generation_config, content-run cap 12).
- ``baberu_ocr/config.rs``: validation rules.
- ``baberu_ocr/processor.rs``: whole-crop bicubic resize to 224 and the
  charset tokenizer.
- ``baberu_ocr/model.rs``: the vision prefix, the decoder and the
  greedy loop with repetition penalty and the content-run cap.

Upstream: ``modeling_baberu.py`` and ``inference.py`` at the pinned revision. The decoder
modules below use the upstream attribute names so the checkpoint loads strictly.

Divergences:
- The DINOv2 encoder is transformers' ``Dinov2Model`` built from the pinned dinov2-base
  config. This port implements the same network in its own tree.
- The image resize uses Pillow's BICUBIC directly (the upstream ``BitImageProcessor``
  path). Upstream uses fast_image_resize's Catmull-Rom, which differs at rounding.
"""

from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from urakatsu.core.registry import register
from urakatsu.providers.base import OcrProvider
from urakatsu.runtimes.store import HuggingFaceFile, WeightSource
from urakatsu.runtimes.torch_runtime import release_cache

from ._common import first_weights_dir, normalize, raw_chw, read_lines

REPOSITORY = "genshiai-daichi/baberu-ocr"
REVISION = "d9cc13153e9a1cd8fdfa3b7b1cc329da2020aeae"
DINOV2_REPOSITORY = "facebook/dinov2-base"
DINOV2_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"

PIL_BICUBIC = 3  # preprocessor_config.json resample
CONTENT_RUN_CAP = 12  # baberu_ocr/mod.rs: the content-run limit passed to the decoder
PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIAL_COUNT = 4


def _repo_file(repo: str, revision: str, name: str) -> HuggingFaceFile:
    return HuggingFaceFile(repo, name, revision)


class BaberuConfig:
    """``config.rs`` ``BaberuOcrConfig`` with the validation applied upstream."""

    def __init__(self, data: dict):
        self.data = data
        self.vision_model_name = data.get("vision_model_name")
        self.vision_hidden_size = int(data["vision_hidden_size"])
        self.vision_image_size = int(data["vision_image_size"])
        self.vision_patch_size = int(data["vision_patch_size"])
        self.vision_num_tokens = int(data["vision_num_tokens"])
        self.projector_act = data["projector_act"]
        self.vocab_size = int(data["vocab_size"])
        self.hidden_size = int(data["hidden_size"])
        self.intermediate_size = int(data["intermediate_size"])
        self.num_hidden_layers = int(data["num_hidden_layers"])
        self.num_attention_heads = int(data["num_attention_heads"])
        self.num_key_value_heads = int(data["num_key_value_heads"])
        self.head_dim = int(data.get("head_dim") or self.hidden_size // self.num_attention_heads)
        self.max_position_embeddings = int(data["max_position_embeddings"])
        self.hidden_act = data["hidden_act"]
        self.rms_norm_eps = float(data["rms_norm_eps"])
        self.sandwich_norm = bool(data["sandwich_norm"])
        self.rope_theta = float(data["rope_theta"])
        self.final_logit_softcap = data.get("final_logit_softcap")
        self.attn_logit_softcap = data.get("attn_logit_softcap")
        if self.attn_logit_softcap is not None:
            # Upstream tanh-caps every decoder layer's scores; the pinned checkpoint sets null.
            raise ValueError("Baberu attention logit softcap is not supported")
        self.tie_word_embeddings = bool(data["tie_word_embeddings"])
        self.pad_token_id = int(data["pad_token_id"])
        self.bos_token_id = int(data["bos_token_id"])
        self.eos_token_id = int(data["eos_token_id"])
        self.unk_token_id = int(data["unk_token_id"])

    def validate(self, vision: dict) -> None:
        if self.vision_model_name != "facebook/dinov2-base":
            raise ValueError(f"unsupported Baberu OCR vision model {self.vision_model_name!r}")
        if self.projector_act != "gelu":
            raise ValueError("unsupported projector activation")
        if self.hidden_act != "silu":
            raise ValueError("unsupported decoder activation")
        if not self.tie_word_embeddings:
            raise ValueError("the released Baberu OCR checkpoint requires tied word embeddings")
        if [self.pad_token_id, self.bos_token_id, self.eos_token_id, self.unk_token_id] != [PAD, BOS, EOS, UNK]:
            raise ValueError("unsupported Baberu OCR special-token layout")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("decoder query heads must be divisible by key/value heads")
        if self.head_dim * self.num_attention_heads != self.hidden_size:
            raise ValueError("decoder head dimensions do not equal its hidden size")
        if self.vision_hidden_size != int(vision["hidden_size"]):
            raise ValueError("Baberu OCR and DINOv2 hidden sizes do not match")
        if self.vision_patch_size != int(vision["patch_size"]):
            raise ValueError("Baberu OCR and DINOv2 patch sizes do not match")
        if self.vision_image_size % self.vision_patch_size != 0:
            raise ValueError("Baberu OCR image size is not divisible by its patch size")
        if self.vision_num_tokens != (self.vision_image_size // self.vision_patch_size) ** 2:
            raise ValueError("Baberu OCR vision token count is inconsistent")
        if vision.get("hidden_act", "gelu") != "gelu":
            raise ValueError("unsupported DINOv2 activation")
        if vision.get("use_swiglu_ffn", False):
            raise ValueError("DINOv2 SwiGLU is not supported")
        if not vision.get("use_mask_token", True):
            raise ValueError("the DINOv2 checkpoint contains a mask token")
        if self.max_position_embeddings <= self.vision_num_tokens:
            raise ValueError("decoder position table cannot contain the vision prefix")


class Tokenizer:
    """``processor.rs`` ``Tokenizer``: four specials, then the deduplicated charset list."""

    def __init__(self, path: Path):
        charset = json.loads(Path(path).read_text(encoding="utf-8"))
        vocabulary = ["<pad>", "<bos>", "<eos>", "<unk>"]
        seen = set(vocabulary)
        for character in charset:
            if character not in seen:
                seen.add(character)
                vocabulary.append(character)
        if len(vocabulary) == SPECIAL_COUNT:
            raise ValueError("Baberu OCR vocabulary is empty")
        self.vocabulary = vocabulary
        self.content_ids = frozenset(i for i, token in enumerate(vocabulary) if self._is_content_token(token, i))

    @staticmethod
    def _is_content_token(token: str, token_id: int) -> bool:
        if token_id < SPECIAL_COUNT or len(token) != 1:
            return False
        if token in "ーｰ〜~":
            return False
        return unicodedata.category(token)[0] in ("L", "N")

    def __len__(self) -> int:
        return len(self.vocabulary)

    def is_content(self, token_id: int) -> bool:
        return token_id in self.content_ids

    def decode(self, token_ids: list[int]) -> str:
        text = []
        for token_id in token_ids:
            if not 0 <= token_id < len(self.vocabulary):
                raise ValueError(f"invalid Baberu OCR token id {token_id}")
            if token_id >= SPECIAL_COUNT:
                text.append(self.vocabulary[token_id])
        return "".join(text)


class _RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x32 = x.to(torch.float32)
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x32).to(input_dtype)


class _MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


class _Attention(nn.Module):
    def __init__(self, cfg: BaberuConfig):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        self.num_kv_heads = cfg.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin, mask, past):
        bsz, q_len, _ = x.shape
        q = self.q_proj(x).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        present = (k, v)
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.head_dim**-0.5)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(out), present


class _DecoderLayer(nn.Module):
    def __init__(self, cfg: BaberuConfig):
        super().__init__()
        eps = cfg.rms_norm_eps
        self.sandwich = cfg.sandwich_norm
        self.input_norm = _RMSNorm(cfg.hidden_size, eps)
        self.attn = _Attention(cfg)
        self.post_attn_norm = _RMSNorm(cfg.hidden_size, eps)
        self.pre_ffn_norm = _RMSNorm(cfg.hidden_size, eps)
        self.mlp = _MLP(cfg.hidden_size, cfg.intermediate_size)
        self.post_ffn_norm = _RMSNorm(cfg.hidden_size, eps)

    def forward(self, h, cos, sin, mask, past):
        residual = h
        x = self.input_norm(h)
        x, present = self.attn(x, cos, sin, mask, past)
        if self.sandwich:
            x = self.post_attn_norm(x)
        h = residual + x
        residual = h
        x = self.pre_ffn_norm(h)
        x = self.mlp(x)
        if self.sandwich:
            x = self.post_ffn_norm(x)
        return residual + x, present


class _Decoder(nn.Module):
    """``modeling_baberu.py`` ``BaberuModel``: embeddings, layers, final norm, 1D RoPE."""

    def __init__(self, cfg: BaberuConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id)
        self.layers = nn.ModuleList([_DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = _RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, inputs_embeds, past):
        q_len = inputs_embeds.shape[1]
        past_len = 0 if past is None else past[0][0].shape[2]
        kv_len = past_len + q_len
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype
        positions = torch.arange(past_len, kv_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(dtype)[None]
        sin = emb.sin().to(dtype)[None]
        mask = None
        if q_len > 1:
            # Causal over the new block; past keys are all visible.
            mask = torch.full((q_len, kv_len), float("-inf"), device=device, dtype=dtype)
            for i in range(q_len):
                mask[i, : past_len + i + 1] = 0.0
        h = inputs_embeds
        presents = []
        for index, layer in enumerate(self.layers):
            layer_past = None if past is None else past[index]
            h, present = layer(h, cos, sin, mask, layer_past)
            presents.append(present)
        return self.norm(h), presents


class _Projector(nn.Module):
    def __init__(self, vision_hidden: int, hidden: int):
        super().__init__()
        self.linear1 = nn.Linear(vision_hidden, hidden, bias=True)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(hidden, hidden, bias=True)

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


_DINOV2_RENAMES = (
    (".attention.attention.query.", ".attention.q_proj."),
    (".attention.attention.key.", ".attention.k_proj."),
    (".attention.attention.value.", ".attention.v_proj."),
    (".attention.output.dense.", ".attention.o_proj."),
)


def _dinov2_key(key: str) -> str:
    """The checkpoint stores DINOv2 attention under the Hugging Face 4.x names; transformers 5 renamed them."""
    for old, new in _DINOV2_RENAMES:
        key = key.replace(old, new)
    return key


class BaberuNetwork(nn.Module):
    """``BaberuOCRModel``: vision_encoder, projector and the decoder under ``model``."""

    def __init__(self, cfg: BaberuConfig, dinov2_config):
        super().__init__()
        from transformers import Dinov2Model

        self.vision_encoder = Dinov2Model(dinov2_config)
        self.projector = _Projector(cfg.vision_hidden_size, cfg.hidden_size)
        self.model = _Decoder(cfg)
        self.final_logit_softcap = cfg.final_logit_softcap

    def logits(self, hidden_last: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden_last, self.model.embed_tokens.weight)
        if self.final_logit_softcap:
            cap = float(self.final_logit_softcap)
            logits = torch.tanh(logits / cap) * cap
        return logits


@register
class BaberuOcrProvider(OcrProvider):
    id = "baberu-ocr"
    name = "Baberu OCR"
    description = "Manga text recognition (DINOv2 prefix, GQA character decoder, greedy with repetition penalty)."
    runtimes = ("torch",)

    catalog_queries = ('baberu',)

    @classmethod
    def accepts(cls, facts) -> Optional[str]:
        return "torch" if facts.runtime == "torch" and facts.model_type == "baberu_ocr" else None
    requires = {"torch": ("transformers", "safetensors")}

    @classmethod
    def weight_sources(cls, runtime: Optional[str] = None) -> tuple[WeightSource, ...]:
        return (
            _repo_file(REPOSITORY, REVISION, "config.json"),
            _repo_file(REPOSITORY, REVISION, "generation_config.json"),
            _repo_file(REPOSITORY, REVISION, "model.safetensors"),
            _repo_file(REPOSITORY, REVISION, "tokenizer/vocab.json"),
            _repo_file(DINOV2_REPOSITORY, DINOV2_REVISION, "config.json"),
            _repo_file(DINOV2_REPOSITORY, DINOV2_REVISION, "preprocessor_config.json"),
        )

    def _load(self) -> None:
        from safetensors.torch import load_file
        from transformers import Dinov2Config

        runtime = self.runtime_instance()
        paths = self.resolve_weights()
        config_path, generation_path, weights_path, vocabulary_path, vision_config_path, processor_path = paths
        self._device = runtime.torch_device
        self._dtype = runtime.dtype()

        config = BaberuConfig(json.loads(config_path.read_text(encoding="utf-8")))
        vision = json.loads(vision_config_path.read_text(encoding="utf-8"))
        config.validate(vision)
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if int(generation[key]) != getattr(config, key):
                raise ValueError(f"{key} differs between Baberu OCR config and generation config")
        if not float(generation["repetition_penalty"]) > 0.0:
            raise ValueError("repetition penalty must be positive")
        if config.vision_num_tokens + 1 + int(generation["max_new_tokens"]) > config.max_position_embeddings:
            raise ValueError("generation can exceed the decoder position table")
        self._max_new_tokens = int(generation["max_new_tokens"])
        self._repetition_penalty = float(generation["repetition_penalty"])

        self.tokenizer = Tokenizer(vocabulary_path)
        if len(self.tokenizer) != config.vocab_size:
            raise ValueError(
                f"Baberu OCR vocabulary has {len(self.tokenizer)} entries but the decoder has {config.vocab_size} outputs"
            )
        self._processor = json.loads(processor_path.read_text(encoding="utf-8"))
        if int(self._processor["resample"]) != PIL_BICUBIC:
            raise ValueError(f"unsupported DINOv2 PIL resampling filter {self._processor['resample']}")
        self._image_size = config.vision_image_size
        self._rescale = float(self._processor["rescale_factor"])
        self._mean = [float(v) for v in self._processor["image_mean"]]
        self._std = [float(v) for v in self._processor["image_std"]]

        dinov2_config = Dinov2Config.from_dict(vision)
        network = BaberuNetwork(config, dinov2_config)
        state = load_file(str(weights_path))
        vision_state = {
            _dinov2_key(k[len("vision_encoder.") :]): v for k, v in state.items() if k.startswith("vision_encoder.")
        }
        other_state = {k: v for k, v in state.items() if not k.startswith("vision_encoder.")}
        # The DINOv2 tree must be complete (including the unused mask token) before loading.
        network.vision_encoder.load_state_dict(vision_state, strict=True)
        result = network.load_state_dict(other_state, strict=False)
        missing = [k for k in result.missing_keys if not k.startswith("vision_encoder.")]
        if missing or result.unexpected_keys:
            raise ValueError(
                f"Baberu OCR checkpoint mismatch: missing={missing[:5]} unexpected={result.unexpected_keys[:5]}"
            )
        self._network = network.to(device=self._device, dtype=self._dtype).eval()

    def _unload(self) -> None:
        self._network = None
        self.tokenizer = None
        release_cache(self._device)

    def _recognize(self, crops: list[Image.Image]) -> list[str]:
        return [self._inference(crop) for crop in crops]

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Whole crop to a fixed square (no centre crop), then rescale and normalise (processor.rs)."""
        if image.width == 0 or image.height == 0:
            raise ValueError("cannot recognize an empty image")
        image = image.convert("RGB")
        if image.size != (self._image_size, self._image_size):
            image = image.resize((self._image_size, self._image_size), Image.BICUBIC)
        pixels = raw_chw(image, self._device) * self._rescale
        return normalize(pixels, self._mean, self._std).to(self._dtype)

    @torch.inference_mode()
    def _inference(self, image: Image.Image) -> str:
        pixel_values = self._preprocess(image)
        token_ids = self._generate(pixel_values)
        return self.tokenizer.decode(token_ids)

    @torch.inference_mode()
    def _generate(self, pixel_values: torch.Tensor) -> list[int]:
        """``model.rs`` ``forward``: prefill with [vision | BOS], then greedy steps with two guards."""
        network = self._network
        device = self._device
        max_new_tokens = self._max_new_tokens
        penalty = self._repetition_penalty
        vision = network.vision_encoder(pixel_values=pixel_values).last_hidden_state[:, 1:, :]
        vision_embeds = network.projector(vision)
        bos = torch.tensor([[BOS]], dtype=torch.long, device=device)
        embeds = torch.cat([vision_embeds, network.model.embed_tokens(bos)], dim=1)
        hidden, cache = network.model(embeds, None)
        logits = network.logits(hidden[:, -1, :])

        tokens: list[int] = []
        unique = {BOS}
        seen = bos.clone()
        for _ in range(max_new_tokens):
            if penalty != 1.0:
                selected = logits.gather(1, seen)
                adjusted = torch.where(selected < 0.0, selected * penalty, selected / penalty)
                logits = logits.scatter(1, seen, adjusted)
            if tokens and self.tokenizer.is_content(tokens[-1]):
                last = tokens[-1]
                run = 0
                for token in reversed(tokens):
                    if token != last:
                        break
                    run += 1
                if run >= CONTENT_RUN_CAP:
                    logits[:, last] = float("-inf")

            next_token = int(logits.argmax(-1)[0].item())
            if next_token == EOS:
                break
            tokens.append(next_token)
            if len(tokens) == max_new_tokens:
                break
            if next_token not in unique:
                unique.add(next_token)
                seen = torch.cat([seen, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)
            next_id = torch.tensor([[next_token]], dtype=torch.long, device=device)
            hidden, cache = network.model(network.model.embed_tokens(next_id), cache)
            logits = network.logits(hidden[:, -1, :])
        return tokens
