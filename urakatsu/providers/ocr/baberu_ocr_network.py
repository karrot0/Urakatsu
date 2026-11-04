"""Baberu OCR's network (baberu_ocr.py), apart so that listing the providers does not import torch."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .baberu_ocr import BaberuConfig


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
