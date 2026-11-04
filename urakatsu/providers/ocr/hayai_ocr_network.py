"""Hayai OCR's network (hayai_ocr.py), apart so that listing the providers does not import torch."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hayai_ocr import HEAD_DIM, KEY_VALUE_HEADS, QUERY_HEADS, RMS_NORM_EPS, ROPE_THETA


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
