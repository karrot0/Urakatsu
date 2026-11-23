"""Hayai OCR's network (hayai_ocr.py), apart so that listing the providers does not import torch."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hayai_ocr import DOWNSCALE, FLOW_HEADS, HEAD_DIM, KEY_VALUE_HEADS, QUERY_HEADS, RMS_NORM_EPS, ROPE_THETA


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




def _block_indices(height: int, width: int, device: torch.device) -> torch.Tensor:
    """``gather_patches``: each merged token's DOWNSCALE x DOWNSCALE patch block, clamped into the
    grid (its bottom and right edges replicate), as flat patch indices ``(tokens, r*r)``."""
    r = DOWNSCALE
    merged_height, merged_width = -(-height // r), -(-width // r)
    y = torch.arange(merged_height, device=device).view(-1, 1, 1, 1) * r
    x = torch.arange(merged_width, device=device).view(1, -1, 1, 1) * r
    dy = torch.arange(r, device=device).view(1, 1, -1, 1)
    dx = torch.arange(r, device=device).view(1, 1, 1, -1)
    rows = (y + dy).clamp(max=height - 1)
    cols = (x + dx).clamp(max=width - 1)
    return (rows * width + cols).reshape(merged_height * merged_width, r * r)


def _neighbour_indices(height: int, width: int, device: torch.device) -> torch.Tensor:
    """``_dw_gather_mix_compiled``: each merged token's 3x3 neighbours, clamped into the grid,
    row-major with the token itself at index 4, ``(tokens, 9)``."""
    y = torch.arange(height, device=device).view(-1, 1, 1, 1)
    x = torch.arange(width, device=device).view(1, -1, 1, 1)
    offsets = torch.tensor([-1, 0, 1], device=device)
    rows = (y + offsets.view(1, 1, -1, 1)).clamp(0, height - 1)
    cols = (x + offsets.view(1, 1, 1, -1)).clamp(0, width - 1)
    return (rows * width + cols).reshape(height * width, 9)


def _dual_causal_mask(height: int, width: int, device: torch.device) -> torch.Tensor:
    """``build_dual_causal_mask``, ``(1, FLOW_HEADS, tokens, tokens)`` (True = attend): the first
    half of the heads sees the grid up to each token row by row, the second half column by column
    from the right."""
    tokens = torch.arange(height * width, device=device)
    horizontal = tokens.view(-1, 1) >= tokens.view(1, -1)
    rank = (width - 1 - tokens % width) * height + tokens // width
    vertical = rank.view(-1, 1) >= rank.view(1, -1)
    half = FLOW_HEADS // 2
    return torch.stack([horizontal] * half + [vertical] * half).unsqueeze(0)


class _CausalFlow(nn.Module):
    """upstream ``VisualCausalFlow``: gated attention over the merged grid in both reading orders."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = _RMSNorm(d_model)
        self.wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.flow_gate = nn.Parameter(torch.zeros(d_model))

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        b, length, d_model = tokens.shape
        head_dim = d_model // FLOW_HEADS
        qkv = self.wqkv(self.norm(tokens)).view(b, length, 3, FLOW_HEADS, head_dim).permute(2, 0, 3, 1, 4)
        mask = _dual_causal_mask(height, width, tokens.device)
        attended = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=mask)
        attended = attended.transpose(1, 2).reshape(b, length, d_model)
        return tokens + self.wo(attended) * self.flow_gate


class _Projector(nn.Module):
    """upstream ``DSCProjector``: 2x2 patch blocks to one token, a depthwise 3x3 mix, the causal flow."""

    def __init__(self, d_vision: int, d_model: int):
        super().__init__()
        unshuffled = d_vision * DOWNSCALE * DOWNSCALE
        self.norm = nn.LayerNorm(unshuffled)
        self.mlp = nn.Sequential(nn.Linear(unshuffled, d_model, bias=False), nn.GELU(), nn.Linear(d_model, d_model, bias=False))
        self.dw_weight = nn.Parameter(torch.zeros(9, d_model))
        self.dw_bias = nn.Parameter(torch.zeros(d_model))
        self.causal_flow = _CausalFlow(d_model)
        self.out_norm = _RMSNorm(d_model)

    def forward(self, features: torch.Tensor, height: int, width: int):
        """``(1, height*width, d_vision)`` patch features: the visual tokens, the merged grid, and
        each token's patch block (which depth fusion pools by)."""
        merged_height, merged_width = -(-height // DOWNSCALE), -(-width // DOWNSCALE)
        blocks = _block_indices(height, width, features.device)
        # (tokens, r*r, d_vision) -> channel c*r*r + k, as ``F.pixel_unshuffle`` orders them.
        tokens = features[0][blocks].transpose(1, 2).reshape(1, blocks.shape[0], -1)
        tokens = self.mlp(self.norm(tokens))
        neighbours = tokens[0][_neighbour_indices(merged_height, merged_width, tokens.device)]
        tokens = (neighbours[:, 4] + (neighbours * self.dw_weight).sum(1) + self.dw_bias).unsqueeze(0)
        tokens = self.causal_flow(tokens, merged_height, merged_width)
        return self.out_norm(tokens), (merged_height, merged_width), blocks


class _DepthFusion(nn.Module):
    """upstream ``StaticDepthFusion``: every encoder depth pooled over the projector's blocks,
    normalised and projected, mixed per decoder layer by a softmax and scaled per channel."""

    def __init__(self, d_vision: int, d_model: int, n_layers: int, n_depths: int):
        super().__init__()
        self.norms = nn.ModuleList([_RMSNorm(d_vision) for _ in range(n_depths)])
        self.wo = nn.Linear(d_vision, d_model, bias=False)
        self.gamma = nn.Parameter(torch.zeros(n_layers, d_model))
        self.depth_logits = nn.Parameter(torch.zeros(n_layers, n_depths))

    def forward(self, states: list[torch.Tensor], blocks: torch.Tensor) -> torch.Tensor:
        """``(layers, tokens, d_model)``, one residual onto the visual tokens per decoder layer."""
        projected = torch.stack(
            [self.wo(norm(state[0][blocks].sum(1) * 0.25)) for norm, state in zip(self.norms, states)]
        )
        alpha = self.depth_logits.float().softmax(-1).to(projected.dtype)
        return torch.einsum("jl,lnd->jnd", alpha, projected) * self.gamma.unsqueeze(1)


class _Decoder(nn.Module):
    """upstream ``VisualCausalOCRDecoder`` (names kept for the checkpoint)."""

    def __init__(self, vocab_size: int, d_model: int, d_vision: int, d_ffn: int, n_layers: int, n_depths: int, ids_classes: int):
        super().__init__()
        self.projector = _Projector(d_vision, d_model)
        self.depth_fusion = _DepthFusion(d_vision, d_model, n_layers, n_depths)
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([_DecoderLayer(d_model, d_ffn) for _ in range(n_layers)])
        self.final_norm = _RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        # The IDS head is trained with the decoder and shipped; it never runs at inference.
        self.ids_norm = _RMSNorm(d_model)
        self.ids_head = nn.Linear(d_model, ids_classes, bias=False)

    def logits(self, hidden_last: torch.Tensor) -> torch.Tensor:
        return self.output_head(self.final_norm(hidden_last))
