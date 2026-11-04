"""Manga OCR CPT's network (manga_ocr_cpt.py), apart so that listing the providers does not import torch."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .manga_ocr_cpt import END, EPSILON, HEADS, HIDDEN, PATCH_BUDGET, START


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
