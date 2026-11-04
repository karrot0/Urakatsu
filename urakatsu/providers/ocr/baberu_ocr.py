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
from typing import TYPE_CHECKING, Optional

from PIL import Image

from urakatsu.core.registry import register
from urakatsu.providers.base import OcrProvider
from urakatsu.runtimes.store import HuggingFaceFile, WeightSource
from urakatsu.runtimes.torch_runtime import inference_mode, release_cache

from ._common import first_weights_dir, normalize, raw_chw, read_lines

if TYPE_CHECKING:  # pragma: no cover
    import torch  # imported where used, and the network on load: the app imports every provider at startup

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

        from .baberu_ocr_network import BaberuNetwork, _dinov2_key

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

    @inference_mode
    def _inference(self, image: Image.Image) -> str:
        pixel_values = self._preprocess(image)
        token_ids = self._generate(pixel_values)
        return self.tokenizer.decode(token_ids)

    @inference_mode
    def _generate(self, pixel_values: torch.Tensor) -> list[int]:
        """``model.rs`` ``forward``: prefill with [vision | BOS], then greedy steps with two guards."""
        import torch

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
