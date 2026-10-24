"""Manga OCR: a ViT encoder with a BERT decoder, run through beam search.

Sources (commit-pinned upstream: mayocream/manga-ocr @ REVISION):
- ``manga_ocr/mod.rs``: load order and ``inference``.
- ``manga_ocr/config.rs``: checkpoint configuration.
- ``manga_ocr/processor.rs``: Pillow grayscale, ViT resize and
  normalisation, vocabulary decoding and ``post_process``.
- ``manga_ocr/model.rs``: ``beam_search``, ``BeamHypotheses``,
  ``no_repeat_ngram_mask``.

Divergences:
- The network is transformers' ``VisionEncoderDecoderModel`` (ViT + BERT) loaded from the
  pinned checkpoint, instead of a hand-built VarStore tree. Its parameter names and
  architecture are the ones this port reproduces. Missing or unexpected checkpoint
  tensors fail the load.
- The decoder runs without a key/value cache, re-reading the whole prefix at each step.
  The cache only reuses values, so the scores match, but this is slower on long outputs.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from urakatsu.core.registry import register
from urakatsu.providers.base import OcrProvider
from urakatsu.runtimes.store import HuggingFaceFile, WeightSource
from urakatsu.runtimes.torch_runtime import release_cache

from ._common import first_weights_dir, normalize, pillow_grayscale, raw_chw, read_lines, is_whitespace

REPOSITORY = "mayocream/manga-ocr"
REVISION = "4380edba990b959c508752350955350c1c80c31c"

# Only Pillow's BILINEAR (PIL resample id 2).
PIL_BILINEAR = 2
SPECIAL_TOKENS = frozenset(["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"])

_HALFWIDTH_KATAKANA = [
    "。", "「", "」", "、", "・", "ヲ", "ァ", "ィ", "ゥ", "ェ", "ォ", "ャ", "ュ", "ョ", "ッ",
    "ー", "ア", "イ", "ウ", "エ", "オ", "カ", "キ", "ク", "ケ", "コ", "サ", "シ", "ス", "セ",
    "ソ", "タ", "チ", "ツ", "テ", "ト", "ナ", "ニ", "ヌ", "ネ", "ノ", "ハ", "ヒ", "フ", "ヘ",
    "ホ", "マ", "ミ", "ム", "メ", "モ", "ヤ", "ユ", "ヨ", "ラ", "リ", "ル", "レ", "ロ", "ワ",
    "ン", "ﾞ", "ﾟ",
]  # index = code point - U+FF61 (jaconv.h2z kana table)


def _config_file(name: str) -> HuggingFaceFile:
    return HuggingFaceFile(REPOSITORY, name, REVISION)


class ViTImageProcessor:
    """``processor.rs`` ``ViTImageProcessor``: grayscale, bilinear resize, rescale, normalise."""

    def __init__(self, path: Path):
        import json

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(data["resample"]) != PIL_BILINEAR:
            raise ValueError(f"unsupported Manga OCR PIL resampling filter {data['resample']}")
        self.do_normalize = bool(data["do_normalize"])
        self.do_resize = bool(data["do_resize"])
        self.image_mean = [float(v) for v in data["image_mean"]]
        self.image_std = [float(v) for v in data["image_std"]]
        self.size = int(data["size"])

    def preprocess(self, image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if image.width == 0 or image.height == 0:
            raise ValueError("cannot recognize an empty image")
        # MangaOcr.__call__ converts to Pillow's L (ITU-R 601-2) and back to RGB first.
        image = pillow_grayscale(image)
        if self.do_resize and (image.width != self.size or image.height != self.size):
            image = image.resize((self.size, self.size), Image.BILINEAR)
        pixels = raw_chw(image, device) / 255.0
        if self.do_normalize:
            pixels = normalize(pixels, self.image_mean, self.image_std)
        return pixels.to(dtype)


class Tokenizer:
    """``processor.rs`` ``Tokenizer``: one vocabulary entry per line, ``##`` subwords joined."""

    def __init__(self, path: Path):
        self.vocabulary = read_lines(path)
        if not self.vocabulary:
            raise ValueError("Manga OCR vocabulary is empty")

    def __len__(self) -> int:
        return len(self.vocabulary)

    def decode(self, token_ids: list[int]) -> str:
        pieces = []
        for token_id in token_ids:
            if not 0 <= token_id < len(self.vocabulary):
                raise ValueError(f"invalid Manga OCR token id {token_id}")
            token = self.vocabulary[token_id]
            if token not in SPECIAL_TOKENS:
                # BertJapaneseTokenizer's character subword tokenizer joins without spaces.
                pieces.append(token[2:] if token.startswith("##") else token)
        return post_process("".join(pieces))


def post_process(text: str) -> str:
    """``processor.rs`` ``post_process``: drop whitespace, ``…`` -> ``...``, runs of dots, full width."""
    text = "".join(c for c in text if not is_whitespace(c))
    text = text.replace("…", "...")
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if len(run) >= 2:
            out.extend("." * len(run))
        else:
            out.extend(run)
        run.clear()

    for character in text:
        if character in (".", "・"):
            run.append(character)
            continue
        flush()
        out.append(character)
    flush()
    return halfwidth_to_fullwidth("".join(out))


_VOICED_BASES = "ウカキクケコサシスセソタチツテトハヒフヘホ"
_VOICED_RESULTS = "ヴガギグゲゴザジズゼゾダヂヅデドバビブベボ"
_SEMI_VOICED_BASES = "ハヒフヘホ"
_SEMI_VOICED_RESULTS = "パピプペポ"


def _compose_katakana(base: str, mark: str) -> Optional[str]:
    bases, results = (_VOICED_BASES, _VOICED_RESULTS) if mark == "ﾞ" else (_SEMI_VOICED_BASES, _SEMI_VOICED_RESULTS)
    index = bases.find(base)
    if index < 0 or len(base) != 1:
        return None
    return results[index]


def halfwidth_to_fullwidth(text: str) -> str:
    """``jaconv.h2z(kana=True, ascii=True, digit=True)`` as ported in ``processor.rs``."""
    out: list[str] = []
    for character in text:
        if "!" <= character <= "~":
            out.append(chr(ord(character) + 0xFEE0))
        elif character == " ":
            out.append("　")
        elif character in ("ﾞ", "ﾟ"):
            previous = out.pop() if out else None
            composed = _compose_katakana(previous, character) if previous is not None else None
            if composed is not None:
                out.append(composed)
            else:
                if previous is not None:
                    out.append(previous)
                out.append(character)
        elif "｡" <= character <= "ﾝ":
            out.append(_HALFWIDTH_KATAKANA[ord(character) - 0xFF61])
        else:
            out.append(character)
    return "".join(out)


class _BeamHypotheses:
    """``model.rs`` ``BeamHypotheses`` (transformers ``BeamHypotheses``)."""

    def __init__(self, num_beams: int, length_penalty: float, early_stopping: bool):
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.beams: list[tuple[float, list[int]]] = []
        self.worst_score = math.inf
        self.done = False

    def add(self, tokens: list[int], sum_logprobs: float) -> None:
        score = sum_logprobs / (len(tokens) ** self.length_penalty)
        if len(self.beams) < self.num_beams or score > self.worst_score:
            self.beams.append((score, tokens))
            if len(self.beams) > self.num_beams:
                worst = min(range(len(self.beams)), key=lambda i: self.beams[i][0])  # first minimum, like Rust
                # Rust swap_remove: move the last element into the removed slot.
                self.beams[worst] = self.beams[-1]
                self.beams.pop()
            self.worst_score = min((s for s, _ in self.beams), default=math.inf)

    def is_done(self, best_sum_logprobs: float, current_length: int) -> bool:
        if len(self.beams) < self.num_beams:
            self.done = False
        elif self.early_stopping:
            self.done = True
        else:
            self.done = self.worst_score >= best_sum_logprobs / (current_length ** self.length_penalty)
        return self.done

    def best(self) -> list[int]:
        self.beams.sort(key=lambda beam: beam[0])
        if not self.beams:
            raise ValueError("Manga OCR beam search produced no hypothesis")
        return self.beams.pop()[1]


def _banned_ngram_tokens(sequence: list[int], ngram_size: int) -> list[int]:
    if ngram_size == 0 or len(sequence) + 1 < ngram_size:
        return []
    prefix_length = ngram_size - 1
    prefix = sequence[len(sequence) - prefix_length:]
    return [
        sequence[i + prefix_length]
        for i in range(len(sequence) - ngram_size + 1)
        if sequence[i : i + prefix_length] == prefix
    ]


@register
class MangaOcrProvider(OcrProvider):
    id = "manga-ocr"
    name = "Manga OCR"
    description = "Manga text recognition (ViT encoder, BERT decoder, beam search)."
    runtimes = ("torch",)

    catalog_queries = ('manga-ocr',)

    @classmethod
    def accepts(cls, facts) -> Optional[str]:
        """manga-ocr style VisionEncoderDecoder models with a BERT vocab and image processor."""
        if facts.runtime == "torch" and facts.model_type == "vision-encoder-decoder" and facts.has_file("vocab.txt") and facts.has_file("preprocessor_config.json"):
            return "torch"
        return None
    requires = {"torch": ("transformers",)}

    @classmethod
    def weight_sources(cls, runtime: Optional[str] = None) -> tuple[WeightSource, ...]:
        return (
            _config_file("config.json"),
            _config_file("preprocessor_config.json"),
            _config_file("vocab.txt"),
            _config_file("model.safetensors"),
        )

    def _load(self) -> None:
        from transformers import VisionEncoderDecoderModel

        runtime = self.runtime_instance()
        config_path, processor_path, vocabulary_path, weights_path = self.resolve_weights()
        self._device = runtime.torch_device
        self._dtype = runtime.dtype()

        import json

        config = json.loads(config_path.read_text(encoding="utf-8"))
        decoder, encoder = config["decoder"], config["encoder"]
        if encoder["hidden_size"] != decoder["hidden_size"]:
            raise ValueError("Manga OCR encoder and decoder hidden sizes must match")
        if config["num_beams"] <= 1:
            raise ValueError("Manga OCR requires beam search")
        if config["max_length"] <= 1:
            raise ValueError("Manga OCR max_length must exceed one")
        self._gen = {
            "decoder_start_token_id": int(config["decoder_start_token_id"]),
            "eos_token_id": int(config["eos_token_id"]),
            "max_length": int(config["max_length"]),
            "num_beams": int(config["num_beams"]),
            "length_penalty": float(config["length_penalty"]),
            "early_stopping": bool(config["early_stopping"]),
            "no_repeat_ngram_size": int(config["no_repeat_ngram_size"]),
            "vocab_size": int(decoder["vocab_size"]),
        }
        self.processor = ViTImageProcessor(processor_path)
        self.tokenizer = Tokenizer(vocabulary_path)
        if len(self.tokenizer) != self._gen["vocab_size"]:
            raise ValueError(
                f"Manga OCR vocabulary has {len(self.tokenizer)} entries but the decoder has {self._gen['vocab_size']} outputs"
            )

        model, info = VisionEncoderDecoderModel.from_pretrained(
            first_weights_dir([config_path]), dtype=self._dtype, output_loading_info=True
        )
        unexpected = [k for k in info["unexpected_keys"]]
        missing = [k for k in info["missing_keys"] if not k.endswith("position_ids")]
        if missing or unexpected:
            raise ValueError(f"Manga OCR checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        self._model = model.to(self._device).eval()

    def _unload(self) -> None:
        self._model = None
        self.tokenizer = None
        release_cache(self._device)

    def _recognize(self, crops: list[Image.Image]) -> list[str]:
        return [self._inference(crop) for crop in crops]

    @torch.inference_mode()
    def _inference(self, image: Image.Image) -> str:
        pixel_values = self.processor.preprocess(image, self._device, self._dtype)
        token_ids = self._beam_search(pixel_values)
        return self.tokenizer.decode(token_ids)

    def _beam_search(self, pixel_values: torch.Tensor) -> list[int]:
        """``model.rs`` ``beam_search``: transformers' BeamSearchScorer order, one image at a time."""
        gen = self._gen
        device = pixel_values.device
        num_beams = gen["num_beams"]
        vocab_size = gen["vocab_size"]
        eos = gen["eos_token_id"]
        encoder_hidden = self._model.encoder(pixel_values=pixel_values).last_hidden_state
        encoder_hidden = (
            encoder_hidden.unsqueeze(1).repeat(1, num_beams, 1, 1).view(num_beams, encoder_hidden.shape[1], -1)
        )

        sequences: list[list[int]] = [[gen["decoder_start_token_id"]] for _ in range(num_beams)]
        beam_scores = [-1.0e9] * num_beams
        beam_scores[0] = 0.0
        hypotheses = _BeamHypotheses(num_beams, gen["length_penalty"], gen["early_stopping"])

        while len(sequences[0]) < gen["max_length"]:
            # BeamSearchScorer evaluates its completion bound against the prefix before the new token.
            current_length = len(sequences[0])
            input_ids = torch.tensor(sequences, dtype=torch.long, device=device)
            logits = self._model.decoder(
                input_ids=input_ids, encoder_hidden_states=encoder_hidden, use_cache=False
            ).logits
            next_scores = logits[:, -1, :].float().log_softmax(-1)
            banned = torch.zeros((num_beams, vocab_size), dtype=torch.bool, device=device)
            if gen["no_repeat_ngram_size"] > 0:
                for row, sequence in enumerate(sequences):
                    for token in _banned_ngram_tokens(sequence, gen["no_repeat_ngram_size"]):
                        banned[row, token] = True
            next_scores = next_scores.masked_fill(banned, float("-inf"))
            next_scores = next_scores + torch.tensor(beam_scores, dtype=torch.float32, device=device).unsqueeze(1)

            candidate_scores, candidate_indices = (
                next_scores.view(1, num_beams * vocab_size).topk(2 * num_beams, dim=1, largest=True, sorted=True)
            )
            candidate_scores = candidate_scores[0].tolist()
            candidate_indices = candidate_indices[0].tolist()

            next_sequences: list[list[int]] = []
            next_scores_list: list[float] = []
            for rank, (score, flat_index) in enumerate(zip(candidate_scores, candidate_indices)):
                beam_index = flat_index // vocab_size
                token_id = flat_index % vocab_size
                if token_id == eos:
                    if rank < num_beams:
                        hypotheses.add(list(sequences[beam_index]), float(score))
                    continue
                next_sequences.append(sequences[beam_index] + [token_id])
                next_scores_list.append(float(score))
                if len(next_sequences) == num_beams:
                    break
            if len(next_sequences) != num_beams:
                raise ValueError("Manga OCR beam search could not fill the next beam")

            sequences = next_sequences
            beam_scores = next_scores_list
            if hypotheses.is_done(beam_scores[0], current_length):
                break

        if not hypotheses.done:
            for sequence, score in zip(sequences, beam_scores):
                hypotheses.add(sequence, float(score))
        best = hypotheses.best()
        if len(best) < gen["max_length"]:
            best.append(eos)
        return best
