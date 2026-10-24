"""PaddleOCR-VL-1.6 text recognition.

Upstream sources:
- ``paddle_ocr_vl/mod.rs``: constants and the torch inference call.
- ``paddle_ocr_vl/processor.rs``: task prompts, ``smart_resize``, the
  antialiased bicubic resize, patchify, prompt encoding and decoding.
- ``paddle_ocr_vl/model.rs``: the torch greedy loop (repetition penalty
  over generated tokens only, EOS stop, ``max_new_tokens``).
- ``paddle_ocr_vl_quantized/mod.rs``: the GGUF path through llama.cpp,
  which is the default runtime. Chat prompt, image attachment, temperature 0,
  repeat penalty 1.2 over the last 512 tokens, 512 new tokens.

Runtimes:
- ``llama``: ``PaddlePaddle/PaddleOCR-VL-1.6-GGUF`` + mmproj through llama-server (newest llama.cpp release). The
  prompt is the GGUF's own chat template rendered with the media marker, so it matches
  ``render_chat_prompt([user: marker + task.prompt()])``. The image goes through llama.cpp's
  mtmd projector and the sampler is greedy with the repeat penalty. It runs on the newest
  llama.cpp release as ``llama-server``: the chat template is rendered here and sent to the
  raw ``/completion`` endpoint with the crop as multimodal data.
- ``torch``: ``PaddlePaddle/PaddleOCR-VL-1.6`` in transformers' native ``paddleocr_vl`` model
  (the port follows the same transformers revision). Generation runs through
  ``generate`` with a logits processor that reproduces the generated-only penalty.

Divergences:
- The torch image resize uses Pillow-free torch ``antialias=True`` bicubic on the uint8 raster,
  as upstream does with ``internal_upsample_bicubic2d_aa``. Tensors are moved to the device after
  resizing.
- The llama chat handler subclass changes the GGUF template in one place: an image content part
  renders as its URL, which the handler replaces with the media marker. For a string message the
  template is unchanged, so the prompt matches upstream.
"""

from __future__ import annotations

import base64
import io
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from PIL import Image

from urakatsu.core.registry import register
from urakatsu.providers.base import OcrProvider
from urakatsu.runtimes.store import HuggingFaceFile, WeightSource

MAX_NEW_TOKENS = 512
REPETITION_PENALTY = 1.2
# llama path: the context must hold the image tokens (<= max_pixels / 28^2) plus the reply.
LLAMA_CONTEXT = 4096
LLAMA_BATCH = 512

GGUF_REPOSITORY = "PaddlePaddle/PaddleOCR-VL-1.6-GGUF"
GGUF_REVISION = "511b09642bb324401f15f97cc23bc67e8f0a291d"
TORCH_REPOSITORY = "PaddlePaddle/PaddleOCR-VL-1.6"
TORCH_REVISION = "66317acc4c9fc17bd154591ce650735cd2855f3e"

# processor.rs: upstream accepts only Transformers' bicubic resampling (PIL id 3).
PIL_BICUBIC = 3


class PaddleOCRVLTask(str, Enum):
    OCR = "Ocr"
    TABLE = "Table"
    FORMULA = "Formula"
    CHART = "Chart"
    SPOTTING = "Spotting"
    SEAL = "Seal"

    @property
    def prompt(self) -> str:
        return {
            PaddleOCRVLTask.OCR: "OCR:",
            PaddleOCRVLTask.TABLE: "Table Recognition:",
            PaddleOCRVLTask.FORMULA: "Formula Recognition:",
            PaddleOCRVLTask.CHART: "Chart Recognition:",
            PaddleOCRVLTask.SPOTTING: "Spotting:",
            PaddleOCRVLTask.SEAL: "Seal Recognition:",
        }[self]


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """processor.rs ``smart_resize`` (Transformers ``smart_resize``): sizes that are multiples of ``factor``."""
    if height < factor:
        width = round(width * factor / height)
        height = factor
    if width < factor:
        height = round(height * factor / width)
        width = factor
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = max(factor, math.floor(height / beta / factor) * factor)
        resized_width = max(factor, math.floor(width / beta / factor) * factor)
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor
    return resized_height, resized_width



@dataclass(frozen=True)
class ImageSettings:
    """``PaddleOCRVLImageProcessor``: values from preprocessor_config.json, defaults otherwise."""

    do_normalize: bool = True
    do_rescale: bool = True
    do_resize: bool = True
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    max_pixels: int = 1_003_520
    min_pixels: int = 112_896
    merge_size: int = 2
    patch_size: int = 14
    rescale_factor: float = 1.0 / 255.0
    resample: int = PIL_BICUBIC
    temporal_patch_size: int = 1

    @classmethod
    def from_json(cls, path: Path) -> "ImageSettings":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {key: data[key] for key in cls.__dataclass_fields__ if key in data}
        for key in ("image_mean", "image_std"):
            if key in known:
                known[key] = tuple(float(v) for v in known[key])
        return cls(**known)


def _task_of(value: "PaddleOCRVLTask | str") -> PaddleOCRVLTask:
    return value if isinstance(value, PaddleOCRVLTask) else PaddleOCRVLTask(value)


def _png_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


# The GGUF's chat template (tokenizer.chat_template of PaddleOCR-VL-1.6-GGUF) with one change:
# an image part renders as its own URL, which is set to llama-server's media marker.
# A string message renders exactly as in the GGUF.
GGUF_CHAT_TEMPLATE = """{%- if not add_generation_prompt is defined -%}
    {%- set add_generation_prompt = true -%}
{%- endif -%}
{%- if not cls_token is defined -%}
    {%- set cls_token = "<|begin_of_sentence|>" -%}
{%- endif -%}
{%- if not eos_token is defined -%}
    {%- set eos_token = "</s>" -%}
{%- endif -%}
{{- cls_token -}}
{%- for message in messages -%}
    {%- if message["role"] == "user" -%}
        {{- "User: " -}}
      {%- if message["content"] is string -%}
        {{- message["content"] }}
      {%- else -%}
        {%- for content in message["content"] -%}
            {%- if content["type"] == "image_url" -%}
                {{ content["image_url"]["url"] }}
            {%- endif -%}
        {%- endfor -%}
        {%- for content in message["content"] -%}
            {%- if content["type"] == "text" -%}
                {{ content["text"] }}
            {%- endif -%}
        {%- endfor -%}
      {%- endif -%}
        {{ "\\n" -}}
    {%- elif message["role"] == "assistant" -%}
        {{- "Assistant:\\n" -}}
      {%- if message["content"] is string -%}
        {{- message["content"] }}
      {%- else -%}
        {%- for content in message["content"] -%}
            {%- if content["type"] == "text" -%}
                {{ content["text"] }}
            {%- endif -%}
        {%- endfor -%}
      {%- endif -%}
        {{ eos_token -}}
    {%- elif message["role"] == "system" -%}
      {%- if message["content"] is string -%}
        {{- message["content"] }}
      {%- else -%}
        {%- for content in message["content"] -%}
            {%- if content["type"] == "text" -%}
                {{ content["text"] + "\\n" }}
            {%- endif -%}
        {%- endfor -%}
      {%- endif -%}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- "Assistant:\\n" -}}
{%- endif -%}
"""


def render_gguf_prompt(messages: list[dict]) -> str:
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    return ImmutableSandboxedEnvironment().from_string(GGUF_CHAT_TEMPLATE).render(messages=messages, add_generation_prompt=True)


@register
class PaddleOcrVlProvider(OcrProvider):
    id = "paddleocr-vl-1.6"
    name = "PaddleOCR-VL-1.6"
    description = "Vision-language OCR (PaddleOCR-VL-1.6): GGUF through llama.cpp, or full precision through torch."
    runtimes = ("llama", "torch")

    weight_roles = {"llama": ("model", "projector")}

    catalog_queries = ('PaddleOCR-VL',)

    @classmethod
    def accepts(cls, facts) -> Optional[str]:
        """PaddleOCR-VL GGUFs with their vision projector, or the full-precision Transformers model."""
        if facts.runtime == "llama" and facts.architecture == "paddleocr" and facts.projector:
            return "llama"
        if facts.runtime == "torch" and "PaddleOCRVLForConditionalGeneration" in facts.architectures:
            return "torch"
        return None
    requires = {"torch": ("transformers", "tokenizers")}

    def __init__(self, settings=None, runtimes=None, runtime=None):
        super().__init__(settings, runtimes, runtime)
        self.task = PaddleOCRVLTask.OCR

    @classmethod
    def weight_sources(cls, runtime: Optional[str] = None) -> tuple[WeightSource, ...]:
        if runtime == "torch":
            return tuple(
                HuggingFaceFile(TORCH_REPOSITORY, name, TORCH_REVISION)
                for name in ("config.json", "model.safetensors", "preprocessor_config.json", "tokenizer.json")
            )
        return (
            HuggingFaceFile(GGUF_REPOSITORY, "PaddleOCR-VL-1.6-GGUF.gguf", GGUF_REVISION),
            HuggingFaceFile(GGUF_REPOSITORY, "PaddleOCR-VL-1.6-GGUF-mmproj.gguf", GGUF_REVISION),
        )

    # -- lifecycle -----------------------------------------------------------------------------

    def _load(self) -> None:
        if self.runtime_name == "torch":
            self._load_torch()
        else:
            self._load_llama()

    def _load_llama(self) -> None:
        model_path, projector_path = self.resolve_weights()
        server = self.runtime_instance().launch(model_path, projector_path, context=LLAMA_CONTEXT, batch=LLAMA_BATCH)
        if not server.vision:
            server.close()
            raise ValueError("PaddleOCR-VL-1.6 projector does not support images")
        self._server = server

    def _load_torch(self) -> None:
        from tokenizers import Tokenizer
        from transformers import PaddleOCRVLForConditionalGeneration

        runtime = self.runtime_instance()
        config_path, weights_path, processor_path, tokenizer_path = self.resolve_weights()
        self._device = runtime.torch_device
        self._dtype = runtime.dtype()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self._image_token_id = int(config["image_token_id"])
        self._eos_token_id = int(config["eos_token_id"]) if "eos_token_id" in config else 2
        self._image = ImageSettings.from_json(processor_path)
        if self._image.resample != PIL_BICUBIC:
            raise ValueError("PaddleOCR-VL only supports Transformers' bicubic resampling mode")
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        model, info = PaddleOCRVLForConditionalGeneration.from_pretrained(
            str(config_path.parent), dtype=self._dtype, output_loading_info=True
        )
        missing = [k for k in info.get("missing_keys", []) if "rotary" not in k]
        if missing or info.get("unexpected_keys"):
            raise ValueError(
                f"PaddleOCR-VL checkpoint mismatch: missing={missing[:5]} unexpected={info.get('unexpected_keys', [])[:5]}"
            )
        self._model = model.to(self._device).eval()

    def _unload(self) -> None:
        if getattr(self, "_server", None) is not None:
            self._server.close()
        self._server = None
        self._model = None
        self._tokenizer = None
        if getattr(self, "_device", None) is not None:
            from urakatsu.runtimes.torch_runtime import release_cache

            release_cache(self._device)

    # -- inference -----------------------------------------------------------------------------

    def _recognize(self, crops: list[Image.Image]) -> list[str]:
        return [self.inference(crop, self.task) for crop in crops]

    def inference(self, image: Image.Image, task: "PaddleOCRVLTask | str" = PaddleOCRVLTask.OCR) -> str:
        task = _task_of(task)
        if self.runtime_name == "torch":
            return self._inference_torch(image, task)
        return self._inference_llama(image, task)

    def _inference_llama(self, image: Image.Image, task: PaddleOCRVLTask) -> str:
        server = self._server
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": server.media_marker}},
                    {"type": "text", "text": task.prompt},
                ],
            }
        ]
        return server.complete(
            server.strip_bos(render_gguf_prompt(messages)),
            images=[buffer.getvalue()],
            temperature=0.0,
            repeat_penalty=REPETITION_PENALTY,
            repeat_last_n=MAX_NEW_TOKENS,
            n_predict=MAX_NEW_TOKENS,
        )

    def _preprocess(self, image: Image.Image, task: PaddleOCRVLTask):
        import numpy as np
        import torch
        import torch.nn.functional as F

        settings = self._image
        if task is PaddleOCRVLTask.SPOTTING and image.width < 1500 and image.height < 1500:
            image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)
        rgb = image.convert("RGB")
        max_pixels = 2048 * 28 * 28 if task is PaddleOCRVLTask.SPOTTING else settings.max_pixels
        height, width = smart_resize(
            rgb.height, rgb.width, settings.patch_size * settings.merge_size, settings.min_pixels, max_pixels
        )
        pixels = torch.from_numpy(np.asarray(rgb, dtype=np.uint8).copy()).permute(2, 0, 1)[None]
        if settings.do_resize:
            # Resizes the uint8 raster on the host (antialiased bicubic), then copies it over.
            pixels = F.interpolate(pixels, size=(height, width), mode="bicubic", align_corners=False, antialias=True)
        pixels = pixels.to(self._device).to(torch.float32)
        if settings.do_rescale:
            pixels = pixels * settings.rescale_factor
        if settings.do_normalize:
            mean = torch.tensor(settings.image_mean, dtype=torch.float32, device=self._device).view(1, 3, 1, 1)
            std = torch.tensor(settings.image_std, dtype=torch.float32, device=self._device).view(1, 3, 1, 1)
            pixels = (pixels - mean) / std
        grid_t = 1
        grid_h = height // settings.patch_size
        grid_w = width // settings.patch_size
        patch = settings.patch_size
        values = (
            pixels.view(1, grid_t, 3, grid_h, patch, grid_w, patch)
            .permute(0, 1, 3, 5, 2, 4, 6)
            .reshape(grid_t * grid_h * grid_w, 3, patch, patch)
        )
        return values, (grid_t, grid_h, grid_w)

    def _encode_prompt(self, task: PaddleOCRVLTask, grid: tuple[int, int, int]) -> list[int]:
        merge = self._image.merge_size
        image_tokens = (grid[0] * grid[1] * grid[2]) // (merge * merge)
        prompt = (
            "<|begin_of_sentence|>User: <|IMAGE_START|>"
            + "<|IMAGE_PLACEHOLDER|>" * image_tokens
            + "<|IMAGE_END|>"
            + task.prompt
            + "\nAssistant:\n"
        )
        return self._tokenizer.encode(prompt, add_special_tokens=False).ids

    def _inference_torch(self, image: Image.Image, task: PaddleOCRVLTask) -> str:
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList

        pixel_values, grid = self._preprocess(image, task)
        ids = self._encode_prompt(task, grid)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self._device)
        prompt_length = len(ids)

        class GeneratedRepetitionPenalty(LogitsProcessor):
            """``greedy_token``: penalise the unique generated tokens, never the prompt."""

            def __call__(self, input_ids, scores):
                generated = input_ids[:, prompt_length:]
                if generated.shape[1] == 0:
                    return scores
                selected = scores.gather(1, generated)
                adjusted = torch.where(selected < 0, selected * REPETITION_PENALTY, selected / REPETITION_PENALTY)
                return scores.scatter(1, generated, adjusted)

        with torch.inference_mode():
            output = self._model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                pixel_values=pixel_values.to(self._dtype),
                image_grid_thw=torch.tensor([list(grid)], dtype=torch.long, device=self._device),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.0,
                logits_processor=LogitsProcessorList([GeneratedRepetitionPenalty()]),
                eos_token_id=self._eos_token_id,
                pad_token_id=self._eos_token_id,
                use_cache=True,
            )
        generated = output[0, prompt_length:].tolist()
        tokens = []
        for token in generated:
            if token == self._eos_token_id:
                break
            tokens.append(token)
        return self._tokenizer.decode(tokens, skip_special_tokens=False)
