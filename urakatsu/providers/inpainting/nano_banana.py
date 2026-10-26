"""Nano Banana 2.1: Google's image model, on your Gemini API key or on Urakatsu Cloud.

Two runtimes:

- ``gemini``: calls Google's Interactions API with your key (``api_key`` in the options, else
  ``GEMINI_API_KEY``). Google bills your account, per image.
- ``cloud``: Urakatsu Cloud runs this same provider with its own key and bills the account's
  image generation quota; ``Provider.__init__`` binds the inference methods to the cloud client.

Nano Banana edits from text and has no mask input, so a call sends the crop, the mask as a
second image (white where to paint) and an instruction that names it. The crop is padded to the
nearest aspect ratio Nano Banana draws, so its answer lines up with the crop, and only the
masked pixels are copied back: nothing outside the selection changes, whatever the model draws
there. The output size is the smallest of 1K, 2K and 4K that covers the crop (``image_size``),
which is also what Google charges for and what Urakatsu Cloud counts.

It follows edit instructions (``instructed``), like FLUX.2 Klein: generative fill sends the
selection as one crop with its surroundings, and sound effects are redrawn from the instruction.
Removing lettering goes through the stage's tiling, one image per tile, so LaMa stays the
cheaper choice for plain cleanup. ``last_usage`` keeps the last reply's token counts, which
Urakatsu Cloud records as what Google charged.
"""

from __future__ import annotations

import base64
import io
import math
import os
from typing import Any, ClassVar, Optional

import httpx
import numpy as np
from PIL import Image
from pydantic import Field

from urakatsu.core.errors import ProcessingError, ProviderUnavailable
from urakatsu.core.registry import register
from urakatsu.core.types import Availability
from urakatsu.remote import CLOUD

from ..base import InpaintingProvider, ProviderSettings

GEMINI = "gemini"
KEY_ENV = "GEMINI_API_KEY"
URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
MODEL = "gemini-nano-banana-2.1"
# Aspect ratios Nano Banana draws (ImageResponseFormat.aspect_ratio in the Interactions API).
ASPECTS = ("1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9", "1:4", "4:1", "1:8", "8:1")
# Output sizes by the longest side they cover.
SIZES = ((1024, "1K"), (2048, "2K"), (4096, "4K"))
TIMEOUT = httpx.Timeout(30.0, read=180.0)

REMOVE = (
    "The first image is a crop of a comic page. The second image is a mask: white marks the area to edit. "
    "Remove all lettering and text inside the white area and redraw the artwork that was behind it, matching the "
    "surrounding lines, screentone and shading. Keep everything outside the white area exactly the same, and keep "
    "the image's size and framing."
)
FILL = (
    "The first image is a crop of a comic page. The second image is a mask: white marks the area to edit. "
    "Edit only the white area, following this instruction: {prompt}\n"
    "Match the page's drawing style, line weight, screentone and shading. Keep everything outside the white area "
    "exactly the same, and keep the image's size and framing."
)


class NanoBananaSettings(ProviderSettings):
    api_key: Optional[str] = Field(
        default=None,
        description="Gemini API key from Google AI Studio, for the gemini runtime (else GEMINI_API_KEY). Urakatsu Cloud needs none.",
    )


def image_size(width: int, height: int) -> str:
    side = max(width, height)
    return next((size for limit, size in SIZES if side <= limit), SIZES[-1][1])


def nearest_aspect(width: int, height: int) -> str:
    ratio = math.log(width / height)
    return min(ASPECTS, key=lambda a: abs(math.log(int(a.split(":")[0]) / int(a.split(":")[1])) - ratio))


def pad_to_aspect(image: Image.Image, mask: np.ndarray, aspect: str) -> tuple[Image.Image, np.ndarray, tuple[int, int]]:
    """``image`` and ``mask`` centred on a canvas of ``aspect``, edges extended; also the crop's offset."""
    a, b = (int(n) for n in aspect.split(":"))
    width, height = image.size
    canvas = (round(height * a / b), height) if width * b < height * a else (width, round(width * b / a))
    left, top = (canvas[0] - width) // 2, (canvas[1] - height) // 2
    pad = ((top, canvas[1] - height - top), (left, canvas[0] - width - left))
    pixels = np.asarray(image)
    padded = np.pad(pixels, pad + ((0, 0),) * (pixels.ndim - 2), mode="edge")
    return Image.fromarray(padded, image.mode), np.pad(mask, pad), (left, top)


def composite(original: Image.Image, generated: Image.Image, mask: np.ndarray) -> Image.Image:
    """``original`` with the masked pixels taken from ``generated`` (already the crop's size)."""
    pixels = np.asarray(original).copy()
    painted = np.asarray(generated.convert(original.mode))
    pixels[mask != 0] = painted[mask != 0]
    return Image.fromarray(pixels, original.mode)


def generated_image(reply: dict[str, Any]) -> bytes:
    """The last image Nano Banana returned, from the interaction's model output steps."""
    data = None
    for step in reply.get("steps") or []:
        for block in step.get("content") or []:
            if block.get("type") == "image" and block.get("data"):
                data = block["data"]
    if data is None:
        raise ProcessingError(f"Nano Banana returned no image ({reply.get('status') or 'no image'}); try another instruction")
    return base64.b64decode(data)


def _png_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=1)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@register
class NanoBananaProvider(InpaintingProvider):
    id: ClassVar[str] = "nano-banana-2.1"
    name: ClassVar[str] = "Nano Banana 2.1"
    description: ClassVar[str] = "Google's image model, on your Gemini API key or Urakatsu Cloud. Generative fill and sound effects from an instruction."
    runtimes: ClassVar[tuple[str, ...]] = (GEMINI, CLOUD)
    remotable = False  # a Urakatsu worker cannot run it; only Google and Urakatsu Cloud can
    prompted: ClassVar[bool] = True
    instructed: ClassVar[bool] = True
    sees_mask: ClassVar[bool] = True  # gets the mask as a second image, so a wide crop is fine
    Settings: ClassVar[type[ProviderSettings]] = NanoBananaSettings
    # Tests swap in a mock; Urakatsu Cloud runs many calls at once on one client.
    transport: ClassVar[Optional[httpx.BaseTransport]] = None

    def __init__(self, settings=None, runtimes=None, runtime=None):
        super().__init__(settings, runtimes, runtime)
        self.last_usage: dict[str, Any] = {}

    @classmethod
    def api_key(cls, settings: Optional[dict] = None) -> Optional[str]:
        """The Gemini key: ``settings``, else the saved options, else ``GEMINI_API_KEY``."""
        if settings is None:
            from urakatsu.core.config import load_config

            try:
                settings = load_config().processor.get(cls.id, {})
            except Exception:  # noqa: BLE001 - an unreadable config leaves the environment
                settings = {}
        value = settings.get("api_key") or os.environ.get(KEY_ENV) or ""
        return value.strip() or None

    @classmethod
    def runtime_availability(cls, runtime: str) -> Availability:
        if runtime == GEMINI:
            return Availability.yes() if cls.api_key() else Availability.no(f"add a Gemini API key in {cls.name}'s options, or set {KEY_ENV}")
        return super().runtime_availability(runtime)

    def attached(self) -> bool:
        """On the gemini runtime with a key in its own options, it needs nothing else here (the key
        may not be saved yet, or comes from Urakatsu Cloud's worker)."""
        return self.runtime_name == GEMINI and self.api_key(self.settings.model_dump()) is not None

    def _load(self) -> None:
        if self.api_key(self.settings.model_dump()) is None:
            raise ProviderUnavailable(f"{self.name}: add a Gemini API key in its options, or set {KEY_ENV}")

    def _inpaint(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        return self.edit(image, mask, REMOVE)

    def _fill(self, image: Image.Image, mask: np.ndarray, prompt: Optional[str]) -> Image.Image:
        return self.edit(image, mask, FILL.format(prompt=prompt) if prompt else REMOVE)

    def edit(self, image: Image.Image, mask: np.ndarray, instruction: str) -> Image.Image:
        """One Nano Banana call: ``image`` with the area ``mask`` marks redrawn per ``instruction``."""
        if mask.shape != (image.height, image.width):
            raise ProcessingError("the mask does not match the image")
        key = self.api_key(self.settings.model_dump())
        if key is None:
            raise ProviderUnavailable(f"{self.name}: add a Gemini API key in its options, or set {KEY_ENV}")
        aspect = nearest_aspect(image.width, image.height)
        padded, padded_mask, (left, top) = pad_to_aspect(image.convert("RGB"), mask, aspect)
        request = {
            "model": MODEL,
            "input": [
                {"type": "text", "text": instruction},
                {"type": "image", "mime_type": "image/png", "data": _png_b64(padded)},
                {"type": "image", "mime_type": "image/png", "data": _png_b64(Image.fromarray(np.where(padded_mask != 0, 255, 0).astype(np.uint8), "L"))},
            ],
            "response_format": {"type": "image", "mime_type": "image/jpeg", "aspect_ratio": aspect, "image_size": image_size(image.width, image.height)},
            # Thinking tokens are billed; an edit needs little.
            "generation_config": {"thinking_level": "minimal"},
            "store": False,
        }
        with httpx.Client(transport=self.transport, timeout=TIMEOUT) as client:
            try:
                response = client.post(URL, json=request, headers={"x-goog-api-key": key})
            except httpx.HTTPError as error:
                raise ProcessingError(f"cannot reach Google: {error}") from error
        if response.status_code in (401, 403):
            raise ProviderUnavailable(f"Google rejected the Gemini API key ({response.status_code})")
        if response.status_code == 429:
            raise ProcessingError("Nano Banana is busy or the key's quota is used up; try again in a minute")
        if response.status_code >= 400:
            raise ProcessingError(f"Nano Banana refused the edit: {response.text[:300]}")
        reply = response.json()
        self.last_usage = reply.get("usage") or {}
        generated = Image.open(io.BytesIO(generated_image(reply))).convert("RGB").resize(padded.size, Image.Resampling.LANCZOS)
        return composite(image, generated.crop((left, top, left + image.width, top + image.height)), mask)
