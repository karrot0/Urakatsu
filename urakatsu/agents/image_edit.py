"""Editing page art with an image model: the agent's ``edit_image`` tool.

Not in Koharu. Typeset text cannot redraw stylized lettering such as sound effects; an
image model can, from an instruction ("translate this SFX into English, keep its style").

- ``ImageEditor`` is the model behind the tool, so providers stay swappable. The first is
  ``ChatGptImageEditor``: the signed-in ChatGPT account's image model through Codex's
  images endpoint (``POST {CODEX_BASE_URL}/images/edits``, as ``codex-rs/ext/image-generation``
  sends it: ``gpt-image-2``, the input as a data URL, the result as ``b64_json``). It has its
  own usage limit on the plan, and no mask: the model redraws everything it is sent.
  The second is ``UrakatsuCloudImageEditor``: Nano Banana 2.1 on Urakatsu Cloud, for a key with
  image permission (``urakatsu.remote.cloud.image_access``), billed to the cloud account.
- Because of that, the tool sends a crop, not the page (``plan_window``): the region plus a
  margin of context, widened over real page pixels to one of the model's output shapes
  (``SIZES``) and padded with white only where the page ends. The answer is scaled back to
  the crop, and only the region (with a soft edge into the margin) is kept, so art outside
  it is never touched.
- Full-page mode sends the whole page the same way and keeps all of it. The model's output
  is at most ``SIZES``' 1536 pixels on the long side, so a larger page comes back softer.
- The result is a new raster layer of kind ``edit`` above the others (``Workspace.
  add_raster_layer``): the art below stays, the layer can be hidden or deleted, a pipeline
  re-run never touches it, and one undo removes it.
"""

from __future__ import annotations

import asyncio
import base64
import io
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from urakatsu.providers.translation.remote.base import decode_response

from .backends.base import AgentError, failure_kind

if TYPE_CHECKING:  # pragma: no cover
    from .chatgpt import ChatGptAuth

IMAGE_MODEL = "gpt-image-2"
# The output shapes the image model draws (width, height).
SIZES = ((1024, 1024), (1536, 1024), (1024, 1536))
QUALITIES = ("auto", "low", "medium", "high")
EDIT_TIMEOUT = httpx.Timeout(30.0, read=300.0)
# Inputs are scaled down to this on the long side; the model works at ``SIZES`` anyway.
MAX_INPUT_SIDE = 2048
# Context around a region: this share of its longer side on every side, at least ``MIN_MARGIN``.
CONTEXT_MARGIN = 0.35
MIN_MARGIN = 32
RETRY_DELAY = 2.0


@dataclass(frozen=True)
class Box:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def tuple(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)


class ImageEditor(ABC):
    """An image model that redraws an image from an instruction."""

    name: str = ""

    @abstractmethod
    async def edit(self, image: Image.Image, prompt: str, size: tuple[int, int], quality: str = "auto") -> Image.Image:
        """``image`` redrawn per ``prompt`` at ``size`` (one of ``SIZES``)."""


class ChatGptImageEditor(ImageEditor):
    name = "ChatGPT"

    def __init__(self, auth: "ChatGptAuth", transport: Optional[httpx.AsyncBaseTransport] = None):
        self.auth = auth
        self.transport = transport

    async def edit(self, image: Image.Image, prompt: str, size: tuple[int, int], quality: str = "auto") -> Image.Image:
        from .chatgpt import CODEX_BASE_URL

        body = {
            "model": IMAGE_MODEL,
            "images": [{"image_url": "data:image/png;base64," + base64.b64encode(_png(image)).decode("ascii")}],
            "prompt": prompt,
            "size": f"{size[0]}x{size[1]}",
            "quality": quality,
            "n": 1,
        }
        url = f"{CODEX_BASE_URL}/images/edits"
        refreshed = retried = False
        async with httpx.AsyncClient(transport=self.transport, timeout=EDIT_TIMEOUT) as client:
            while True:
                headers = await asyncio.to_thread(self.auth.headers, refreshed)
                try:
                    response = await client.post(url, headers=headers, json=body)
                    if response.status_code == 401 and not refreshed:
                        refreshed = True
                        continue
                    data = decode_response("chatgpt", response.status_code, response.text)
                except httpx.HTTPError as error:
                    failure: Exception = AgentError(f"ChatGPT image edit failed: {error}", transient=isinstance(error, httpx.TransportError))
                except Exception as error:
                    failure = error
                else:
                    break
                if retried or failure_kind(failure) != "transient":
                    raise failure
                retried = True
                await asyncio.sleep(RETRY_DELAY)
        try:
            encoded = data["data"][0]["b64_json"]
            result = Image.open(io.BytesIO(base64.b64decode(encoded)))
            result.load()
        except (KeyError, IndexError, TypeError, ValueError, OSError) as error:
            raise AgentError(f"ChatGPT returned no image ({error})") from error
        return result.convert("RGB")


# Nano Banana draws 1K images; a crop no longer than this is one image on the cloud's meter.
CLOUD_SIDE = 1024


class UrakatsuCloudImageEditor(ImageEditor):
    """Nano Banana 2.1 on Urakatsu Cloud, through the ``nano-banana-2.1`` provider's cloud client.

    The whole crop is the edit area (the tool keeps only the region afterwards), sent at most
    ``CLOUD_SIDE`` pixels on its long side so it counts as one image, and scaled back to ``size``.
    Nano Banana has no quality setting, so ``quality`` is not sent."""

    name = "Urakatsu Cloud"

    async def edit(self, image: Image.Image, prompt: str, size: tuple[int, int], quality: str = "auto") -> Image.Image:
        from urakatsu.providers.inpainting.nano_banana import NanoBananaProvider
        from urakatsu.remote import CLOUD

        sent = image.convert("RGB")
        if max(sent.size) > CLOUD_SIDE:
            scale = CLOUD_SIDE / max(sent.size)
            sent = sent.resize((max(1, round(sent.width * scale)), max(1, round(sent.height * scale))), Image.Resampling.LANCZOS)
        everywhere = np.full((sent.height, sent.width), 255, np.uint8)
        provider = NanoBananaProvider({}, None, CLOUD)
        try:
            await provider.load()
            result = await asyncio.to_thread(provider._fill, sent, everywhere, prompt)
        except Exception as error:
            raise AgentError(f"Urakatsu Cloud image edit failed: {error}") from error
        finally:
            provider.unload()
        return result.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def _png(image: Image.Image) -> bytes:
    if max(image.size) > MAX_INPUT_SIDE:
        scale = MAX_INPUT_SIDE / max(image.size)
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def output_size(width: int, height: int) -> tuple[int, int]:
    """The model output shape closest to ``width``:``height``."""
    aspect = math.log(width / height)
    return min(SIZES, key=lambda size: abs(math.log(size[0] / size[1]) - aspect))


def _grow(low: int, high: int, length: int, limit: int) -> tuple[int, int]:
    """Widens ``[low, high)`` to ``length`` around its centre, within ``[0, limit)``."""
    extra = max(length - (high - low), 0)
    low, high = low - extra // 2, high + extra - extra // 2
    if low < 0:
        high, low = min(high - low, limit), 0
    if high > limit:
        low, high = max(low - (high - limit), 0), limit
    return low, high


def plan_window(region: Box, page: tuple[int, int], margin: bool = True) -> tuple[Box, tuple[int, int]]:
    """The crop to send for ``region`` and the output size to ask for: the region plus context,
    widened over the page towards the output shape."""
    width, height = page
    pad = max(round(max(region.width, region.height) * CONTEXT_MARGIN), MIN_MARGIN) if margin else 0
    box = Box(max(region.left - pad, 0), max(region.top - pad, 0), min(region.right + pad, width), min(region.bottom + pad, height))
    size = output_size(box.width, box.height)
    target = size[0] / size[1]
    if box.width / box.height < target:
        left, right = _grow(box.left, box.right, round(box.height * target), width)
        box = Box(left, box.top, right, box.bottom)
    else:
        top, bottom = _grow(box.top, box.bottom, round(box.width / target), height)
        box = Box(box.left, top, box.right, bottom)
    return box, size


def pad_to(image: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, tuple[int, int]]:
    """``image`` centred on white to ``size``'s shape; returns it and where the image sits."""
    target = size[0] / size[1]
    width, height = image.size
    if width / height < target:
        width = round(height * target)
    else:
        height = round(image.width / target)
    if (width, height) == image.size:
        return image, (0, 0)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    offset = ((width - image.width) // 2, (height - image.height) // 2)
    canvas.paste(image, offset)
    return canvas, offset


def feather_for(region: Box, window: Box) -> int:
    """The soft edge: about a tenth of the region's short side, no wider than the margin around it."""
    room = min(region.left - window.left, region.top - window.top, window.right - region.right, window.bottom - region.bottom)
    return max(0, min(round(min(region.width, region.height) * 0.1), 24, room))


def composite(result: Image.Image, padded: tuple[int, int], offset: tuple[int, int], window: Box, region: Box, page: tuple[int, int]) -> Image.Image:
    """A page-sized RGBA layer holding ``result`` (the model's answer for the padded window)
    inside ``region`` only, opaque there and fading out across the margin."""
    scaled = result.resize(padded, Image.Resampling.LANCZOS)
    crop = scaled.crop((offset[0], offset[1], offset[0] + window.width, offset[1] + window.height)).convert("RGBA")
    feather = feather_for(region, window)
    mask = Image.new("L", (window.width, window.height), 0)
    inner = (region.left - window.left, region.top - window.top, region.right - window.left - 1, region.bottom - window.top - 1)
    if feather:
        grown = (inner[0] - feather // 2, inner[1] - feather // 2, inner[2] + feather // 2, inner[3] + feather // 2)
        ImageDraw.Draw(mask).rectangle(grown, fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(feather / 3))
        ImageDraw.Draw(mask).rectangle(inner, fill=255)
    else:
        ImageDraw.Draw(mask).rectangle(inner, fill=255)
    crop.putalpha(mask)
    layer = Image.new("RGBA", page, (0, 0, 0, 0))
    layer.paste(crop, (window.left, window.top))
    return layer


async def edit_region(
    editor: ImageEditor, art: Image.Image, region: Optional[Box], prompt: str, quality: str = "auto"
) -> tuple[Image.Image, Box, tuple[int, int]]:
    """Runs one edit; returns the page-sized layer, the window sent and the output size asked for.
    ``region`` ``None`` is the whole page."""
    page = art.size
    whole = Box(0, 0, *page)
    target = whole if region is None else region
    window, size = plan_window(target, page, margin=region is not None)
    padded, offset = pad_to(art.crop(window.tuple()), size)
    result = await editor.edit(padded, prompt, size, quality)
    return composite(result, padded.size, offset, window, target, page), window, size
