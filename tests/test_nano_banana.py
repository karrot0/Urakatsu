"""Nano Banana 2.1 on a Gemini API key, against a mock of Google's Interactions API."""

from __future__ import annotations

import base64
import io
import json

import httpx
import numpy as np
import pytest
from PIL import Image

from urakatsu.core.errors import ProcessingError, ProviderUnavailable
from urakatsu.providers.inpainting import nano_banana
from urakatsu.providers.inpainting.nano_banana import NanoBananaProvider, image_size, nearest_aspect, pad_to_aspect


def google(seen: list[dict], status: int = 200):
    """Answers with a solid red image of the size the request's aspect ratio implies."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append({"body": body, "key": request.headers.get("x-goog-api-key")})
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "nope"}})
        a, b = (int(n) for n in body["response_format"]["aspect_ratio"].split(":"))
        out = io.BytesIO()
        Image.new("RGB", (1024, round(1024 * b / a)), (255, 0, 0)).save(out, format="JPEG", quality=95)
        content = [{"type": "image", "data": base64.b64encode(out.getvalue()).decode("ascii")}]
        reply = {"status": "completed", "steps": [{"type": "model_output", "content": content}], "usage": {"total_input_tokens": 1200, "total_output_tokens": 1120}}
        return httpx.Response(200, json=reply)

    return httpx.MockTransport(handler)


@pytest.fixture
def crop():
    image = Image.new("RGB", (300, 500), (10, 20, 30))
    mask = np.zeros((500, 300), np.uint8)
    mask[100:200, 50:150] = 255
    return image, mask


def provider(monkeypatch, seen, status=200, **settings):
    monkeypatch.setattr(NanoBananaProvider, "transport", google(seen, status))
    return NanoBananaProvider({"api_key": "g-key", **settings}, None, "gemini")


async def test_a_fill_changes_only_the_selection(monkeypatch, crop):
    seen: list[dict] = []
    image, mask = crop
    model = provider(monkeypatch, seen)
    await model.load()
    result = np.asarray(model._fill(image, mask, "replace the sound effect with BOOM"))
    assert result.shape == (500, 300, 3)
    assert (result[mask != 0][:, 0] > 200).all()
    assert np.array_equal(result[mask == 0], np.asarray(image)[mask == 0])
    body = seen[0]["body"]
    assert seen[0]["key"] == "g-key" and body["model"] == "gemini-nano-banana-2.1"
    assert body["response_format"] == {"type": "image", "mime_type": "image/jpeg", "aspect_ratio": nearest_aspect(300, 500), "image_size": "1K"}
    assert body["generation_config"] == {"thinking_level": "minimal"} and body["store"] is False
    assert [block["type"] for block in body["input"]] == ["text", "image", "image"]
    assert "replace the sound effect with BOOM" in body["input"][0]["text"]
    assert model.last_usage == {"total_input_tokens": 1200, "total_output_tokens": 1120}


async def test_removing_lettering_uses_the_removal_instruction(monkeypatch, crop):
    seen: list[dict] = []
    model = provider(monkeypatch, seen)
    model._inpaint(*crop)
    assert "Remove all lettering" in seen[0]["body"]["input"][0]["text"]


def test_the_key_comes_from_the_options_or_the_environment(monkeypatch):
    monkeypatch.delenv(nano_banana.KEY_ENV, raising=False)
    availability = NanoBananaProvider.runtime_availability("gemini")
    assert not availability.ok and "Gemini API key" in availability.reason
    monkeypatch.setenv(nano_banana.KEY_ENV, "g-env")
    assert NanoBananaProvider.runtime_availability("gemini").ok
    assert NanoBananaProvider.api_key({"api_key": " g-opt "}) == "g-opt"


async def test_google_errors_read_as_such(monkeypatch, crop):
    with pytest.raises(ProviderUnavailable, match="rejected the Gemini API key"):
        provider(monkeypatch, [], 403)._fill(*crop, "x")
    with pytest.raises(ProcessingError, match="busy"):
        provider(monkeypatch, [], 429)._fill(*crop, "x")


def test_sizes_and_padding():
    assert (image_size(1024, 700), image_size(1025, 10), image_size(3000, 5000)) == ("1K", "2K", "4K")
    image = Image.new("RGB", (300, 500))
    aspect = nearest_aspect(300, 500)
    padded, padded_mask, (left, top) = pad_to_aspect(image, np.ones((500, 300), np.uint8), aspect)
    a, b = (int(n) for n in aspect.split(":"))
    assert abs(padded.width / padded.height - a / b) < 0.01 and padded_mask[top : top + 500, left : left + 300].all()
