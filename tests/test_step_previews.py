"""Live diffusion previews: sd-server frames reach the step sink, placed on the page."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from urakatsu.core.progress import StepPreview, report_step, reporting_steps
from urakatsu.pipeline.stages.inpainting import placed_previews
from urakatsu.runtimes.diffusion import SdServer


def _png(color: tuple[int, int, int], size: tuple[int, int] = (8, 8)) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class _Response:
    def __init__(self, body: dict, status: int = 200):
        self.body = body
        self.status_code = status

    def json(self) -> dict:
        return self.body

    def raise_for_status(self) -> "_Response":
        return self


class _Client:
    def __init__(self, modes: list[str], polls: list[dict]):
        self.modes = modes
        self.polls = list(polls)
        self.request: dict = {}

    def get(self, path: str) -> _Response:
        if path == "/sdcpp/v1/capabilities":
            return _Response({"preview_modes": self.modes})
        return _Response(self.polls.pop(0))

    def post(self, path: str, json: dict) -> _Response:
        self.request = json
        return _Response({"id": "job"})


class _Process:
    running = True

    def __init__(self, client: _Client):
        self.client = client


def _polls() -> list[dict]:
    frame = {"pass": 1, "step": 1, "total_steps": 2, "b64_json": _png((255, 0, 0))}
    return [
        {"status": "generating", "preview": frame},
        {"status": "generating", "preview": frame},  # the same frame polled twice is reported once
        {"status": "generating", "preview": {**frame, "step": 2}},
        {"status": "completed", "result": {"images": [{"index": 0, "b64_json": _png((0, 0, 255))}]}},
    ]


def test_sd_server_reports_each_new_preview_frame(monkeypatch):
    monkeypatch.setattr("urakatsu.runtimes.diffusion.POLL_INTERVAL", 0)
    client = _Client(["none", "proj"], _polls())
    seen: list[StepPreview] = []
    with reporting_steps(seen.append):
        images = SdServer(_Process(client)).generate({"prompt": "x"})
    assert client.request["preview"] == "proj"
    assert [(s.step, s.total) for s in seen] == [(1, 2), (2, 2)]
    assert seen[0].image.getpixel((0, 0)) == (255, 0, 0)
    assert images[0].getpixel((0, 0)) == (0, 0, 255)


def test_sd_server_skips_previews_nobody_listens_for_or_the_release_lacks(monkeypatch):
    monkeypatch.setattr("urakatsu.runtimes.diffusion.POLL_INTERVAL", 0)
    unheard = _Client(["none", "proj"], _polls())
    SdServer(_Process(unheard)).generate({"prompt": "x"})
    assert "preview" not in unheard.request

    old = _Client([], _polls())
    seen: list[StepPreview] = []
    with reporting_steps(seen.append):
        SdServer(_Process(old)).generate({"prompt": "x"})
    assert "preview" not in old.request and seen == []


def test_placed_previews_show_the_frame_only_inside_the_mask_at_the_crop():
    crop = Image.new("RGB", (40, 20), (255, 255, 255))
    mask = np.zeros((20, 40), dtype=np.uint8)
    mask[:, 20:] = 255
    seen: list[StepPreview] = []
    with reporting_steps(seen.append), placed_previews(crop, mask, (100, 50)):
        report_step(StepPreview(3, 10, Image.new("RGB", (8, 4), (0, 0, 0))))
    (placed,) = seen
    assert placed.box == (100, 50, 40, 20) and placed.step == 3
    assert placed.image.getpixel((2, 10)) == (255, 255, 255)  # unmasked: the crop as it is
    assert placed.image.getpixel((37, 10)) == (0, 0, 0)  # masked: the preview frame
