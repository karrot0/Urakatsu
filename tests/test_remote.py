"""The remote runtime (a worker running providers for another app) and attached servers."""

from __future__ import annotations

import base64
import io
import json
import time

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from urakatsu.core.config import PipelineConfig
from urakatsu.core.errors import OutOfMemory, ProviderUnavailable, is_out_of_memory
from urakatsu.core.registry import Registry
from urakatsu.core.results import LayoutDetection, LayoutDetections, LayoutMask
from urakatsu.providers.base import DetectionProvider, InpaintingProvider, OcrProvider, TranslationProvider
from urakatsu.remote import REMOTE, client as remote_client, wire
from urakatsu.remote.worker import Worker, create_app
from urakatsu.runtimes.manager import RuntimeManager

TOKEN = "s3cret"


class EchoDetector(DetectionProvider):
    id = "echo-detector"
    name = "Echo detector"
    loads = 0

    def _load(self):
        type(self).loads += 1

    def _detect(self, image):
        pixels = (np.asarray(image.convert("L")) < 128).astype(np.uint8)[10:30, 5:25]
        mask = LayoutMask(x=5, y=10, width=20, height=20, pixels=pixels)
        return LayoutDetections(image.width, image.height, [LayoutDetection(1, "text", 0.875, (5.0, 10.0, 25.0, 30.0), int(pixels.sum()), mask)])


class EchoOcr(OcrProvider):
    id = "echo-ocr"
    name = "Echo OCR"

    def _recognize(self, crops):
        return [f"{crop.width}x{crop.height}:{crop.mode}" for crop in crops]


class EchoInpainter(InpaintingProvider):
    id = "echo-inpainter"
    name = "Echo inpainter"
    prompted = True

    def _inpaint(self, image, mask):
        pixels = np.asarray(image.convert("RGB")).copy()
        pixels[mask != 0] = 255 - pixels[mask != 0]
        return Image.fromarray(pixels)

    def _fill(self, image, mask, prompt):
        pixels = np.asarray(image.convert("RGB")).copy()
        pixels[mask != 0] = len(prompt or "")
        return Image.fromarray(pixels)


class SlowOcr(OcrProvider):
    """Takes longer than a poll waits, as a diffusion model on a notebook GPU takes longer than a tunnel keeps a request."""

    id = "slow-ocr"
    name = "Slow OCR"
    seconds = 0.4

    def _recognize(self, crops):
        time.sleep(self.seconds)
        return ["slow" for _ in crops]


class ExhaustedInpainter(InpaintingProvider):
    id = "exhausted-inpainter"
    name = "Exhausted inpainter"

    def _inpaint(self, image, mask):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")


class LocalTranslator(TranslationProvider):
    id = "local-translator"


def page() -> Image.Image:
    rng = np.random.default_rng(7)
    return Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8), "RGB")


class SharedClient:
    """One in-process worker connection for every ``RemoteModel``; closing it is a no-op."""

    def __init__(self, client: TestClient):
        self.client = client
        self.requests: list[tuple[str, str]] = []

    def request(self, method, path, *args, **kwargs):
        self.requests.append((method, path))
        return self.client.request(method, path, *args, **kwargs)

    def delete(self, *args, **kwargs):
        return self.client.delete(*args, **kwargs)

    def close(self):
        pass


@pytest.fixture
def worker(monkeypatch):
    registry = Registry()
    registry._discovered = True
    for cls in (EchoDetector, EchoOcr, SlowOcr, EchoInpainter, ExhaustedInpainter):
        registry.add(cls)
    instance = Worker(PipelineConfig(device="cpu"), RuntimeManager("cpu"), registry)
    with TestClient(create_app(instance, TOKEN)) as http:
        sent: list[dict] = []
        shared = SharedClient(http)

        def factory(url, headers):
            sent.append(headers)
            http.headers.clear()
            http.headers.update(headers)
            return shared

        monkeypatch.setattr(remote_client, "client_factory", factory)
        monkeypatch.setattr(remote_client, "POLL_WAIT", 0.05)
        instance.shared = shared
        yield instance, http, sent


def remote(cls, **settings):
    return cls({"remote_url": "http://worker.test/", "remote_api_key": TOKEN, **settings}, None, REMOTE)


# ---- declaration -----------------------------------------------------------------------------


def test_vision_providers_declare_remote_but_never_default_to_it():
    assert EchoOcr.runtimes_declared() == (REMOTE,)
    assert EchoOcr.default_runtime() is None
    info = EchoInpainter.info()
    assert [r.name for r in info.runtimes] == [REMOTE]
    assert info.settings_schema["properties"]["remote_url"]["runtimes"] == [REMOTE]
    assert REMOTE not in LocalTranslator.runtimes_declared()
    assert "remote_url" not in LocalTranslator.info().settings_schema.get("properties", {})


# ---- wire --------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_images_cross_the_wire_unchanged(mode):
    image = page().convert(mode)
    back = wire.decode_image(wire.encode_image(image))
    assert back.mode == mode and np.array_equal(np.asarray(back), np.asarray(image))


@pytest.mark.parametrize("dtype", [np.uint8, np.bool_, np.float32])
def test_arrays_keep_dtype_and_values(dtype):
    array = (np.arange(35).reshape(5, 7) % 3).astype(dtype)
    back = wire.decode_array(wire.encode_array(array))
    assert back.dtype == array.dtype and np.array_equal(back, array)


def test_detections_round_trip():
    original = EchoDetector()._detect(page())
    back = wire.decode_detections(json.loads(json.dumps(wire.encode_detections(original))))
    assert back.image_width == 64 and back.image_height == 48
    (a,), (b,) = original.detections, back.detections
    assert (a.label_id, a.label, a.score, a.bbox, a.area) == (b.label_id, b.label, b.score, b.bbox, b.area)
    assert np.array_equal(a.mask.pixels, b.mask.pixels) and (a.mask.x, a.mask.y) == (b.mask.x, b.mask.y)


# ---- worker round trips -----------------------------------------------------------------------


async def test_remote_providers_match_local_runs(worker):
    _, _, sent = worker
    image = page()
    mask = np.zeros((48, 64), np.uint8)
    mask[8:20, 10:40] = 255

    detector = remote(EchoDetector)
    await detector.load()
    assert detector.loaded and sent[-1] == {"Authorization": f"Bearer {TOKEN}"}
    got = await detector.detect(image)
    want = EchoDetector()._detect(image)
    assert np.array_equal(got.detections[0].mask.pixels, want.detections[0].mask.pixels)

    ocr = remote(EchoOcr)
    await ocr.load()
    assert await ocr.recognize([image, image.convert("L").crop((0, 0, 9, 5))]) == ["64x48:RGB", "9x5:L"]
    assert await ocr.recognize([]) == []

    inpainter = remote(EchoInpainter)
    await inpainter.load()
    assert inpainter.instructed is False  # what the worker's loaded model says
    # The stage calls these synchronously, per tile.
    assert np.array_equal(np.asarray(inpainter._inpaint(image, mask)), np.asarray(EchoInpainter()._inpaint(image, mask)))
    filled = np.asarray(inpainter._fill(image, mask, "abc"))
    assert (filled[8:20, 10:40] == 3).all() and np.array_equal(filled[30:], np.asarray(image)[30:])
    assert inpainter.unload() and not inpainter.loaded


async def test_settings_reach_the_worker_without_the_connection(worker):
    instance, _, _ = worker
    provider = remote(EchoDetector, model=None)
    await provider.load()
    (model,) = instance.models.values()
    dumped = model.provider.settings.model_dump()
    assert dumped["remote_url"] is None and dumped["remote_api_key"] is None
    assert model.provider.runtime_name is None  # the worker's own choice, never remote


async def test_same_model_is_shared_and_another_for_the_stage_replaces_it(worker):
    instance, _, _ = worker
    first, second = remote(EchoOcr), remote(EchoOcr)
    await first.load()
    await second.load()
    assert len(instance.models) == 1 and first._remote.model_id == second._remote.model_id


async def test_wrong_token_is_refused(worker):
    provider = EchoOcr({"remote_url": "http://worker.test", "remote_api_key": "nope"}, None, REMOTE)
    with pytest.raises(ProviderUnavailable, match="token"):
        await provider.load()


async def test_missing_url_says_where_to_set_it(worker, monkeypatch):
    monkeypatch.delenv(remote_client.URL_ENV, raising=False)
    with pytest.raises(ProviderUnavailable, match="worker URL"):
        await EchoOcr({}, None, REMOTE).load()


async def test_a_lost_model_is_loaded_again(worker):
    _, http, _ = worker
    EchoDetector.loads = 0
    provider = remote(EchoDetector)
    await provider.load()
    assert http.delete(f"/v1/models/{provider._remote.model_id}").status_code == 204  # the worker swapped it out
    result = await provider.detect(page())
    assert len(result.detections) == 1 and EchoDetector.loads == 2


async def test_worker_out_of_memory_reaches_the_pipeline_as_such(worker):
    provider = remote(ExhaustedInpainter)
    await provider.load()
    with pytest.raises(OutOfMemory) as raised:
        provider._inpaint(page(), np.ones((48, 64), np.uint8))
    assert is_out_of_memory(raised.value)


async def test_unknown_provider_fails_the_load(worker):
    class Missing(OcrProvider):
        id = "not-on-the-worker"

    with pytest.raises(ProviderUnavailable, match="not-on-the-worker"):
        await remote(Missing).load()


# ---- attached servers ---------------------------------------------------------------------------


def _png_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def fake_llama(seen: list[httpx.Request]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/props":
            return httpx.Response(200, json={"media_marker": "<m>", "modalities": {"vision": True}})
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": []})
        if request.url.path == "/completion":
            return httpx.Response(200, json={"content": "こんにちは"})
        return httpx.Response(404)

    return httpx.MockTransport(handle)


def fake_sd(seen: list[dict]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sdcpp/v1/models/status":
            return httpx.Response(200, json={"state": "ready"})
        if request.url.path == "/sdcpp/v1/capabilities":
            return httpx.Response(200, json={"preview_modes": []})
        if request.url.path == "/sdcpp/v1/img_gen":
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"id": "job"})
        if request.url.path == "/sdcpp/v1/jobs/job":
            body = seen[-1]
            image = Image.new("RGB", (body["width"], body["height"]), "gray")
            return httpx.Response(200, json={"status": "completed", "result": {"images": [{"index": 0, "b64_json": _png_b64(image)}]}})
        return httpx.Response(404)

    return httpx.MockTransport(handle)


async def test_paddle_ocr_uses_an_attached_llama_server(monkeypatch):
    from urakatsu.providers.ocr import paddle_ocr_vl
    from urakatsu.runtimes.llama import LlamaServer

    seen: list[httpx.Request] = []
    original = LlamaServer.attach.__func__
    monkeypatch.setattr(paddle_ocr_vl.LlamaServer, "attach", classmethod(lambda cls, url, key=None: original(cls, url, key, fake_llama(seen))))
    provider = paddle_ocr_vl.PaddleOcrVlProvider({"server_url": "https://colab.example/", "server_api_key": "k"}, None, "llama")
    assert provider.attached()
    await provider.load()  # offline and without llama.cpp: nothing is installed or downloaded
    assert await provider.recognize([Image.new("RGB", (40, 20), "white")]) == ["こんにちは"]
    completion = next(r for r in seen if r.url.path == "/completion")
    assert completion.headers["authorization"] == "Bearer k"
    assert "<m>" in json.loads(completion.content)["prompt"]["prompt_string"]
    provider.unload()


def test_server_url_counts_only_on_its_runtime():
    from urakatsu.providers.ocr.paddle_ocr_vl import PaddleOcrVlProvider

    assert not PaddleOcrVlProvider({"server_url": "http://x"}, None, "torch").attached()
    assert not PaddleOcrVlProvider({"server_url": "  "}, None, "llama").attached()


async def test_flux2_klein_uses_an_attached_sd_server(monkeypatch):
    from urakatsu.providers.inpainting import flux2_klein
    from urakatsu.runtimes.diffusion import SdServer

    seen: list[dict] = []
    original = SdServer.attach.__func__
    monkeypatch.setattr(flux2_klein.SdServer, "attach", classmethod(lambda cls, url, key=None: original(cls, url, key, fake_sd(seen))))
    provider = flux2_klein.Flux2KleinProvider({"server_url": "http://gpu-box:1234"}, None, "diffusion")
    await provider.load()
    mask = np.zeros((64, 96), np.uint8)
    mask[16:48, 16:80] = 255
    result = provider._inpaint(Image.new("RGB", (96, 64), "white"), mask)
    assert result.size == (96, 64) and seen[0]["prompt"] == provider.settings.prompt
    provider.unload()


def test_attached_server_reports_an_unreachable_address(monkeypatch):
    from urakatsu.core.errors import ModelLoadError
    from urakatsu.runtimes import server_process

    monkeypatch.setattr(server_process, "ATTACH_TIMEOUT", 0.0)
    refuse = httpx.MockTransport(lambda request: (_ for _ in ()).throw(httpx.ConnectError("refused")))
    with pytest.raises(ModelLoadError, match="refused"):
        server_process.AttachedServer("sd-server", "http://nowhere:1", lambda c: c.get("/").status_code == 200, transport=refuse)
    with pytest.raises(ModelLoadError, match="http"):
        server_process.AttachedServer("sd-server", "nowhere:1", lambda c: True)


async def test_a_call_longer_than_a_poll_is_waited_for_in_short_requests(worker):
    instance, _, _ = worker
    provider = remote(SlowOcr)
    await provider.load()
    started = len(instance.shared.requests)
    assert await provider.recognize([page()]) == ["slow"]
    calls = instance.shared.requests[started:]
    polls = [path for method, path in calls if method == "GET" and path.startswith("/v1/jobs/") and not path.endswith("/result")]
    assert calls[0][1].endswith("/recognize") and calls[-1][1].endswith("/result")
    assert len(polls) >= 2  # each poll returned after POLL_WAIT while the model was still working
    assert not instance.jobs  # a fetched result is gone


def test_a_job_whose_model_was_unloaded_while_it_waited_reports_it(worker):
    instance, http, _ = worker
    http.headers.update({"Authorization": f"Bearer {TOKEN}"})
    load = http.post("/v1/models", json={"stage": "ocr", "provider": "slow-ocr", "settings": {}}).json()
    while http.get(f"/v1/models/{load['id']}", params={"wait": 1}).json()["state"] == "loading":
        pass
    crop = {"crops": ("0.png", wire.encode_image(page()), wire.PNG)}
    first = http.post(f"/v1/models/{load['id']}/recognize", files=crop)
    second = http.post(f"/v1/models/{load['id']}/recognize", files=crop)
    assert first.status_code == second.status_code == 202
    assert http.delete(f"/v1/models/{load['id']}").status_code == 204  # waits for the lane
    assert http.get(f"/v1/jobs/{first.json()['job']}", params={"wait": 1}).json()["state"] == "done"
    lost = http.get(f"/v1/jobs/{second.json()['job']}", params={"wait": 1}).json()
    assert lost["state"] == "failed" and lost["failure"]["kind"] == "gone"
    assert http.get(f"/v1/jobs/{second.json()['job']}/result").status_code == 409
    assert http.get("/v1/jobs/nope").status_code == 404
