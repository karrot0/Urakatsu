"""How long loaded models live: idle unload, what unloading gives back, and native servers ending
with Urakatsu."""

import asyncio
import gc
import os
import subprocess
import sys
import threading
import time
import weakref
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from urakatsu.core.config import PipelineConfig
from urakatsu.core.errors import ProcessingError
from urakatsu.core.stage import Stage
from urakatsu.pipeline import Pipeline, ProjectStore, Request
from urakatsu.pipeline import pipeline as pipeline_module
from urakatsu.remote import worker as worker_module
from urakatsu.remote.worker import Model, Worker
from urakatsu.runtimes import RuntimeManager
from urakatsu.runtimes.store import ModelStore
from urakatsu.service import convert
from urakatsu.service.gen import urakatsu_pb2 as pb

from fakes import FakeDetector, FakeInpainter, FakeOcr, FakeTranslator, fake_config, fake_processors, fake_registry, page_png


@pytest.fixture
def project(isolated_home):
    store = ProjectStore(isolated_home / "projects")
    project_id = store.create("test")
    store.import_page(project_id, "p0.png", page_png())
    return store, project_id


def make_pipeline(minutes: int) -> Pipeline:
    config = fake_config()
    config.idle_unload_minutes = minutes
    return Pipeline(config, RuntimeManager("cpu"), fake_registry(), fake_processors())


def idle_for(pipeline: Pipeline, minutes: float) -> None:
    pipeline._last_used -= minutes * 60


@pytest.mark.asyncio
async def test_models_unload_after_the_idle_timeout(project):
    store, project_id = project
    pipeline = make_pipeline(1)
    await pipeline.execute(store.session(project_id), Request())
    assert pipeline.loaded()
    idle_for(pipeline, 1.5)
    pipeline._watch_idle(restart=True)
    await asyncio.wait_for(pipeline._idle_watch, 5)
    assert pipeline.loaded() == {}


@pytest.mark.asyncio
async def test_recently_used_models_stay_loaded(project):
    store, project_id = project
    pipeline = make_pipeline(1)
    await pipeline.execute(store.session(project_id), Request())
    await asyncio.sleep(0.05)
    assert pipeline.loaded()
    assert not pipeline._idle_watch.done()  # still counting down
    pipeline._idle_watch.cancel()


@pytest.mark.asyncio
async def test_zero_keeps_models_loaded(project):
    store, project_id = project
    pipeline = make_pipeline(0)
    await pipeline.execute(store.session(project_id), Request())
    idle_for(pipeline, 600)
    await asyncio.wait_for(pipeline._idle_watch, 5)
    assert pipeline.loaded()


@pytest.mark.asyncio
async def test_models_unload_on_request_one_stage_or_all(project):
    store, project_id = project
    pipeline = make_pipeline(0)
    await pipeline.execute(store.session(project_id), Request())
    loaded = set(pipeline.loaded())
    assert Stage.OCR in loaded
    assert await pipeline.unload(Stage.OCR) == [Stage.OCR]
    assert set(pipeline.loaded()) == loaded - {Stage.OCR}
    assert await pipeline.unload(Stage.OCR) == []  # nothing left to unload there
    assert set(await pipeline.unload()) == loaded - {Stage.OCR}
    assert pipeline.loaded() == {}


@pytest.mark.asyncio
async def test_a_manual_unload_never_pulls_a_model_from_under_a_run(project):
    from urakatsu.core.errors import ProjectBusy

    store, project_id = project
    pipeline = make_pipeline(0)
    await pipeline.execute(store.session(project_id), Request())
    with pipeline.in_use():  # a restitch or an agent turn using a model
        with pytest.raises(ProjectBusy):
            await pipeline.unload()
    assert pipeline.loaded()


@pytest.mark.asyncio
async def test_turning_the_timeout_on_starts_the_countdown(project):
    store, project_id = project
    pipeline = make_pipeline(0)
    await pipeline.execute(store.session(project_id), Request())
    idle_for(pipeline, 10)
    config = pipeline.config.model_copy(deep=True)
    config.idle_unload_minutes = 5
    pipeline.configure(config)
    await asyncio.wait_for(pipeline._idle_watch, 5)
    assert pipeline.loaded() == {}


def test_idle_timeout_round_trips_through_the_api():
    config = PipelineConfig(idle_unload_minutes=30)
    assert convert.config_from_proto(convert.config_to_proto(config), PipelineConfig()).idle_unload_minutes == 30
    assert convert.config_from_proto(pb.Config(), config).idle_unload_minutes == 30  # unset keeps it
    assert convert.config_from_proto(pb.Config(idle_unload_minutes=0), config).idle_unload_minutes == 0
    with pytest.raises(ValueError):
        convert.config_from_proto(pb.Config(idle_unload_minutes=-1), config)


# -- what unloading gives back --------------------------------------------------------------------


class NetDetector(FakeDetector):
    """Fails like a real provider when its network is gone between two images."""

    id = "net-detector"
    delay = 0.0
    net = None

    def _load(self):
        self.net = object()

    def _unload(self):
        self.net = None

    def _detect(self, image):
        time.sleep(self.delay)
        if self.net is None:
            raise ProcessingError(f"{self.id} is not loaded")
        return []


class StuckDetector(FakeDetector):
    id = "stuck-detector"

    def _unload(self):
        raise subprocess.TimeoutExpired("server", 10)


class MemoryTranslator(FakeTranslator):
    """Runs out of memory once, holding something in the frame that fails, as a batch's tensors are."""

    id = "memory-translator"
    attempt = None
    pinned = None

    async def translate(self, request):
        cls = type(self)
        if cls.attempt is None:
            tensors = Tensors()
            cls.attempt = weakref.ref(tensors)
            raise RuntimeError(f"CUDA out of memory with {tensors}")
        cls.pinned = cls.attempt() is not None
        return [f"EN:{segment}" for segment in request]


class Tensors:
    pass


def pipeline_with(config, *providers) -> Pipeline:
    registry = fake_registry()
    for cls in providers:
        registry.add(cls)
    return Pipeline(config, RuntimeManager("cpu"), registry, fake_processors())


@pytest.fixture
def releases(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline_module, "release_memory", lambda: calls.append(True))
    return calls


@pytest.mark.asyncio
async def test_a_settings_change_releases_what_it_unloads(releases):
    pipeline = pipeline_with(fake_config(), NetDetector)
    provider = pipeline.provider(Stage.DETECTION)
    await provider.load()
    config = fake_config()
    config.detection.model = "net-detector"
    pipeline.configure(config)
    assert not provider.loaded and len(releases) == 1


@pytest.mark.asyncio
async def test_swapping_a_provider_releases_the_old_one(releases):
    pipeline = pipeline_with(fake_config(), NetDetector)
    provider = pipeline.provider(Stage.DETECTION)
    pipeline.provider(Stage.DETECTION)
    assert not releases  # nothing was loaded, nothing to give back
    await provider.load()
    pipeline.config.detection.model = "net-detector"
    pipeline.provider(Stage.DETECTION)
    assert not provider.loaded and len(releases) == 1


@pytest.mark.asyncio
async def test_a_failed_stage_releases_its_model(releases):
    pipeline = pipeline_with(fake_config())
    provider = pipeline.provider(Stage.DETECTION)
    await provider.load()
    pipeline._stage_error(Stage.DETECTION, provider, pipeline_module.ErrorKind.PROCESSING, RuntimeError("model exploded"))
    assert not provider.loaded and len(releases) == 1


@pytest.mark.asyncio
async def test_unload_all_releases_once_and_only_when_something_unloaded(releases):
    pipeline = pipeline_with(fake_config())
    detector, ocr = pipeline.provider(Stage.DETECTION), pipeline.provider(Stage.OCR)
    assert pipeline.unload_all() == [] and not releases
    await detector.load()
    await ocr.load()
    assert set(pipeline.unload_all()) == {Stage.DETECTION, Stage.OCR}
    assert len(releases) == 1


@pytest.mark.asyncio
async def test_the_retry_after_out_of_memory_starts_clean(project, monkeypatch):
    store, project_id = project
    monkeypatch.setattr(pipeline_module, "OOM_SETTLE_SECONDS", 0.0)
    released = []
    real = pipeline_module.release_memory
    monkeypatch.setattr(pipeline_module, "release_memory", lambda: (released.append(True), real()))
    MemoryTranslator.attempt = MemoryTranslator.pinned = None
    pipeline = pipeline_with(fake_config("memory-translator"), MemoryTranslator)
    await pipeline.execute(store.session(project_id), Request())
    assert MemoryTranslator.pinned is False  # the failed attempt's tensors were gone before it ran again
    assert released  # and what the other stages' models held went back to the device
    pipeline._idle_watch.cancel()


@pytest.mark.asyncio
async def test_the_worker_retry_starts_clean(monkeypatch):
    released = []
    real = worker_module.release_memory
    monkeypatch.setattr(worker_module, "release_memory", lambda: (released.append(True), real()))
    pinned = []
    attempts = []

    def inference():
        if not attempts:
            tensors = Tensors()
            attempts.append(weakref.ref(tensors))
            raise RuntimeError(f"CUDA out of memory with {tensors}")
        pinned.append(attempts[0]() is not None)
        return "text"

    worker = Worker(PipelineConfig(device="cpu"), RuntimeManager("cpu"), fake_registry())
    used, other = FakeOcr({}, worker.runtimes), FakeDetector({}, worker.runtimes)
    await used.load()
    await other.load()
    model = Model("used", Stage.OCR, used, state="ready")
    worker.models = {"used": model, "other": Model("other", Stage.DETECTION, other, state="ready")}
    assert await worker.run(model, inference) == "text"
    assert pinned == [False] and released and not other.loaded


@pytest.mark.asyncio
async def test_unloading_on_a_worker_releases(monkeypatch):
    released = []
    monkeypatch.setattr(worker_module, "release_memory", lambda: released.append(True))
    worker = Worker(PipelineConfig(device="cpu"), RuntimeManager("cpu"), fake_registry())
    provider = FakeOcr({}, worker.runtimes)
    await provider.load()
    worker.models = {"used": Model("used", Stage.OCR, provider, state="ready")}
    assert await worker.unload("used") and released


@pytest.mark.asyncio
async def test_idle_unload_leaves_a_restitch_alone():
    config = fake_config()
    config.detection.model = "net-detector"
    config.idle_unload_minutes = 1
    pipeline = pipeline_with(config, NetDetector)
    provider = pipeline.provider(Stage.DETECTION)
    provider.delay = 0.15
    await provider.load()
    idle_for(pipeline, 59.95 / 60)  # the timeout runs out while the first tile is on the model
    pipeline._watch_idle(restart=True)
    assert len(await pipeline.detect_layout([Image.new("RGB", (32, 32))] * 4)) == 4
    assert provider.loaded
    pipeline._idle_watch.cancel()


@pytest.mark.asyncio
async def test_models_in_use_outlive_the_timeout():
    pipeline = make_pipeline(1)
    provider = pipeline.provider(Stage.OCR)
    await provider.load()
    with pipeline.in_use():
        idle_for(pipeline, 10)
        pipeline._watch_idle(restart=True)
        await asyncio.sleep(0.05)
        assert provider.loaded  # and the watcher waits; it must not spin on the event loop
    idle_for(pipeline, 10)
    pipeline._watch_idle(restart=True)
    await asyncio.wait_for(pipeline._idle_watch, 5)
    assert not provider.loaded


@pytest.mark.asyncio
async def test_a_model_that_will_not_unload_does_not_keep_the_others(releases):
    config = fake_config()
    config.detection.model = "stuck-detector"
    config.idle_unload_minutes = 1
    pipeline = pipeline_with(config, StuckDetector)
    stuck, ocr = pipeline.provider(Stage.DETECTION), pipeline.provider(Stage.OCR)
    await stuck.load()
    await ocr.load()
    idle_for(pipeline, 10)
    pipeline._watch_idle(restart=True)
    await asyncio.wait_for(pipeline._idle_watch, 5)  # raises if the failure ended the watcher
    assert not ocr.loaded and stuck.loaded and len(releases) == 1


@pytest.mark.asyncio
async def test_a_cancelled_load_can_still_be_unloaded():
    finish = threading.Event()

    class SlowLoad(NetDetector):
        id = "slow-load"

        def _load(self):
            finish.wait(5)
            super()._load()

    provider = SlowLoad({}, RuntimeManager("cpu"))
    task = asyncio.create_task(provider.load())
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    finish.set()  # a thread cannot be cancelled: the model still arrives
    for _ in range(100):
        if provider.loaded:
            break
        await asyncio.sleep(0.01)
    assert provider.unload() and provider.net is None


def test_unloading_hands_the_gpu_memory_back():
    """Real weights, on the GPU; set URAKATSU_WEIGHTS_HOME to a model store that holds LaMa."""
    torch = pytest.importorskip("torch")
    from urakatsu.providers.inpainting.lama import LamaProvider

    root = os.environ.get("URAKATSU_WEIGHTS_HOME")
    if not root:
        pytest.skip("set URAKATSU_WEIGHTS_HOME to a model store that holds LaMa")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    store = ModelStore(root=Path(root), offline=True)
    if not store.cached(LamaProvider.weight_sources()[0]):
        pytest.skip(f"LaMa weights are not cached under {root}")
    config = PipelineConfig()
    config.inpainting.model = "lama"
    pipeline = Pipeline(config, RuntimeManager("auto", store=store))
    image = Image.new("RGB", (512, 512), (235, 235, 235))
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[100:200, 100:200] = 255
    gc.collect()
    torch.cuda.empty_cache()
    allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    for _ in range(3):
        provider = pipeline.provider(Stage.INPAINTING)
        asyncio.run(provider.load())
        provider._inpaint(image, mask)
        assert torch.cuda.memory_allocated() > allocated  # it really was on the GPU
        assert Stage.INPAINTING in pipeline.unload_all()
        assert torch.cuda.memory_allocated() == allocated
        assert torch.cuda.memory_reserved() <= reserved + 64 * 1024 * 1024


PARENT = """
import subprocess, sys, time
from urakatsu.runtimes.server_process import _tie_to_this_process
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
_tie_to_this_process(child)
print(child.pid, flush=True)
time.sleep(120)
"""


def _exits_within(pid: int, seconds: float) -> bool:
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if not handle:
        return True  # already gone
    try:
        return kernel32.WaitForSingleObject(ctypes.c_void_p(handle), int(seconds * 1000)) == 0  # WAIT_OBJECT_0
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


@pytest.mark.skipif(sys.platform != "win32", reason="the kill-on-close job object is Windows-only")
def test_servers_die_when_urakatsu_is_killed():
    parent = subprocess.Popen([sys.executable, "-c", PARENT], stdout=subprocess.PIPE, text=True)
    try:
        child = int(parent.stdout.readline())
        assert not _exits_within(child, 0.5)  # running while Urakatsu is
        parent.kill()  # TerminateProcess: no atexit, no cleanup in the parent
        parent.wait(10)
        assert _exits_within(child, 10)
    finally:
        parent.kill()
