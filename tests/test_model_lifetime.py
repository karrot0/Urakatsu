"""How long loaded models live: idle unload, and native servers ending with Urakatsu."""

import asyncio
import subprocess
import sys

import pytest

from urakatsu.core.config import PipelineConfig
from urakatsu.pipeline import Pipeline, ProjectStore, Request
from urakatsu.runtimes import RuntimeManager
from urakatsu.service import convert
from urakatsu.service.gen import urakatsu_pb2 as pb

from fakes import fake_config, fake_processors, fake_registry, page_png


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
