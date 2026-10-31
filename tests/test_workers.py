"""Workers people run themselves: the notebook worker's tunnel and connection line, idle exit and
remote stop, starting one on Kaggle and finding it through the session's log, and pasting a
Colab worker's connection line."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import threading
import time

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from urakatsu.core.config import PipelineConfig
from urakatsu.core.errors import ProviderUnavailable
from urakatsu.core.registry import Registry
from urakatsu.providers.base import OcrProvider
from urakatsu.remote import REMOTE, client as remote_client, kaggle, tunnel, workers
from urakatsu.remote.worker import Worker, create_app
from urakatsu.runtimes.manager import RuntimeManager
from urakatsu.runtimes.packages import CLOUDFLARED, Asset, PackageManager, Release


class SizeOcr(OcrProvider):
    id = "size-ocr"
    name = "Size OCR"

    def _recognize(self, crops):
        return [f"{crop.width}x{crop.height}" for crop in crops]


@pytest.fixture(autouse=True)
def fresh_addresses():
    workers._addresses.clear()
    yield
    workers._addresses.clear()


def test_connection_lines_carry_the_address_and_token():
    line = tunnel.connection_line("https://quiet-words.trycloudflare.com", "tok_1-x")
    assert line == "Connection: https://quiet-words.trycloudflare.com#tok_1-x"
    assert tunnel.parse_connection(line) == ("https://quiet-words.trycloudflare.com", "tok_1-x")
    assert tunnel.parse_connection("  https://quiet-words.trycloudflare.com/#tok  ") == ("https://quiet-words.trycloudflare.com", "tok")
    assert tunnel.parse_connection("https://quiet-words.trycloudflare.com") is None
    assert tunnel.parse_connection("Token: abc") is None


# ---- the notebook worker -----------------------------------------------------------------------


def test_tunnel_reports_each_new_address(tmp_path, monkeypatch):
    # A stand-in for cloudflared: logs an address to stderr the way it does, then keeps running.
    (tmp_path / "tunnel").write_text(
        "import sys, time\nprint('INF |  https://first-words-here.trycloudflare.com  |', file=sys.stderr, flush=True)\ntime.sleep(30)\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    seen: list[str] = []
    opened = threading.Event()
    running = tunnel.Tunnel(8100, lambda url: (seen.append(url), opened.set()), executable=sys.executable)
    running.start()
    try:
        assert opened.wait(10)
        assert seen == ["https://first-words-here.trycloudflare.com"] and running.url == seen[0]
    finally:
        running.stop()


def test_cloudflared_installs_from_its_bare_executable(tmp_path):
    binary = b"\x7fELF fake cloudflared"
    digest = "sha256:" + hashlib.sha256(binary).hexdigest()
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=binary))
    manager = PackageManager(root=tmp_path / "runtimes", offline=False, transport=transport)
    name = "cloudflared-windows-amd64.exe" if sys.platform == "win32" else "cloudflared-linux-amd64"
    variant = "win-cpu" if sys.platform == "win32" else "linux-cpu"
    installed = manager._install(CLOUDFLARED, Release("2026.10.3", variant, "", (Asset(name, "https://github.test/cf", len(binary), digest),)), None)
    assert installed.executable.read_bytes() == binary
    assert installed.executable.stem == "cloudflared"


@pytest.fixture
def notebook():
    registry = Registry()
    registry._discovered = True
    registry.add(SizeOcr)
    instance = Worker(PipelineConfig(device="cpu"), RuntimeManager("cpu"), registry)
    with TestClient(create_app(instance, "t0ken"), headers={"Authorization": "Bearer t0ken"}) as http:
        yield instance, http


def test_status_checks_do_not_keep_a_worker_alive(notebook):
    instance, http = notebook
    instance.active -= 600
    assert http.get("/v1/info").status_code == 200
    assert instance.idle() >= 600
    http.post("/v1/models", json={"stage": "ocr", "provider": "size-ocr", "settings": {}})
    assert instance.idle() < 5


def test_shutdown_needs_a_worker_started_to_allow_it(notebook):
    instance, http = notebook
    assert http.get("/v1/info").json()["stoppable"] is False
    assert http.post("/v1/shutdown").status_code == 501
    stopped = threading.Event()
    instance.stop = stopped.set
    assert http.get("/v1/info").json()["stoppable"] is True
    assert http.post("/v1/shutdown").status_code == 202
    assert stopped.wait(3)


# ---- Kaggle ------------------------------------------------------------------------------------


class KaggleApi:
    def __init__(self, save_error: str = ""):
        self.calls: list[tuple[str, dict, httpx.Request]] = []
        self.save_error = save_error
        self.status = "RUNNING"
        self.log_lines: list[str] = []

    def logs(self, request: httpx.Request) -> httpx.Response:
        """A running session's log: server-sent events of {stream_name, time, data}."""
        events = "".join(f"data: {json.dumps({'stream_name': 'stdout', 'time': 1.0, 'data': line})}\n\n" for line in self.log_lines)
        return httpx.Response(200, content=events.encode(), headers={"content-type": "text/event-stream"})

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/v1/kernels/logs/stream/"):
            self.calls.append(("logs", {}, request))
            return self.logs(request)
        method = request.url.path.rsplit("/", 1)[-1]
        body = json.loads(request.content or b"{}")
        self.calls.append((method, body, request))
        if method == "IntrospectToken":
            return httpx.Response(200, json={"active": True, "username": "reader"})
        if method == "SaveKernel":
            return httpx.Response(200, json={"error": self.save_error} if self.save_error else {"ref": f"/code/{body['slug']}", "versionNumber": 1})
        if method == "GetKernelSessionStatus":
            return httpx.Response(200, json={"status": self.status})
        if method == "GetAcceleratorQuotaStatistics":
            return httpx.Response(200, json={"gpuQuota": {"timeUsed": "16200.000s", "timeReserved": "1800s", "totalTimeAllowed": "108000s"}})
        return httpx.Response(404, json={"code": 404, "message": "no such method"})


def test_kaggle_runs_the_worker_script_on_a_private_t4():
    api = KaggleApi()
    client = kaggle.Kaggle("KGAT_abc", client=httpx.Client(transport=httpx.MockTransport(api)))
    ref = client.push("urakatsu-worker-1a2b", "Urakatsu worker 1a2b", kaggle.worker_script("w0rker-token", ref="main"))
    assert ref == "reader/urakatsu-worker-1a2b"
    (_, introspect, first), (_, body, request) = api.calls
    assert introspect == {"token": "KGAT_abc"} and first.headers["authorization"] == "Bearer KGAT_abc"
    assert request.url.host == "api.kaggle.com" and request.url.path == "/v1/kernels.KernelsApiService/SaveKernel"
    assert body["slug"] == "reader/urakatsu-worker-1a2b" and body["kernelType"] == "script" and body["language"] == "python"
    assert body["isPrivate"] is True and body["enableGpu"] is True and body["enableInternet"] is True and body["machineShape"] == "NvidiaTeslaT4"
    assert '"--tunnel", "--token", "w0rker-token"' in body["text"] and '"--idle-exit"' in body["text"]
    assert "git+https://github.com/karrot0/Urakatsu@main" in body["text"]
    compile(body["text"], "worker.py", "exec")
    assert client.status(ref) == ("RUNNING", "")
    assert client.quota() == {"used_hours": 4.5, "reserved_hours": 0.5, "allowed_hours": 30.0}


def test_kaggle_accounts_by_legacy_key_or_kaggle_json():
    api = KaggleApi()
    transport = httpx.Client(transport=httpx.MockTransport(api))
    with pytest.raises(kaggle.KaggleError, match="username"):
        kaggle.Kaggle("0123abcd", client=transport).push("s", "S", "pass")
    legacy = kaggle.Kaggle(json.dumps({"username": "old-hand", "key": "0123abcd"}), client=transport)
    assert legacy.whoami() == "old-hand" and legacy.token == "0123abcd"
    legacy.push("s", "S", "pass")
    _, _, request = api.calls[-1]
    assert request.headers["authorization"].startswith("Basic ")


def test_kaggle_says_why_it_will_not_run():
    api = KaggleApi(save_error="Phone verification is required to use accelerators")
    client = kaggle.Kaggle("KGAT_abc", client=httpx.Client(transport=httpx.MockTransport(api)))
    with pytest.raises(kaggle.KaggleError, match="verify your phone"):
        client.push("s", "S", "pass")


def test_the_tunnel_address_comes_from_the_session_log():
    api = KaggleApi()
    client = kaggle.Kaggle("KGAT_abc", "reader", client=httpx.Client(transport=httpx.MockTransport(api)))
    assert client.tunnel_url("reader/urakatsu-worker-1a2b") is None
    api.log_lines = ["Installing\n", "Connection: https://first.trycloudflare.com#t\n", "cloudflared restarted\n", "Connection: https://second.trycloudflare.com#t\n"]
    assert client.tunnel_url("reader/urakatsu-worker-1a2b") == "https://second.trycloudflare.com"
    _, _, request = api.calls[-1]
    assert request.url.path == "/v1/kernels/logs/stream/reader/urakatsu-worker-1a2b" and request.headers["authorization"] == "Bearer KGAT_abc"
    # A finished session answers with the whole log as JSON.
    finished = [{"stream_name": "stdout", "time": 1.0, "data": "Connection: https://done.trycloudflare.com#t\n"}]
    api.logs = lambda request: httpx.Response(200, json=finished)
    assert client.tunnel_url("reader/urakatsu-worker-1a2b") == "https://done.trycloudflare.com"


# ---- the app finding its workers ----------------------------------------------------------------


def test_no_workers_says_where_to_add_one():
    with pytest.raises(ProviderUnavailable, match="Settings, Workers"):
        workers.connect()


class Dead:
    """A connection to an address that stopped answering (a notebook that restarted)."""

    def request(self, *args, **kwargs):
        raise httpx.ConnectError("connection refused")

    def delete(self, *args, **kwargs):
        raise httpx.ConnectError("connection refused")

    def close(self):
        pass


def kaggle_through(api, monkeypatch):
    real = kaggle.Kaggle
    monkeypatch.setattr(kaggle, "Kaggle", lambda token, username="": real(token, username, client=httpx.Client(transport=httpx.MockTransport(api))))


def test_a_model_with_no_worker_url_uses_a_kaggle_worker_and_follows_it(monkeypatch):
    api = KaggleApi()
    api.log_lines = ["Connection: https://old.trycloudflare.com#w0rker\n"]
    kaggle_through(api, monkeypatch)
    entry = workers.WorkerEntry(kind="kaggle", token="w0rker", kaggle=workers.KaggleAccount(token="KGAT_abc", username="reader", ref="reader/urakatsu-worker-1"))
    workers.save([entry])
    registry = Registry()
    registry._discovered = True
    registry.add(SizeOcr)
    instance = Worker(PipelineConfig(device="cpu"), RuntimeManager("cpu"), registry)
    monkeypatch.setattr(workers, "info", lambda point: {"protocol": 2})  # the app's status check saw it up
    with TestClient(create_app(instance, "w0rker")) as http:
        connected: list[tuple[str, str]] = []

        def factory(url, headers):
            connected.append((url, headers.get("Authorization", "")))
            if url == "https://old.trycloudflare.com":
                return Dead()
            http.headers.update(headers)
            return http

        monkeypatch.setattr(remote_client, "client_factory", factory)
        monkeypatch.setattr(remote_client, "POLL_WAIT", 0.05)
        assert workers.kaggle_address(entry) == "https://old.trycloudflare.com"  # what the app last read
        # The session's tunnel restarted on a new address before the app noticed.
        api.log_lines.append("Connection: https://new.trycloudflare.com#w0rker\n")
        ocr = SizeOcr({}, None, REMOTE)
        asyncio.run(ocr.load())
        assert ocr.loaded and ocr._remote.endpoint.url == "https://new.trycloudflare.com"
        assert [url for url, _ in connected] == ["https://old.trycloudflare.com", "https://new.trycloudflare.com"]
        assert {token for _, token in connected} == {"Bearer w0rker"}
        assert asyncio.run(ocr.recognize([Image.fromarray(np.zeros((5, 9, 3), np.uint8))])) == ["9x5"]


def workers_api():
    from fastapi import FastAPI

    from urakatsu.service.workers_api import workers_router

    app = FastAPI()
    app.include_router(workers_router(), prefix="/api/v1")
    return TestClient(app)


def test_settings_add_connect_order_and_remove_workers(monkeypatch):
    api = KaggleApi()
    kaggle_through(api, monkeypatch)
    http = workers_api()
    colab = http.post("/api/v1/workers", json={"kind": "colab"}).json()
    assert colab["name"] == "Colab" and colab["url"] is None
    assert http.patch(f"/api/v1/workers/{colab['id']}", json={"connection": "Token: nope"}).status_code == 400
    pasted = http.patch(f"/api/v1/workers/{colab['id']}", json={"connection": "Connection: https://colab-words.trycloudflare.com#c0lab"}).json()
    assert pasted["url"] == "https://colab-words.trycloudflare.com" and pasted["token_set"]
    assert workers.find(colab["id"]).token == "c0lab"
    found = http.post("/api/v1/workers", json={"kind": "kaggle", "kaggle_token": "KGAT_abc"}).json()
    assert found["kaggle_username"] == "reader" and found["kaggle_token_set"] and "KGAT_abc" not in json.dumps(found)
    assert http.post("/api/v1/workers", json={"kind": "manual"}).status_code == 400
    listed = http.get("/api/v1/workers").json()
    assert [w["kind"] for w in listed["workers"]] == ["colab", "kaggle"]
    colab_status = listed["workers"][0]["status"]
    assert colab_status["state"] == "offline" and "paste the new connection line" in colab_status["detail"]
    assert "KGAT_abc" not in json.dumps(listed)
    assert http.post(f"/api/v1/workers/{found['id']}/start").json()["status"]["state"] == "starting"
    started = workers.find(found["id"])
    assert started.kaggle.ref == f"reader/urakatsu-worker-{found['id']}" and started.token
    _, save, _ = next(call for call in api.calls if call[0] == "SaveKernel")
    assert f'"--token", "{started.token}"' in save["text"]
    http.put("/api/v1/workers/order", json={"ids": [found["id"], colab["id"]]})
    assert [w.kind for w in workers.load()] == ["kaggle", "colab"]
    assert http.delete(f"/api/v1/workers/{colab['id']}").json() == {"ok": True}
    assert [w.kind for w in workers.load()] == ["kaggle"]
    assert http.post(f"/api/v1/workers/{colab['id']}/start").status_code == 404


def test_one_failing_check_does_not_fail_the_list(monkeypatch):
    workers.save([workers.WorkerEntry(kind="manual", url="https://box.test", token="t")])

    def broken(entry):
        raise ValueError("not JSON")

    monkeypatch.setattr(workers, "status", broken)
    (listed,) = workers_api().get("/api/v1/workers").json()["workers"]
    assert listed["status"]["state"] == "offline" and "not JSON" in listed["status"]["detail"]


# ---- which models each worker runs -------------------------------------------------------------


class Fetched:
    """A weight file that records being fetched instead of downloading."""

    fetched: list[str] = []

    def __init__(self, name: str):
        self.name = name

    def describe(self) -> str:
        return self.name

    def resolve(self, store, progress=None):
        Fetched.fetched.append(self.name)
        return store.root / self.name


class WeightedOcr(OcrProvider):
    id = "weighted-ocr"
    name = "Weighted OCR"

    @classmethod
    def weight_sources(cls, runtime=None):
        return (Fetched("weighted-ocr.bin"),)

    def _load(self):
        self.resolve_weights()

    def _recognize(self, crops):
        return ["w" for _ in crops]


def test_preload_loads_one_model_per_stage_and_downloads_the_rest(notebook):
    instance, http = notebook
    instance.registry.add(WeightedOcr)
    Fetched.fetched = []
    requests = [{"stage": "ocr", "provider": "size-ocr", "settings": {}}, {"stage": "ocr", "provider": "weighted-ocr", "settings": {}}]
    assert http.post("/v1/preload", json=requests).status_code == 202
    deadline = time.monotonic() + 5
    while (not Fetched.fetched or not any(m.state == "ready" for m in instance.models.values())) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert [m.provider.id for m in instance.models.values() if m.state == "ready"] == ["size-ocr"]
    assert Fetched.fetched == ["weighted-ocr.bin"]  # downloaded, not loaded: one model per stage


def installed_entry(model_id: str) -> dict:
    return {"id": model_id, "repo": "someone/ocr-model", "revision": "a" * 40, "runtime": "torch", "files": [{"path": "model.safetensors", "role": "model", "size": 10}]}


def test_a_load_request_carries_the_installed_model_and_the_worker_registers_it(notebook, isolated_home):
    from urakatsu.catalog.installed import InstalledModel, InstalledModels
    from urakatsu.remote.client import load_request
    from urakatsu.runtimes.store import ModelStore

    InstalledModels(ModelStore()).add(InstalledModel.model_validate(installed_entry("my-ocr")))
    body = load_request(SizeOcr({"model": "my-ocr", "remote_worker": "w1", "remote_url": "https://x"}, None, REMOTE))
    assert body["settings"]["model"] == "my-ocr" and "remote_worker" not in body["settings"] and "remote_url" not in body["settings"]
    assert body["installed"]["repo"] == "someone/ocr-model"
    with pytest.raises(ProviderUnavailable, match="was removed"):
        load_request(SizeOcr({"model": "gone"}, None, REMOTE))

    instance, http = notebook
    instance.runtimes.store = ModelStore(isolated_home / "worker-models")
    assert http.post("/v1/models", json={**body, "installed": installed_entry("other")}).status_code == 400
    assert http.post("/v1/models", json=body).status_code == 200
    assert InstalledModels(instance.runtimes.store).get("my-ocr") is not None


def assigned_config(**workers_by_slot):
    """A pipeline whose default models run on the given workers ({"ocr": "w1", ...})."""
    from urakatsu.core.config import PipelineConfig

    config = PipelineConfig()
    config.inpainting.model = "lama"
    config.fill.model = "flux2-klein"
    config.sound_effects.mode = "redraw"
    slots = {"detection": config.detection.model, "ocr": config.ocr.model, "inpainting": "lama", "fill": "flux2-klein"}
    for slot, worker in workers_by_slot.items():
        provider = slots[slot]
        config.runtimes[provider] = REMOTE
        config.processor[provider] = {"remote_worker": worker}
    return config


def test_assignments_follow_each_models_runtime_menu():
    config = assigned_config(ocr="colab", inpainting="kaggle", fill="kaggle")
    config.runtimes[config.detection.model] = REMOTE  # a Worker URL model: not assigned to a worker
    found = workers.assignments(config)
    # The redraw model is the fill model here, so FLUX.2 Klein shows once.
    assert [(a.worker, a.role, a.provider) for a in found] == [("colab", "ocr", config.ocr.model), ("kaggle", "inpainting", "lama"), ("kaggle", "fill", "flux2-klein")]
    assert all("remote_worker" not in a.request["settings"] for a in found)
    assert [r["provider"] for r in workers.requests_for("kaggle", config)] == ["lama", "flux2-klein"]


def test_kaggle_carries_its_models_to_preload(monkeypatch):
    api = KaggleApi()
    kaggle_through(api, monkeypatch)
    entry = workers.WorkerEntry(id="kaggle", kind="kaggle", kaggle=workers.KaggleAccount(token="KGAT_abc", username="reader"))
    workers.save([entry])
    monkeypatch.setattr(workers, "load_config", lambda: assigned_config(inpainting="kaggle"))
    workers.start(entry)
    _, save, _ = next(call for call in api.calls if call[0] == "SaveKernel")
    assert '"--preload"' in save["text"] and '\\"provider\\":\\"lama\\"' in save["text"]
    compile(save["text"], "worker.py", "exec")


def test_a_model_goes_only_to_its_worker(monkeypatch):
    workers.save([workers.WorkerEntry(id="colab", kind="colab", name="My Colab"), workers.WorkerEntry(id="box", kind="manual", url="https://box.test", token="t")])
    monkeypatch.setattr(workers, "info", lambda point: {"protocol": 2})
    with pytest.raises(ProviderUnavailable, match="My Colab is offline; run the notebook"):
        workers.connect("colab")  # the other worker answers, but this model is not assigned to it
    with pytest.raises(ProviderUnavailable, match="was removed"):
        workers.connect("gone")
    assert workers.connect("box").url == "https://box.test"
    assert workers.connect().url == "https://box.test"  # no worker picked: the first that answers


def test_an_offline_kaggle_worker_starts_for_its_models(monkeypatch):
    entry = workers.WorkerEntry(id="kaggle", kind="kaggle", token="t", kaggle=workers.KaggleAccount(token="KGAT_abc", username="reader"))
    workers.save([entry])
    started: list[str] = []
    monkeypatch.setattr(workers, "_start_and_wait", lambda starting, wait: started.append(starting.id) or workers.Endpoint(url="https://k.test", worker=starting.id))
    assert workers.connect("kaggle").url == "https://k.test" and started == ["kaggle"]


def test_saving_settings_preloads_only_what_changed(monkeypatch):
    workers.save([workers.WorkerEntry(id="colab", kind="colab", url="https://c.test", token="t")])
    workers._sent.clear()
    sent: list[tuple[str, list]] = []

    class Inline:
        def __init__(self, target, args, **kwargs):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(workers.threading, "Thread", Inline)
    monkeypatch.setattr(workers, "preload", lambda entry, requests: sent.append((entry.id, [r["provider"] for r in requests])) or workers._sent.__setitem__(entry.id, json.dumps(requests, sort_keys=True)))
    workers.sync(assigned_config(ocr="colab"))
    workers.sync(assigned_config(ocr="colab"))  # nothing changed: nothing sent
    workers.sync(assigned_config(ocr="colab", inpainting="colab"))
    assert [providers for _, providers in sent] == [["paddleocr-vl-1.6"], ["paddleocr-vl-1.6", "lama"]]


def test_the_workers_page_says_what_each_worker_runs(monkeypatch):
    workers.save([workers.WorkerEntry(id="kaggle", kind="kaggle", kaggle=workers.KaggleAccount(token="KGAT_abc", username="reader"))])
    monkeypatch.setattr(workers, "load_config", lambda: assigned_config(inpainting="kaggle", fill="kaggle"))
    (listed,) = workers_api().get("/api/v1/workers?status=false").json()["workers"]
    assert listed["status"] is None
    assert [(r["role"], r["provider"]) for r in listed["runs"]] == [("Inpainting", "lama"), ("Generative fill", "flux2-klein")]
