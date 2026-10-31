"""Starting a worker on Kaggle without opening Kaggle: the app pushes a private script and Kaggle runs it.

Kaggle's API (``https://api.kaggle.com/v1/<service>/<method>``, JSON POSTs, as its own
``kagglesdk`` calls it) can save a kernel and run it in the background on a T4 GPU with the
internet on. The script installs Urakatsu and runs ``urakatsu worker --tunnel`` with a token the
app made, which prints its tunnel's connection line (``tunnel``); the app reads that line from
the session's live log (``tunnel_url``), so nobody copies an address. ``--idle-exit`` ends the
session once nobody uses it, so it stops spending the account's weekly GPU hours; the next run
that needs it starts it again.

The account needs a phone-verified Kaggle account (Kaggle allows GPUs and the internet only then)
and an API token from kaggle.com/settings: a new ``KGAT_`` token, or a legacy key with its username.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

import httpx

from urakatsu.updates import build_info

from .tunnel import CONNECTION

API = "https://api.kaggle.com/v1"
SETTINGS_URL = "https://www.kaggle.com/settings"
TIMEOUT = httpx.Timeout(30.0)
MACHINE = "NvidiaTeslaT4"  # Kaggle's default GPU; the P100 is retired
REPOSITORY = "https://github.com/karrot0/Urakatsu"
# Minutes without a request before the worker ends its Kaggle session.
IDLE_EXIT = 30
# Kaggle ends a session after at most 12 hours.
SESSION_SECONDS = 12 * 3600
# Seconds the session log may stay quiet before reading it stops: a running session's log stream
# stays open, replaying what was written so far and then waiting for more.
LOG_IDLE = 3.0
# Session states (KernelWorkerStatus) in which a worker is starting or up.
RUNNING = frozenset({"QUEUED", "RUNNING", "NEW_SCRIPT"})


class KaggleError(RuntimeError):
    pass


def install_ref() -> str:
    """The Urakatsu commit a worker installs: this build's on a released build, else main."""
    info = build_info()
    return info.commit if info.channel != "dev" and re.fullmatch(r"[0-9a-f]{7,40}", info.commit or "") else "main"


def worker_script(token: str, ref: Optional[str] = None, preload: Optional[list[dict[str, Any]]] = None) -> str:
    """The notebook's script; ``preload`` lists the models assigned to it (``workers.assignments``)."""
    spec = f"urakatsu[worker] @ git+{REPOSITORY}@{ref or install_ref()}"
    command = ["urakatsu", "worker", "--tunnel", "--token", token, "--idle-exit", str(IDLE_EXIT)]
    if preload:
        command += ["--preload", json.dumps(preload, separators=(",", ":"))]
    return f'''# Urakatsu worker, pushed by the Urakatsu app (Settings, Workers). It runs detection, OCR and
# inpainting models for the app on this GPU and ends itself after {IDLE_EXIT} minutes without requests.
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", {json.dumps(spec)}])
subprocess.check_call({json.dumps(command)})
'''


def _seconds(value: Any) -> float:
    """A protobuf duration as Kaggle sends it ("151.500s")."""
    try:
        return float(str(value).rstrip("s")) if value is not None else 0.0
    except ValueError:
        return 0.0


class Kaggle:
    def __init__(self, token: str, username: str = "", client: Optional[httpx.Client] = None):
        token, username = token.strip(), username.strip()
        if token.startswith("{"):  # a pasted kaggle.json
            try:
                data = json.loads(token)
                token, username = str(data["key"]).strip(), username or str(data["username"]).strip()
            except (ValueError, KeyError) as error:
                raise KaggleError("that kaggle.json has no username and key") from error
        if not token:
            raise KaggleError(f"add a Kaggle API token (from {SETTINGS_URL})")
        self.token = token
        self.username = username
        self._client = client

    def _auth(self) -> dict[str, Any]:
        if self.token.startswith("KGAT_"):
            return {"headers": {"Authorization": f"Bearer {self.token}"}}
        if not self.username:
            raise KaggleError("a legacy Kaggle key needs its username; or make a new API token at " + SETTINGS_URL)
        return {"auth": (self.username, self.token)}

    def call(self, service: str, method: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{API}/{service}/{method}"
        try:
            if self._client is not None:
                response = self._client.post(url, json=body, **self._auth())
            else:
                with httpx.Client(timeout=TIMEOUT) as client:
                    response = client.post(url, json=body, **self._auth())
        except httpx.HTTPError as error:
            raise KaggleError(f"cannot reach Kaggle: {error}") from error
        try:
            data = response.json() if response.content else {}
        except ValueError:
            data = {}
        if response.status_code == 401:
            raise KaggleError(f"Kaggle rejected the API token; make a new one at {SETTINGS_URL}")
        if response.status_code >= 400 or (isinstance(data, dict) and int(data.get("code") or 0) >= 400):
            message = data.get("message") if isinstance(data, dict) else None
            raise KaggleError(f"Kaggle: {message or response.text[:300] or response.status_code}")
        return data if isinstance(data, dict) else {}

    def whoami(self) -> str:
        """The account's username (asked of Kaggle for a ``KGAT_`` token)."""
        if not self.username:
            data = self.call("security.OAuthService", "IntrospectToken", {"token": self.token})
            if not data.get("active") or not data.get("username"):
                raise KaggleError(f"Kaggle does not accept this API token; make a new one at {SETTINGS_URL}")
            self.username = str(data["username"])
        return self.username

    def push(self, slug: str, title: str, script: str) -> str:
        """Saves the script as a private kernel and runs it on a GPU; returns its "user/slug"."""
        body = {
            "slug": f"{self.whoami()}/{slug}",
            "newTitle": title,
            "text": script,
            "language": "python",
            "kernelType": "script",
            "isPrivate": True,
            "enableGpu": True,
            "enableInternet": True,
            "machineShape": MACHINE,
            "sessionTimeoutSeconds": SESSION_SECONDS,
        }
        data = self.call("kernels.KernelsApiService", "SaveKernel", body)
        if data.get("error"):
            error = str(data["error"])
            if "phone" in error.lower() or "verif" in error.lower():
                error += f" (verify your phone number at {SETTINGS_URL} to use GPUs and the internet)"
            raise KaggleError(f"Kaggle: {error}")
        ref = str(data.get("ref") or "").strip("/")
        return "/".join(ref.split("/")[-2:]) if ref.count("/") >= 1 else f"{self.username}/{slug}"

    def status(self, ref: str) -> tuple[str, str]:
        """The latest session's state (QUEUED, RUNNING, COMPLETE, ERROR, ...) and failure message."""
        user, _, slug = ref.partition("/")
        data = self.call("kernels.KernelsApiService", "GetKernelSessionStatus", {"userName": user, "kernelSlug": slug})
        return str(data.get("status") or "QUEUED"), str(data.get("failureMessage") or "")

    def quota(self) -> dict[str, float]:
        """This week's GPU hours: used, reserved by running sessions, and allowed."""
        data = self.call("kernels.KernelsApiService", "GetAcceleratorQuotaStatistics", {})
        gpu = data.get("gpuQuota") or {}
        return {
            "used_hours": _seconds(gpu.get("timeUsed")) / 3600,
            "reserved_hours": _seconds(gpu.get("timeReserved")) / 3600,
            "allowed_hours": _seconds(gpu.get("totalTimeAllowed")) / 3600,
        }

    def tunnel_url(self, ref: str) -> Optional[str]:
        """The worker's latest tunnel address, from the running session's log, or None before it opens.

        ``GET /v1/kernels/logs/stream/{owner}/{slug}`` streams a running session's log as
        server-sent events of ``{stream_name, time, data}``, starting from the beginning; a
        finished session answers with the whole log as JSON instead (the ``kaggle`` CLI's
        ``kernels_logs_stream``).
        """
        url = f"{API}/kernels/logs/stream/{ref}"
        timeout = httpx.Timeout(TIMEOUT.connect, read=LOG_IDLE)
        found: Optional[str] = None
        auth = self._auth()
        headers = {"Accept": "text/event-stream, */*", **auth.pop("headers", {})}
        try:
            client = self._client or httpx.Client(timeout=timeout)
            try:
                with client.stream("GET", url, headers=headers, timeout=timeout, **auth) as response:
                    if response.status_code in (401, 403, 404):
                        return None
                    response.raise_for_status()
                    if response.headers.get("content-type", "").startswith("text/event-stream"):
                        for line in response.iter_lines():
                            if line.startswith("data:"):
                                found = _connection(line[5:].strip()) or found
                    else:
                        found = _connection(response.read().decode("utf-8", "replace"))
            finally:
                if client is not self._client:
                    client.close()
        except httpx.ReadTimeout:
            pass  # the log went quiet: everything written so far has been read
        except httpx.HTTPError as error:
            raise KaggleError(f"cannot read the Kaggle session's log: {error}") from error
        return found


def _connection(text: str) -> Optional[str]:
    """The last tunnel address in a piece of the log (an event's data, or a whole log)."""
    try:
        data = json.loads(text)
        text = "\n".join(str(e.get("data", "")) for e in (data if isinstance(data, list) else [data]) if isinstance(e, dict))
    except ValueError:
        pass
    matches = CONNECTION.findall(text)
    return matches[-1][0].rstrip("/") if matches else None
