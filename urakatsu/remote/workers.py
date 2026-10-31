"""The workers this app uses (Settings, Workers): notebooks and machines that run its models.

Kept in ``URAKATSU_HOME/workers.json``, in the order the app tries them:

- ``kaggle``: the app starts it itself, as a private Kaggle script on a T4 (``kaggle``), with the
  account's API token and a worker token the app made. Its tunnel address is read from the
  session's log, so a session that restarts on a new address is found again by itself. With
  ``auto_start`` a run that needs a worker starts it when none answers.
- ``colab``: the Colab notebook (``notebooks/worker.ipynb``), opened from the app with one click.
  It prints a connection line (``tunnel.connection_line``) that the user pastes back here, and
  pastes again if the notebook restarts on a new address.
- ``manual``: any machine running ``urakatsu worker``, at a fixed address with its token.

Kaggle and Colab workers are reached through a Cloudflare tunnel (``tunnel``), so Cloudflare
relays their pages.

Each model picks its worker in the runtime menu (Settings, Pipeline), saved as ``remote_worker``
in its settings; ``assignments`` reads them back. A model goes only to its worker (``connect``):
an offline Kaggle worker that may start itself starts, any other says it is offline. The models
assigned to a worker are preloaded so the first page does not wait for weights: a Kaggle script
carries them (``urakatsu worker --preload``), and a Colab or other-machine worker gets them as
soon as it answers or the assignments change (``preload``, ``sync``). A remote model with no
worker picked uses its Worker URL, else the first worker here that answers.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Literal, Optional

import httpx
from pydantic import BaseModel, Field

from urakatsu.core.config import PipelineConfig, home, load_config
from urakatsu.core.errors import ProviderUnavailable
from urakatsu.core.stage import Stage

from . import PROTOCOL, REMOTE, kaggle

log = logging.getLogger(__name__)

COLAB_URL = "https://colab.research.google.com/github/karrot0/Urakatsu/blob/main/notebooks/worker.ipynb"
INFO_TIMEOUT = 5.0
# Seconds a Kaggle worker's address read from its log is trusted before it is read again.
LOOKUP_TTL = 30.0
# Seconds a run waits for a Kaggle worker it started: Kaggle queues it, then Urakatsu installs.
START_WAIT = 15 * 60.0
START_POLL = 10.0

Kind = Literal["kaggle", "colab", "manual"]


class KaggleAccount(BaseModel):
    token: str = ""
    username: str = ""
    # "user/slug" of the script the app pushed, once it has.
    ref: str = ""


class WorkerEntry(BaseModel):
    id: str = Field(default_factory=lambda: secrets.token_hex(4))
    name: str = ""
    kind: Kind = "manual"
    url: str = ""  # colab (pasted), manual
    token: str = ""  # the worker's bearer token; for kaggle, made by the app
    auto_start: bool = True  # kaggle: start it when a run needs a worker and none answers
    kaggle: Optional[KaggleAccount] = None


class Assignment(BaseModel):
    worker: str  # the entry's id
    role: str  # detection | ocr | inpainting | fill | sound_effects
    provider: str
    name: str
    request: dict[str, Any]  # the load request (``client.load_request``)


# The pipeline's model slots, in the order a worker preloads them.
ROLES: tuple[tuple[str, Stage], ...] = (
    ("detection", Stage.DETECTION),
    ("ocr", Stage.OCR),
    ("inpainting", Stage.INPAINTING),
    ("fill", Stage.INPAINTING),
    ("sound_effects", Stage.INPAINTING),
)


class Endpoint(BaseModel):
    url: str
    token: str = ""
    worker: str = ""  # the entry's id


def path() -> Path:
    return home() / "workers.json"


_lock = threading.RLock()
# Kaggle worker id -> (when read, the tunnel address its log last showed).
_addresses: dict[str, tuple[float, Optional[str]]] = {}
# Worker id -> the preload last sent to it (``sync`` sends only what changed).
_sent: dict[str, str] = {}


def load() -> list[WorkerEntry]:
    try:
        raw = json.loads(path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [WorkerEntry.model_validate(item) for item in raw if isinstance(item, dict)]


def save(entries: list[WorkerEntry]) -> None:
    target = path()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps([e.model_dump(mode="json") for e in entries], indent=2), encoding="utf-8")
    tmp.replace(target)


def find(identifier: str) -> WorkerEntry:
    entry = next((e for e in load() if e.id == identifier), None)
    if entry is None:
        raise KeyError(f"no worker {identifier}")
    return entry


def update(entry: WorkerEntry) -> None:
    with _lock:
        entries = load()
        save([entry if e.id == entry.id else e for e in entries])


# -- what each worker runs -----------------------------------------------------------------------


def _slot(config: PipelineConfig, role: str) -> str:
    if role == "fill":
        return config.fill_provider()
    if role == "sound_effects":
        return config.redraw_provider() if config.sound_effects.mode == "redraw" else ""
    return config.provider_for(Stage.parse(role))


def assignments(config: Optional[PipelineConfig] = None) -> list[Assignment]:
    """The models set to run on one of the app's workers, each once, in preload order."""
    from urakatsu.core.registry import registry

    from .client import load_request

    config = config or load_config()
    found: list[Assignment] = []
    seen: set[str] = set()
    for role, stage in ROLES:
        provider_id = _slot(config, role)
        if not provider_id or provider_id in seen or config.runtime_for(provider_id) != REMOTE:
            continue
        settings = config.settings_for(stage, provider_id)
        worker = settings.get("remote_worker")
        if not worker:
            continue
        seen.add(provider_id)
        try:
            cls = registry.get(stage, provider_id)
            request = load_request(cls(settings, None, REMOTE))
        except (KeyError, ValueError, ProviderUnavailable) as error:
            log.info("not preloading %s: %s", provider_id, error)
            continue
        found.append(Assignment(worker=worker, role=role, provider=provider_id, name=cls.name or cls.id, request=request))
    return found


def requests_for(worker: str, config: Optional[PipelineConfig] = None) -> list[dict[str, Any]]:
    return [a.request for a in assignments(config) if a.worker == worker]


# -- where a worker is ---------------------------------------------------------------------------


def kaggle_address(entry: WorkerEntry, fresh: bool = False) -> Optional[str]:
    """The tunnel address a Kaggle worker's session log last showed, or None before it opened one."""
    if entry.kaggle is None or not entry.kaggle.ref:
        return None
    cached = _addresses.get(entry.id)
    if cached is not None and not fresh and time.monotonic() - cached[0] < LOOKUP_TTL:
        return cached[1]
    try:
        found = account(entry).tunnel_url(entry.kaggle.ref)
    except kaggle.KaggleError as error:
        log.info("could not read Kaggle worker %s's log: %s", entry.name or entry.id, error)
        found = None
    _addresses[entry.id] = (time.monotonic(), found)
    return found


def endpoint(entry: WorkerEntry, fresh: bool = False) -> Optional[Endpoint]:
    url = kaggle_address(entry, fresh) if entry.kind == "kaggle" else entry.url.strip()
    return Endpoint(url=url.rstrip("/"), token=entry.token.strip(), worker=entry.id) if url else None


def info(point: Endpoint) -> Optional[dict[str, Any]]:
    """The worker's ``/v1/info``, or None when it does not answer there."""
    headers = {"Authorization": f"Bearer {point.token}"} if point.token else {}
    try:
        response = httpx.get(f"{point.url}/v1/info", headers=headers, timeout=INFO_TIMEOUT)
        return response.json() if response.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None


def status(entry: WorkerEntry) -> dict[str, Any]:
    """What Settings shows for a worker: online (and on what), starting, or offline, and why."""
    state: dict[str, Any] = {"state": "offline", "device": None, "url": None, "detail": None}
    point = endpoint(entry, fresh=True)
    answer = info(point) if point is not None else None
    if answer is not None:
        state.update(state="online", device=answer.get("device"), url=point.url, stoppable=bool(answer.get("stoppable")))
        if answer.get("protocol") != PROTOCOL:
            state.update(state="outdated", detail=f"it speaks protocol {answer.get('protocol')}, this app {PROTOCOL}; update Urakatsu there")
        return state
    if entry.kind == "kaggle" and entry.kaggle is not None and entry.kaggle.ref:
        try:
            session, failure = account(entry).status(entry.kaggle.ref)
        except kaggle.KaggleError as error:
            state["detail"] = str(error)
            return state
        if session in kaggle.RUNNING:
            state.update(state="starting", detail="Kaggle is queuing the session" if session == "QUEUED" else "installing Urakatsu and opening the tunnel")
        elif session == "ERROR":
            state["detail"] = failure or "the last Kaggle session failed"
    elif entry.kind == "colab":
        if point is not None:  # before a line is pasted, the Connection row says what to do
            state["detail"] = "it does not answer there; if the notebook restarted, paste the new connection line it printed"
    elif point is not None:
        state["detail"] = f"it does not answer at {point.url}"
    return state


# -- starting and stopping ------------------------------------------------------------------------


def account(entry: WorkerEntry) -> kaggle.Kaggle:
    if entry.kaggle is None:
        raise kaggle.KaggleError("this worker has no Kaggle account")
    return kaggle.Kaggle(entry.kaggle.token, entry.kaggle.username)


def start(entry: WorkerEntry) -> WorkerEntry:
    """Pushes and runs the Kaggle script (a no-op while a session is already starting or up)."""
    if entry.kind != "kaggle":
        raise ValueError(f"the app starts only Kaggle workers; start a {entry.kind} worker where it runs")
    client = account(entry)
    if entry.kaggle.ref:
        try:
            session, _ = client.status(entry.kaggle.ref)
        except kaggle.KaggleError:
            session = ""
        if session in kaggle.RUNNING:
            return entry
    if not entry.token:
        entry.token = secrets.token_urlsafe(24)
    slug = f"urakatsu-worker-{entry.id}"
    preload = requests_for(entry.id)
    entry.kaggle.ref = client.push(slug, f"Urakatsu worker {entry.id}", kaggle.worker_script(entry.token, preload=preload))
    _sent[entry.id] = json.dumps(preload, sort_keys=True)
    entry.kaggle.username = client.username
    update(entry)
    _addresses.pop(entry.id, None)  # the new session opens a new tunnel
    log.info("started Kaggle worker %s (%s)", entry.name or entry.id, entry.kaggle.ref)
    return entry


def stop(entry: WorkerEntry) -> bool:
    """Asks the worker to end itself (which ends its notebook session); False if it does not answer."""
    point = endpoint(entry, fresh=True)
    if point is None:
        return False
    headers = {"Authorization": f"Bearer {point.token}"} if point.token else {}
    try:
        response = httpx.post(f"{point.url}/v1/shutdown", headers=headers, timeout=INFO_TIMEOUT)
    except httpx.HTTPError:
        return False
    _addresses.pop(entry.id, None)
    return response.status_code == 202


def preload(entry: WorkerEntry, requests: Optional[list[dict[str, Any]]] = None) -> bool:
    """Sends the worker its assigned models to load ahead of use; False if it does not answer."""
    requests = requests_for(entry.id) if requests is None else requests
    point = endpoint(entry)
    if not requests or point is None:
        return False
    headers = {"Authorization": f"Bearer {point.token}"} if point.token else {}
    try:
        response = httpx.post(f"{point.url}/v1/preload", json=requests, headers=headers, timeout=INFO_TIMEOUT)
    except httpx.HTTPError as error:
        log.info("could not preload worker %s: %s", entry.name or entry.id, error)
        return False
    if response.status_code != 202:
        return False
    _sent[entry.id] = json.dumps(requests, sort_keys=True)
    return True


def sync(config: PipelineConfig) -> None:
    """Preloads the models whose assignment changed onto workers that answer (after Settings saves)."""
    try:
        entries = {e.id: e for e in load()}
        wanted: dict[str, list[dict[str, Any]]] = {}
        for assignment in assignments(config):
            wanted.setdefault(assignment.worker, []).append(assignment.request)
    except Exception as error:  # noqa: BLE001 - saving settings never fails over a preload
        log.info("could not work out the workers' models: %s", error)
        return
    for worker, requests in wanted.items():
        entry = entries.get(worker)
        if entry is not None and _sent.get(worker) != json.dumps(requests, sort_keys=True):
            threading.Thread(target=preload, args=(entry, requests), name="worker-preload", daemon=True).start()


def connect(worker: Optional[str] = None, fresh: bool = False, wait: float = START_WAIT) -> Endpoint:
    """``worker`` (an entry's id) once it answers, starting it first if it is a Kaggle worker that
    may start itself; without one, the first worker that answers, else an auto-start Kaggle one."""
    entries = load()
    if worker is not None:
        entry = next((e for e in entries if e.id == worker), None)
        if entry is None:
            raise ProviderUnavailable("the worker this model was assigned to was removed; pick another runtime in Settings, Pipeline")
        point = endpoint(entry, fresh)
        if point is not None and info(point) is not None:
            return point
        if entry.kind == "kaggle" and entry.auto_start and entry.kaggle is not None:
            return _start_and_wait(entry, wait)
        name = entry.name or entry.kind
        if entry.kind == "kaggle":
            raise ProviderUnavailable(f"{name} is offline; start it in Settings, Workers")
        if entry.kind == "colab":
            raise ProviderUnavailable(f"{name} is offline; run the notebook and paste its connection line in Settings, Workers")
        raise ProviderUnavailable(f"{name} does not answer at {entry.url or 'its address'}")
    if not entries:
        raise ProviderUnavailable("no worker: add one in Settings, Workers, or set this model's Worker URL")
    for entry in entries:
        point = endpoint(entry, fresh)
        if point is not None and info(point) is not None:
            return point
    starting = next((e for e in entries if e.kind == "kaggle" and e.auto_start and e.kaggle is not None), None)
    if starting is None:
        names = ", ".join(e.name or e.kind for e in entries)
        raise ProviderUnavailable(f"no worker answers ({names}); start one in Settings, Workers")
    return _start_and_wait(starting, wait)


def _start_and_wait(starting: WorkerEntry, wait: float) -> Endpoint:
    try:
        start(starting)
    except kaggle.KaggleError as error:
        raise ProviderUnavailable(f"could not start the Kaggle worker: {error}") from error
    log.info("waiting for the Kaggle worker to come up (up to %.0f minutes)", wait / 60)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        time.sleep(START_POLL)
        point = endpoint(starting, fresh=True)
        if point is not None and info(point) is not None:
            return point
        try:
            session, failure = account(starting).status(starting.kaggle.ref)
        except kaggle.KaggleError as error:
            log.info("could not ask Kaggle about the worker: %s", error)
            continue
        if session not in kaggle.RUNNING:
            raise ProviderUnavailable(f"the Kaggle worker stopped before it came up: {failure or session.lower()}")
    raise ProviderUnavailable(f"the Kaggle worker did not come up within {wait / 60:.0f} minutes; see Settings, Workers")
