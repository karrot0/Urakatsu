"""A Cloudflare quick tunnel to the worker's port, for a worker on a notebook nobody can reach.

``cloudflared tunnel --url http://127.0.0.1:<port>`` needs no account and gives a random
``https://<words>.trycloudflare.com`` address, a new one each time it starts. ``Tunnel`` keeps
it running (restarting it if it exits) and calls ``on_url`` with every new address; the worker
prints it as a connection line, which the app reads from a Kaggle session's log or the user
pastes from Colab (``workers``). cloudflared is a native runtime like llama.cpp: installed on
first use and kept on its newest release (``runtimes.packages``).

Cloudflare relays the traffic: it is encrypted on the way, but Cloudflare could read the pages.

Cloudflare closes a request after 100 s, which the worker protocol already allows for (no
request is held longer than ``worker.MAX_WAIT``).
"""

from __future__ import annotations

import logging
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)

ADDRESS = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
# What the worker prints for each tunnel address: the line a Kaggle session's log carries and the
# one a Colab user pastes into the app (Settings, Workers).
CONNECTION = re.compile(r"Connection: (https://\S+?)#(\S+)")


def connection_line(url: str, token: str) -> str:
    return f"Connection: {url}#{token}"


def parse_connection(text: str) -> Optional[tuple[str, str]]:
    """(address, token) from a connection line, or from just "<address>#<token>"."""
    text = text.strip()
    match = CONNECTION.search(text if text.startswith("Connection: ") else f"Connection: {text}")
    return (match.group(1).rstrip("/"), match.group(2)) if match else None


# Seconds before starting cloudflared again after it exits, doubling up to RESTART_MAX.
RESTART_DELAY = 2.0
RESTART_MAX = 60.0


def cloudflared() -> Path:
    from urakatsu.runtimes.hardware import discover
    from urakatsu.runtimes.packages import CLOUDFLARED, PackageManager

    return PackageManager().ensure(CLOUDFLARED, discover().candidates()).executable


class Tunnel:
    def __init__(self, port: int, on_url: Callable[[str], None], executable: Optional[Path] = None):
        self.port = port
        self.on_url = on_url
        self.executable = executable
        self.url: Optional[str] = None
        self._process: Optional[subprocess.Popen] = None
        self._stopped = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.executable is None:
            self.executable = cloudflared()
        self._thread = threading.Thread(target=self._run, name="tunnel", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()

    def _run(self) -> None:
        delay = RESTART_DELAY
        while not self._stopped.is_set():
            started = time.monotonic()
            command = [str(self.executable), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{self.port}"]
            self._process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
            assert self._process.stderr is not None
            for line in self._process.stderr:  # cloudflared logs to stderr, the address among it
                match = ADDRESS.search(line)
                if match and match.group(0) != self.url:
                    self.url = match.group(0)
                    log.info("tunnel open at %s", self.url)
                    try:
                        self.on_url(self.url)
                    except Exception as error:  # noqa: BLE001 - a failed publish must not stop the tunnel
                        log.warning("could not announce the tunnel address: %s", error)
                elif "ERR" in line:
                    log.debug("cloudflared: %s", line.rstrip())
            code = self._process.wait()
            if self._stopped.is_set():
                return
            # A tunnel that ran a while starts again quickly; one that keeps failing backs off.
            delay = RESTART_DELAY if time.monotonic() - started > RESTART_MAX else min(delay * 2, RESTART_MAX)
            log.warning("cloudflared exited with %s; starting it again in %.0f s", code, delay)
            self.url = None
            self._stopped.wait(delay)
