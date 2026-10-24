"""``urakatsu dev``: the service and the Vite dev server in one terminal, for working on Urakatsu.

- The service (``urakatsu serve``) runs as a child process and restarts when a Python file
  under ``urakatsu/`` changes. It is stopped the way Ctrl+C stops it (CTRL_BREAK_EVENT on
  Windows, SIGINT elsewhere), so ``Workspace.close`` still stops the llama-servers, Suwayomi
  and anything else it started. watchfiles' own ``run_process`` cannot be used for this: on
  Windows its SIGINT is ``TerminateProcess``, which leaves those processes running.
- The Vite dev server (hot reload for ``web/src``) proxies to the service through
  ``URAKATSU_API`` and keeps running across service restarts.
- Ctrl+C stops both. Both children sit in their own process group, so the terminal's Ctrl+C
  reaches only this supervisor, which then stops them in order.
- Ports not given on the command line start at 8090 and 5173 and move to the next free one
  (when another program holds one). A port that was given must be free.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "urakatsu"
WEB = ROOT / "web"
VITE = WEB / "node_modules" / "vite" / "bin" / "vite.js"
STOP_TIMEOUT = 30.0  # closing the workspace waits for its llama-servers and Suwayomi to exit
DEFAULT_PORT = 8090
DEFAULT_UI_PORT = 5173
PORT_TRIES = 20


class DevError(RuntimeError):
    pass


def port_free(host: str, port: int) -> bool:
    """Nothing answers on ``host:port``, and it can be bound.

    Both checks are needed: Windows lets a specific address bind beside a program that holds
    the wildcard one, and the bind check alone would miss it.
    """
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return False
    except OSError:
        pass
    with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def _held_by(port: int) -> str:
    """Names the program listening on ``port`` (" by docker.exe"), or "" when the OS does not say."""
    try:
        import psutil

        for connection in psutil.net_connections("inet"):
            if connection.status == psutil.CONN_LISTEN and connection.laddr and connection.laddr.port == port and connection.pid:
                return f" by {psutil.Process(connection.pid).name()}"
    except Exception:  # noqa: BLE001 - only words a message; access can be denied
        pass
    return ""


def pick_port(host: str, port: Optional[int], default: int, flag: str) -> int:
    """``port`` if it was given (it must be free), else ``default`` or the next free port after it."""
    if port is not None:
        if not port_free(host, port):
            raise DevError(f"Port {port} is in use{_held_by(port)}. Stop that program, or pass another {flag}.")
        return port
    for candidate in range(default, default + PORT_TRIES):
        if port_free(host, candidate):
            if candidate != default:
                print(f"[dev] port {default} is in use{_held_by(default)}; using {candidate}", flush=True)
            return candidate
    raise DevError(f"Ports {default} to {default + PORT_TRIES - 1} are all in use. Pass a free {flag}.")


def _own_group() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def stop(process: subprocess.Popen, timeout: float = STOP_TIMEOUT) -> None:
    """Asks ``process`` to stop as Ctrl+C would, and kills it if it has not after ``timeout``."""
    if process.poll() is not None:
        return
    try:
        process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
        process.wait(timeout)
    except OSError:
        pass
    except subprocess.TimeoutExpired:
        print(f"[dev] process {process.pid} did not stop in {timeout:.0f} s; killing it", flush=True)
    if process.poll() is None:
        process.kill()
        process.wait()


def service_command(args: list[str]) -> list[str]:
    return [sys.executable, "-m", "urakatsu.service.cli", "serve", *args]


def start_service(args: list[str]) -> subprocess.Popen:
    return subprocess.Popen(service_command(args), cwd=ROOT, **_own_group())


def start_vite(ui_port: int, api: str) -> subprocess.Popen:
    # node runs vite.js directly: through npm.cmd, cmd.exe would ask "Terminate batch job?".
    node = shutil.which("node")
    if not node:
        raise DevError("Node.js is not on PATH. Install Node.js 22 or newer, or pass --no-web.")
    if not VITE.exists():
        raise DevError(f"The web UI's packages are not installed. Run `npm install` in {WEB}, or pass --no-web.")
    env = {**os.environ, "URAKATSU_API": api}
    return subprocess.Popen([node, str(VITE), "--port", str(ui_port), "--strictPort"], cwd=WEB, env=env, **_own_group())


def _changed(changes: set[tuple[object, str]]) -> str:
    names = sorted({Path(path).relative_to(ROOT).as_posix() if Path(path).is_relative_to(ROOT) else path for _, path in changes})
    return ", ".join(names[:3]) + (f" and {len(names) - 3} more" if len(names) > 3 else "")


def run(
    host: str = "127.0.0.1",
    port: Optional[int] = None,
    ui_port: Optional[int] = None,
    service_args: Optional[list[str]] = None,
    web: bool = True,
    reload: bool = True,
) -> int:
    """``service_args`` are passed to ``urakatsu serve`` after ``--host`` and ``--port``."""
    from watchfiles import PythonFilter, watch

    port = pick_port(host, port, DEFAULT_PORT, "--port")
    api = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"
    args = ["--host", host, "--port", str(port), *(service_args or [])]
    ui_port = pick_port("localhost", ui_port, DEFAULT_UI_PORT, "--ui-port") if web else None
    vite: Optional[subprocess.Popen] = start_vite(ui_port, api) if ui_port else None
    service = start_service(args)
    print(f"[dev] service on {api}" + (", restarted when a file in urakatsu/ changes" if reload else ""), flush=True)
    if vite:
        print(f"[dev] open http://localhost:{ui_port} (the UI, with hot reload)", flush=True)
    print("[dev] Ctrl+C stops both", flush=True)
    try:
        if not reload:
            service.wait()
            print(f"[dev] the service stopped (exit code {service.returncode})", flush=True)
            return service.returncode or 0
        reported = False
        # Wakes every second without changes, to notice a service that stopped by itself.
        for changes in watch(PACKAGE, watch_filter=PythonFilter(), rust_timeout=1000, yield_on_timeout=True):
            if not changes:
                if service.poll() is not None and not reported:
                    print(f"[dev] the service stopped (exit code {service.returncode}); save a file in urakatsu/ to start it again", flush=True)
                    reported = True
                continue
            print(f"[dev] {_changed(changes)} changed; restarting the service", flush=True)
            stop(service)
            service = start_service(args)
            reported = False
    except KeyboardInterrupt:
        print("[dev] stopping", flush=True)
    finally:
        stop(service)
        if vite:
            stop(vite, timeout=5)
    return 0
