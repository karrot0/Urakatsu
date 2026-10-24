"""The desktop app: the workspace server on a background thread, shown in a pywebview window.

Started by the launcher (``Urakatsu.exe`` on Windows) as ``pythonw -m urakatsu.desktop``
(``python`` on macOS and Linux). On first start
it installs PyTorch for this machine's GPU (``gpu_setup``) with progress in the window, then
loads the same UI ``urakatsu serve`` serves. Closing the window stops the server and every
runtime it started.
"""

from __future__ import annotations

import asyncio
import ctypes
import json
import logging
import os
import socket
import sys
import threading
import time
from importlib import resources
from pathlib import Path
from typing import Any, Optional

import httpx

from urakatsu import __version__
from urakatsu.core.config import home

from . import chrome, gpu_setup

log = logging.getLogger("urakatsu.desktop")

TITLE = "Urakatsu"
PREFERRED_PORT = 47811  # fixed, so the window's origin (and its saved theme and shortcuts) stays the same
APP_ID = "Urakatsu.Desktop"


# ---- the server ---------------------------------------------------------------------------------


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def pick_port() -> int:
    if _port_free(PREFERRED_PORT):
        return PREFERRED_PORT
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Server:
    """``urakatsu serve`` (UI, gRPC-web, JSON API) on its own thread and event loop."""

    def __init__(self, port: int):
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.error: Optional[BaseException] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutdown: Optional[asyncio.Event] = None
        self._thread = threading.Thread(target=self._run, name="urakatsu-server", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except BaseException as error:  # noqa: BLE001 - reported in the window
            log.exception("server stopped")
            self.error = error

    async def _main(self) -> None:
        from urakatsu.runtimes.packages import start_updater
        from urakatsu.service.app import serve
        from urakatsu.service.workspace import Workspace

        self._loop = asyncio.get_running_loop()
        self._shutdown = asyncio.Event()
        start_updater()  # newest llama.cpp and stable-diffusion.cpp, as `urakatsu serve` does
        await serve(Workspace(), port=self.port, grpc_port=None, log_level="warning", shutdown=self._shutdown)

    def wait_ready(self, timeout: float = 120.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.error is not None:
                raise RuntimeError(f"The server stopped: {self.error}")
            try:
                if httpx.get(f"{self.url}/api/health", timeout=1.0).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        raise RuntimeError("The server did not start in time")

    def stop(self) -> None:
        if self._loop is not None and self._shutdown is not None:
            self._loop.call_soon_threadsafe(self._shutdown.set)
        self._thread.join(timeout=20)


# ---- one instance -------------------------------------------------------------------------------


def _instance_path() -> Path:
    return home() / "app" / "desktop.json"


def _alive(pid: int) -> bool:
    if not _windows():
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.restype = ctypes.c_void_p
    handle = kernel32.OpenProcess(0x00100000 | 0x1000, False, pid)  # SYNCHRONIZE | QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        return kernel32.WaitForSingleObject(ctypes.c_void_p(handle), 0) == 0x102  # WAIT_TIMEOUT: still running
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


def running_instance() -> bool:
    """Whether another Urakatsu window is open (or still setting up) for this data folder."""
    try:
        record = json.loads(_instance_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    pid = record.get("pid")
    return isinstance(pid, int) and pid != os.getpid() and _alive(pid)


def _claim_instance(port: Optional[int] = None) -> None:
    """Recorded at startup (so a second launch during the GPU setup finds this one), then with the port."""
    path = _instance_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": os.getpid(), "port": port}), encoding="utf-8")


def _release_instance() -> None:
    try:
        _instance_path().unlink()
    except OSError:
        pass


# ---- Windows niceties ---------------------------------------------------------------------------


def _windows() -> bool:
    return sys.platform == "win32"


def _focus_existing() -> None:
    if not _windows():
        return
    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, TITLE)
    if hwnd:
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.SetForegroundWindow(hwnd)


def _set_icon() -> None:
    """pythonw.exe has Python's icon; give the window and its taskbar button Urakatsu's."""
    if not _windows():
        return
    user32 = ctypes.windll.user32
    icon = str(resources.files("urakatsu.desktop") / "urakatsu.ico")
    hwnd = 0
    for _ in range(50):
        hwnd = user32.FindWindowW(None, TITLE)
        if hwnd:
            break
        time.sleep(0.1)
    if not hwnd:
        return
    for which, size in ((1, 32), (0, 16)):  # ICON_BIG, ICON_SMALL
        handle = user32.LoadImageW(None, icon, 1, size, size, 0x10)  # IMAGE_ICON, LR_LOADFROMFILE
        if handle:
            user32.SendMessageW(hwnd, 0x80, which, handle)  # WM_SETICON


# ---- the window ---------------------------------------------------------------------------------


class Bridge:
    """Methods the pages call (``window.pywebview.api``): the setup page's buttons, and the window
    controls of the title bar the pages draw (chrome.py)."""

    def __init__(self) -> None:
        self.choice = threading.Event()
        self.skip_setup = False
        self._window: Any = None  # private: pywebview exposes only public attributes to the page

    def minimize(self) -> None:
        chrome.minimize(self._window)

    def toggle_maximize(self) -> None:
        chrome.toggle_maximize(self._window)

    def close(self) -> None:
        chrome.close(self._window)

    def drag(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        chrome.drag(self._window, x, y)

    def resize(self, edge: str, x: Optional[int] = None, y: Optional[int] = None) -> None:
        chrome.resize(self._window, edge, x, y)

    def system_menu(self) -> None:
        chrome.system_menu(self._window)

    def window_state(self) -> dict:
        return {"maximized": chrome.maximized(self._window), "double_click_ms": chrome.double_click_ms()}

    def retry(self) -> None:
        self.skip_setup = False
        self.choice.set()

    def skip(self) -> None:
        self.skip_setup = True
        self.choice.set()


class Desktop:
    def __init__(self) -> None:
        self.bridge = Bridge()
        self.server: Optional[Server] = None
        self.window: Any = None

    def render(self, **state: Any) -> None:
        try:
            self.window.evaluate_js(f"render({json.dumps(state)})")
        except Exception:  # noqa: BLE001 - the window may be closing
            pass

    def window_changed(self, *_: Any) -> None:
        """Tells the page whether the window is maximized, for the maximize button's icon."""
        try:
            state = json.dumps({"maximized": chrome.maximized(self.window)})
            self.window.evaluate_js(f"window.dispatchEvent(new CustomEvent('urakatsu:window', {{ detail: {state} }}))")
        except Exception:  # noqa: BLE001 - the window may be closing
            pass

    def boot(self) -> None:
        """Runs on pywebview's worker thread once the window exists."""
        _set_icon()
        self.window.events.loaded.wait(30)
        chrome.attach(self.window)
        self.window.events.resized += self.window_changed
        try:
            if gpu_setup.needs_setup():
                self.setup_gpu()
            self.render(title="Starting Urakatsu", message="Loading the workspace.")
            port = pick_port()
            self.server = Server(port)
            self.server.start()
            self.server.wait_ready()
            _claim_instance(port)
            if _windows():
                from urakatsu.updates import updater

                updater.installer = self.run_installer
            log.info("workspace ready at %s", self.server.url)
            self.window.load_url(f"{self.server.url}/?shell={chrome.SHELL}")
        except Exception as error:  # noqa: BLE001 - shown in the window
            log.exception("startup failed")
            self.render(title="Urakatsu could not start", message=f"Details are in {gpu_setup.default_log_path()}.", error=str(error))

    def setup_gpu(self) -> None:
        while True:
            self.render(title="Setting up GPU support", message="Checking your graphics driver.")
            try:
                build = gpu_setup.choose_build(gpu_setup.driver_cuda(), device=gpu_setup.primary_device())
                steps = gpu_setup.plan(build)
                size = "about 3 GB" if build.tag != "cpu" else "about 300 MB"
                message = f"Downloading PyTorch for {build.label} ({size}). This happens once."
                status = {step.id: "pending" for step in steps}

                def show(line: str = "") -> None:
                    self.render(
                        title="Setting up GPU support",
                        message=message,
                        steps=[{"title": s.title, "status": status[s.id]} for s in steps],
                        line=line,
                    )

                for step in steps:
                    status[step.id] = "active"
                    show()
                    try:
                        gpu_setup.run_step(step, show)
                    except Exception:
                        status[step.id] = "failed"
                        raise
                    status[step.id] = "done"
                gpu_setup.record(build)
                return
            except Exception as error:  # noqa: BLE001 - offered a retry
                log.exception("GPU setup failed")
                self.bridge.choice.clear()
                self.render(
                    title="Setting up GPU support",
                    message="Detection, some OCR models and LaMa need PyTorch. llama.cpp and stable-diffusion.cpp models work without it.",
                    error=str(error),
                )
                self.bridge.choice.wait()
                if self.bridge.skip_setup:
                    return

    def run_installer(self, installer: Path) -> None:
        """Updates the app: the new installer runs once this process has exited (an open app is
        never updated underneath, see the launcher), then starts the updated app itself."""
        quoted = str(installer).replace("'", "''")  # PowerShell single-quoted string
        script = f"Wait-Process -Id {os.getpid()} -ErrorAction SilentlyContinue; Start-Process -FilePath '{quoted}' -ArgumentList '--quiet'"
        import subprocess

        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", script],
            creationflags=flags,
            close_fds=True,
        )
        log.info("installing %s once the app quits", installer.name)
        self.window.destroy()

    def closed(self) -> None:
        log.info("window closed")
        if self.server is not None:
            self.server.stop()
        _release_instance()


def _configure_logging() -> None:
    path = gpu_setup.default_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8")],
    )


def main() -> int:
    _configure_logging()
    log.info("Urakatsu %s desktop, Python %s, %s", __version__, sys.version.split()[0], sys.executable)
    if running_instance():
        _focus_existing()
        return 0
    _claim_instance()
    if _windows():
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)

    import webview

    page = (resources.files("urakatsu.desktop") / "setup.html").read_text(encoding="utf-8").replace("{shell}", chrome.SHELL)
    desktop = Desktop()
    webview.settings.update(chrome.settings())
    desktop.window = webview.create_window(
        TITLE,
        html=page,
        js_api=desktop.bridge,
        width=1440,
        height=900,
        min_size=(960, 640),
        text_select=True,
        **chrome.window_options(),
    )
    desktop.bridge._window = desktop.window
    desktop.window.events.closed += desktop.closed
    storage = home() / "app" / "webview"
    storage.mkdir(parents=True, exist_ok=True)
    webview.settings["ALLOW_DOWNLOADS"] = True
    icon = str(resources.files("urakatsu.desktop") / "urakatsu.png") if sys.platform.startswith("linux") else None
    webview.start(desktop.boot, private_mode=False, storage_path=str(storage), icon=icon)
    return 0
