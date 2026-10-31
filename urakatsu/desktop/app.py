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
import hashlib
import json
import logging
import os
import shutil
import socket
import sys
import threading
import time
from importlib import resources
from pathlib import Path
from typing import Any, Optional

import httpx
import psutil

from urakatsu import __version__
from urakatsu.core.config import home

from . import chrome, gpu_setup

log = logging.getLogger("urakatsu.desktop")

TITLE = "Urakatsu"
PREFERRED_PORT = 47811  # fixed, so the window's origin (and its saved theme and shortcuts) stays the same
APP_ID = "Urakatsu.Desktop"
START_TOLERANCE = 2.0  # seconds a process's start time may differ from its record (the launcher reads it from FILETIME)
GPU_SHARE = 90.0  # the startup bar's share (in percent) for GPU setup; loading the workspace fills the rest


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
        workspace = Workspace()
        # A scheduled automation that wakes the computer starts this app if it is closed.
        workspace.automations.wake.launch = background_command()
        await serve(workspace, port=self.port, grpc_port=None, log_level="warning", shutdown=self._shutdown)

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


BACKGROUND = "--background"


def background_command() -> tuple[str, list[str]]:
    """Starts the app minimized, or does nothing when it is already open (``BACKGROUND``)."""
    python = Path(sys.executable)
    windowless = python.with_name("pythonw.exe")
    return str(windowless if windowless.exists() else python), ["-m", "urakatsu.desktop", BACKGROUND]


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


def _started(pid: int) -> Optional[float]:
    """When a process began (seconds since the epoch), or None when it is gone or unreadable."""
    try:
        return psutil.Process(pid).create_time()
    except psutil.Error:
        return None


def _is_urakatsu(pid: int) -> bool:
    """Whether a process was started as this app, for a record that predates ``started``."""
    try:
        return "urakatsu.desktop" in psutil.Process(pid).cmdline()[1:]  # pythonw -m urakatsu.desktop
    except psutil.AccessDenied:
        return True  # alive but unreadable: believe the record
    except psutil.Error:
        return False


def running_instance() -> bool:
    """Whether another Urakatsu window is open (or still setting up) for this data folder.

    An app that was killed leaves its record behind, and Windows hands its process id to the next
    program that starts (a terminal, a browser), so the id alone proves nothing: the process must
    also have begun when the record says this app did.
    """
    try:
        record = json.loads(_instance_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    pid = record.get("pid")
    if not isinstance(pid, int) or pid == os.getpid() or not _alive(pid):
        return False
    started = record.get("started")
    if not isinstance(started, (int, float)):
        return _is_urakatsu(pid)
    actual = _started(pid)
    return actual is not None and abs(actual - started) < START_TOLERANCE


def _claim_instance(port: Optional[int] = None) -> None:
    """Recorded at startup (so a second launch during the GPU setup finds this one), then with the port."""
    path = _instance_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"pid": os.getpid(), "port": port, "started": _started(os.getpid())}
    path.write_text(json.dumps(record), encoding="utf-8")


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


def drop_stale_cache(storage: Path) -> None:
    """Empties the webview's HTTP cache when the UI build changed since the last start.

    Earlier builds served the app page without ``cache-control``, and WebView2 kept such a page as
    fresh for hours after an update, so the window ran the old bundle it names. The cache is
    only that: emptying it keeps the saved theme, shortcuts and sign-ins (Local Storage, cookies).
    Runs before the webview starts, while nothing holds the files (one instance, see above).
    """
    from urakatsu.service.app import DEFAULT_UI_DIR

    page = DEFAULT_UI_DIR / "index.html"
    if not page.is_file():
        return
    build = hashlib.sha256(page.read_bytes()).hexdigest()[:16]
    stamp = storage / "ui-build"
    try:
        if stamp.read_text(encoding="utf-8").strip() == build:
            return
    except OSError:
        pass
    shutil.rmtree(storage / "EBWebView" / "Default" / "Cache", ignore_errors=True)
    try:
        stamp.write_text(build, encoding="utf-8")
    except OSError:
        log.warning("could not record the UI build in %s", stamp)


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


def format_bytes(count: float) -> str:
    for unit, size in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if count >= size:
            return f"{count / size:.1f} {unit}"
    return f"{count:.0f} B"


class NetworkMeter:
    """The bytes this machine has received since the meter started, and the rate over the last
    second or so. Counted for the whole machine (psutil has no per-download count), which during
    setup is the downloads themselves."""

    def __init__(self) -> None:
        self.start = psutil.net_io_counters().bytes_recv
        self.mark = self.start
        self.at = time.monotonic()
        self.rate = 0.0

    def reading(self) -> str:
        now = time.monotonic()
        received = psutil.net_io_counters().bytes_recv
        if now - self.at >= 1.0:
            self.rate = (received - self.mark) / (now - self.at)
            self.mark, self.at = received, now
        return f"{format_bytes(self.rate)}/s · {format_bytes(received - self.start)} received"


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
            self.render(title="Starting Urakatsu", message="Loading the workspace.", percent=GPU_SHARE)
            port = pick_port()
            self.server = Server(port)
            self.server.start()
            self.server.wait_ready()
            _claim_instance(port)
            if _windows():
                from urakatsu.updates import updater

                updater.installer = self.run_installer
            log.info("workspace ready at %s", self.server.url)
            self.render(title="Starting Urakatsu", message="Loading the workspace.", percent=100)
            self.window.load_url(f"{self.server.url}/?shell={chrome.SHELL}")
        except Exception as error:  # noqa: BLE001 - shown in the window
            log.exception("startup failed")
            self.render(title="Urakatsu could not start", message=f"Details are in {gpu_setup.default_log_path()}.", error=str(error))

    def setup_gpu(self) -> None:
        while True:
            self.render(title="Setting up GPU support", message="Checking your graphics driver.", percent=0)
            try:
                build = gpu_setup.choose_build(gpu_setup.driver_cuda(), device=gpu_setup.primary_device())
                steps = gpu_setup.plan(build)
                size = "about 3 GB" if build.tag != "cpu" else "about 300 MB"
                message = f"Downloading PyTorch for {build.label} ({size}). This happens once."
                status = {step.id: "pending" for step in steps}
                fractions = {step.id: 0.0 for step in steps}
                weight = sum(step.weight for step in steps)
                meter = NetworkMeter()

                def show(line: str = "") -> None:
                    done = sum(step.weight * (1.0 if status[step.id] == "done" else fractions[step.id]) for step in steps)
                    self.render(
                        title="Setting up GPU support",
                        message=message,
                        percent=GPU_SHARE * done / weight,
                        network=meter.reading(),
                        steps=[{"title": s.title, "status": status[s.id]} for s in steps],
                        line=line,
                    )

                for step in steps:
                    status[step.id] = "active"
                    show()
                    progress = gpu_setup.StepProgress()

                    def on_line(line: str, step: gpu_setup.Step = step, progress: gpu_setup.StepProgress = progress) -> None:
                        fractions[step.id] = progress.feed(line)
                        show(line)

                    try:
                        gpu_setup.run_step(step, on_line)
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

        # A hidden console, not DETACHED_PROCESS: Windows PowerShell started without a console
        # exits at once (status 0) without running its command, so the installer never ran.
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
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
    background = BACKGROUND in sys.argv[1:]  # started for a scheduled automation
    if running_instance():
        if not background:
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
        minimized=background,
        **chrome.window_options(),
    )
    desktop.bridge._window = desktop.window
    desktop.window.events.closed += desktop.closed
    storage = home() / "app" / "webview"
    storage.mkdir(parents=True, exist_ok=True)
    drop_stale_cache(storage)
    webview.settings["ALLOW_DOWNLOADS"] = True
    icon = str(resources.files("urakatsu.desktop") / "urakatsu.png") if sys.platform.startswith("linux") else None
    webview.start(desktop.boot, private_mode=False, storage_path=str(storage), icon=icon)
    return 0
