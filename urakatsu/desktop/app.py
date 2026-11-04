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
        self._started = threading.Event()  # set by serve() once requests are answered
        self._thread = threading.Thread(target=self._run, name="urakatsu-server", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            from urakatsu.service.event_loop import run_server

            run_server(self._main())
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
        await serve(workspace, port=self.port, grpc_port=None, log_level="warning", shutdown=self._shutdown, started=self._started.set)

    def wait_ready(self, timeout: float = 120.0) -> None:
        """Waits for serve() to say it answers requests. Not by asking it over HTTP: on Windows each try
        before the socket is open takes about a second to fail, so the window opened that much late."""
        deadline = time.monotonic() + timeout
        while not self._started.wait(0.05):
            if self.error is not None:
                raise RuntimeError(f"The server stopped: {self.error}")
            if not self._thread.is_alive() and self._thread.ident is not None:
                raise RuntimeError("The server stopped")
            if time.monotonic() >= deadline:
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
    """pythonw.exe has Python's icon and name; give the window and its taskbar button Urakatsu's."""
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
    try:
        _set_taskbar_identity(hwnd, icon)
    except OSError:
        log.exception("could not set the taskbar identity")


def _launcher() -> Optional[Path]:
    """The installed launcher (``Urakatsu.exe``), which a pinned taskbar button should start."""
    local = os.environ.get("LOCALAPPDATA")
    exe = Path(local) / "Programs" / "Urakatsu" / "Urakatsu.exe" if local else None
    return exe if exe is not None and exe.is_file() else None


class _PropertyKey(ctypes.Structure):
    _fields_ = [("fmtid", ctypes.c_byte * 16), ("pid", ctypes.c_ulong)]


class _PropVariant(ctypes.Structure):
    # VT_LPWSTR only: vt, three reserved words, then the string pointer (24 bytes on 64-bit).
    _fields_ = [
        ("vt", ctypes.c_ushort),
        ("reserved", ctypes.c_ushort * 3),
        ("value", ctypes.c_wchar_p),
        ("padding", ctypes.c_void_p),
    ]


def _set_taskbar_identity(hwnd: int, icon: str) -> None:
    """Names the taskbar button and its jump list "Urakatsu", and makes "Pin to taskbar" pin the app.

    The process AppUserModelID alone leaves Windows to describe the button from the executable,
    pythonw.exe, so the jump list said "Python" and a pin started a bare Python. The window's own
    relaunch properties (System.AppUserModel.*) override that.
    """
    import uuid

    launcher = _launcher()
    if launcher is not None:
        command, icon_resource = f'"{launcher}"', f"{launcher},0"
    else:
        python, _ = background_command()
        command, icon_resource = f'"{python}" -m urakatsu.desktop', icon
    fmtid = (ctypes.c_byte * 16).from_buffer_copy(uuid.UUID("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3").bytes_le)
    values = [(2, command), (3, icon_resource), (4, TITLE), (5, APP_ID)]

    ole32 = ctypes.windll.ole32
    initialized = ole32.CoInitializeEx(None, 2) in (0, 1)  # COINIT_APARTMENTTHREADED; S_OK or S_FALSE
    try:
        iid = (ctypes.c_byte * 16).from_buffer_copy(uuid.UUID("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99").bytes_le)
        store = ctypes.c_void_p()
        shell32 = ctypes.windll.shell32
        shell32.SHGetPropertyStoreForWindow.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        result = shell32.SHGetPropertyStoreForWindow(ctypes.c_void_p(hwnd), ctypes.byref(iid), ctypes.byref(store))
        if result != 0 or not store:
            raise OSError(f"SHGetPropertyStoreForWindow failed: {result:#x}")
        vtable = ctypes.cast(ctypes.cast(store, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
        hresult = ctypes.HRESULT if hasattr(ctypes, "HRESULT") else ctypes.c_long
        set_value = ctypes.WINFUNCTYPE(hresult, ctypes.c_void_p, ctypes.POINTER(_PropertyKey), ctypes.POINTER(_PropVariant))(vtable[6])
        commit = ctypes.WINFUNCTYPE(hresult, ctypes.c_void_p)(vtable[7])
        release = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtable[2])
        try:
            for pid, text in values:
                key = _PropertyKey(fmtid, pid)
                value = _PropVariant(vt=31, value=text)  # VT_LPWSTR; the store copies the string
                set_value(store, ctypes.byref(key), ctypes.byref(value))
            commit(store)
        finally:
            release(store)
    finally:
        if initialized:
            ole32.CoUninitialize()


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
    linux = sys.platform.startswith("linux")
    if linux:
        qt_downloads(storage)
    icon = str(resources.files("urakatsu.desktop") / "urakatsu.png") if linux else None
    webview.start(desktop.boot, private_mode=False, storage_path=str(storage), icon=icon, gui="qt" if linux else None)
    return 0


def qt_downloads(storage: Path) -> None:
    """Replaces pywebview's Qt download handler, which crashes the app on every export.

    pywebview 6.2 calls Qt 5's ``QWebEngineDownloadItem.setPath``; Qt 6's download request has no
    such method, and an exception in a PyQt6 slot aborts the process. Its dialog also suggested
    the blob URL's id rather than the name the page gave the file (``<a download>``).
    """
    try:
        from webview.platforms import qt
    except ImportError:
        log.exception("Qt cannot be loaded; downloads use pywebview's own handler")
        return
    # Read when the module loads, before ``webview.start`` sets the storage path: set it here,
    # or the window's profile (saved theme, shortcuts, sign-ins) would move to ~/.pywebview.
    qt._profile_storage_path = str(storage)

    def on_download_requested(view: Any, download: Any) -> None:
        try:
            name = download.suggestedFileName() if hasattr(download, "suggestedFileName") else Path(download.url().path()).name
            folder = download.downloadDirectory() if hasattr(download, "downloadDirectory") else str(Path.home() / "Downloads")
            path, _ = qt.QFileDialog.getSaveFileName(view, view.localization["global.saveFile"], str(Path(folder) / name))
            if not path:
                download.cancel()
                return
            target = Path(path)
            if hasattr(download, "setDownloadFileName"):  # Qt 6
                download.setDownloadDirectory(str(target.parent))
                download.setDownloadFileName(target.name)
            else:
                download.setPath(str(target))
            download.accept()
        except Exception:  # noqa: BLE001 - an exception here would abort the whole app
            log.exception("could not save a download")

    qt.BrowserView.on_download_requested = on_download_requested
