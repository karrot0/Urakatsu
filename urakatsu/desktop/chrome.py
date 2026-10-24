"""The app window's own title bar: the page draws it, the system moves and sizes the window.

- Windows: the window keeps its native frame, so resizing from the edges, the shadow and the
  rounded corners stay native, but its caption is removed (WM_NCCALCSIZE, as Windows Terminal
  does). A press on the page's title bar is handed to Windows as a press on the caption, so it
  runs its own move loop (with Snap, and dragging a maximized window restores it); a press on
  the page's top-edge strip is handed over as one on the top edge. The page draws minimize,
  maximize and close (WindowControls.tsx), maximizes on a double-click, and opens the system
  menu on a right-click. (WebView2's own ``app-region: drag`` support stops working once the
  window has been minimized, so it is not used.)
- macOS: pywebview's frameless window (full-size content under a transparent title bar, still
  resizable) with the native traffic lights shown again over the page's header.
- Linux: the system title bar. Qt's frameless windows lose their resize edges, and moving one
  from the page needs calls on Qt's GUI thread that pywebview does not offer.

The page learns which from ``?shell=`` (web/src/desktop.ts) and calls the functions below
through the window's JavaScript API (``app.Bridge``).
"""

from __future__ import annotations

import ctypes
import logging
import sys
from typing import Any, Callable

log = logging.getLogger("urakatsu.desktop")

if sys.platform == "win32":
    SHELL = "windows"
elif sys.platform == "darwin":
    SHELL = "macos"
else:
    SHELL = "linux"


def window_options() -> dict[str, Any]:
    """Extra ``webview.create_window`` arguments for this system."""
    if SHELL == "macos":
        return {"frameless": True, "easy_drag": False}
    return {}


def settings() -> dict[str, Any]:
    """pywebview settings: on macOS its drag script moves the window from the page's title bar."""
    if SHELL == "macos":
        return {"DRAG_REGION_SELECTOR": "[data-titlebar]", "DRAG_REGION_DIRECT_TARGET_ONLY": True}
    return {}


def attach(window: Any) -> None:
    """Turns the window's frame into the page's title bar. Call once the first page has loaded."""
    try:
        if SHELL == "windows":
            _on_ui_thread(window, lambda: _windows_attach(window))
        elif SHELL == "macos":
            _macos_attach(window)
    except Exception:  # noqa: BLE001 - the app works with the system frame too
        log.exception("could not set up the window's title bar")


def minimize(window: Any) -> None:
    window.minimize()


def toggle_maximize(window: Any) -> None:
    if maximized(window):
        window.restore()
    else:
        window.maximize()


def close(window: Any) -> None:
    window.destroy()


def maximized(window: Any) -> bool:
    try:
        if SHELL == "windows":
            return bool(_user32().IsZoomed(_hwnd(window)))
        if SHELL == "macos":
            return bool(window.native.isZoomed())
    except Exception:  # noqa: BLE001 - not shown yet
        pass
    return False


def double_click_ms() -> int:
    """The system's double-click time: two presses on the title bar this close maximize."""
    if SHELL == "windows":
        try:
            return int(_user32().GetDoubleClickTime())
        except Exception:  # noqa: BLE001
            pass
    return 500


def drag(window: Any, x: int | None = None, y: int | None = None) -> None:
    """Starts moving the window from a press on the page's title bar (Windows). ``x``, ``y`` are
    where the press happened, in screen pixels, so the window keeps its grip on the pointer even
    when it moved before this call arrived."""
    if SHELL == "windows":
        _frame_press(window, _WM_NCLBUTTONDOWN, _HTCAPTION, x, y)


def resize(window: Any, edge: str, x: int | None = None, y: int | None = None) -> None:
    """Starts a native resize from the page's top-edge strip: ``top``, ``top-left`` or ``top-right``."""
    hit = {"top": _HTTOP, "top-left": _HTTOPLEFT, "top-right": _HTTOPRIGHT}.get(edge)
    if SHELL == "windows" and hit is not None and not maximized(window):
        _frame_press(window, _WM_NCLBUTTONDOWN, hit, x, y)


def system_menu(window: Any) -> None:
    """Opens the window's system menu (restore, move, size, minimize, maximize, close) at the pointer."""
    if SHELL == "windows":
        _frame_press(window, _WM_POPUPSYSTEMMENU, 0)


# ---- Windows ---------------------------------------------------------------------------------------

_WM_NCCALCSIZE = 0x0083
_WM_SIZE = 0x0005
_WM_WINDOWPOSCHANGING, _WM_WINDOWPOSCHANGED = 0x0046, 0x0047
_SIZE_MINIMIZED, _SIZE_MAXIMIZED = 1, 2
_WM_NCLBUTTONDOWN = 0x00A1
_WM_PUT_BACK_SIZE = 0x8000 + 0x55  # WM_APP + 0x55: chrome.py's own message
_WM_POPUPSYSTEMMENU = 0x0313
_HTCAPTION, _HTTOP, _HTTOPLEFT, _HTTOPRIGHT = 2, 12, 13, 14
_GWLP_WNDPROC = -4
_SWP_FRAMECHANGED = 0x0020
_SWP_NOMOVE, _SWP_NOSIZE, _SWP_NOZORDER, _SWP_NOACTIVATE = 0x0002, 0x0001, 0x0004, 0x0010
_SM_CYFRAME, _SM_CXPADDEDBORDER = 33, 92

# The window procedure must outlive the window: ctypes frees a callback with its last reference.
_hooks: list[Any] = []


def _user32() -> Any:
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    lresult = ctypes.c_ssize_t
    user32.CallWindowProcW.argtypes = [ctypes.c_void_p, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.CallWindowProcW.restype = lresult
    user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
    user32.SetWindowLongPtrW.restype = ctypes.c_void_p
    user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT]
    user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.c_void_p]
    user32.GetWindowPlacement.argtypes = [wintypes.HWND, ctypes.c_void_p]
    user32.IsZoomed.argtypes = [wintypes.HWND]
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.GetDpiForWindow.argtypes = [wintypes.HWND]
    user32.GetSystemMetricsForDpi.argtypes = [ctypes.c_int, wintypes.UINT]
    return user32


def _frame_press(window: Any, message: int, hit: int, x: int | None = None, y: int | None = None) -> None:
    """Sends ``message`` for the frame part ``hit`` at (``x``, ``y``), else at the pointer, on the
    window's thread: for a press Windows then runs its own move or sizing loop until release."""

    def press() -> None:
        from ctypes import wintypes

        user32 = _user32()
        point = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(point))
        if x is not None and y is not None:
            point.x, point.y = int(x), int(y)
        user32.ReleaseCapture()
        user32.SendMessageW(_hwnd(window), message, hit, (point.y & 0xFFFF) << 16 | (point.x & 0xFFFF))

    _on_ui_thread(window, press)


def _hwnd(window: Any) -> int:
    return int(window.native.Handle.ToInt64())


def _on_ui_thread(window: Any, work: Callable[[], None]) -> None:
    """Runs ``work`` on the window's thread, which WebView2 and window procedures require."""
    from System import Func, Type  # pythonnet, which pywebview's WinForms backend loads

    def call() -> Any:
        work()
        return None

    window.native.Invoke(Func[Type](call))


def _windows_attach(window: Any) -> None:
    from ctypes import wintypes

    class NcCalcSizeParams(ctypes.Structure):
        _fields_ = [("rgrc", wintypes.RECT * 3), ("lppos", ctypes.c_void_p)]

    class WindowPos(ctypes.Structure):
        _fields_ = [
            ("hwnd", wintypes.HWND), ("after", wintypes.HWND), ("x", ctypes.c_int), ("y", ctypes.c_int),
            ("cx", ctypes.c_int), ("cy", ctypes.c_int), ("flags", wintypes.UINT),
        ]  # fmt: skip

    user32 = _user32()
    hwnd = _hwnd(window)
    procedure = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
    previous = ctypes.c_void_p()
    # WinForms restores a maximized or minimized form from its client size plus the caption it
    # believes the form has, which grows the window by a caption's height every time. Coming back
    # from maximized, its resize is held to the size Windows restored while it handles the restore
    # (WM_WINDOWPOSCHANGED). Coming back from minimized, holding it there keeps WinForms from laying
    # the web view out again (it then gets no input), so the size the window had when it was
    # minimized is put back just afterwards.
    state: dict[str, Any] = {"from": None, "hold": None, "put_back": None, "posted": False}

    class Placement(ctypes.Structure):
        _fields_ = [
            ("length", wintypes.UINT), ("flags", wintypes.UINT), ("show", wintypes.UINT),
            ("minimized", wintypes.POINT), ("maximized", wintypes.POINT), ("normal", wintypes.RECT),
        ]  # fmt: skip

    def normal_size(handle: int) -> tuple[int, int]:
        placement = Placement()
        placement.length = ctypes.sizeof(Placement)
        user32.GetWindowPlacement(handle, ctypes.byref(placement))
        rect = placement.normal
        return rect.right - rect.left, rect.bottom - rect.top

    def window_procedure(handle: int, message: int, wparam: int, lparam: int) -> int:
        if message == _WM_SIZE:
            if wparam == _SIZE_MINIMIZED and state["from"] is None:
                state["put_back"], state["posted"] = normal_size(handle), False
            state["from"] = {_SIZE_MAXIMIZED: "maximized", _SIZE_MINIMIZED: "minimized"}.get(wparam)
        elif message == _WM_WINDOWPOSCHANGED and state["from"] and not user32.IsZoomed(handle) and not user32.IsIconic(handle):
            if state["from"] == "minimized":
                if state["put_back"] and not state["posted"]:
                    state["posted"] = True
                    user32.PostMessageW(handle, _WM_PUT_BACK_SIZE, 0, 0)
                return user32.CallWindowProcW(previous, handle, message, wparam, lparam)
            position = WindowPos.from_address(lparam)
            state["hold"] = (position.cx, position.cy)
            try:
                return user32.CallWindowProcW(previous, handle, message, wparam, lparam)
            finally:
                state["hold"] = None
        elif message == _WM_WINDOWPOSCHANGING and state["hold"] is not None:
            position = WindowPos.from_address(lparam)
            if not position.flags & _SWP_NOSIZE:
                position.cx, position.cy = state["hold"]
        elif message == _WM_PUT_BACK_SIZE:
            size, state["put_back"] = state["put_back"], None
            current = wintypes.RECT()
            user32.GetWindowRect(handle, ctypes.byref(current))
            if size and (current.right - current.left, current.bottom - current.top) != size and not user32.IsZoomed(handle):
                user32.SetWindowPos(handle, None, 0, 0, size[0], size[1], _SWP_NOMOVE | _SWP_NOZORDER | _SWP_NOACTIVATE)
            return 0
        if message == _WM_NCCALCSIZE and wparam:
            # Let Windows compute the frame, then give the caption's height back to the client
            # area. A maximized window overhangs the screen by its frame, so keep that inset.
            params = NcCalcSizeParams.from_address(lparam)
            top = params.rgrc[0].top
            result = user32.CallWindowProcW(previous, handle, message, wparam, lparam)
            if user32.IsZoomed(handle):
                dpi = user32.GetDpiForWindow(handle)
                top += user32.GetSystemMetricsForDpi(_SM_CYFRAME, dpi) + user32.GetSystemMetricsForDpi(_SM_CXPADDEDBORDER, dpi)
            params.rgrc[0].top = top
            return result
        return user32.CallWindowProcW(previous, handle, message, wparam, lparam)

    hook = procedure(window_procedure)
    _hooks.append(hook)
    previous.value = user32.SetWindowLongPtrW(hwnd, _GWLP_WNDPROC, ctypes.cast(hook, ctypes.c_void_p))
    flags = _SWP_FRAMECHANGED | _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOZORDER | _SWP_NOACTIVATE
    user32.SetWindowPos(hwnd, None, 0, 0, 0, 0, flags)


# ---- macOS -----------------------------------------------------------------------------------------


def _macos_attach(window: Any) -> None:
    import AppKit
    from PyObjCTools import AppHelper

    def show_traffic_lights() -> None:
        for button in (AppKit.NSWindowCloseButton, AppKit.NSWindowMiniaturizeButton, AppKit.NSWindowZoomButton):
            window.native.standardWindowButton_(button).setHidden_(False)

    AppHelper.callAfter(show_traffic_lights)
