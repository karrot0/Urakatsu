/**
 * The desktop app's own title bar (urakatsu/desktop/chrome.py).
 *
 * The app window opens the workspace with ?shell=windows|macos|linux; the shell is kept in
 * sessionStorage (the router drops the query) and set as <html data-shell>, which turns on the
 * insets in styles/theme.css and WindowControls. In a browser there is no shell.
 *
 * Any element marked data-titlebar acts as the window's title bar, except the controls inside it:
 * - Windows: a press is handed to the system's move loop (with Snap); a second press within the
 *   double-click time maximizes or restores; a right-click opens the system menu.
 * - macOS: pywebview moves the window from data-titlebar itself; a double-click zooms.
 */

export type Shell = "windows" | "macos" | "linux";

type WindowApi = {
  minimize(): Promise<void>;
  toggle_maximize(): Promise<void>;
  close(): Promise<void>;
  drag(x: number, y: number): Promise<void>;
  resize(edge: string, x: number, y: number): Promise<void>;
  system_menu(): Promise<void>;
  window_state(): Promise<WindowState>;
};

export type WindowState = { maximized: boolean; double_click_ms?: number };

const KEY = "urakatsu.shell";

function readShell(): Shell | null {
  const fromQuery = new URLSearchParams(location.search).get("shell");
  let value: string | null = fromQuery;
  try {
    if (fromQuery) sessionStorage.setItem(KEY, fromQuery);
    value ??= sessionStorage.getItem(KEY);
  } catch {
    // Storage blocked: the query and the webview are left.
  }
  return value === "windows" || value === "macos" || value === "linux" ? value : hostShell();
}

/**
 * The app window's webview, for a page that lost the query and its session storage: going back
 * to the setup page (an about:blank document) and forward again reloads the workspace without
 * either. WebView2 gives every page window.chrome.webview, and pywebview's WebKit windows a
 * jsBridge message handler; a browser has neither.
 */
function hostShell(): Shell | null {
  const host = window as unknown as { chrome?: { webview?: unknown }; webkit?: { messageHandlers?: { jsBridge?: unknown } } };
  if (host.chrome?.webview) return "windows";
  if (host.webkit?.messageHandlers?.jsBridge) return navigator.userAgent.includes("Mac") ? "macos" : "linux";
  return null;
}

export const shell = readShell();
if (shell) document.documentElement.dataset.shell = shell;

/** The window's API, once pywebview has injected it. */
export function windowApi(): WindowApi | undefined {
  return (window as unknown as { pywebview?: { api?: WindowApi } }).pywebview?.api;
}

/** Where a press happened, in screen pixels: the window keeps its grip on the pointer. */
export function screenPoint(event: { screenX: number; screenY: number }): [number, number] {
  return [Math.round(event.screenX * devicePixelRatio), Math.round(event.screenY * devicePixelRatio)];
}

const INTERACTIVE = 'button, a, input, select, textarea, label, [role="button"], [role="tab"], [role="menuitem"], [role="combobox"], [contenteditable="true"]';

function titlebarTarget(event: Event): boolean {
  const target = event.target;
  return target instanceof Element && !!target.closest("[data-titlebar]") && !target.closest(INTERACTIVE);
}

let doubleClickMs = 500;
let lastPress = -Infinity;

/** Window state from the app: the maximize button's icon, and the double-click time. */
export function onWindowState(listener: (state: WindowState) => void): () => void {
  const handle = (event: Event) => {
    const state = (event as CustomEvent<WindowState>).detail;
    if (state.double_click_ms) doubleClickMs = state.double_click_ms;
    listener(state);
  };
  window.addEventListener("urakatsu:window", handle);
  const ask = () => void windowApi()?.window_state().then((state) => handle(new CustomEvent("urakatsu:window", { detail: state })));
  if (windowApi()) ask();
  else window.addEventListener("pywebviewready", ask, { once: true });
  return () => {
    window.removeEventListener("urakatsu:window", handle);
    window.removeEventListener("pywebviewready", ask);
  };
}

if (shell === "windows") {
  document.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || !titlebarTarget(event)) return;
    const api = windowApi();
    if (!api) return;
    event.preventDefault();
    if (event.timeStamp - lastPress < doubleClickMs) {
      lastPress = -Infinity;
      void api.toggle_maximize();
    } else {
      lastPress = event.timeStamp;
      void api.drag(...screenPoint(event));
    }
  });
  document.addEventListener("contextmenu", (event) => {
    if (!titlebarTarget(event)) return;
    event.preventDefault();
    void windowApi()?.system_menu();
  });
} else if (shell === "macos") {
  document.addEventListener("dblclick", (event) => {
    if (titlebarTarget(event)) void windowApi()?.toggle_maximize();
  });
}
