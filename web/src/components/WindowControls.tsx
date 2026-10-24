import { useEffect, useState } from "react";

import { cn } from "@/lib/utils";
import { onWindowState, screenPoint, shell, windowApi } from "../desktop";

/**
 * The desktop window's controls, drawn by the page because the window has no system title bar
 * (urakatsu/desktop/chrome.py). Windows: minimize, maximize and close at the top right, in the
 * system's own glyphs, and strips along the top edge that resize the window where the caption
 * used to. macOS keeps its native traffic lights; Linux keeps its system title bar.
 */
export function WindowControls() {
  const [maximized, setMaximized] = useState(false);
  const [active, setActive] = useState(() => document.hasFocus());

  useEffect(() => {
    if (shell !== "windows") return;
    const stop = onWindowState((state) => setMaximized(state.maximized));
    const focus = () => setActive(true);
    const blur = () => setActive(false);
    window.addEventListener("focus", focus);
    window.addEventListener("blur", blur);
    return () => {
      stop();
      window.removeEventListener("focus", focus);
      window.removeEventListener("blur", blur);
    };
  }, []);

  if (shell !== "windows") return null;
  const glyph = cn(
    "flex h-12 w-[46px] cursor-default items-center justify-center font-[Segoe_Fluent_Icons,Segoe_MDL2_Assets] text-[10px] transition-colors duration-[20ms] ease-in focus-visible:shadow-[inset_0_0_0_2px_var(--brand)] focus-visible:outline-none",
    active ? "text-fg" : "text-fg-3",
  );
  return (
    <>
      {!maximized && (
        <>
          <ResizeEdge edge="top-left" className="left-0 w-2 cursor-nwse-resize" />
          <ResizeEdge edge="top" className="right-2 left-2 cursor-ns-resize" />
          <ResizeEdge edge="top-right" className="right-0 w-2 cursor-nesw-resize" />
        </>
      )}
      <div className="fixed top-0 right-0 z-[60] flex" role="group" aria-label="Window">
        <button type="button" className={cn(glyph, "hover:bg-hover active:bg-press")} title="Minimize" aria-label="Minimize" onClick={() => void windowApi()?.minimize()}>
          {"\uE921"}
        </button>
        <button
          type="button"
          className={cn(glyph, "hover:bg-hover active:bg-press")}
          title={maximized ? "Restore" : "Maximize"}
          aria-label={maximized ? "Restore" : "Maximize"}
          onClick={() => void windowApi()?.toggle_maximize()}
        >
          {maximized ? "\uE923" : "\uE922"}
        </button>
        <button
          type="button"
          className={cn(glyph, "hover:bg-(--window-close) hover:text-white active:bg-(--window-close) active:text-white/80")}
          title="Close"
          aria-label="Close"
          onClick={() => void windowApi()?.close()}
        >
          {"\uE8BB"}
        </button>
      </div>
    </>
  );
}

function ResizeEdge({ edge, className }: { edge: "top" | "top-left" | "top-right"; className: string }) {
  return (
    <div
      aria-hidden
      className={cn("fixed top-0 z-[61] h-[5px]", className)}
      onPointerDown={(event) => {
        if (event.button === 0) void windowApi()?.resize(edge, ...screenPoint(event));
      }}
    />
  );
}

/** Keeps a header's content clear of the window controls: put it at the header's left or right end. */
export function TitleBarInset({ side }: { side: "left" | "right" }) {
  if (!shell) return null;
  return <span aria-hidden className={cn("h-full flex-none", side === "left" ? "w-(--window-controls-left)" : "w-(--window-controls-right)")} />;
}
