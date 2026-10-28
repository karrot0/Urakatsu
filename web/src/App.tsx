import { useRef } from "react";
import { CircleAlert, X } from "lucide-react";
import type { PanelImperativeHandle } from "react-resizable-panels";

import { Button } from "@/components/ui/button";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { Canvas } from "./components/Canvas";
import { ModelsPage } from "./components/ModelsPage";
import { CommandBar } from "./components/CommandBar";
import { TooltipProvider } from "./components/primitives";
import { RightPanel } from "./components/RightPanel";
import { SettingsPage } from "./components/SettingsPage";
import { SourcesPage } from "./components/SourcesPage";
import { BrowserPanel } from "./components/BrowserPanel";
import { Sidebar } from "./components/Sidebar";
import { Start } from "./components/Start";
import { WindowControls } from "./components/WindowControls";
import { MobileApp } from "./components/mobile/MobileApp";
import { useMobile } from "./components/mobile/useMobile";
import { UpdatePopup } from "./components/Updates";
import { useWorkspace } from "./state/workspace";

export function App() {
  const ws = useWorkspace();
  // Phone widths get the mobile layer: a tab bar, navigation stacks and sheets.
  const mobile = useMobile();
  return (
    <TooltipProvider>
      {mobile ? (
        <MobileApp />
      ) : ws.settingsOpen ? (
        <SettingsPage />
      ) : ws.modelsOpen ? (
        <ModelsPage />
      ) : ws.sourcesOpen ? (
        <SourcesPage />
      ) : ws.project ? (
        <Editor projectId={ws.project.id} />
      ) : (
        <Start />
      )}
      <UpdatePopup />
      {ws.browser && <BrowserPanel key={`${ws.browser.system}:${ws.browser.url}`} />}
      <WindowControls />
      {ws.error && (
        <div
          className="fixed bottom-5 left-1/2 z-50 flex max-w-[min(560px,calc(100vw_-_32px))] -translate-x-1/2 animate-[fade-in_var(--t-in)] items-center gap-2 rounded-card bg-surface-over py-2 pr-2 pl-3 shadow-overlay max-md:glass-thick max-md:top-[calc(var(--safe-top)+64px)] max-md:bottom-auto max-md:w-[calc(100vw_-_24px)] max-md:rounded-[18px] max-md:py-2.5"
          role="alert"
        >
          <CircleAlert size={16} className="text-danger" />
          <span>{ws.error}</span>
          <Button variant="ghost" size="icon-sm" onClick={ws.dismissError} aria-label="Dismiss">
            <X size={14} />
          </Button>
        </div>
      )}
    </TooltipProvider>
  );
}

const PANEL_WIDTHS_KEY = "urakatsu.panels";
const DEFAULT_WIDTHS = { sidebar: 264, inspector: 340 };

function loadWidths(): typeof DEFAULT_WIDTHS {
  try {
    return { ...DEFAULT_WIDTHS, ...JSON.parse(localStorage.getItem(PANEL_WIDTHS_KEY) ?? "{}") };
  } catch {
    return DEFAULT_WIDTHS;
  }
}

/** Chapters, canvas and inspector. The side panels keep their pixel width when the window resizes; double-click an edge to reset it. */
function Editor({ projectId }: { projectId: string }) {
  const widths = useRef(loadWidths());
  const sidebar = useRef<PanelImperativeHandle>(null);
  const inspector = useRef<PanelImperativeHandle>(null);
  const panel = { className: "flex min-h-0 flex-col", style: { overflow: "hidden" } };

  return (
    <ResizablePanelGroup
      orientation="horizontal"
      className="h-screen overflow-hidden"
      onLayoutChanged={(_, meta) => {
        if (!meta.isUserInteraction || !sidebar.current || !inspector.current) return;
        widths.current = { sidebar: Math.round(sidebar.current.getSize().inPixels), inspector: Math.round(inspector.current.getSize().inPixels) };
        try {
          localStorage.setItem(PANEL_WIDTHS_KEY, JSON.stringify(widths.current));
        } catch {
          // Storage is unavailable; the widths still apply for this session.
        }
      }}
    >
      <ResizablePanel id="sidebar" panelRef={sidebar} defaultSize={`${widths.current.sidebar}px`} minSize="200px" maxSize="480px" groupResizeBehavior="preserve-pixel-size" {...panel}>
        <Sidebar key={projectId} />
      </ResizablePanel>
      <ResizableHandle aria-label="Resize chapters" />
      {/* 600px keeps the whole command bar visible. */}
      <ResizablePanel id="canvas" minSize="600px" {...panel}>
        <main className="flex min-h-0 min-w-0 flex-1 flex-col">
          <CommandBar />
          <Canvas />
        </main>
      </ResizablePanel>
      <ResizableHandle aria-label="Resize inspector" />
      <ResizablePanel id="inspector" panelRef={inspector} defaultSize={`${widths.current.inspector}px`} minSize="280px" maxSize="560px" groupResizeBehavior="preserve-pixel-size" {...panel}>
        <RightPanel />
      </ResizablePanel>
    </ResizablePanelGroup>
  );
}
