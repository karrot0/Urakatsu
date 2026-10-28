// The page editor on a phone: the canvas fills the screen (pinch to zoom, two fingers to pan,
// Canvas.tsx), glass bars float over it. Top: back, a pager and undo/redo. Bottom: the tools in
// a glass toolbar, brush options above it while a brush is out, and Run. The inspector, the run
// options and the chapter's pages open in sheets.

import { Brush, Check, ChevronLeft, ChevronRight, Eraser, Eye, EyeOff, Layers, MousePointer2, Play, Redo2, SlidersHorizontal, Sparkles, Square, SquareDashed, Lasso, Type, Undo2, WandSparkles, type LucideIcon } from "lucide-react";
import { useMemo, useState } from "react";

import { cn } from "cn";

import { assetHref, assetVersion } from "../../api/client";
import type { Config, TranslationModel } from "../../gen/urakatsu/v1/urakatsu_pb";
import { MIN_BRUSH, STAGES, chapterOf, currentPage, stageDone, useWorkspace, type RunScope, type StageName, type Tool, type ViewMode } from "../../state/workspace";
import { Canvas } from "../Canvas";
import { STAGE_LABEL } from "../CommandBar";
import { ColorWell, Slider } from "../primitives";
import { RightPanel } from "../RightPanel";
import { isBrushTool, maxBrush } from "../ToolBelt";
import { Segmented } from "../ui";
import { BarIcon, GlassButton, GlassGroup, ListRow, ListSection, SearchField, Sheet } from "./glass";

const TOOLS: { tool: Tool; label: string; icon: LucideIcon }[] = [
  { tool: "select", label: "Select", icon: MousePointer2 },
  { tool: "text", label: "Text", icon: Type },
  { tool: "draw", label: "Brush", icon: Brush },
  { tool: "eraser", label: "Eraser", icon: Eraser },
  { tool: "remove", label: "Remove text", icon: Sparkles },
  { tool: "fill", label: "Generative fill", icon: WandSparkles },
];

const VIEWS: { value: ViewMode; label: string }[] = [
  { value: "source", label: "Source" },
  { value: "cleanup", label: "Layers" },
  { value: "render", label: "Rendered" },
];

export function MobileEditor({ onBack }: { onBack: () => void }) {
  const ws = useWorkspace();
  const project = ws.project!;
  const page = currentPage(ws);
  const chapter = chapterOf(ws, ws.pageId);
  const [sheet, setSheet] = useState<"inspector" | "run" | "pages">();
  const order = chapter?.pageIds ?? [];
  const index = page ? order.indexOf(page.id) : -1;
  const prev = index > 0 ? order[index - 1] : undefined;
  const next = index >= 0 && index < order.length - 1 ? order[index + 1] : undefined;
  const running = ws.job?.status === "running";
  const progress = ws.job && ws.job.total ? ws.job.completed / ws.job.total : 0;

  return (
    <div className="fixed inset-0 flex flex-col bg-canvas">
      <Canvas />

      <header className="pointer-events-none fixed inset-x-0 top-0 z-30 pt-safe-top">
        <div className="flex h-[60px] items-center gap-2 pr-[max(12px,var(--safe-right))] pl-[max(12px,var(--safe-left))] [&>*]:pointer-events-auto">
          <GlassButton label={chapter ? `Back to ${chapter.name}` : "Back"} onClick={onBack}>
            <ChevronLeft size={24} strokeWidth={2.25} className="-ml-0.5" />
          </GlassButton>
          <GlassGroup label="Pages" className="min-w-0 flex-1">
            <BarIcon label="Previous page" disabled={!prev} onClick={() => prev && ws.selectPage(prev)}>
              <ChevronLeft size={20} />
            </BarIcon>
            <button
              type="button"
              className="flex h-touch min-w-0 flex-1 cursor-pointer flex-col items-center justify-center rounded-full leading-4 outline-none select-none focus-visible:outline-2 focus-visible:outline-brand active:bg-glass-selected [-webkit-tap-highlight-color:transparent]"
              aria-label="Pages in this chapter"
              onClick={() => setSheet("pages")}
            >
              <span className="max-w-full truncate text-[15px] font-semibold tabular-nums">{index >= 0 ? `${index + 1} of ${order.length}` : page?.name}</span>
              <span className="max-w-full truncate text-[11px] text-fg-2">{chapter?.name}</span>
            </button>
            <BarIcon label="Next page" disabled={!next} onClick={() => next && ws.selectPage(next)}>
              <ChevronRight size={20} />
            </BarIcon>
          </GlassGroup>
          <GlassGroup label="History">
            <BarIcon label="Undo" disabled={!project.canUndo} onClick={() => void ws.undo()}>
              <Undo2 size={20} />
            </BarIcon>
            <BarIcon label="Redo" disabled={!project.canRedo} onClick={() => void ws.redo()}>
              <Redo2 size={20} />
            </BarIcon>
          </GlassGroup>
        </div>
      </header>

      <footer className="pointer-events-none fixed inset-x-0 bottom-0 z-30 flex flex-col items-center gap-2 pr-[max(12px,var(--safe-right))] pb-[max(calc(var(--safe-bottom)_-_10px),12px)] pl-[max(12px,var(--safe-left))] [&>*]:pointer-events-auto">
        {running && (
          <div className="glass flex h-10 w-full max-w-[440px] items-center gap-3 rounded-full pr-1 pl-4" role="status">
            <span className="min-w-0 flex-1 truncate text-[13px] font-medium">Translating{ws.job?.total ? ` ${ws.job.completed} of ${ws.job.total}` : ""}</span>
            <span className="relative h-1 w-20 overflow-hidden rounded-full bg-line-strong">
              <span className="absolute inset-y-0 left-0 rounded-[inherit] bg-brand transition-[width] duration-200 ease-settle" style={{ width: `${Math.round(progress * 100)}%` }} />
            </span>
            <button type="button" className="h-8 cursor-pointer rounded-full px-3 text-[13px] font-medium text-danger active:bg-glass-selected" onClick={() => void ws.stop()}>
              Stop
            </button>
          </div>
        )}
        {page && <ToolOptions />}
        <div className="flex w-full max-w-[440px] items-center gap-2">
          <div className="glass flex min-w-0 flex-1 items-center overflow-x-auto rounded-full p-1 [scrollbar-width:none]" role="toolbar" aria-label="Tools">
            {/* Under 400px the buttons narrow to 40pt so all seven fit; they stay 44pt tall. */}
            {TOOLS.map(({ tool, label, icon: Icon }) => (
              <BarIcon key={tool} label={label} active={ws.tool === tool} aria-pressed={ws.tool === tool} disabled={!page} className="max-[400px]:w-10" onClick={() => ws.setTool(tool)}>
                <Icon size={20} />
              </BarIcon>
            ))}
            <span className="mx-0.5 h-6 w-px flex-none bg-line" aria-hidden />
            <BarIcon label="Inspector" className="max-[400px]:w-10" onClick={() => setSheet("inspector")}>
              <SlidersHorizontal size={20} />
            </BarIcon>
          </div>
          <GlassButton label="Run" tone="prominent" onClick={() => setSheet("run")}>
            <Play size={20} fill="currentColor" className="ml-0.5" />
          </GlassButton>
        </div>
      </footer>

      <Sheet open={sheet === "inspector"} onOpenChange={(open) => setSheet(open ? "inspector" : undefined)} title="Inspector" contentClassName="flex flex-col px-0 pb-0">
        <div className="flex flex-none flex-col gap-2 px-4 pb-2">
          <Segmented<ViewMode> label="View" fill value={ws.view} onChange={ws.setView} options={VIEWS} />
          <div className="flex gap-2">
            <ToggleChip on={ws.showBlocks} icon={ws.showBlocks ? Eye : EyeOff} label="Text frames" onClick={() => ws.toggleOverlay("blocks")} />
            <ToggleChip on={ws.showRegions} icon={Layers} label="Balloons and panels" onClick={() => ws.toggleOverlay("regions")} />
          </div>
        </div>
        <div className="flex min-h-0 flex-1 flex-col pb-[var(--safe-bottom)]">
          <RightPanel />
        </div>
      </Sheet>

      <RunSheet open={sheet === "run"} onOpenChange={(open) => setSheet(open ? "run" : undefined)} />

      <Sheet open={sheet === "pages"} onOpenChange={(open) => setSheet(open ? "pages" : undefined)} title={chapter?.name ?? "Pages"}>
        <ul className="grid grid-cols-3 gap-3 pt-1" aria-label="Pages">
          {order.map((id, i) => {
            const item = project.pages.find((p) => p.id === id);
            if (!item) return null;
            const current = id === ws.pageId;
            return (
              <li key={id}>
                <button
                  type="button"
                  className="flex w-full cursor-pointer flex-col gap-1 outline-none active:opacity-70 [-webkit-tap-highlight-color:transparent]"
                  aria-current={current ? "page" : undefined}
                  onClick={() => {
                    ws.selectPage(id);
                    setSheet(undefined);
                  }}
                >
                  <img
                    className={cn("block aspect-[5/7] w-full rounded-control bg-surface-fill object-cover object-top", current ? "shadow-[0_0_0_2.5px_var(--brand)]" : "shadow-[0_0_0_0.5px_var(--border)]")}
                    src={assetHref(project.id, id, "thumbnail", assetVersion(item, "thumbnail"))}
                    alt=""
                    loading="lazy"
                  />
                  <span className={cn("text-[13px] tabular-nums", current ? "font-semibold text-brand" : "text-fg-2")}>Page {i + 1}</span>
                </button>
              </li>
            );
          })}
        </ul>
      </Sheet>
    </div>
  );
}

function ToggleChip({ on, icon: Icon, label, onClick }: { on: boolean; icon: LucideIcon; label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      aria-pressed={on}
      className={cn(
        "inline-flex h-9 min-w-0 flex-1 cursor-pointer items-center justify-center gap-1.5 rounded-full px-3 text-[13px] font-medium outline-none select-none focus-visible:shadow-focus [-webkit-tap-highlight-color:transparent]",
        on ? "bg-brand-weak text-brand" : "bg-grouped-fill text-fg-2",
      )}
      onClick={onClick}
    >
      <Icon size={16} /> <span className="truncate">{label}</span>
    </button>
  );
}

/** Brush color and size while a brush is out; the selection shape for generative fill. */
function ToolOptions() {
  const ws = useWorkspace();
  const page = currentPage(ws)!;
  if (isBrushTool(ws.tool)) {
    const max = maxBrush(page.width, page.height);
    const size = Math.round(ws.brush.diameter);
    return (
      <div className="glass flex h-12 w-full max-w-[440px] items-center gap-3 rounded-full pr-4 pl-2" role="group" aria-label="Brush">
        {ws.tool === "draw" ? (
          <ColorWell label="Brush color" value={ws.brush.color} onChange={(color) => color && ws.setBrush({ ...ws.brush, color })} />
        ) : (
          <span className="w-1" />
        )}
        <div className="min-w-0 flex-1">
          <Slider label="Brush size" min={MIN_BRUSH} max={max} value={size} onChange={(diameter) => ws.setBrush({ ...ws.brush, diameter: Math.min(max, Math.max(MIN_BRUSH, Math.round(diameter))) })} />
        </div>
        <span className="w-8 text-right text-[13px] font-medium text-fg-2 tabular-nums">{size}</span>
      </div>
    );
  }
  if (ws.tool === "fill") {
    return (
      <div className="glass flex items-center rounded-full p-1" role="group" aria-label="Selection shape">
        <BarIcon label="Rectangle selection" active={ws.fillShape === "rect"} aria-pressed={ws.fillShape === "rect"} onClick={() => ws.setFillShape("rect")}>
          <SquareDashed size={20} />
        </BarIcon>
        <BarIcon label="Lasso selection" active={ws.fillShape === "lasso"} aria-pressed={ws.fillShape === "lasso"} onClick={() => ws.setFillShape("lasso")}>
          <Lasso size={20} />
        </BarIcon>
      </div>
    );
  }
  return null;
}

// ---- run ----------------------------------------------------------------------------------------

const SCOPES: { scope: RunScope; label: string }[] = [
  { scope: "page", label: "This page" },
  { scope: "chapter", label: "This chapter" },
  { scope: "project", label: "Entire project" },
];

const STAGE_HINT: Record<StageName, string> = {
  detection: "Find text, balloons and panels",
  ocr: "Read the source text",
  translation: "Translate every text block",
  inpainting: "Remove the original lettering",
};

/** What Run does: the translation model, the scope and the stages, then Run or Stop. */
function RunSheet({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const ws = useWorkspace();
  const [choosing, setChoosing] = useState(false);
  const [query, setQuery] = useState("");
  const running = ws.job?.status === "running";
  const page = currentPage(ws);
  const translation = ws.config?.translation;
  const selected = ws.translationModels.find((m) => m.provider === translation?.provider && m.model === translation?.model);
  const models = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return ws.translationModels.filter((m) => !needle || `${m.name} ${m.model} ${m.providerName}`.toLowerCase().includes(needle));
  }, [ws.translationModels, query]);

  const run = () => {
    const stages = STAGES.filter((s) => ws.runStages.includes(s));
    if (stages.length === STAGES.length) void ws.run({ kind: "full", scope: ws.runScope });
    else if (stages.length === 1) void ws.run({ kind: "only", stage: stages[0], scope: ws.runScope });
    else void ws.run({ kind: "stages", stages, scope: ws.runScope });
    onOpenChange(false);
  };
  const toggleStage = (stage: StageName) => {
    const has = ws.runStages.includes(stage);
    if (has && ws.runStages.length === 1) return;
    ws.setRunStages(has ? ws.runStages.filter((s) => s !== stage) : STAGES.filter((s) => s === stage || ws.runStages.includes(s)));
  };
  const choose = (model: TranslationModel) => {
    if (!ws.config || !translation) return;
    void ws.saveConfig({
      ...ws.config,
      translation: { ...translation, provider: model.provider, model: model.model, quantization: model.quantizations[0]?.id ?? "", vision: model.vision, reasoning: model.reasoning },
    } as Config);
    setChoosing(false);
  };

  return (
    <Sheet
      open={open}
      onOpenChange={(next) => {
        onOpenChange(next);
        if (!next) setChoosing(false);
      }}
      title={choosing ? "Model" : "Run"}
      leading={
        choosing && (
          <GlassButton label="Back" className="size-9 w-9" onClick={() => setChoosing(false)}>
            <ChevronLeft size={20} />
          </GlassButton>
        )
      }
    >
      {choosing ? (
        <>
          <SearchField value={query} onChange={setQuery} placeholder="Filter models" />
          <ListSection>
            {models.map((model) => {
              const current = selected?.provider === model.provider && selected?.model === model.model;
              return (
                <ListRow
                  key={`${model.provider}:${model.model}`}
                  title={model.name || model.model}
                  subtitle={[model.providerName || model.provider, model.vision && "Vision", model.reasoning && "Reasoning"].filter(Boolean).join(" · ")}
                  trailing={current && <Check size={20} className="text-brand" />}
                  onClick={() => choose(model)}
                />
              );
            })}
            {models.length === 0 && <ListRow title="No models match." />}
          </ListSection>
        </>
      ) : (
        <>
          <ListSection>
            <ListRow title="Model" detail={selected?.name || translation?.model || "None"} chevron onClick={() => setChoosing(true)} />
          </ListSection>
          <ListSection header="Scope">
            {SCOPES.map(({ scope, label }) => (
              <ListRow
                key={scope}
                title={label}
                subtitle={scope === "chapter" ? chapterOf(ws, ws.pageId)?.name : scope === "project" ? `${ws.project?.pages.length ?? 0} pages` : page?.name}
                trailing={ws.runScope === scope && <Check size={20} className="text-brand" />}
                onClick={() => ws.setRunScope(scope)}
              />
            ))}
          </ListSection>
          <ListSection header="Stages" footer={page ? `This page: ${STAGES.filter((s) => stageDone(page, s)).length} of ${STAGES.length} stages done.` : undefined}>
            {STAGES.map((stage) => (
              <ListRow
                key={stage}
                title={STAGE_LABEL[stage]}
                subtitle={STAGE_HINT[stage]}
                trailing={ws.runStages.includes(stage) && <Check size={20} className="text-brand" />}
                onClick={() => toggleStage(stage)}
              />
            ))}
          </ListSection>
          <div className="mt-6">
            {running ? (
              <GlassButton label="Stop" shape="capsule" className="h-[50px] w-full text-danger" onClick={() => void ws.stop()}>
                <Square size={14} fill="currentColor" /> Stop
              </GlassButton>
            ) : (
              <GlassButton label="Run" shape="capsule" tone="prominent" className="h-[50px] w-full" disabled={!ws.pageId} onClick={run}>
                <Play size={16} fill="currentColor" /> Run
              </GlassButton>
            )}
          </div>
        </>
      )}
    </Sheet>
  );
}
