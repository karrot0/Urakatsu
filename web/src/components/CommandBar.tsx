import {
  BookOpen,
  Boxes,
  Bot,
  Check,
  ChevronLeft,
  ChevronRight,
  Eye,
  EyeOff,
  FileImage,
  Layers,
  Monitor,
  Moon,
  Play,
  Redo2,
  Settings,
  Square,
  Sun,
  Undo2,
} from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "cn";
import type { Config, TranslationModel } from "../gen/urakatsu/v1/urakatsu_pb";
import { STAGES, chapterOf, currentPage, selectedPages, useWorkspace, type RunScope, type StageName, type Theme, type ViewMode } from "../state/workspace";
import { Popover, Tooltip } from "./primitives";
import { Segmented } from "./ui";
import { ScrollArea } from "@/components/ui/scroll-area";

export const STAGE_LABEL: Record<StageName, string> = {
  detection: "Detection",
  ocr: "OCR",
  translation: "Translation",
  inpainting: "Inpainting",
};

const STAGE_DETAIL: Record<StageName, string> = {
  detection: "Find text, balloons and panels",
  ocr: "Read the source text",
  translation: "Translate every text block",
  inpainting: "Remove the original lettering",
};

const SCOPE_LABEL: Record<RunScope, string> = {
  page: "Current page",
  chapter: "Current chapter",
  project: "Entire project",
};

const popupItem =
  "flex min-h-[30px] w-full cursor-default items-center gap-2 rounded-control bg-transparent px-2 py-0.5 text-left text-fg transition-colors duration-[20ms] ease-in outline-none select-none hover:bg-hover focus-visible:bg-hover data-[highlighted]:bg-hover [&_svg]:shrink-0";

export function CommandBar() {
  const ws = useWorkspace();
  const page = currentPage(ws);
  const chapter = chapterOf(ws, ws.pageId);
  const project = ws.project!;
  const hasLayers = !!page && page.rasterLayers.length > 0;

  return (
    <header className="flex h-12 items-center gap-2 py-0 pr-2 pl-3 shadow-[inset_0_-1px_0_var(--border-soft)]">
      <nav className="flex min-w-0 flex-1 items-center gap-1 text-fg-2 [&_svg]:shrink-0 [&_svg]:text-icon-2" aria-label="Location">
        {chapter && page ? (
          <>
            <BookOpen size={14} />
            <span className="truncate">{chapter.name}</span>
            <ChevronRight size={14} />
            <FileImage size={14} />
            <span className="truncate text-fg">{page.name}</span>
          </>
        ) : (
          <span className="truncate text-fg">{project.name}</span>
        )}
      </nav>

      {page && (
        <>
          <Segmented<ViewMode>
            label="View"
            value={ws.view}
            onChange={ws.setView}
            options={[
              { value: "source", label: "Source", title: "Original page (1)" },
              { value: "cleanup", label: "Layers", title: hasLayers ? "Page with cleanup and paint layers (2)" : "Page with your layers (2)" },
              { value: "render", label: "Rendered", title: "Typeset result (3)" },
            ]}
          />
          <div className="flex items-center gap-0.5">
            <Tooltip content={ws.showBlocks ? "Hide text frames" : "Show text frames"} side="bottom">
              <Button variant="ghost" size="icon-sm" aria-pressed={ws.showBlocks} aria-label="Text frames" onClick={() => ws.toggleOverlay("blocks")}>
                {ws.showBlocks ? <Eye size={16} /> : <EyeOff size={16} />}
              </Button>
            </Tooltip>
            <Tooltip content={ws.showRegions ? "Hide balloons and panels" : "Show balloons and panels"} side="bottom">
              <Button variant="ghost" size="icon-sm" aria-pressed={ws.showRegions} aria-label="Balloons and panels" onClick={() => ws.toggleOverlay("regions")}>
                <Layers size={16} />
              </Button>
            </Tooltip>
            <span className="mx-0.5 h-[18px] w-px bg-line-soft" />
            <Tooltip content="Undo (Ctrl+Z)" side="bottom">
              <Button variant="ghost" size="icon-sm" onClick={() => void ws.undo()} disabled={!project.canUndo} aria-label="Undo">
                <Undo2 size={16} />
              </Button>
            </Tooltip>
            <Tooltip content="Redo (Ctrl+Shift+Z)" side="bottom">
              <Button variant="ghost" size="icon-sm" onClick={() => void ws.redo()} disabled={!project.canRedo} aria-label="Redo">
                <Redo2 size={16} />
              </Button>
            </Tooltip>
          </div>
        </>
      )}

      <InferenceControl />
    </header>
  );
}

type View = "root" | "model" | "scope" | "stages";

function InferenceControl() {
  const ws = useWorkspace();
  const [open, setOpen] = useState(false);
  const [view, setView] = useState<View>("root");
  const running = ws.job?.status === "running";
  const config = ws.config;
  const translation = config?.translation;
  const selected = ws.translationModels.find((m) => m.provider === translation?.provider && m.model === translation?.model);
  const modelName = selected?.name || translation?.model || "No model";
  const progress = ws.job && ws.job.total ? Math.round((ws.job.completed / ws.job.total) * 100) : 0;

  // Pages picked in the sidebar (Shift/Ctrl) take over the scope until the selection is cleared.
  const picked = selectedPages(ws).length;
  const scope = picked > 1 ? "selection" : ws.runScope;
  const run = () => {
    const stages = STAGES.filter((s) => ws.runStages.includes(s));
    if (stages.length === STAGES.length) void ws.run({ kind: "full", scope });
    else if (stages.length === 1) void ws.run({ kind: "only", stage: stages[0], scope });
    else void ws.run({ kind: "stages", stages, scope });
  };

  const toggleStage = (stage: StageName) => {
    const has = ws.runStages.includes(stage);
    if (has && ws.runStages.length === 1) return;
    ws.setRunStages(has ? ws.runStages.filter((s) => s !== stage) : STAGES.filter((s) => s === stage || ws.runStages.includes(s)));
  };

  const chooseModel = (model: TranslationModel) => {
    if (!config || !translation) return;
    void ws.saveConfig({
      ...config,
      translation: {
        ...translation,
        provider: model.provider,
        model: model.model,
        quantization: model.quantizations[0]?.id ?? "",
        vision: model.vision,
        reasoning: model.reasoning,
      },
    } as Config);
    setView("root");
  };

  const stagesLabel = ws.runStages.length === STAGES.length ? "All stages" : ws.runStages.length === 1 ? STAGE_LABEL[ws.runStages[0]] : `${ws.runStages.length} stages`;

  return (
    <div className="ml-1 flex items-center gap-1">
      {running && (
        <div
          className="relative h-1 w-24 overflow-hidden rounded-full bg-surface-fill"
          role="progressbar"
          aria-label="Run progress"
          aria-valuenow={progress}
        >
          <span
            className={cn("absolute inset-y-0 left-0 rounded-[inherit] bg-brand transition-[width] duration-200 ease-settle", !ws.job?.total && "w-[30%]! animate-[slide_1.1s_ease-in-out_infinite]")}
            style={{ width: `${progress}%` }}
          />
        </div>
      )}
      <Popover
        open={open}
        onOpenChange={(next) => {
          setOpen(next);
          if (!next) setView("root");
        }}
        align="end"
        className="w-80 p-0.5"
        trigger={
          <Button variant="outline" size="sm" aria-label="Run settings" className="group/model max-w-60 font-normal">
            <Bot size={14} />
            <span className="truncate">{modelName}</span>
            <ChevronRight size={14} className="rotate-90 transition-transform duration-150 ease-settle group-data-[popup-open]/model:-rotate-90" />
          </Button>
        }
      >
        {view === "root" && (
          <div className="flex flex-col">
            <SelectorRow label="Model" value={modelName} onClick={() => setView("model")} />
            <SelectorRow label="Scope" value={picked > 1 ? `${picked} selected pages` : SCOPE_LABEL[ws.runScope]} onClick={() => setView("scope")} />
            <SelectorRow label="Stages" value={stagesLabel} onClick={() => setView("stages")} />
            <hr />
            <button
              className={popupItem}
              onClick={() => {
                setOpen(false);
                ws.setSettingsOpen(true);
              }}
            >
              <Settings size={14} className="text-icon-2" /> Settings
            </button>
            <button
              className={popupItem}
              onClick={() => {
                setOpen(false);
                ws.setModelsOpen(true);
              }}
            >
              <Boxes size={14} className="text-icon-2" /> Find more models
            </button>
          </div>
        )}
        {view === "model" && (
          <SelectorPanel title="Model" onBack={() => setView("root")}>
            <ModelList current={selected} onChoose={chooseModel} />
          </SelectorPanel>
        )}
        {view === "scope" && (
          <SelectorPanel title="Scope" onBack={() => setView("root")}>
            {picked > 1 && (
              <p className="px-2 pt-1 pb-1.5 text-xs leading-4 text-fg-2">
                {picked} pages are selected in the sidebar, so Run uses them. Clear the selection to use a scope below.
              </p>
            )}
            {(Object.keys(SCOPE_LABEL) as RunScope[]).map((scope) => (
              <SelectorOption
                key={scope}
                label={SCOPE_LABEL[scope]}
                detail={
                  scope === "page" ? "Only the page on the canvas" : scope === "chapter" ? `Every page in ${chapterOf(ws, ws.pageId)?.name ?? "this chapter"}` : `All ${ws.project?.pages.length ?? 0} pages`
                }
                selected={ws.runScope === scope}
                onClick={() => {
                  ws.setRunScope(scope);
                  setView("root");
                }}
              />
            ))}
          </SelectorPanel>
        )}
        {view === "stages" && (
          <SelectorPanel title="Pipeline stages" onBack={() => setView("root")}>
            {STAGES.map((stage) => (
              <SelectorOption key={stage} label={STAGE_LABEL[stage]} detail={STAGE_DETAIL[stage]} selected={ws.runStages.includes(stage)} onClick={() => toggleStage(stage)} />
            ))}
          </SelectorPanel>
        )}
      </Popover>
      {running ? (
        <Button variant="outline" size="sm" className="min-w-18" onClick={() => void ws.stop()}>
          <Square size={12} /> Stop
        </Button>
      ) : (
        <Button size="sm" className="min-w-18" disabled={!ws.pageId} onClick={run} title={picked > 1 ? `Run on the ${picked} selected pages` : undefined}>
          <Play size={14} /> {picked > 1 ? `Run ${picked} pages` : "Run"}
        </Button>
      )}
    </div>
  );
}

function ModelList({ current, onChoose }: { current?: TranslationModel; onChoose: (model: TranslationModel) => void }) {
  const ws = useWorkspace();
  const [query, setQuery] = useState("");
  const groups = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const map = new Map<string, TranslationModel[]>();
    for (const model of ws.translationModels) {
      if (needle && !`${model.name} ${model.model} ${model.providerName}`.toLowerCase().includes(needle)) continue;
      map.set(model.providerName || model.provider, [...(map.get(model.providerName || model.provider) ?? []), model]);
    }
    return [...map.entries()];
  }, [ws.translationModels, query]);
  return (
    <>
      <Input
        className="mx-0.5 mt-0.5 mb-1 h-7 w-[calc(100%-4px)] focus:shadow-focus"
        placeholder="Filter models"
        aria-label="Filter models"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        autoFocus
      />
      <ScrollArea viewportClassName="max-h-80">
        {groups.length === 0 && <p className="p-2 text-xs tracking-[0.005em] text-fg-2">No models match.</p>}
        {groups.map(([provider, models]) => (
          <div key={provider}>
            <div className="px-2 pt-2 pb-1 text-xs text-fg-3">{provider}</div>
            {models.map((model) => (
              <SelectorOption
                key={`${model.provider}:${model.model}`}
                label={model.name || model.model}
                detail={[model.vision && "Vision", model.reasoning && "Reasoning"].filter(Boolean).join(" · ") || undefined}
                selected={current?.provider === model.provider && current?.model === model.model}
                onClick={() => onChoose(model)}
              />
            ))}
          </div>
        ))}
      </ScrollArea>
    </>
  );
}

function SelectorRow({ label, value, onClick }: { label: string; value: string; onClick: () => void }) {
  return (
    <button className={popupItem} onClick={onClick}>
      <span className="text-fg-2">{label}</span>
      <span className="min-w-0 flex-1 truncate text-right">{value}</span>
      <ChevronRight size={14} className="text-icon-2" />
    </button>
  );
}

function SelectorPanel({ title, onBack, children }: { title: string; onBack: () => void; children: ReactNode }) {
  return (
    <div className="flex flex-col">
      <button className={cn(popupItem, "font-medium")} onClick={onBack}>
        <ChevronLeft size={14} className="text-icon-2" /> {title}
      </button>
      <hr />
      {children}
    </div>
  );
}

function SelectorOption({ label, detail, selected, onClick }: { label: string; detail?: string; selected: boolean; onClick: () => void }) {
  return (
    <button className={popupItem} aria-pressed={selected} onClick={onClick}>
      <span className="flex min-w-0 flex-1 flex-col py-0.5">
        <span>{label}</span>
        {detail && <span className="text-[12px] whitespace-normal text-fg-3">{detail}</span>}
      </span>
      {selected && <Check size={14} className="cursor-pointer text-brand" />}
    </button>
  );
}

export function ThemeSwitch({ value, onChange }: { value: Theme; onChange: (theme: Theme) => void }) {
  return (
    <Segmented<Theme>
      label="Theme"
      value={value}
      onChange={onChange}
      options={[
        { value: "light", label: <Sun size={14} />, title: "Light" },
        { value: "dark", label: <Moon size={14} />, title: "Dark" },
        { value: "system", label: <Monitor size={14} />, title: "Match system" },
      ]}
    />
  );
}
