// Right panel: Properties (Type + Layers) and Activity, plus the Agent chat.

import {
  Activity as ActivityIcon,
  AlignCenter,
  AlignLeft,
  AlignRight,
  ArrowDown,
  ArrowUp,
  Brush,
  ChevronDown,
  Eye,
  EyeOff,
  Image as ImageIcon,
  Layers3,
  Lock,
  Sparkles,
  SlidersHorizontal,
  Trash2,
  Type,
} from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";

import { cn } from "cn";

import { client } from "../api/client";
import type { RasterLayer, TextBlock, Typography } from "../gen/urakatsu/v1/urakatsu_pb";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { currentBlock, currentPage, useWorkspace } from "../state/workspace";
import { ActivityPanel } from "./ActivityTrace";
import { AgentPanel } from "./AgentPanel";
import { ColorWell, NumberField, Popover, RadioMenu, Select, Slider, Tooltip, type Option } from "./primitives";
import { Badge } from "./ui";
import { ScrollArea } from "@/components/ui/scroll-area";
import { TitleBarInset } from "./WindowControls";

const PANEL_TAB = "inline-flex h-[30px] flex-1 cursor-pointer items-center justify-center gap-1 rounded-control text-fg-2 transition-colors duration-[20ms] ease-in focus-visible:outline-none focus-visible:shadow-focus";
const TAB_LABEL = "in-data-[shell=windows]:@max-[26rem]:sr-only";
const panelTab = (active: boolean) => cn(PANEL_TAB, active ? "bg-surface font-medium text-fg shadow-raised" : "hover:bg-hover hover:text-fg");

export function RightPanel() {
  const ws = useWorkspace();
  const [tab, setTab] = useState<"properties" | "activity" | "agent">("properties");
  useEffect(() => {
    if (ws.job?.status === "running") setTab((current) => (current === "agent" ? current : "activity"));
  }, [ws.job?.status]);
  return (
    <aside className="flex min-h-0 flex-1 flex-col bg-surface-sunk shadow-[inset_1px_0_0_var(--border-soft)]">
      <div className="@container flex h-12 items-center gap-0.5 px-2 shadow-[inset_0_-1px_0_var(--border-soft)]" role="tablist" data-titlebar>
        <button role="tab" aria-selected={tab === "properties"} className={panelTab(tab === "properties")} title="Properties" onClick={() => setTab("properties")}>
          <SlidersHorizontal size={14} /> <span className={TAB_LABEL}>Properties</span>
        </button>
        <button role="tab" aria-selected={tab === "activity"} className={panelTab(tab === "activity")} title="Activity" onClick={() => setTab("activity")}>
          <ActivityIcon size={14} /> <span className={TAB_LABEL}>Activity</span>{" "}
          {ws.job?.status === "running" && <span className="size-1.5 animate-[soft-pulse_1s_ease-in-out_infinite_alternate] rounded-full bg-brand" aria-label="Running" />}
        </button>
        <button role="tab" aria-selected={tab === "agent"} className={panelTab(tab === "agent")} title="Agent" onClick={() => setTab("agent")}>
          <Sparkles size={14} /> <span className={TAB_LABEL}>Agent</span>
        </button>
        <span aria-hidden className="flex-1" />
        <TitleBarInset side="right" />
      </div>
      {tab === "properties" ? (
        <div className="flex min-h-0 flex-1 flex-col">
          <header className="flex h-[34px] flex-none items-center gap-1 px-3 shadow-[inset_0_-1px_0_var(--border-soft)]">
            <Type size={14} className="text-brand" />
            <h2 className="text-xs font-semibold">Type</h2>
          </header>
          <TypeInspector />
          <LayersInspector />
        </div>
      ) : tab === "agent" ? (
        <AgentPanel />
      ) : (
        <ScrollArea className="flex-1" contentClassName="p-4">
          <ActivityPanel />
        </ScrollArea>
      )}
    </aside>
  );
}

// ---- type ------------------------------------------------------------------------------------

const DEFAULT_TYPOGRAPHY = {
  $typeName: "urakatsu.v1.Typography",
  preferredFont: "",
  fontStyle: "",
  autoFit: true,
  alignment: "",
  writingMode: "",
} as Typography;

const WEIGHTS: Option[] = [100, 200, 300, 400, 500, 600, 700, 800, 900].map((w) => ({ value: String(w), label: String(w) }));
const STYLES: Option[] = [
  { value: "normal", label: "Regular" },
  { value: "italic", label: "Italic" },
  { value: "oblique", label: "Oblique" },
];
const DIRECTIONS: Option[] = [
  { value: "auto", label: "Auto" },
  { value: "Horizontal", label: "Horizontal" },
  { value: "Vertical", label: "Vertical" },
];
const SIZE_PRESETS = [8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 48, 64, 72, 96];

const rgb = (value?: string) => (value && value.length >= 7 ? value.slice(0, 7) : null);
const alpha = (value?: string) => (value && value.length === 9 ? parseInt(value.slice(7), 16) : 255);

function TypeInspector() {
  const ws = useWorkspace();
  const block = currentBlock(ws);
  const disabled = !block;
  const typography = block?.typography ?? DEFAULT_TYPOGRAPHY;
  // Only the properties named here merge into the layer's style, so a guess the pipeline made
  // (a vertical writing mode for vertical Japanese) never becomes the user's choice by accident.
  const apply = (patch: Partial<Typography>) => {
    if (!block) return;
    const typographyFields = Object.keys(patch).map((key) => key.replace(/[A-Z]/g, (letter) => `_${letter.toLowerCase()}`));
    void ws.updateBlock(block.layerId, { typography: { ...typography, ...patch } as Typography, typographyFields });
  };
  const defaultFont = useDefaultFont();
  // A pipeline-made style's writing mode is only a hint the renderer ignores for horizontal
  // scripts, so it shows as Auto until the user picks one.
  const writingMode = typography.authored ? typography.writingMode : "";
  const vertical = writingMode === "Vertical";
  const alignment = typography.alignment || (vertical ? "Start" : "Center");
  const strokeOn = (typography.strokeWidth ?? 0) > 0 && alpha(typography.strokeColor) > 0;

  return (
    <section className="flex flex-none flex-col gap-2 px-3 pt-2 pb-3 shadow-[inset_0_-1px_0_var(--border-soft)] aria-disabled:opacity-[0.55]" aria-disabled={disabled}>
      <div className="grid min-w-0 grid-cols-[minmax(0,1fr)_40px] gap-1">
        <InspectorField label="Font">
          <FontPicker value={typography.preferredFont} disabled={disabled} defaultFont={defaultFont} onChange={(preferredFont) => apply({ preferredFont })} />
        </InspectorField>
        <InspectorField label="Color">
          <ColorWell size="sm" label="Text color" disabled={disabled} value={rgb(typography.color) ?? "#000000"} onChange={(color) => color && apply({ color: `${color}ff` })} />
        </InspectorField>
      </div>
      <div className="grid min-w-0 grid-cols-[minmax(0,1fr)_72px_84px] gap-1">
        <InspectorField label="Size">
          <NumberField
            label="Font size"
            stepper={false}
            disabled={disabled}
            value={typography.autoFit ? null : (typography.size ?? null)}
            placeholder="Auto"
            min={0.5}
            max={300}
            step={0.5}
            onCommit={(size) => (size === null ? apply({ autoFit: true }) : apply({ size, autoFit: false }))}
            suffix={
              <RadioMenu
                value={typography.autoFit ? "auto" : String(typography.size ?? "")}
                options={[{ value: "auto", label: "Auto" }, ...SIZE_PRESETS.map((size) => ({ value: String(size), label: String(size) }))]}
                onChange={(choice) => (choice === "auto" ? apply({ autoFit: true }) : apply({ size: Number(choice), autoFit: false }))}
                trigger={
                  <button
                    type="button"
                    className="grid w-6 flex-none cursor-default place-items-center text-icon-2 shadow-[inset_1px_0_0_var(--border)] transition-colors duration-[20ms] ease-in hover:bg-hover hover:text-icon disabled:pointer-events-none disabled:opacity-[0.35]"
                    aria-label="Choose font size"
                    disabled={disabled}
                  >
                    <ChevronDown size={12} />
                  </button>
                }
              />
            }
          />
        </InspectorField>
        <InspectorField label="Weight">
          <Select size="sm" label="Font weight" disabled={disabled} value={String(typography.fontWeight ?? 400)} options={WEIGHTS} onChange={(w) => apply({ fontWeight: Number(w) })} />
        </InspectorField>
        <InspectorField label="Style">
          <Select size="sm" label="Font style" disabled={disabled} value={typography.fontStyle || "normal"} options={STYLES} onChange={(fontStyle) => apply({ fontStyle })} />
        </InspectorField>
      </div>
      <div className="grid min-w-0 grid-cols-2 gap-1">
        <InspectorField label="Alignment">
          <div className="grid h-[26px] grid-cols-[repeat(3,1fr)] rounded-control bg-surface p-0.5 shadow-[inset_0_0_0_1px_var(--border)]" role="group" aria-label="Alignment">
            {(
              [
                ["Start", AlignLeft, vertical ? "Align top" : "Align left"],
                ["Center", AlignCenter, vertical ? "Align middle" : "Align center"],
                ["End", AlignRight, vertical ? "Align bottom" : "Align right"],
              ] as const
            ).map(([value, Icon, label]) => (
              <button
                key={value}
                type="button"
                aria-label={label}
                aria-pressed={alignment === value}
                disabled={disabled}
                className={cn(
                  "grid place-items-center rounded-[4px] transition-colors duration-[20ms] ease-in focus-visible:outline-none focus-visible:shadow-focus",
                  alignment === value ? "bg-brand text-on-brand" : "text-icon-2 hover:text-icon",
                )}
                onClick={() => apply({ alignment: value })}
              >
                <Icon size={14} />
              </button>
            ))}
          </div>
        </InspectorField>
        <InspectorField label="Direction">
          <Select
            size="sm"
            label="Text direction"
            disabled={disabled}
            value={writingMode || "auto"}
            options={DIRECTIONS}
            onChange={(mode) => apply({ writingMode: mode === "auto" ? "" : mode })}
          />
        </InspectorField>
      </div>
      <div className="grid min-w-0 grid-cols-[44px_minmax(0,1fr)] gap-1">
        <InspectorField label="Border">
          <ColorWell
            size="sm"
            label="Border color"
            allowNone
            disabled={disabled}
            value={strokeOn ? rgb(typography.strokeColor) : null}
            onChange={(color) =>
              apply(
                color
                  ? { strokeColor: `${color}ff`, strokeWidth: (typography.strokeWidth ?? 0) > 0 ? typography.strokeWidth : 1.5 }
                  : { strokeColor: `${rgb(typography.strokeColor) ?? "#ffffff"}00` },
              )
            }
          />
        </InspectorField>
        <InspectorField label="Width">
          <NumberField
            label="Border width"
            disabled={disabled}
            value={(typography.strokeWidth ?? 0) > 0 ? (typography.strokeWidth ?? 1.5) : 1.5}
            min={0.5}
            max={32}
            step={0.5}
            onCommit={(width) => width !== null && apply({ strokeWidth: width })}
          />
        </InspectorField>
      </div>
    </section>
  );
}

const FIELD_LABEL = "text-[10px] leading-3.5 font-medium tracking-[0.06em] text-fg-3 uppercase";

function InspectorField({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex min-w-0 flex-col gap-0.5">
      <span className={FIELD_LABEL}>{label}</span>
      {children}
    </div>
  );
}

let fontCache: Promise<string[]> | null = null;
function useFontFamilies(): string[] {
  const [families, setFamilies] = useState<string[]>([]);
  useEffect(() => {
    fontCache ??= client.listFonts({}).then((list) => list.families.map((f) => f.name).sort((a, b) => a.localeCompare(b)));
    void fontCache.then(setFamilies).catch(() => setFamilies([]));
  }, []);
  return families;
}

/** The font a block without its own font typesets in: the first of Settings > Typesetting's default fonts that is installed. */
function useDefaultFont(): string | undefined {
  const ws = useWorkspace();
  const installed = useFontFamilies();
  const preferred = ws.config?.fontFamilies ?? [];
  const names = new Set(installed.map((family) => family.toLowerCase()));
  return preferred.find((family) => names.has(family.toLowerCase())) ?? preferred[0];
}

const popupItem = (selected: boolean) =>
  cn(
    "flex w-full min-h-[30px] cursor-default items-center gap-2 rounded-control px-2 py-0.5 text-left text-sm text-fg outline-none transition-colors duration-[20ms] ease-in select-none",
    selected ? "bg-selected" : "hover:bg-hover focus-visible:bg-hover",
  );

export function FontPicker({
  value,
  onChange,
  disabled,
  placeholder = "Typesetting default",
  defaultFont,
}: {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  placeholder?: string;
  /** The font an empty value falls back to; shown by name instead of the placeholder. */
  defaultFont?: string;
}) {
  const families = useFontFamilies();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const shown = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return (needle ? families.filter((f) => f.toLowerCase().includes(needle)) : families).slice(0, 300);
  }, [families, query]);
  return (
    <Popover
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (!next) setQuery("");
      }}
      align="start"
      className="w-[280px] p-1"
      trigger={
        <button
          type="button"
          className="inline-flex h-[26px] w-full min-w-0 cursor-default items-center justify-between gap-1 rounded-control bg-surface py-0 pr-0.5 pl-1 text-left text-xs text-fg shadow-[inset_0_0_0_1px_var(--border)] [transition:background-color_20ms_ease-in,box-shadow_100ms_ease-out] hover:bg-[color-mix(in_oklab,var(--surface),var(--n-950)_3%)] focus-visible:shadow-focus focus-visible:outline-none data-[popup-open]:shadow-focus disabled:pointer-events-none disabled:opacity-50"
          disabled={disabled}
          aria-label="Font"
        >
          <span className="min-w-0 flex-1 truncate text-sm" style={value || defaultFont ? { fontFamily: `"${value || defaultFont}"` } : undefined}>
            {value ||
              (defaultFont ? (
                <>
                  {defaultFont}
                  <span className="font-sans text-xs text-fg-3"> default</span>
                </>
              ) : (
                <span className="text-fg-3">{placeholder}</span>
              ))}
          </span>
          <ChevronDown size={14} className="flex-none text-icon-2" />
        </button>
      }
    >
      <Input className="mb-1 h-7" placeholder="Search fonts" aria-label="Search fonts" value={query} onChange={(e) => setQuery(e.target.value)} autoFocus />
      <ScrollArea viewportClassName="max-h-[300px]" role="listbox" aria-label="Fonts">
        <button
          role="option"
          aria-selected={!value}
          className={popupItem(false)}
          onClick={() => {
            onChange("");
            setOpen(false);
          }}
        >
          <span className="text-fg-2">{defaultFont ? `Default (${defaultFont})` : placeholder}</span>
        </button>
        {shown.map((family) => (
          <button
            key={family}
            role="option"
            aria-selected={family === value}
            className={popupItem(family === value)}
            style={{ fontFamily: `"${family}"` }}
            onClick={() => {
              onChange(family);
              setOpen(false);
            }}
          >
            {family}
          </button>
        ))}
        {families.length > 0 && shown.length === 0 && <p className="p-2 text-xs tracking-[0.005em] text-fg-2">No fonts match.</p>}
      </ScrollArea>
    </Popover>
  );
}

// ---- layers ----------------------------------------------------------------------------------

type Row =
  | { kind: "text"; id: string; block: TextBlock; index: number }
  | { kind: "raster"; id: string; layer: RasterLayer }
  | { kind: "artwork"; id: "artwork" };

function roleName(block: TextBlock): string {
  const role = block.role.split(/[./]/).at(-1) ?? "";
  if (role === "dialogue") return "Dialogue";
  if (role === "free-text" || role === "free_text") return "Free text";
  if (role) return role[0].toUpperCase() + role.slice(1).replace(/-/g, " ");
  return block.balloonId ? "Dialogue" : "Text";
}

const LAYER_MAIN = "flex min-w-0 flex-1 items-center gap-2 rounded-control px-1 py-1.5 text-left transition-colors duration-[20ms] ease-in hover:bg-hover-faint focus-visible:outline-none focus-visible:shadow-focus";
const LAYER_EYE = "grid size-7 flex-none cursor-pointer place-items-center rounded-control text-icon-2 transition-colors duration-[20ms] ease-in hover:bg-hover hover:text-icon focus-visible:outline-none focus-visible:shadow-focus";
const LAYER_MOVE_BUTTON = "grid size-5 cursor-pointer place-items-center rounded-pill text-icon-2 hover:bg-hover hover:text-icon disabled:pointer-events-none disabled:opacity-30";

function LayersInspector() {
  const ws = useWorkspace();
  const page = currentPage(ws);
  const rows = useMemo<Row[]>(() => {
    if (!page) return [];
    return [
      ...page.blocks.map((block, index) => ({ kind: "text" as const, id: block.layerId, block, index })),
      ...[...page.rasterLayers].reverse().map((layer) => ({ kind: "raster" as const, id: layer.id, layer })),
      { kind: "artwork" as const, id: "artwork" as const },
    ];
  }, [page]);

  if (!page) return <p className="px-3 py-8 text-center text-fg-2">Choose a page to see its layers.</p>;
  const textCount = page.blocks.length;
  const rasterCount = page.rasterLayers.length;

  return (
    <section className="flex min-h-0 flex-1 flex-col">
      <header className="flex h-[34px] flex-none items-center gap-1 px-3 shadow-[inset_0_-1px_0_var(--border-soft)]">
        <Layers3 size={14} className="text-brand" />
        <h2 className="text-xs font-semibold">Layers</h2>
        <span className="text-xs text-fg-3 tabular-nums">{textCount + rasterCount + 1}</span>
      </header>
      <ScrollArea className="flex-1" contentClassName="px-1 pt-0.5 pb-3">
        {rows.map((row) => {
          if (row.kind === "artwork") {
            return (
              <div key="artwork" className="relative flex items-center gap-0.5 rounded-card pr-0.5 transition-colors duration-[20ms] ease-in">
                <div className={cn(LAYER_MAIN, "cursor-default")}>
                  <ImageIcon size={14} className="flex-none text-icon-2" />
                  <span className="flex min-w-0 flex-1 flex-col">
                    <span className="truncate">Original artwork</span>
                    <span className="text-xs text-fg-3">Image</span>
                  </span>
                </div>
                <span className="w-9 flex-none text-right text-xs text-fg-3 tabular-nums">100%</span>
                <Tooltip content="Locked" side="left">
                  <span className="grid size-7 flex-none cursor-default place-items-center rounded-control text-icon-2" role="img" aria-label="Original artwork is locked">
                    <Lock size={14} />
                  </span>
                </Tooltip>
              </div>
            );
          }
          const isText = row.kind === "text";
          const visible = isText ? row.block.visible : row.layer.visible;
          const opacity = (isText ? row.block.opacity : row.layer.opacity) || 1;
          const name = isText ? (row.block.translation || row.block.sourceText || "").trim() || `Text ${row.index + 1}` : row.layer.name;
          const detail = isText ? roleName(row.block) : row.layer.kind === "cleanup" ? "Cleanup" : row.layer.kind === "edit" ? "Image edit" : "Paint";
          const selected = isText ? ws.blockId === row.id : ws.rasterId === row.id;
          // Text layers display in stored order; raster layers are shown top-most first.
          const siblings = isText ? textCount : rasterCount;
          const position = isText ? row.index : rasterCount - 1 - page.rasterLayers.findIndex((l) => l.id === row.id);
          const up = isText ? -1 : 1;
          return (
            <div key={row.id} className={cn("rounded-card transition-colors duration-[20ms] ease-in", selected && "bg-selected")}>
              <div className="group/line relative flex items-center gap-0.5 pr-0.5">
                <button
                  type="button"
                  className={cn(LAYER_MAIN, "cursor-pointer")}
                  aria-expanded={selected}
                  aria-label={`Edit ${name}`}
                  onClick={() => (isText ? ws.selectBlock(selected ? undefined : row.id) : ws.selectRaster(selected ? undefined : row.id))}
                >
                  {isText ? <Type size={14} className="flex-none text-icon-2" /> : <Brush size={14} className="flex-none text-icon-2" />}
                  <span className="flex min-w-0 flex-1 flex-col">
                    <span className="truncate">{name}</span>
                    <span className="text-xs text-fg-3">{detail}</span>
                  </span>
                </button>
                <div
                  className={cn(
                    "absolute top-1/2 flex -translate-y-1/2 rounded-control bg-surface-over p-0.5 shadow-raised opacity-0 pointer-events-none transition-opacity duration-100 ease-out group-hover/line:pointer-events-auto group-hover/line:opacity-100 group-focus-within/line:pointer-events-auto group-focus-within/line:opacity-100",
                    selected ? "right-[34px]" : "right-[72px]",
                  )}
                >
                  <button type="button" className={LAYER_MOVE_BUTTON} aria-label={`Move ${name} up`} disabled={position <= 0} onClick={() => void ws.moveLayer(row.id, up)}>
                    <ArrowUp size={12} />
                  </button>
                  <button type="button" className={LAYER_MOVE_BUTTON} aria-label={`Move ${name} down`} disabled={position >= siblings - 1} onClick={() => void ws.moveLayer(row.id, -up)}>
                    <ArrowDown size={12} />
                  </button>
                </div>
                {!selected && <span className="w-9 flex-none text-right text-xs text-fg-3 tabular-nums">{Math.round(opacity * 100)}%</span>}
                <button type="button" className={LAYER_EYE} aria-label={visible ? `Hide ${name}` : `Show ${name}`} onClick={() => void ws.setLayerVisibility([row.id], !visible)}>
                  {visible ? <Eye size={14} /> : <EyeOff size={14} />}
                </button>
              </div>
              {selected && <LayerEditor row={row} name={name} opacity={opacity} />}
            </div>
          );
        })}
      </ScrollArea>
    </section>
  );
}

function LayerEditor({ row, name, opacity }: { row: Exclude<Row, { kind: "artwork" }>; name: string; opacity: number }) {
  const ws = useWorkspace();
  const [value, setValue] = useState(opacity * 100);
  useEffect(() => setValue(opacity * 100), [opacity]);
  const block = row.kind === "text" ? row.block : undefined;
  return (
    <div className="flex animate-[fade-in_150ms] flex-col gap-2 pt-0.5 pb-2 px-2">
      <div className="flex items-center gap-2">
        <span className={FIELD_LABEL}>Opacity</span>
        <Slider label={`${name} opacity`} min={0} max={100} value={value} onChange={setValue} onCommit={(next) => void ws.setLayerVisibility([row.id], undefined, next / 100)} className="min-w-0 flex-1" />
        <span className="w-9 flex-none text-right text-xs text-fg-3 tabular-nums">{Math.round(value)}%</span>
        <Tooltip content="Delete layer" side="left">
          <Button type="button" variant="ghost" size="icon-xs" aria-label={`Delete ${name}`} onClick={() => void ws.deleteLayer(row.id)}>
            <Trash2 size={14} />
          </Button>
        </Tooltip>
      </div>
      {block && (
        <>
          <div className="flex items-center gap-2">
            <span className={FIELD_LABEL}>Placement</span>
            <Badge>{block.regionId ? "Detected region" : "Custom frame"}</Badge>
            {(block.sourceOrigin === "user" || block.translationOrigin === "user") && <Badge tone="warning">Edited</Badge>}
          </div>
          <CommitText label="Source" value={block.sourceText ?? ""} source onCommit={(text) => void ws.updateBlock(block.layerId, { sourceText: text })} />
          <CommitText label="Translation" value={block.translation ?? ""} onCommit={(text) => void ws.updateBlock(block.layerId, { translation: text })} />
        </>
      )}
    </div>
  );
}

function CommitText({ label, value, onCommit, source }: { label: string; value: string; onCommit: (value: string) => void; source?: boolean }) {
  const [text, setText] = useState(value);
  useEffect(() => setText(value), [value]);
  return (
    <label className="flex min-w-0 flex-col gap-0.5">
      <span className={FIELD_LABEL}>{label}</span>
      <Textarea
        className={cn(
          "max-h-[120px] min-h-11 p-1 text-xs leading-4",
          source ? "text-sm leading-5" : "shadow-[inset_0_0_0_1px_var(--brand-line)] focus:shadow-focus",
        )}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onBlur={() => text !== value && onCommit(text)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) (e.target as HTMLTextAreaElement).blur();
        }}
      />
    </label>
  );
}
