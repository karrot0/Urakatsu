// Floating tool belt: one tool active at a time; brush tools add color and size below,
// generative fill its selection shape.

import { memo } from "react";

import { Brush, Eraser, Hand, Lasso, MousePointer2, Pipette, Sparkles, SquareDashed, Type, WandSparkles, type LucideIcon } from "lucide-react";

import { cn } from "cn";

import { format } from "../shortcuts";
import { currentPage, MIN_BRUSH, useWorkspace, type FillShape, type Tool } from "../state/workspace";
import { ColorWell, NumberField, Popover, Slider, Switch, Tooltip } from "./primitives";

export const TOOL_INFO: Record<Tool, { label: string; icon: LucideIcon }> = {
  select: { label: "Select", icon: MousePointer2 },
  text: { label: "Text", icon: Type },
  draw: { label: "Brush", icon: Brush },
  eraser: { label: "Eraser", icon: Eraser },
  color_picker: { label: "Color picker", icon: Pipette },
  remove: { label: "Remove text", icon: Sparkles },
  fill: { label: "Generative fill", icon: WandSparkles },
  pan: { label: "Hand", icon: Hand },
};

const ORDER: Tool[] = ["select", "text", "draw", "eraser", "color_picker", "remove", "fill", "pan"];

const FILL_SHAPES: { shape: FillShape; label: string; icon: LucideIcon }[] = [
  { shape: "rect", label: "Rectangle selection", icon: SquareDashed },
  { shape: "lasso", label: "Lasso selection", icon: Lasso },
];

const TOOL_BUTTON =
  "my-px grid cursor-pointer place-items-center rounded-[10px] bg-transparent [transition:background-color_20ms_ease-in,color_100ms_ease-out] focus-visible:shadow-focus focus-visible:outline-none disabled:pointer-events-none disabled:opacity-30";

export const isBrushTool = (tool: Tool) => tool === "draw" || tool === "eraser" || tool === "remove";

export function maxBrush(width: number, height: number): number {
  return Math.max(128, Math.round(Math.max(width, height) / 8));
}

// Memoized: the canvas re-renders on every pointer move of a drag, the tool belt only when the workspace changes.
export const ToolBelt = memo(function ToolBelt() {
  const ws = useWorkspace();
  const page = currentPage(ws);
  return (
    <aside
      className="absolute top-3 left-3 z-20 flex w-11 flex-col items-center rounded-[14px] bg-surface-over p-0.5 shadow-floating max-md:hidden"
      aria-label="Tools"
      role="toolbar"
      aria-orientation="vertical"
    >
      {ORDER.map((tool, index) => {
        const { label, icon: Icon } = TOOL_INFO[tool];
        const active = ws.tool === tool;
        return (
          <div key={tool} className="contents">
            {index === ORDER.length - 1 && <span className="my-0.5 h-px w-5 bg-line" />}
            <Tooltip
              side="right"
              content={
                <>
                  {label} <kbd>{format(ws.shortcuts[tool])}</kbd>
                </>
              }
            >
              <button
                type="button"
                className={cn(TOOL_BUTTON, "h-9 w-9", active ? "bg-brand-weak text-brand" : "text-icon-2 hover:bg-hover hover:text-icon")}
                aria-label={label}
                aria-pressed={active}
                disabled={!page}
                onClick={() => ws.setTool(tool)}
              >
                <Icon size={16} />
              </button>
            </Tooltip>
          </div>
        );
      })}

      {isBrushTool(ws.tool) && page && (
        <div className="mt-0.5 flex w-full flex-col items-center gap-0.5 pt-1 shadow-[inset_0_1px_0_var(--border-soft)]">
          {ws.tool === "draw" && (
            <ColorWell size="sm" label="Brush color" value={ws.brush.color} onChange={(color) => color && ws.setBrush({ ...ws.brush, color })} />
          )}
          <BrushSize max={maxBrush(page.width, page.height)} />
        </div>
      )}

      {ws.tool === "fill" && page && (
        <div className="mt-0.5 flex w-full flex-col items-center gap-0.5 pt-1 shadow-[inset_0_1px_0_var(--border-soft)]" role="group" aria-label="Selection shape">
          {FILL_SHAPES.map(({ shape, label, icon: Icon }) => (
            <Tooltip key={shape} side="right" content={label}>
              <button
                type="button"
                className={cn(TOOL_BUTTON, "h-8 w-8", ws.fillShape === shape ? "bg-selected text-fg" : "text-icon-2 hover:bg-hover hover:text-icon")}
                aria-label={label}
                aria-pressed={ws.fillShape === shape}
                onClick={() => ws.setFillShape(shape)}
              >
                <Icon size={15} />
              </button>
            </Tooltip>
          ))}
        </div>
      )}
    </aside>
  );
});

function BrushSize({ max }: { max: number }) {
  const ws = useWorkspace();
  const size = Math.round(ws.brush.diameter);
  const set = (diameter: number) => ws.setBrush({ ...ws.brush, diameter: Math.min(max, Math.max(MIN_BRUSH, Math.round(diameter))) });
  const keys = `${format(ws.shortcuts.brush_smaller)} and ${format(ws.shortcuts.brush_larger)}`;
  return (
    <Popover
      side="right"
      trigger={
        <button
          type="button"
          className={cn(TOOL_BUTTON, "h-8 w-8 text-[11px] font-medium tabular-nums text-fg-2 hover:bg-hover")}
          aria-label={`Brush size ${size} pixels`}
          title={`Brush size (${keys})`}
        >
          {size}
        </button>
      }
    >
      <div className="flex w-[200px] flex-col gap-2">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs tracking-[0.005em] text-fg-2">Brush size</span>
          <div style={{ width: 96 }}>
            <NumberField label="Brush size" value={size} min={MIN_BRUSH} max={max} onCommit={(value) => value !== null && set(value)} />
          </div>
        </div>
        <Slider label="Brush size" min={MIN_BRUSH} max={max} value={size} onChange={set} />
        <p className="text-xs tracking-[0.005em] text-fg-3">{keys}, or Alt + scroll on the canvas, also change the size.</p>
        <div className="flex flex-col gap-1.5 pt-2 shadow-[inset_0_1px_0_var(--border-soft)]">
          <span className="text-xs tracking-[0.005em] text-fg-2">Pen pressure</span>
          <label className="flex items-center justify-between gap-2 text-xs text-fg-2">
            Size
            <Switch label="Pen pressure changes the size" checked={ws.brush.pressureSize} onChange={(on) => ws.setBrush({ ...ws.brush, pressureSize: on })} />
          </label>
          <label className="flex items-center justify-between gap-2 text-xs text-fg-2">
            Opacity
            <Switch label="Pen pressure changes the opacity" checked={ws.brush.pressureOpacity} onChange={(on) => ws.setBrush({ ...ws.brush, pressureOpacity: on })} />
          </label>
          <p className="text-xs tracking-[0.005em] text-fg-3">With a drawing tablet: press harder for a bigger, darker stroke.</p>
        </div>
      </div>
    </Popover>
  );
}
