// A gradient for text (Photoshop's Gradient Overlay) or its border (gradient Stroke): none, linear at an
// angle, or radial, with colour stops. The swatch shows it; the popover edits it.

import { ArrowLeftRight, Plus, X } from "lucide-react";

import { cn } from "cn";

import type { Gradient, GradientStop } from "../gen/urakatsu/v1/urakatsu_pb";
import { ColorWell, NumberField, Popover, Tooltip } from "./primitives";

type Kind = "none" | "linear" | "radial";

const stop = (offset: number, color: string): GradientStop => ({ $typeName: "urakatsu.v1.GradientStop", offset, color });
const gradient = (kind: string, angle: number, stops: GradientStop[]): Gradient => ({ $typeName: "urakatsu.v1.Gradient", kind, angle, stops });

const PRESETS: { name: string; stops: [number, string][] }[] = [
  { name: "Sunset", stops: [[0, "#ff512fff"], [1, "#ffd200ff"]] },
  { name: "Fire", stops: [[0, "#7f1d1dff"], [0.5, "#ef4444ff"], [1, "#fde047ff"]] },
  { name: "Ocean", stops: [[0, "#1e3a8aff"], [1, "#22d3eeff"]] },
  { name: "Neon", stops: [[0, "#2563ebff"], [0.5, "#9333eaff"], [1, "#db2777ff"]] },
  { name: "Gold", stops: [[0, "#78350fff"], [0.5, "#fbbf24ff"], [1, "#fef3c7ff"]] },
  { name: "Mono", stops: [[0, "#000000ff"], [1, "#ffffffff"]] },
];

/** The stops as a left-to-right CSS gradient, for previews. */
function css(stops: readonly GradientStop[]): string {
  const sorted = [...stops].sort((a, b) => a.offset - b.offset);
  return `linear-gradient(90deg, ${sorted.map((s) => `${s.color} ${Math.round(s.offset * 100)}%`).join(", ")})`;
}

const SEGMENT = "grid place-items-center rounded-[4px] text-xs transition-colors duration-[20ms] ease-in focus-visible:outline-none focus-visible:shadow-focus";

export function GradientField({
  label,
  value,
  onChange,
  disabled,
  startColor,
}: {
  label: string;
  value: Gradient | undefined;
  onChange: (value: Gradient | undefined) => void;
  disabled?: boolean;
  /** The first stop of a new gradient (the solid colour it replaces). */
  startColor: string;
}) {
  const kind: Kind = value && value.stops.length >= 2 ? (value.kind === "radial" ? "radial" : "linear") : "none";
  const stops = value?.stops ?? [];
  const set = (next: Partial<{ kind: string; angle: number; stops: GradientStop[] }>) =>
    onChange(gradient(next.kind ?? value?.kind ?? "linear", next.angle ?? value?.angle ?? 90, next.stops ?? stops));

  const choose = (next: Kind) => {
    if (next === "none") return onChange(undefined);
    if (kind === "none") return onChange(gradient(next, 90, [stop(0, startColor), stop(1, "#ffffffff")]));
    set({ kind: next });
  };
  const setStop = (index: number, patch: Partial<Pick<GradientStop, "offset" | "color">>) =>
    set({ stops: stops.map((s, i) => (i === index ? stop(patch.offset ?? s.offset, patch.color ?? s.color) : s)) });
  const addStop = () => {
    const sorted = [...stops].sort((a, b) => a.offset - b.offset);
    // In the widest gap, in the colour of the stop before it.
    let at = 0;
    for (let i = 1; i < sorted.length; i++) if (sorted[i].offset - sorted[i - 1].offset > sorted[at + 1].offset - sorted[at].offset) at = i - 1;
    set({ stops: [...stops, stop((sorted[at].offset + sorted[at + 1].offset) / 2, sorted[at].color)] });
  };

  return (
    <Popover
      side="left"
      align="start"
      className="p-2"
      trigger={
        <button
          type="button"
          aria-label={label}
          disabled={disabled}
          className="grid size-[26px] cursor-pointer place-items-center rounded-control bg-surface p-0 shadow-[inset_0_0_0_1px_var(--border)] outline-none transition-colors duration-[20ms] hover:bg-hover focus-visible:shadow-focus disabled:pointer-events-none disabled:opacity-50 data-popup-open:shadow-focus"
        >
          <span
            className={cn("size-3.5 rounded-[3px] shadow-[inset_0_0_0_1px_rgb(0_0_0/0.12)]", kind === "none" && "bg-[linear-gradient(135deg,transparent_45%,var(--danger)_45%,var(--danger)_55%,transparent_55%)]")}
            style={kind === "none" ? undefined : { background: kind === "radial" ? `radial-gradient(${[...stops].sort((a, b) => a.offset - b.offset).map((s) => `${s.color} ${Math.round(s.offset * 100)}%`).join(", ")})` : css(stops) }}
          />
        </button>
      }
    >
      <div className="flex w-[232px] flex-col gap-2">
        <div className="grid h-[26px] grid-cols-3 rounded-control bg-surface p-0.5 shadow-[inset_0_0_0_1px_var(--border)]" role="group" aria-label={`${label} type`}>
          {(["none", "linear", "radial"] as const).map((option) => (
            <button key={option} type="button" aria-pressed={kind === option} className={cn(SEGMENT, kind === option ? "bg-brand text-on-brand" : "text-fg-2 hover:text-fg")} onClick={() => choose(option)}>
              {option === "none" ? "None" : option === "linear" ? "Linear" : "Radial"}
            </button>
          ))}
        </div>
        {kind !== "none" && (
          <>
            <div className="h-5 rounded-control shadow-[inset_0_0_0_1px_rgb(0_0_0/0.12)]" style={{ background: css(stops) }} aria-hidden />
            {kind === "linear" && (
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs text-fg-2">Angle</span>
                <div className="w-[96px]">
                  <NumberField label={`${label} angle`} value={value?.angle ?? 90} min={-360} max={360} step={15} suffix={<span className="pr-1.5 text-xs text-fg-3">°</span>} onCommit={(angle) => angle !== null && set({ angle })} />
                </div>
              </div>
            )}
            <div className="flex flex-col gap-1" role="list" aria-label={`${label} stops`}>
              {stops.map((s, index) => (
                <div key={index} className="flex items-center gap-1" role="listitem">
                  <ColorWell size="sm" label={`Stop ${index + 1} colour`} value={s.color.slice(0, 7)} onChange={(color) => color && setStop(index, { color: `${color}ff` })} />
                  <div className="min-w-0 flex-1">
                    <NumberField label={`Stop ${index + 1} position`} value={Math.round(s.offset * 100)} min={0} max={100} step={5} suffix={<span className="pr-1.5 text-xs text-fg-3">%</span>} onCommit={(position) => position !== null && setStop(index, { offset: position / 100 })} />
                  </div>
                  <button
                    type="button"
                    className="grid size-6 flex-none cursor-pointer place-items-center rounded-control text-icon-2 hover:bg-hover hover:text-icon disabled:pointer-events-none disabled:opacity-30"
                    aria-label={`Remove stop ${index + 1}`}
                    disabled={stops.length <= 2}
                    onClick={() => set({ stops: stops.filter((_, i) => i !== index) })}
                  >
                    <X size={13} />
                  </button>
                </div>
              ))}
            </div>
            <div className="flex items-center gap-1">
              <button type="button" className="inline-flex h-6 cursor-pointer items-center gap-1 rounded-control px-1.5 text-xs text-fg-2 hover:bg-hover hover:text-fg" onClick={addStop}>
                <Plus size={12} /> Add stop
              </button>
              <button
                type="button"
                className="inline-flex h-6 cursor-pointer items-center gap-1 rounded-control px-1.5 text-xs text-fg-2 hover:bg-hover hover:text-fg"
                onClick={() => set({ stops: stops.map((s) => stop(1 - s.offset, s.color)) })}
              >
                <ArrowLeftRight size={12} /> Reverse
              </button>
            </div>
          </>
        )}
        <div className="flex flex-col gap-1 pt-1.5 shadow-[inset_0_1px_0_var(--border-soft)]">
          <span className="text-xs text-fg-2">Presets</span>
          <div className="flex flex-wrap gap-1">
            {PRESETS.map((preset) => (
              <Tooltip key={preset.name} content={preset.name}>
                <button
                  type="button"
                  aria-label={`${preset.name} gradient`}
                  className="h-[18px] w-[34px] cursor-pointer rounded-[4px] shadow-[inset_0_0_0_1px_rgb(0_0_0/0.12)] focus-visible:shadow-focus focus-visible:outline-none"
                  style={{ background: css(preset.stops.map(([offset, color]) => stop(offset, color))) }}
                  onClick={() => onChange(gradient(kind === "radial" ? "radial" : "linear", value?.angle ?? 90, preset.stops.map(([offset, color]) => stop(offset, color))))}
                />
              </Tooltip>
            ))}
          </div>
        </div>
      </div>
    </Popover>
  );
}
