// The project's format, from the project header: manga pages, or webtoon/manhwa/manhua strips that
// are stitched and cut again where no text is cut (urakatsu/pipeline/stitch.py), with the cut settings.

import { BookOpen, GalleryVertical } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { useWorkspace, type ProjectFormatInit } from "../state/workspace";
import { CheckRow, NumberField, Popover, Slider } from "./primitives";
import { Segmented } from "./ui";

type Kind = "manga" | "webtoon";

/** The server's defaults (StitchSettings in pipeline/stitch.py), for a project that never set them. */
const DEFAULTS = { height: 0, sensitivity: 90, scanStep: 5, margin: 0, detect: true };

export function ProjectFormatButton() {
  const ws = useWorkspace();
  const project = ws.project!;
  const kind: Kind = project.format?.kind === "webtoon" ? "webtoon" : "manga";
  const stored = project.format?.stitch;
  const stitch = stored ? { height: stored.height, sensitivity: stored.sensitivity, scanStep: stored.scanStep, margin: stored.margin, detect: stored.detect } : DEFAULTS;
  // The slider moves locally and saves when let go.
  const [sensitivity, setSensitivity] = useState(stitch.sensitivity);
  useEffect(() => setSensitivity(stitch.sensitivity), [stitch.sensitivity]);

  const save = (next: { kind?: Kind } & Partial<typeof DEFAULTS>) => {
    const { kind: nextKind = kind, ...settings } = next;
    const format: ProjectFormatInit = { kind: nextKind, stitch: { ...stitch, ...settings } };
    void ws.setProjectFormat(format);
  };

  const label = kind === "webtoon" ? "Format: webtoon" : "Format: manga";
  return (
    <Popover
      align="end"
      className="w-80 p-3"
      trigger={
        <Button variant="ghost" size="icon-sm" aria-label={label} title={label}>
          {kind === "webtoon" ? <GalleryVertical size={16} /> : <BookOpen size={16} />}
        </Button>
      }
    >
      <div className="flex flex-col gap-3">
        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-fg-2">Format</span>
          <Segmented
            fill
            label="Project format"
            value={kind}
            onChange={(value) => save({ kind: value })}
            options={[
              { value: "manga", label: "Manga", title: "Manga: pages as they come" },
              { value: "webtoon", label: "Webtoon", title: "Webtoon, manhwa and manhua: long strips" },
            ]}
          />
          <p className="text-xs leading-4 text-fg-3">
            {kind === "webtoon"
              ? "Webtoon, manhwa and manhua. A chapter's images are joined into one strip and cut again where no text, bubble or drawing is cut."
              : "Pages are kept as they come. Choose Webtoon for manhwa and manhua that arrive as one long strip."}
          </p>
        </div>
        {kind === "webtoon" && (
          <>
            <Field label="Page height" detail="Auto is 1.5 times the strip width">
              <NumberField
                label="Page height"
                value={stitch.height || null}
                placeholder="Auto"
                min={200}
                max={20000}
                step={100}
                format={{ useGrouping: false }}
                onCommit={(value) => save({ height: value ?? 0 })}
              />
            </Field>
            <Field label="Sensitivity" detail={`${sensitivity}. Higher cuts only flatter rows`}>
              <Slider label="Sensitivity" value={sensitivity} min={0} max={100} onChange={setSensitivity} onCommit={(value) => save({ sensitivity: value })} />
            </Field>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Scan step">
                <NumberField label="Scan step" value={stitch.scanStep} min={1} max={100} onCommit={(value) => save({ scanStep: value ?? DEFAULTS.scanStep })} />
              </Field>
              <Field label="Side margin">
                <NumberField label="Side margin" value={stitch.margin} min={0} max={1000} onCommit={(value) => save({ margin: value ?? 0 })} />
              </Field>
            </div>
            <div className="-mx-2">
              <CheckRow checked={stitch.detect} label="Avoid detected text" detail="Detection model" onChange={(detect) => save({ detect })} />
            </div>
            <p className="text-xs leading-4 text-fg-3">
              Chapters downloaded from now on are restitched as they arrive. Restitch a chapter already here from its row in the sidebar.
            </p>
          </>
        )}
      </div>
    </Popover>
  );
}

function Field({ label, detail, children }: { label: string; detail?: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="flex items-baseline justify-between gap-2 text-xs">
        <span className="font-medium text-fg-2">{label}</span>
        {detail && <span className="truncate text-fg-3">{detail}</span>}
      </span>
      {children}
    </label>
  );
}

