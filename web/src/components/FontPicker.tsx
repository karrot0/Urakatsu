// Font choice: the installed families (from the service) with search, and the typesetting default.

import { useEffect, useMemo, useState } from "react";

import { ChevronDown } from "lucide-react";

import { cn } from "cn";

import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { client } from "../api/client";
import { useWorkspace } from "../state/workspace";
import { Popover } from "./primitives";

let fontCache: Promise<string[]> | null = null;
export function useFontFamilies(): string[] {
  const [families, setFamilies] = useState<string[]>([]);
  useEffect(() => {
    fontCache ??= client.listFonts({}).then((list) => list.families.map((f) => f.name).sort((a, b) => a.localeCompare(b)));
    void fontCache.then(setFamilies).catch(() => setFamilies([]));
  }, []);
  return families;
}

/** The font a block without its own font typesets in: the first of Settings > Typesetting's default fonts that is installed. */
export function useDefaultFont(): string | undefined {
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
