import { cn } from "cn";
import { useEffect, useRef, useState, type ReactNode, type RefObject } from "react";

import { Badge as BadgeBase } from "./ui/badge";
import { ToggleGroup, ToggleGroupItem } from "./ui/toggle-group";

export type Tone = "neutral" | "success" | "warning" | "danger" | "info";

/** A segmented control: one choice out of a few, always one selected. */
export function Segmented<T extends string>({
  value,
  options,
  onChange,
  fill,
  label,
}: {
  value: T;
  options: { value: T; label: ReactNode; disabled?: boolean; title?: string }[];
  onChange: (value: T) => void;
  fill?: boolean;
  label?: string;
}) {
  return (
    <ToggleGroup
      aria-label={label}
      value={[value]}
      // A pressed segment cannot be unpressed: an empty choice keeps the current one.
      onValueChange={(next) => next.length && onChange(next[next.length - 1] as T)}
      className={cn(fill && "flex w-full")}
    >
      {options.map((option) => (
        <ToggleGroupItem
          key={option.value}
          value={option.value}
          disabled={option.disabled}
          title={option.title}
          aria-label={option.title}
          className={cn(fill && "flex-1")}
        >
          {option.label}
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  );
}

export function Field({ label, hint, htmlFor, children }: { label: string; hint?: string; htmlFor?: string; children: ReactNode }) {
  return (
    <div className="flex min-w-0 flex-1 flex-col gap-1">
      <label htmlFor={htmlFor} className="text-xs text-fg-2">
        {label}
      </label>
      {children}
      {hint && <span className="text-xs text-fg-3">{hint}</span>}
    </div>
  );
}

const BADGE_VARIANT = { neutral: "default", success: "success", warning: "warning", danger: "destructive", info: "info" } as const;

export function Badge({ tone = "neutral", children, title, className }: { tone?: Tone; children: ReactNode; title?: string; className?: string }) {
  return (
    <BadgeBase variant={BADGE_VARIANT[tone]} title={title} className={className}>
      {children}
    </BadgeBase>
  );
}

export function Section({ title, action, children }: { title: string; action?: ReactNode; children: ReactNode }) {
  return (
    <section className="[&+&]:mt-5">
      <div className="mb-2 flex items-center justify-between gap-2 text-xs font-medium text-fg-2">
        <span>{title}</span>
        {action}
      </div>
      {children}
    </section>
  );
}

/** Open state for a popover that closes on an outside click or Escape. */
export function usePopover<T extends HTMLElement>(): [boolean, (open: boolean | ((open: boolean) => boolean)) => void, RefObject<T | null>] {
  const [open, setOpen] = useState(false);
  const anchor = useRef<T>(null);
  useEffect(() => {
    if (!open) return;
    const close = (event: MouseEvent) => {
      if (!anchor.current?.contains(event.target as Node)) setOpen(false);
    };
    const escape = (event: KeyboardEvent) => event.key === "Escape" && setOpen(false);
    window.addEventListener("mousedown", close);
    window.addEventListener("keydown", escape);
    return () => {
      window.removeEventListener("mousedown", close);
      window.removeEventListener("keydown", escape);
    };
  }, [open]);
  return [open, setOpen, anchor];
}
