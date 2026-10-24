// The app's controls, as small wrappers over the shadcn components in ./ui (Base UI underneath,
// styled with Tailwind and the Urakatsu tokens). Every dropdown is a Select or a menu from here;
// no native <select>.

import { NumberField as NumberFieldPrimitive } from "@base-ui/react/number-field";
import { cn } from "cn";
import { Minus, Plus } from "lucide-react";
import { useEffect, useRef, useState, type ReactElement, type ReactNode } from "react";
import { HexColorPicker } from "react-colorful";

import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuRadioGroup, DropdownMenuRadioItem, DropdownMenuTrigger } from "./ui/dropdown-menu";
import { Input } from "./ui/input";
import { Popover as PopoverRoot, PopoverContent, PopoverTrigger } from "./ui/popover";
import { Select as SelectRoot, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from "./ui/select";
import { Slider as SliderBase } from "./ui/slider";
import { Switch as SwitchBase } from "./ui/switch";
import { TooltipProvider as TooltipProviderBase, Tooltip as TooltipRoot, TooltipContent, TooltipTrigger } from "./ui/tooltip";

// ---- select --------------------------------------------------------------------------------

export interface Option<T extends string = string> {
  value: T;
  label: ReactNode;
  detail?: ReactNode;
  disabled?: boolean;
  group?: string;
}

export function Select<T extends string>({
  value,
  options,
  onChange,
  label,
  placeholder,
  size = "md",
  disabled,
  className,
  renderValue,
}: {
  value: T | null | undefined;
  options: Option<T>[];
  onChange: (value: T) => void;
  label: string;
  placeholder?: string;
  size?: "sm" | "md";
  disabled?: boolean;
  className?: string;
  renderValue?: (option: Option<T> | undefined) => ReactNode;
}) {
  const selected = options.find((option) => option.value === value);
  const groups = [...new Set(options.map((option) => option.group ?? ""))];
  return (
    <SelectRoot
      value={value ?? null}
      onValueChange={(next) => next !== null && onChange(next as T)}
      disabled={disabled}
      items={options.map((option) => ({ value: option.value, label: option.label }))}
    >
      <SelectTrigger size={size === "sm" ? "sm" : "default"} className={className} aria-label={label}>
        <SelectValue>
          {() => (selected ? (renderValue ? renderValue(selected) : selected.label) : <span className="text-fg-3">{placeholder ?? "Choose"}</span>)}
        </SelectValue>
      </SelectTrigger>
      <SelectContent>
        {groups.map((group) => (
          <SelectGroup key={group || "_"}>
            {group && <SelectLabel>{group}</SelectLabel>}
            {options
              .filter((option) => (option.group ?? "") === group)
              .map((option) => (
                <SelectItem key={option.value} value={option.value} disabled={option.disabled}>
                  {option.label}
                  {option.detail && <span className="text-xs whitespace-nowrap text-fg-3">{option.detail}</span>}
                </SelectItem>
              ))}
          </SelectGroup>
        ))}
      </SelectContent>
    </SelectRoot>
  );
}

// ---- popover -------------------------------------------------------------------------------

export function Popover({
  trigger,
  children,
  side = "bottom",
  align = "center",
  open,
  onOpenChange,
  className,
}: {
  trigger: ReactElement;
  children: ReactNode;
  side?: "top" | "bottom" | "left" | "right";
  align?: "start" | "center" | "end";
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  className?: string;
}) {
  return (
    <PopoverRoot open={open} onOpenChange={onOpenChange ? (next) => onOpenChange(next) : undefined}>
      <PopoverTrigger render={trigger} />
      <PopoverContent side={side} align={align} className={className}>
        {children}
      </PopoverContent>
    </PopoverRoot>
  );
}

// ---- tooltip -------------------------------------------------------------------------------

export const TooltipProvider = ({ children }: { children: ReactNode }) => <TooltipProviderBase delay={400}>{children}</TooltipProviderBase>;

export function Tooltip({
  content,
  children,
  side = "top",
}: {
  content: ReactNode;
  children: ReactElement;
  side?: "top" | "bottom" | "left" | "right";
}) {
  return (
    <TooltipRoot>
      <TooltipTrigger render={children} />
      <TooltipContent side={side} sideOffset={8}>
        {content}
      </TooltipContent>
    </TooltipRoot>
  );
}

// ---- slider --------------------------------------------------------------------------------

export function Slider({
  value,
  min,
  max,
  step = 1,
  label,
  onChange,
  onCommit,
  disabled,
  className,
}: {
  value: number;
  min: number;
  max: number;
  step?: number;
  label: string;
  onChange: (value: number) => void;
  onCommit?: (value: number) => void;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <SliderBase
      className={className}
      value={value}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      aria-label={label}
      onValueChange={(next) => onChange(next as number)}
      onValueCommitted={onCommit ? (next) => onCommit(next as number) : undefined}
    />
  );
}

// ---- switch --------------------------------------------------------------------------------

export function Switch({ checked, onChange, label, disabled }: { checked: boolean; onChange: (checked: boolean) => void; label: string; disabled?: boolean }) {
  return <SwitchBase checked={checked} disabled={disabled} aria-label={label} onCheckedChange={(next) => onChange(next)} />;
}

// ---- number field --------------------------------------------------------------------------

export function NumberField({
  value,
  onCommit,
  label,
  min,
  max,
  step = 1,
  placeholder,
  disabled,
  stepper = true,
  suffix,
  format,
}: {
  value: number | null;
  onCommit: (value: number | null) => void;
  label: string;
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
  disabled?: boolean;
  stepper?: boolean;
  suffix?: ReactNode;
  /** How the number is shown, e.g. { useGrouping: false } for a port. */
  format?: Intl.NumberFormatOptions;
}) {
  const [draft, setDraft] = useState<number | null>(value);
  useEffect(() => setDraft(value), [value]);
  const stepButton =
    "grid w-6 flex-none cursor-default place-items-center bg-transparent text-icon-2 transition-colors duration-[20ms] hover:bg-hover hover:text-icon data-disabled:pointer-events-none data-disabled:opacity-35";
  return (
    <NumberFieldPrimitive.Root
      className="min-w-0"
      value={draft}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      format={format}
      onValueChange={(next) => setDraft(next)}
      onValueCommitted={(next) => onCommit(next)}
    >
      {/* Compact by default (inspector); full height and left-aligned inside Settings. */}
      <NumberFieldPrimitive.Group className="flex h-[26px] items-stretch overflow-hidden rounded-control bg-surface shadow-[inset_0_0_0_1px_var(--border)] transition-shadow duration-100 focus-within:shadow-focus in-data-disabled:opacity-50 [.settings_&]:h-8">
        {stepper && (
          <NumberFieldPrimitive.Decrement className={stepButton} aria-label={`Decrease ${label}`}>
            <Minus size={12} />
          </NumberFieldPrimitive.Decrement>
        )}
        <NumberFieldPrimitive.Input
          className="min-w-0 flex-1 bg-transparent px-1 text-center text-xs text-fg tabular-nums outline-none placeholder:text-fg-3 [.settings_&]:px-2 [.settings_&]:text-left [.settings_&]:text-sm"
          aria-label={label}
          placeholder={placeholder}
        />
        {stepper && (
          <NumberFieldPrimitive.Increment className={stepButton} aria-label={`Increase ${label}`}>
            <Plus size={12} />
          </NumberFieldPrimitive.Increment>
        )}
        {suffix}
      </NumberFieldPrimitive.Group>
    </NumberFieldPrimitive.Root>
  );
}

// ---- menus ---------------------------------------------------------------------------------

export function RadioMenu<T extends string>({
  trigger,
  value,
  options,
  onChange,
  align = "end",
}: {
  trigger: ReactElement;
  value: T;
  options: Option<T>[];
  onChange: (value: T) => void;
  align?: "start" | "center" | "end";
}) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger render={trigger} />
      <DropdownMenuContent align={align}>
        <DropdownMenuRadioGroup value={value} onValueChange={(next) => onChange(next as T)}>
          {options.map((option) => (
            <DropdownMenuRadioItem key={option.value} value={option.value} disabled={option.disabled}>
              {option.label}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export interface MenuAction {
  label: ReactNode;
  onSelect: () => void;
  danger?: boolean;
  disabled?: boolean;
}

export function ActionMenu({ trigger, actions, align = "end" }: { trigger: ReactElement; actions: MenuAction[]; align?: "start" | "center" | "end" }) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger render={trigger} />
      <DropdownMenuContent align={align}>
        {actions.map((action, index) => (
          <DropdownMenuItem key={index} variant={action.danger ? "destructive" : "default"} disabled={action.disabled} onClick={action.onSelect}>
            {action.label}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

// ---- color well ----------------------------------------------------------------------------

const COLOR_SETTLE_MS = 200;
const SWATCHES = ["#000000", "#ffffff", "#e5484d", "#f76b15", "#ffc53d", "#30a46c", "#0090ff", "#8e4ec6"];

const swatch = "block cursor-pointer rounded-pill p-0 shadow-[inset_0_0_0_1px_color-mix(in_oklab,var(--n-950)_18%,transparent)]";
// A "no color" swatch: the surface crossed by a red diagonal.
const noneSwatch =
  "bg-[linear-gradient(to_top_right,transparent_calc(50%-1px),var(--danger)_calc(50%-1px),var(--danger)_calc(50%+1px),transparent_calc(50%+1px)),var(--surface)]";

/** A swatch that opens a picker. `value` is #rrggbb; null means "none" when `allowNone` is set. */
export function ColorWell({
  value,
  onChange,
  label,
  allowNone,
  disabled,
  size = "md",
}: {
  value: string | null;
  onChange: (value: string | null) => void;
  label: string;
  allowNone?: boolean;
  disabled?: boolean;
  size?: "sm" | "md";
}) {
  const [draft, setDraft] = useState(value ?? "#000000");
  const [hex, setHex] = useState(value ?? "");
  // Dragging applies once the pointer rests, so the canvas follows without an undo step per pixel.
  const timer = useRef(0);
  const sent = useRef(value);
  const dirty = useRef(false); // the picker moved since it opened
  useEffect(() => {
    sent.current = value;
    setDraft(value ?? "#000000");
    setHex(value ?? "");
  }, [value]);
  useEffect(() => () => window.clearTimeout(timer.current), []);
  const commit = (next: string | null) => {
    window.clearTimeout(timer.current);
    if ((next ?? "").toLowerCase() === (sent.current ?? "").toLowerCase()) return;
    sent.current = next;
    onChange(next);
  };
  return (
    <Popover
      side="left"
      align="start"
      className="p-2"
      onOpenChange={(open) => {
        if (!open && dirty.current) commit(draft);
        dirty.current = false;
      }}
      trigger={
        <button
          type="button"
          aria-label={label}
          disabled={disabled}
          className={cn(
            "grid cursor-pointer place-items-center rounded-control bg-surface p-0 shadow-[inset_0_0_0_1px_var(--border)] outline-none transition-colors duration-[20ms] hover:bg-hover focus-visible:shadow-focus disabled:pointer-events-none disabled:opacity-50 data-popup-open:shadow-focus",
            size === "sm" ? "size-[26px]" : "size-8",
          )}
        >
          <span className={cn(swatch, size === "sm" ? "size-3.5" : "size-4", value === null && noneSwatch)} style={value ? { background: value } : undefined} />
        </button>
      }
    >
      <div className="flex w-[200px] flex-col gap-2 [&_.react-colorful]:h-40 [&_.react-colorful]:w-full [&_.react-colorful__last-control]:rounded-b-control [&_.react-colorful__pointer]:size-4 [&_.react-colorful__saturation]:rounded-t-control">
        <HexColorPicker
          color={draft}
          onChange={(next) => {
            setDraft(next);
            setHex(next);
            dirty.current = true;
            window.clearTimeout(timer.current);
            timer.current = window.setTimeout(() => commit(next), COLOR_SETTLE_MS);
          }}
        />
        <div className="flex flex-wrap gap-1">
          {SWATCHES.map((color) => (
            <button
              key={color}
              type="button"
              className={cn(swatch, "size-[18px]")}
              style={{ background: color }}
              aria-label={color}
              onClick={() => {
                setDraft(color);
                setHex(color);
                commit(color);
              }}
            />
          ))}
          {allowNone && <button type="button" className={cn(swatch, noneSwatch, "size-[18px]")} aria-label="No color" onClick={() => commit(null)} />}
        </div>
        <Input
          className="h-7 font-mono tabular-nums"
          aria-label={`${label} hex`}
          value={hex}
          placeholder="#000000"
          onChange={(event) => setHex(event.target.value)}
          onKeyDown={(event) => event.key === "Enter" && event.currentTarget.blur()}
          onBlur={() => {
            const next = hex.trim().startsWith("#") ? hex.trim() : `#${hex.trim()}`;
            if (/^#[0-9a-f]{6}$/i.test(next)) {
              setDraft(next);
              commit(next);
            } else setHex(value ?? "");
          }}
        />
      </div>
    </Popover>
  );
}
