// The Agent tab's prompt box: an auto-growing message field with the open page as context,
// a searchable model picker, and one send / stop button.

import { ArrowUp, Check, ChevronDown, Cloud, Cpu, FileImage, Search, Sparkles, Square } from "lucide-react";
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";

import { cn } from "cn";

import type { AgentCommand, AgentModel } from "../api/agent";
import { Popover, Tooltip } from "./primitives";

const MIN_HEIGHT = 40;
const MAX_HEIGHT = 200;

export function AgentComposer({
  value,
  onChange,
  onSend,
  onStop,
  running,
  disabled,
  placeholder,
  lastSent,
  page,
  models,
  model,
  onModel,
  commands,
}: {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onStop: () => void;
  running: boolean;
  disabled: boolean;
  placeholder: string;
  /** The last message sent, recalled with the Up arrow in an empty box. */
  lastSent?: string;
  /** The open page the assistant works on, shown as context. */
  page?: string;
  models: AgentModel[] | null;
  model: string;
  onModel: (model: string) => void;
  /** Slash commands, offered while the message is just ``/`` and a partial name. */
  commands: AgentCommand[];
}) {
  const field = useRef<HTMLTextAreaElement>(null);
  const canSend = !disabled && !running && !!value.trim() && !!model;
  const [active, setActive] = useState(0);
  const [dismissed, setDismissed] = useState(false);
  const typed = /^\/(\S*)$/.exec(value)?.[1];
  const matches = useMemo(
    () => (typed === undefined || dismissed ? [] : commands.filter((c) => c.name.startsWith(typed.toLowerCase()))),
    [typed, dismissed, commands],
  );
  const current = Math.min(active, Math.max(matches.length - 1, 0));

  useEffect(() => {
    if (!value.startsWith("/")) setDismissed(false);
  }, [value]);

  const complete = (command: AgentCommand) => {
    onChange(`/${command.name} `);
    setActive(0);
    field.current?.focus();
  };

  // Grow with the text up to MAX_HEIGHT, then scroll.
  useLayoutEffect(() => {
    const element = field.current;
    if (!element) return;
    element.style.height = "0px";
    element.style.height = `${Math.min(MAX_HEIGHT, Math.max(MIN_HEIGHT, element.scrollHeight))}px`;
  }, [value]);

  useEffect(() => {
    if (!disabled) field.current?.focus();
  }, [disabled]);

  const keyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (matches.length && !event.nativeEvent.isComposing) {
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        setActive((current + (event.key === "ArrowDown" ? 1 : matches.length - 1)) % matches.length);
        return;
      }
      if (event.key === "Tab" || (event.key === "Enter" && !event.shiftKey)) {
        event.preventDefault();
        complete(matches[current]);
        return;
      }
      if (event.key === "Escape") {
        event.preventDefault();
        setDismissed(true);
        return;
      }
    }
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      if (canSend) onSend();
    } else if (event.key === "Escape" && running) {
      event.preventDefault();
      onStop();
    } else if (event.key === "ArrowUp" && !value && lastSent) {
      event.preventDefault();
      onChange(lastSent);
    }
  };

  return (
    <div className="flex-none p-2">
      <div
        className={cn(
          "flex flex-col rounded-card bg-surface shadow-[inset_0_0_0_1px_var(--border)] transition-shadow duration-100 ease-out",
          "focus-within:shadow-[inset_0_0_0_1px_var(--brand-line),var(--focus-ring)]",
          disabled && "opacity-60",
        )}
        onMouseDown={(event) => {
          // A click on the box's padding or toolbar background still lands in the field.
          if (event.target === event.currentTarget) {
            event.preventDefault();
            field.current?.focus();
          }
        }}
      >
        {page && (
          <div className="flex px-2 pt-2">
            <Tooltip content="The assistant works on the open page unless you name another" side="top">
              <span className="inline-flex h-[22px] max-w-full cursor-default items-center gap-1 rounded-pill bg-surface-fill px-1.5 text-xs text-fg-2">
                <FileImage size={12} className="flex-none text-icon-2" />
                <span className="truncate">{page}</span>
              </span>
            </Tooltip>
          </div>
        )}
        {matches.length > 0 && (
          <div role="listbox" aria-label="Commands" className="flex flex-col gap-px p-1 shadow-[inset_0_-1px_0_var(--border-soft)]">
            {matches.map((command, index) => (
              <button
                key={command.name}
                type="button"
                role="option"
                aria-selected={index === current}
                className={cn(
                  "flex min-w-0 cursor-pointer flex-col items-start rounded-control px-2 py-1 text-left transition-colors duration-[20ms] ease-in",
                  index === current ? "bg-hover-faint" : "hover:bg-hover-faint",
                )}
                onMouseDown={(event) => {
                  event.preventDefault();
                  complete(command);
                }}
                onMouseEnter={() => setActive(index)}
              >
                <span className="text-sm leading-5 text-fg">
                  <span className="font-medium">/{command.name}</span>
                  {command.argument && <span className="text-fg-3"> {command.argument}</span>}
                </span>
                <span className="w-full truncate text-xs leading-4 text-fg-3">{command.description}</span>
              </button>
            ))}
          </div>
        )}
        <textarea
          ref={field}
          rows={1}
          aria-label="Message"
          aria-autocomplete="list"
          aria-expanded={matches.length > 0}
          className="block w-full resize-none bg-transparent px-3 pt-2 pb-1 text-sm leading-5 text-fg outline-none placeholder:text-fg-3 disabled:cursor-not-allowed"
          style={{ minHeight: MIN_HEIGHT, maxHeight: MAX_HEIGHT }}
          placeholder={placeholder}
          value={value}
          disabled={disabled}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={keyDown}
        />
        <div className="flex items-center gap-1 px-1.5 pb-1.5">
          <ModelPicker models={models} value={model} onChange={onModel} disabled={disabled} />
          <span className="min-w-0 flex-1" />
          <span className="hidden flex-none text-[11px] text-fg-3 @[320px]/agent:inline">{running ? "Esc to stop" : "Enter to send"}</span>
          {running ? (
            <Tooltip content="Stop (Esc)" side="top">
              <button
                type="button"
                aria-label="Stop"
                className="grid size-7 flex-none cursor-pointer place-items-center rounded-full bg-fg text-surface transition-colors duration-[20ms] ease-in hover:bg-fg-2 focus-visible:shadow-focus focus-visible:outline-none"
                onClick={onStop}
              >
                <Square size={10} fill="currentColor" />
              </button>
            </Tooltip>
          ) : (
            <Tooltip content="Send (Enter)" side="top">
              <button
                type="button"
                aria-label="Send"
                disabled={!canSend}
                className="grid size-7 flex-none cursor-pointer place-items-center rounded-full bg-brand text-on-brand transition-colors duration-[20ms] ease-in hover:bg-brand-hover active:bg-brand-press focus-visible:shadow-focus focus-visible:outline-none disabled:cursor-default disabled:bg-surface-fill disabled:text-fg-3"
                onClick={onSend}
              >
                <ArrowUp size={14} />
              </button>
            </Tooltip>
          )}
        </div>
      </div>
    </div>
  );
}

// ---- model picker ------------------------------------------------------------------------------

const SOURCE_ORDER: Record<AgentModel["source"], number> = { chatgpt: 0, local: 1, provider: 2 };

function SourceIcon({ source, className }: { source: AgentModel["source"]; className?: string }) {
  const Icon = source === "chatgpt" ? Sparkles : source === "local" ? Cpu : Cloud;
  return <Icon size={12} className={cn("flex-none", className)} />;
}

type Row = { kind: "group"; label: string } | { kind: "model"; model: AgentModel; index: number };

function ModelPicker({ models, value, onChange, disabled }: { models: AgentModel[] | null; value: string; onChange: (model: string) => void; disabled: boolean }) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const list = useRef<HTMLDivElement>(null);
  const selected = models?.find((m) => m.id === value);

  // Matches grouped by provider: ChatGPT first, then the local model, then hosted providers.
  const { rows, matches } = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const found = (models ?? [])
      .filter((m) => !needle || `${m.name} ${m.model} ${m.provider_name}`.toLowerCase().includes(needle))
      .sort((a, b) => SOURCE_ORDER[a.source] - SOURCE_ORDER[b.source]);
    const rows: Row[] = [];
    const matches: AgentModel[] = [];
    let group = "";
    for (const model of found.slice(0, 300)) {
      if (model.provider_name !== group) {
        group = model.provider_name;
        rows.push({ kind: "group", label: group });
      }
      rows.push({ kind: "model", model, index: matches.length });
      matches.push(model);
    }
    return { rows, matches };
  }, [models, query]);

  useEffect(() => {
    if (!open) return;
    const index = matches.findIndex((m) => m.id === value);
    setActive(index >= 0 ? index : 0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);
  useEffect(() => {
    list.current?.querySelector(`[data-index="${active}"]`)?.scrollIntoView({ block: "nearest" });
  }, [active, open]);

  const choose = (model: AgentModel) => {
    onChange(model.id);
    setOpen(false);
    setQuery("");
  };

  const keyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const step = event.key === "ArrowDown" ? 1 : -1;
      setActive((index) => (matches.length ? (index + step + matches.length) % matches.length : 0));
    } else if (event.key === "Enter" && matches[active]) {
      event.preventDefault();
      choose(matches[active]);
    }
  };

  const loading = models === null;
  return (
    <Popover
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (!next) setQuery("");
      }}
      side="top"
      align="start"
      className="w-[300px] p-1"
      trigger={
        <button
          type="button"
          aria-label="Model"
          disabled={disabled || loading || !models?.length}
          className="inline-flex h-7 max-w-[70%] min-w-0 cursor-pointer items-center gap-1 rounded-control px-1.5 text-xs text-fg-2 transition-colors duration-[20ms] ease-in hover:bg-hover hover:text-fg focus-visible:shadow-focus focus-visible:outline-none disabled:cursor-default disabled:hover:bg-transparent data-[popup-open]:bg-hover data-[popup-open]:text-fg"
        >
          {selected && <SourceIcon source={selected.source} className="text-icon-2" />}
          <span className="truncate">{selected?.name ?? (loading ? "Loading models" : "No models")}</span>
          <ChevronDown size={12} className="flex-none text-icon-2" />
        </button>
      }
    >
      <div className="flex items-center gap-1.5 px-2 pb-1 shadow-[inset_0_-1px_0_var(--border-soft)]">
        <Search size={14} className="flex-none text-icon-2" />
        <input
          className="h-8 min-w-0 flex-1 bg-transparent text-sm text-fg outline-none placeholder:text-fg-3"
          placeholder="Search models"
          aria-label="Search models"
          value={query}
          onChange={(event) => {
            setQuery(event.target.value);
            setActive(0);
          }}
          onKeyDown={keyDown}
          autoFocus
        />
      </div>
      <div ref={list} className="max-h-[320px] overflow-y-auto pt-1" role="listbox" aria-label="Models">
        {rows.map((row) =>
          row.kind === "group" ? (
            <div key={`group:${row.label}`} className="px-2 pt-2 pb-0.5 text-[10px] leading-3.5 font-medium tracking-[0.06em] text-fg-3 uppercase">
              {row.label}
            </div>
          ) : (
            <button
              key={row.model.id}
              type="button"
              role="option"
              data-index={row.index}
              aria-selected={row.model.id === value}
              className={cn(
                "flex min-h-[30px] w-full cursor-default items-center gap-2 rounded-control px-2 py-1 text-left text-sm text-fg outline-none select-none",
                row.index === active ? "bg-hover" : "hover:bg-hover-faint",
              )}
              onMouseMove={() => row.index !== active && setActive(row.index)}
              onClick={() => choose(row.model)}
            >
              <SourceIcon source={row.model.source} className="text-icon-2" />
              <span className="flex min-w-0 flex-1 flex-col">
                <span className="truncate">{row.model.name}</span>
                {row.model.name.toLowerCase() !== row.model.model.toLowerCase() && <span className="truncate text-[11px] text-fg-3">{row.model.model}</span>}
              </span>
              {row.model.id === value && <Check size={14} className="flex-none text-brand" />}
            </button>
          ),
        )}
        {models && matches.length === 0 && <p className="p-2 text-xs text-fg-2">No models match.</p>}
        {matches.length === 300 && <p className="px-2 py-1.5 text-[11px] text-fg-3">Showing the first 300. Search to find others.</p>}
      </div>
    </Popover>
  );
}
