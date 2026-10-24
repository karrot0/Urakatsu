// A text field with character formatting (rich text): bold, italic, underline, strikethrough and
// colour for any part of the text, as in Photoshop's Character panel. The browser's editing
// commands do the formatting; richText.ts reads the result back as the text and its styles.

import { useEffect, useRef, useState, type CSSProperties } from "react";

import { Bold, Italic, RemoveFormatting, Strikethrough, Underline, type LucideIcon } from "lucide-react";

import { cn } from "cn";

import { fieldChrome } from "@/components/ui/input";
import { fromDom, sameStyles, toHtml, type RichStyle } from "../richText";
import { FontPicker } from "./FontPicker";
import { ColorWell, Select, Tooltip } from "./primitives";

const COMMANDS: { command: string; label: string; keys: string; icon: LucideIcon }[] = [
  { command: "bold", label: "Bold", keys: "Ctrl+B", icon: Bold },
  { command: "italic", label: "Italic", keys: "Ctrl+I", icon: Italic },
  { command: "underline", label: "Underline", keys: "Ctrl+U", icon: Underline },
  { command: "strikeThrough", label: "Strikethrough", keys: "", icon: Strikethrough },
];

/** Characters before a point in the editor (a <br> counts as one). */
function offsetOf(root: HTMLElement, node: Node, offset: number): number {
  const before = document.createRange();
  before.setStart(root, 0);
  before.setEnd(node, offset);
  const fragment = before.cloneContents();
  return (fragment.textContent ?? "").length + fragment.querySelectorAll("br").length;
}

/** The point ``index`` characters into the editor. */
function pointAt(root: HTMLElement, index: number): [Node, number] {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT);
  let count = 0;
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    if (node.nodeType === Node.TEXT_NODE) {
      const length = (node as Text).data.length;
      if (index <= count + length) return [node, index - count];
      count += length;
    } else if (node instanceof HTMLBRElement) {
      count += 1;
      if (index <= count) return [node.parentNode!, Array.prototype.indexOf.call(node.parentNode!.childNodes, node) + 1];
    }
  }
  return [root, root.childNodes.length];
}

// Sizes relative to the layer's (which auto-fit may choose), so a run keeps its proportion when the text is fitted.
const SIZES = ["0.5", "0.75", "0.9", "1", "1.25", "1.5", "2", "3"] as const;
const SIZE_OPTIONS = SIZES.map((value) => ({ value, label: value === "1" ? "100% (layer size)" : `${Math.round(Number(value) * 100)}%` }));

/** The family and size the editor set at a node (em sizes nest, so they multiply). */
function explicitAt(root: HTMLElement, node: Node | null): { font: string; scale: number } {
  let font = "";
  let scale = 1;
  for (let el = node instanceof Element ? node : (node?.parentElement ?? null); el && el !== root; el = el.parentElement) {
    if (!(el instanceof HTMLElement)) continue;
    if (!font && el.style.fontFamily && el.style.fontFamily !== "inherit") font = el.style.fontFamily.split(",")[0].trim().replace(/^["']|["']$/g, "");
    const size = el.style.fontSize.match(/^([\d.]+)em$/);
    if (size) scale *= Number(size[1]);
  }
  return { font, scale };
}

const TOOL_BUTTON =
  "grid size-6 cursor-pointer place-items-center rounded-control text-icon-2 transition-colors duration-[20ms] ease-in hover:bg-hover hover:text-icon focus-visible:shadow-focus focus-visible:outline-none aria-pressed:bg-selected aria-pressed:text-fg";

export function RichTextField({
  label,
  labelClassName,
  text,
  styles,
  onCommit,
  overlay,
  onDone,
}: {
  label: string;
  labelClassName?: string;
  text: string;
  styles: readonly RichStyle[];
  onCommit: (text: string, styles: RichStyle[]) => void;
  /** On the canvas: the editable area fills its box with this look, the toolbar floats above it. */
  overlay?: { style?: CSSProperties; className?: string };
  /** Esc or Ctrl+Enter in the on-canvas editor: saved, now close it. */
  onDone?: () => void;
}) {
  const editor = useRef<HTMLDivElement>(null);
  // The selection last made in the editor, as character offsets: the colour picker takes focus (which
  // commits, and the editor is drawn again from the saved text), then formats it.
  const selected = useRef<[number, number] | null>(null);
  const [active, setActive] = useState<Record<string, boolean>>({});
  const [at, setAt] = useState({ font: "", scale: 1 });
  const value = useRef({ text, styles });
  value.current = { text, styles };

  // Shows the stored text, unless it is being edited.
  const shown = JSON.stringify([text, styles.map((s) => [s.start, s.end, s.bold, s.italic, s.underline, s.strikethrough, s.color ?? ""])]);
  useEffect(() => {
    const el = editor.current;
    if (el && document.activeElement !== el) el.innerHTML = toHtml(text, styles);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shown]);

  // The formatting at the caret, for the toolbar, and the selection to restore. Formatting changes
  // without moving the selection, so commands and key presses refresh it too.
  const refresh = () => {
    const el = editor.current;
    const selection = document.getSelection();
    if (!el || !selection || selection.rangeCount === 0 || !el.contains(selection.anchorNode)) return;
    const range = selection.getRangeAt(0);
    selected.current = [offsetOf(el, range.startContainer, range.startOffset), offsetOf(el, range.endContainer, range.endOffset)];
    setActive(Object.fromEntries(COMMANDS.map(({ command }) => [command, document.queryCommandState(command)])));
    setAt(explicitAt(el, selection.anchorNode));
  };
  useEffect(() => {
    document.addEventListener("selectionchange", refresh);
    return () => document.removeEventListener("selectionchange", refresh);
  }, []);

  // The on-canvas editor opens ready to type, at the end of the text.
  const opened = !!overlay;
  useEffect(() => {
    const el = editor.current;
    if (!opened || !el) return;
    el.focus({ preventScroll: true });
    const selection = document.getSelection();
    selection?.selectAllChildren(el);
    selection?.collapseToEnd();
  }, [opened]);

  const commit = () => {
    const el = editor.current;
    if (!el) return;
    const next = fromDom(el);
    if (next.text !== value.current.text || !sameStyles(next.styles, value.current.styles)) onCommit(next.text, next.styles);
  };

  const format = (command: string, argument?: string, after?: (el: HTMLElement) => void) => {
    const el = editor.current;
    if (!el) return;
    const focused = document.activeElement === el;
    if (!focused) {
      // Back to the selection the picker took focus from.
      el.focus();
      const selection = document.getSelection();
      if (selected.current && selection) {
        const range = document.createRange();
        range.setStart(...pointAt(el, selected.current[0]));
        range.setEnd(...pointAt(el, selected.current[1]));
        selection.removeAllRanges();
        selection.addRange(range);
      }
    }
    document.execCommand("styleWithCSS", false, "true");
    document.execCommand(command, false, argument);
    after?.(el);
    refresh();
    if (!focused) {
      commit();
      // On the canvas the editor stays open for more typing.
      if (!overlay) el.blur();
    }
  };

  // The browser has no em sizes: mark the selection with its largest size, then turn that into the
  // relative size, divided by any size around it so the run ends up at exactly `scale`.
  const resize = (scale: number) =>
    format("fontSize", "7", (el) => {
      for (const marked of Array.from(el.querySelectorAll<HTMLElement>("[style*='xxx-large'], font[size='7']"))) {
        const span = marked.tagName === "FONT" ? document.createElement("span") : marked;
        if (span !== marked) {
          span.append(...Array.from(marked.childNodes));
          marked.replaceWith(span);
        }
        const outer = explicitAt(el, span.parentElement).scale;
        span.style.fontSize = scale === 1 && outer === 1 ? "" : `${Math.round((scale / outer) * 1000) / 1000}em`;
        if (!span.getAttribute("style")) span.replaceWith(...Array.from(span.childNodes));
      }
    });

  const toolbar = (
    <>
      <div className="flex items-center gap-0.5" role="toolbar" aria-label={`${label} formatting`}>
        {!overlay && <span className={cn(labelClassName, "mr-auto")}>{label}</span>}
        {COMMANDS.map(({ command, label: name, keys, icon: Icon }) => (
          <Tooltip key={command} content={keys ? `${name} (${keys})` : name}>
            <button
              type="button"
              className={TOOL_BUTTON}
              aria-label={name}
              aria-pressed={!!active[command]}
              // Keep the editor's focus and selection.
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => format(command)}
            >
              <Icon size={13} />
            </button>
          </Tooltip>
        ))}
        <ColorWell size="sm" label="Text colour" value={null} onChange={(color) => color && format("foreColor", color)} />
        <Tooltip content="Clear formatting">
          <button type="button" className={TOOL_BUTTON} aria-label="Clear formatting" onMouseDown={(event) => event.preventDefault()} onClick={() => format("removeFormat")}>
            <RemoveFormatting size={13} />
          </button>
        </Tooltip>
      </div>
      <div className="flex items-center gap-1">
        <div className="min-w-0 flex-1" onMouseDown={(event) => event.preventDefault()}>
          <FontPicker value={at.font} placeholder="Layer font" onChange={(family) => format("fontName", family || "inherit")} />
        </div>
        <Select
          size="sm"
          className="w-[88px] flex-none"
          label="Text size"
          value={(SIZES.find((size) => Math.abs(Number(size) - at.scale) < 0.01) ?? null) as (typeof SIZES)[number] | null}
          placeholder={`${Math.round(at.scale * 100)}%`}
          options={SIZE_OPTIONS}
          onChange={(value) => resize(Number(value))}
        />
      </div>
    </>
  );

  const field = (
    <div
      ref={editor}
      role="textbox"
      aria-multiline
      aria-label={label}
      contentEditable
      suppressContentEditableWarning
      className={
        overlay
          ? cn("h-full w-full overflow-hidden whitespace-pre-wrap break-words outline-none", overlay.className)
          : cn(fieldChrome, "max-h-[120px] min-h-11 overflow-y-auto whitespace-pre-wrap break-words p-1 text-xs leading-4 shadow-[inset_0_0_0_1px_var(--brand-line)] focus:shadow-focus")
      }
      style={overlay?.style}
      onBlur={commit}
      onKeyUp={refresh}
      onKeyDown={(event) => {
        if ((event.key === "Enter" && (event.ctrlKey || event.metaKey)) || (overlay && event.key === "Escape")) {
          // Done: save, and close the on-canvas editor.
          event.preventDefault();
          event.stopPropagation();
          event.currentTarget.blur();
          onDone?.();
        } else if (event.key === "Enter") {
          // A line break, not a <div> per line.
          event.preventDefault();
          document.execCommand("insertLineBreak");
        }
      }}
      onPaste={(event) => {
        // Plain text only: formatting is what the toolbar sets.
        event.preventDefault();
        document.execCommand("insertText", false, event.clipboardData.getData("text/plain"));
      }}
    />
  );

  if (overlay) {
    return (
      <div className="relative h-full w-full">
        <div className="absolute bottom-full left-0 mb-1.5 flex w-[264px] flex-col gap-1 rounded-[12px] bg-surface-over p-1 shadow-floating">{toolbar}</div>
        {field}
      </div>
    );
  }
  return (
    <div className="flex min-w-0 flex-col gap-0.5">
      {toolbar}
      {field}
    </div>
  );
}
