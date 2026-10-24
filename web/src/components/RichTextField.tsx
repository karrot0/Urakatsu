// A text field with character formatting (rich text): bold, italic, underline, strikethrough and
// colour for any part of the text, as in Photoshop's Character panel. The browser's editing
// commands do the formatting; richText.ts reads the result back as the text and its styles.

import { useEffect, useRef, useState, type CSSProperties } from "react";

import { Bold, Italic, RemoveFormatting, Strikethrough, Underline, type LucideIcon } from "lucide-react";

import { cn } from "cn";

import { fieldChrome } from "@/components/ui/input";
import type { Gradient } from "../gen/urakatsu/v1/urakatsu_pb";
import { cssGradient, fromDom, gradientData, gradientFrom, sameStyles, toHtml, type RichStyle } from "../richText";
import { GradientField } from "./GradientField";
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

/** The family, size and gradient the editor set at a node (em sizes nest, so they multiply). */
function explicitAt(root: HTMLElement, node: Node | null): { font: string; scale: number; gradient?: Gradient } {
  let font = "";
  let scale = 1;
  let gradient: Gradient | undefined;
  let gradientDone = false;
  for (let el = node instanceof Element ? node : (node?.parentElement ?? null); el && el !== root; el = el.parentElement) {
    if (!(el instanceof HTMLElement)) continue;
    if (!font && el.style.fontFamily && el.style.fontFamily !== "inherit") font = el.style.fontFamily.split(",")[0].trim().replace(/^["']|["']$/g, "");
    const size = el.style.fontSize.match(/^([\d.]+)em$/);
    if (size) scale *= Number(size[1]);
    if (!gradientDone && el.dataset.gradient !== undefined) {
      gradientDone = true;
      try {
        gradient = el.dataset.gradient === "none" ? undefined : gradientFrom(JSON.parse(el.dataset.gradient));
      } catch {
        gradient = undefined;
      }
    }
  }
  return { font, scale, gradient };
}

// The colour that marks a selection while it becomes a gradient (never left in the text).
const GRADIENT_MARK = "#010203";

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
  /** On the canvas: the editable area sits in its box with this look (soft line breaks where the
   * render's lines start), the toolbar floats above it. */
  overlay?: {
    /** The text's look, shared by the editable text and the border behind it. */
    style?: CSSProperties;
    /** The fill, on the editable text only. */
    fillStyle?: CSSProperties;
    /** The border, drawn by a copy of the text behind it; none without a border. */
    underlayStyle?: CSSProperties;
    className?: string;
    boxClassName?: string;
    vertical?: boolean;
    softBreaks?: readonly number[];
  };
  /** Esc or Ctrl+Enter in the on-canvas editor: saved, now close it. */
  onDone?: () => void;
}) {
  const editor = useRef<HTMLDivElement>(null);
  // The selection last made in the editor, as character offsets: the colour picker takes focus (which
  // commits, and the editor is drawn again from the saved text), then formats it.
  const selected = useRef<[number, number] | null>(null);
  const [active, setActive] = useState<Record<string, boolean>>({});
  const [at, setAt] = useState<{ font: string; scale: number; gradient?: Gradient }>({ font: "", scale: 1 });
  // On the canvas: an exact copy behind the editable text that draws only the border, as the renderer
  // draws it under the fill (a text stroke on the editable text would cover a gradient fill).
  const underlay = useRef<HTMLDivElement>(null);
  const syncUnderlay = () => {
    const copy = underlay.current;
    const el = editor.current;
    if (!copy || !el) return;
    copy.innerHTML = el.innerHTML;
    // The border is one paint: runs' colours and gradients belong to the fill.
    for (const node of Array.from(copy.querySelectorAll<HTMLElement>("[style], font"))) {
      for (const property of ["color", "background-image", "background-clip", "-webkit-background-clip"]) node.style.removeProperty(property);
      node.removeAttribute("color");
    }
  };
  const value = useRef({ text, styles });
  value.current = { text, styles };

  // Shows the stored text, unless it is being edited.
  const softBreaks = overlay?.softBreaks;
  const shown = JSON.stringify([text, styles.map((s) => [s.start, s.end, s.bold, s.italic, s.underline, s.strikethrough, s.color ?? "", s.font, s.scale ?? 1]), softBreaks ?? []]);
  useEffect(() => {
    const el = editor.current;
    if (el && document.activeElement !== el) el.innerHTML = toHtml(text, styles, softBreaks);
    syncUnderlay();
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
    syncUnderlay();
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

  // The browser has no gradient command: colour the selection with a marker, then turn what it marked
  // into gradient spans (or, for none, spans that keep out the gradient around them).
  const paintGradient = (gradient: Gradient | undefined) =>
    format("foreColor", GRADIENT_MARK, (el) => {
      const plain = getComputedStyle(el).color;
      for (const marked of Array.from(el.querySelectorAll<HTMLElement>("[style*='rgb(1, 2, 3)'], font[color='#010203']"))) {
        const span = marked.tagName === "FONT" ? document.createElement("span") : marked;
        if (span !== marked) {
          span.append(...Array.from(marked.childNodes));
          marked.replaceWith(span);
        }
        if (gradient) {
          span.dataset.gradient = gradientData(gradient);
          span.style.backgroundImage = cssGradient(gradient);
          span.style.setProperty("-webkit-background-clip", "text");
          span.style.backgroundClip = "text";
          span.style.color = "transparent";
        } else {
          span.dataset.gradient = "none";
          span.style.backgroundImage = "none";
          span.style.color = plain;
        }
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
        <GradientField label="Selection gradient" value={at.gradient} startColor="#000000ff" onChange={paintGradient} />
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
      // On the canvas, no spellcheck: lettering is names and sound effects, and Chrome paints marked words
      // without their text stroke (the border), so flagged words lost theirs.
      spellCheck={overlay ? false : undefined}
      autoCorrect={overlay ? "off" : undefined}
      className={
        overlay
          ? cn("relative overflow-hidden whitespace-pre-wrap break-words outline-none", overlay.vertical ? "h-full" : "w-full")
          : cn(fieldChrome, "max-h-[120px] min-h-11 overflow-y-auto whitespace-pre-wrap break-words p-1 text-xs leading-4 shadow-[inset_0_0_0_1px_var(--brand-line)] focus:shadow-focus")
      }
      style={overlay ? { ...overlay.style, ...overlay.fillStyle } : undefined}
      onInput={syncUnderlay}
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
      <div className={cn("relative h-full w-full", overlay.boxClassName)}>
        <div className="absolute bottom-full left-0 mb-1.5 flex w-[264px] flex-col gap-1 rounded-[12px] bg-surface-over p-1 shadow-floating">{toolbar}</div>
        <div className={cn("relative", overlay.className)}>
          {overlay.underlayStyle && (
            <div
              ref={underlay}
              aria-hidden
              className="pointer-events-none absolute inset-0 overflow-hidden whitespace-pre-wrap break-words select-none"
              style={{ ...overlay.style, ...overlay.underlayStyle }}
            />
          )}
          {field}
        </div>
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
