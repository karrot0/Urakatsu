// Rich text in the translation editor: a text with its styles <-> the editor's HTML.
// Offsets are code points, as the server counts them (JS strings count UTF-16 units), so
// characters outside the BMP keep their formatting.

import type { TextStyle } from "./gen/urakatsu/v1/urakatsu_pb";

export type RichStyle = Omit<TextStyle, "$typeName" | "$unknown">;
type Format = Pick<RichStyle, "bold" | "italic" | "underline" | "strikethrough" | "color" | "font" | "scale">;

const PLAIN: Format = { bold: false, italic: false, underline: false, strikethrough: false, color: undefined, font: "", scale: undefined };

/** A size relative to the layer's; 100% is none. */
const scaleOf = (format: Format | undefined) => (format?.scale && Math.abs(format.scale - 1) > 1e-6 ? Math.round(format.scale * 1000) / 1000 : undefined);

const sameFormat = (a: Format | undefined, b: Format | undefined) =>
  (a ?? PLAIN).bold === (b ?? PLAIN).bold &&
  (a ?? PLAIN).italic === (b ?? PLAIN).italic &&
  (a ?? PLAIN).underline === (b ?? PLAIN).underline &&
  (a ?? PLAIN).strikethrough === (b ?? PLAIN).strikethrough &&
  ((a ?? PLAIN).color ?? "") === ((b ?? PLAIN).color ?? "") &&
  ((a ?? PLAIN).font ?? "") === ((b ?? PLAIN).font ?? "") &&
  scaleOf(a) === scaleOf(b);

const escapeHtml = (text: string) => text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function css(format: Format): string {
  const decorations = [format.underline && "underline", format.strikethrough && "line-through"].filter(Boolean).join(" ");
  return [
    format.bold && "font-weight:700",
    format.italic && "font-style:italic",
    decorations && `text-decoration-line:${decorations}`,
    format.color && `color:${format.color}`,
    format.font && `font-family:"${format.font.replace(/"/g, "")}"`,
    scaleOf(format) && `font-size:${scaleOf(format)}em`,
  ]
    .filter(Boolean)
    .join(";");
}

/** The editor's HTML for `text` formatted by `styles`. */
export function toHtml(text: string, styles: readonly RichStyle[]): string {
  const chars = Array.from(text);
  const formats: (Format | undefined)[] = new Array(chars.length);
  for (const style of styles) for (let i = Math.max(0, style.start); i < Math.min(style.end, chars.length); i++) formats[i] = style;
  let html = "";
  for (let i = 0; i < chars.length; ) {
    let j = i + 1;
    while (j < chars.length && sameFormat(formats[j], formats[i])) j++;
    const piece = escapeHtml(chars.slice(i, j).join("")).replace(/\n/g, "<br>");
    const format = formats[i];
    html += format && !sameFormat(format, PLAIN) ? `<span style="${css(format)}">${piece}</span>` : piece;
    i = j;
  }
  // A last empty line only shows with a second <br>; fromDom drops it again.
  return text.endsWith("\n") ? `${html}<br>` : html;
}

/** "#rrggbbaa" for a CSS colour ("rgb(…)", "rgba(…)"). */
function hexOf(color: string): string | undefined {
  const parts = color.match(/[\d.]+/g)?.map(Number);
  if (!parts || parts.length < 3) return undefined;
  const alpha = parts.length > 3 ? Math.round(parts[3] * 255) : 255;
  return `#${[parts[0], parts[1], parts[2], alpha].map((v) => Math.round(v).toString(16).padStart(2, "0")).join("")}`;
}

/** The formatting a text node shows. Decorations do not inherit in CSS, so they are read up to the root;
 * family and size count only where the editor set them (em sizes nest, so they multiply). */
function formatOf(element: Element, root: HTMLElement): Format {
  const computed = getComputedStyle(element);
  const format: Format = { ...PLAIN, bold: Number(computed.fontWeight) >= 600, italic: computed.fontStyle !== "normal" };
  let scale = 1;
  for (let node: Element | null = element; node && node !== root; node = node.parentElement) {
    if (node instanceof HTMLElement) {
      const family = node.style.fontFamily === "inherit" ? "" : node.style.fontFamily || node.getAttribute("face") || "";
      if (!format.font && family) format.font = family.split(",")[0].trim().replace(/^["']|["']$/g, "");
      const size = node.style.fontSize.match(/^([\d.]+)em$/);
      if (size) scale *= Number(size[1]);
    }
    const line = getComputedStyle(node).textDecorationLine;
    if (line.includes("underline")) format.underline = true;
    if (line.includes("line-through")) format.strikethrough = true;
    // Only a colour set in the editor (style or <font color>) is formatting; the rest is the theme's.
    if (format.color === undefined && node instanceof HTMLElement && (node.style.color || node.getAttribute("color"))) format.color = hexOf(getComputedStyle(node).color);
  }
  format.scale = scale === 1 ? undefined : scale;
  return format;
}

/** The text and styles the editor shows. */
export function fromDom(root: HTMLElement): { text: string; styles: RichStyle[] } {
  const chars: string[] = [];
  const formats: Format[] = [];
  const walk = (node: Node) => {
    for (const child of Array.from(node.childNodes)) {
      if (child.nodeType === Node.TEXT_NODE) {
        const format = formatOf(child.parentElement ?? root, root);
        for (const char of Array.from(child.textContent ?? "")) {
          chars.push(char);
          formats.push(format);
        }
      } else if (child instanceof HTMLBRElement) {
        chars.push("\n");
        formats.push(PLAIN);
      } else if (child instanceof HTMLElement) {
        // Chrome wraps a new line in a <div> when Enter is not intercepted.
        if ((child.tagName === "DIV" || child.tagName === "P") && chars.length > 0 && chars[chars.length - 1] !== "\n") {
          chars.push("\n");
          formats.push(PLAIN);
        }
        walk(child);
      }
    }
  };
  walk(root);
  // The <br> that holds an empty last line open is not a character.
  let last: Node | null = root.lastChild;
  while (last && last.lastChild) last = last.lastChild;
  if (last instanceof HTMLBRElement && chars[chars.length - 1] === "\n") {
    chars.pop();
    formats.pop();
  }
  const styles: RichStyle[] = [];
  for (let i = 0; i < chars.length; ) {
    let j = i + 1;
    while (j < chars.length && sameFormat(formats[j], formats[i])) j++;
    if (!sameFormat(formats[i], PLAIN)) styles.push({ start: i, end: j, ...formats[i], font: formats[i].font ?? "", scale: scaleOf(formats[i]) });
    i = j;
  }
  return { text: chars.join(""), styles };
}

/** Whether two style lists format the same characters the same way. */
export function sameStyles(a: readonly RichStyle[], b: readonly RichStyle[]): boolean {
  return a.length === b.length && a.every((style, i) => style.start === b[i].start && style.end === b[i].end && sameFormat(style, b[i]));
}
