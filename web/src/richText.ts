// Rich text in the translation editor: a text with its styles <-> the editor's HTML.
// Offsets are code points, as the server counts them (JS strings count UTF-16 units), so
// characters outside the BMP keep their formatting.

import type { Gradient, TextStyle } from "./gen/urakatsu/v1/urakatsu_pb";

export type RichStyle = Omit<TextStyle, "$typeName" | "$unknown">;
type Format = Pick<RichStyle, "bold" | "italic" | "underline" | "strikethrough" | "color" | "font" | "scale" | "gradient">;

const PLAIN: Format = { bold: false, italic: false, underline: false, strikethrough: false, color: undefined, font: "", scale: undefined, gradient: undefined };

type GradientShape = { kind: string; angle: number; stops: readonly { offset: number; color: string }[] };

/** A gradient's identity, for comparing formats ("" for none; fewer than two stops is none). */
const gradientKey = (gradient: GradientShape | undefined) =>
  gradient && gradient.stops.length >= 2 ? JSON.stringify([gradient.kind, gradient.angle, gradient.stops.map((s) => [s.offset, s.color])]) : "";

/** A gradient as CSS (Photoshop's angle: 90 runs bottom to top, CSS's 0deg does). */
export function cssGradient(gradient: GradientShape): string {
  const stops = [...gradient.stops].sort((a, b) => a.offset - b.offset).map((s) => `${s.color} ${Math.round(s.offset * 100)}%`).join(", ");
  return gradient.kind === "radial" ? `radial-gradient(circle farthest-side, ${stops})` : `linear-gradient(${90 - gradient.angle}deg, ${stops})`;
}

/** A gradient message from its JSON form (the editor keeps a run's gradient in `data-gradient`). */
export function gradientFrom(shape: GradientShape): Gradient {
  return {
    $typeName: "urakatsu.v1.Gradient",
    kind: shape.kind,
    angle: shape.angle,
    stops: shape.stops.map((s) => ({ $typeName: "urakatsu.v1.GradientStop", offset: s.offset, color: s.color })),
  };
}

/** What `data-gradient` holds for a gradient. */
export const gradientData = (gradient: GradientShape) => JSON.stringify({ kind: gradient.kind, angle: gradient.angle, stops: gradient.stops.map((s) => ({ offset: s.offset, color: s.color })) });

/** A size relative to the layer's; 100% is none. */
const scaleOf = (format: Format | undefined) => (format?.scale && Math.abs(format.scale - 1) > 1e-6 ? Math.round(format.scale * 1000) / 1000 : undefined);

const sameFormat = (a: Format | undefined, b: Format | undefined) =>
  (a ?? PLAIN).bold === (b ?? PLAIN).bold &&
  (a ?? PLAIN).italic === (b ?? PLAIN).italic &&
  (a ?? PLAIN).underline === (b ?? PLAIN).underline &&
  (a ?? PLAIN).strikethrough === (b ?? PLAIN).strikethrough &&
  ((a ?? PLAIN).color ?? "") === ((b ?? PLAIN).color ?? "") &&
  ((a ?? PLAIN).font ?? "") === ((b ?? PLAIN).font ?? "") &&
  scaleOf(a) === scaleOf(b) &&
  gradientKey(a?.gradient) === gradientKey(b?.gradient);

const escapeHtml = (text: string) => text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const escapeAttribute = (text: string) => escapeHtml(text).replace(/"/g, "&quot;");

function css(format: Format): string {
  const decorations = [format.underline && "underline", format.strikethrough && "line-through"].filter(Boolean).join(" ");
  return [
    format.bold && "font-weight:700",
    format.italic && "font-style:italic",
    decorations && `text-decoration-line:${decorations}`,
    // A gradient fills the characters (the text itself is see-through); otherwise their colour.
    gradientKey(format.gradient) && `background-image:${cssGradient(format.gradient!)};-webkit-background-clip:text;background-clip:text;color:transparent`,
    !gradientKey(format.gradient) && format.color && `color:${format.color}`,
    format.font && `font-family:"${format.font.replace(/"/g, "")}"`,
    scaleOf(format) && `font-size:${scaleOf(format)}em`,
  ]
    .filter(Boolean)
    .join(";");
}

// A line break the renderer chose (the text has none there): shown, but not part of the text.
const SOFT_BREAK = '<br data-soft="">';

/** The editor's HTML for `text` formatted by `styles`, with soft line breaks before the characters at
 * `softBreaks` (where the render's lines start). */
export function toHtml(text: string, styles: readonly RichStyle[], softBreaks: readonly number[] = []): string {
  const chars = Array.from(text);
  const formats: (Format | undefined)[] = new Array(chars.length);
  for (const style of styles) for (let i = Math.max(0, style.start); i < Math.min(style.end, chars.length); i++) formats[i] = style;
  // A line that starts after a newline already breaks.
  const soft = new Set(softBreaks.filter((i) => i > 0 && i < chars.length && chars[i - 1] !== "\n"));
  let html = "";
  for (let i = 0; i < chars.length; ) {
    let j = i + 1;
    while (j < chars.length && sameFormat(formats[j], formats[i]) && !soft.has(j)) j++;
    const piece = escapeHtml(chars.slice(i, j).join("")).replace(/\n/g, "<br>");
    const format = formats[i];
    if (soft.has(i)) html += SOFT_BREAK;
    const data = format && gradientKey(format.gradient) ? ` data-gradient="${escapeAttribute(gradientData(format.gradient!))}"` : "";
    html += format && !sameFormat(format, PLAIN) ? `<span style="${css(format)}"${data}>${piece}</span>` : piece;
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
  let gradientDone = false;
  for (let node: Element | null = element; node && node !== root; node = node.parentElement) {
    if (node instanceof HTMLElement) {
      // The nearest gradient mark wins; "none" is a selection taken out of a gradient around it.
      if (!gradientDone && node.dataset.gradient !== undefined) {
        gradientDone = true;
        if (node.dataset.gradient !== "none") {
          try {
            format.gradient = gradientFrom(JSON.parse(node.dataset.gradient));
          } catch {
            // a damaged mark is no gradient
          }
        }
      }
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
        if (child.hasAttribute("data-soft")) continue; // the renderer's line break, not a character
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
  if (last instanceof HTMLBRElement && !last.hasAttribute("data-soft") && chars[chars.length - 1] === "\n") {
    chars.pop();
    formats.pop();
  }
  const styles: RichStyle[] = [];
  for (let i = 0; i < chars.length; ) {
    let j = i + 1;
    while (j < chars.length && sameFormat(formats[j], formats[i])) j++;
    if (!sameFormat(formats[i], PLAIN))
      styles.push({ start: i, end: j, ...formats[i], font: formats[i].font ?? "", scale: scaleOf(formats[i]), gradient: gradientKey(formats[i].gradient) ? formats[i].gradient : undefined });
    i = j;
  }
  return { text: chars.join(""), styles };
}

/** Whether two style lists format the same characters the same way. */
export function sameStyles(a: readonly RichStyle[], b: readonly RichStyle[]): boolean {
  return a.length === b.length && a.every((style, i) => style.start === b[i].start && style.end === b[i].end && sameFormat(style, b[i]));
}
