// Lettering styles in the browser: the kinds of face a line is lettered in, the font each is set
// in (Settings > Typesetting), and the face a block takes from its style. Mirrors
// urakatsu/core/styles.py, which the renderer draws with.

import { useMemo } from "react";

import type { TextBlock } from "./gen/urakatsu/v1/urakatsu_pb";
import { useWorkspace } from "./state/workspace";

export interface StyleFace {
  family?: string;
  weight?: number;
  italic?: boolean;
}
export type StyleTable = Record<string, Record<string, StyleFace>>;

/** The styles, in the order the font reader's table lists them. */
export const STYLES = [
  "antique",
  "mincho",
  "fangsong",
  "gothic",
  "rounded",
  "kaisho",
  "gyokai",
  "gyosho",
  "sosho",
  "reisho",
  "tensho",
  "kointai",
  "weibei",
  "textbook",
  "brush",
  "handwriting",
  "pop",
  "design",
  "decorative",
] as const;
export const DEFAULT_STYLE = "antique";

const NAMES: Record<string, string> = {
  antique: "Antique",
  mincho: "Mincho",
  fangsong: "Fangsong",
  gothic: "Gothic",
  rounded: "Rounded",
  kaisho: "Kaisho",
  gyokai: "Gyokai",
  gyosho: "Gyosho",
  sosho: "Sosho",
  reisho: "Reisho",
  tensho: "Tensho",
  kointai: "Kointai",
  weibei: "Weibei",
  textbook: "Textbook",
  brush: "Brush",
  handwriting: "Handwriting",
  pop: "Pop",
  design: "Design",
  decorative: "Decorative",
};

/** What each of the styles a page usually has is used for; these are listed first. */
export const STYLE_USES: Record<string, string> = {
  antique: "Dialogue",
  mincho: "Narration",
  gothic: "Shouts",
  rounded: "Thoughts",
  handwriting: "Handwriting",
  brush: "Brush",
  design: "Special voices",
  pop: "Side text",
  kointai: "Horror",
  decorative: "Decorative",
};
export const COMMON_STYLES = Object.keys(STYLE_USES);

/** "Dialogue (Antique)", or the kernel's name for a style with no usual use. */
export function styleLabel(style: string): string {
  const name = NAMES[style] ?? style;
  return STYLE_USES[style] && STYLE_USES[style] !== name ? `${STYLE_USES[style]} (${name})` : name;
}

export const SCRIPTS: { value: string; label: string }[] = [
  { value: "Latn", label: "Latin" },
  { value: "Cyrl", label: "Cyrillic" },
  { value: "Grek", label: "Greek" },
  { value: "Hans", label: "Chinese (Simplified)" },
  { value: "Hant", label: "Chinese (Traditional)" },
  { value: "Jpan", label: "Japanese" },
];

export function parseStyles(json: string | undefined): StyleTable {
  if (!json) return {};
  try {
    const value = JSON.parse(json);
    return value && typeof value === "object" ? (value as StyleTable) : {};
  } catch {
    return {};
  }
}

/** The face `style` sets for `script`: its own, with what that leaves out from antique's. */
export function styleFace(styles: StyleTable, style: string | undefined, script: string): StyleFace {
  const antique = styles[DEFAULT_STYLE]?.[script] ?? {};
  const own = styles[style || DEFAULT_STYLE]?.[script];
  if (!own) return antique;
  return { family: own.family || antique.family, weight: own.weight ?? antique.weight, italic: own.italic ?? antique.italic };
}

const CYRILLIC = new Set(["ru", "uk", "be", "bg", "sr", "mk", "kk", "ky", "mn", "tg"]);
const TRADITIONAL = ["zh-tw", "zh-hk", "zh-mo", "zh-hant"];

/** The script a text is set in: its language's, else what its characters imply, else Latin. */
export function scriptOf(text: string, language?: string): string {
  const tag = (language ?? "").trim().toLowerCase().replace(/_/g, "-");
  const primary = tag.split("-")[0];
  if (tag) {
    if (TRADITIONAL.some((value) => tag === value || tag.startsWith(`${value}-`))) return "Hant";
    if (primary === "zh") return "Hans";
    if (primary === "ja") return "Jpan";
    if (CYRILLIC.has(primary)) return "Cyrl";
    if (primary === "el") return "Grek";
  }
  let han = false;
  for (const character of text) {
    const code = character.codePointAt(0) ?? 0;
    if (code >= 0x3040 && code <= 0x30ff) return "Jpan";
    if ((code >= 0x4e00 && code <= 0x9fff) || (code >= 0x3400 && code <= 0x4dbf)) han = true;
    else if (code >= 0x0400 && code <= 0x04ff) return "Cyrl";
    else if (code >= 0x0370 && code <= 0x03ff) return "Grek";
  }
  return han ? "Hans" : "Latn";
}

/** The style table in Settings > Typesetting. */
export function useStyles(): StyleTable {
  const json = useWorkspace().config?.stylesJson;
  return useMemo(() => parseStyles(json), [json]);
}

/** The style a block is set in: the one chosen for it, else the one matched from the original. */
export function blockStyle(block: TextBlock | undefined): string {
  return block?.typography?.style || block?.letteringStyle || "";
}

/** What a block's style sets where the block sets none: its font, weight and slant. Empty for a
 *  block without a style, which is set in the default fonts. */
export function blockFace(styles: StyleTable, block: TextBlock | undefined): StyleFace {
  const style = blockStyle(block);
  if (!block || !style) return {};
  const face = styleFace(styles, style, scriptOf(block.translation, block.translationLanguage));
  // The weight the lettering stage set (bold for a source heavier than the book's dialogue).
  return { ...face, weight: block.letteringWeight ?? face.weight };
}
