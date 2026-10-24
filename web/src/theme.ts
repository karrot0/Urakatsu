// Color themes: a neutral temperature and an accent, on top of light/dark. styles/theme.css
// reads them from <html data-neutral="…" data-accent="…">; the defaults are the original look.

export const NEUTRALS = [
  { id: "warm", label: "Warm", hint: "Paper-like greys" },
  { id: "cool", label: "Cool", hint: "Blue-shifted greys" },
  { id: "gray", label: "Gray", hint: "No tint" },
] as const;

// Swatch colors approximate --brand in light mode for each accent (oklch(0.58 c h)).
export const ACCENTS = [
  { id: "blue", label: "Blue", swatch: "oklch(0.58 0.17 250)" },
  { id: "violet", label: "Violet", swatch: "oklch(0.58 0.17 295)" },
  { id: "green", label: "Green", swatch: "oklch(0.58 0.145 155)" },
  { id: "teal", label: "Teal", swatch: "oklch(0.58 0.136 195)" },
  { id: "orange", label: "Orange", swatch: "oklch(0.58 0.162 50)" },
  { id: "rose", label: "Rose", swatch: "oklch(0.58 0.17 10)" },
] as const;

export type Neutral = (typeof NEUTRALS)[number]["id"];
export type Accent = (typeof ACCENTS)[number]["id"];

export interface Palette {
  neutral: Neutral;
  accent: Accent;
}

export const DEFAULT_PALETTE: Palette = { neutral: "warm", accent: "blue" };
const PALETTE_KEY = "urakatsu.palette";

export function loadPalette(): Palette {
  try {
    const stored = JSON.parse(localStorage.getItem(PALETTE_KEY) ?? "{}");
    return {
      neutral: NEUTRALS.some((n) => n.id === stored.neutral) ? stored.neutral : DEFAULT_PALETTE.neutral,
      accent: ACCENTS.some((a) => a.id === stored.accent) ? stored.accent : DEFAULT_PALETTE.accent,
    };
  } catch {
    return DEFAULT_PALETTE;
  }
}

export function applyPalette(palette: Palette): void {
  const root = document.documentElement;
  root.dataset.neutral = palette.neutral;
  root.dataset.accent = palette.accent;
  try {
    localStorage.setItem(PALETTE_KEY, JSON.stringify(palette));
  } catch {
    /* storage unavailable */
  }
}
