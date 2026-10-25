import { useId } from "react";

export type SoundEffectMode = "off" | "typeset" | "redraw";

const INK = "#16171a";
const PAPER = "#ffffff";

/** Hand-lettered sound effect: ink letters with a white halo and an ink edge, as manga letterers draw them. */
function Lettering({ text, x, y, size, rotate, font, italic = false }: { text: string; x: number; y: number; size: number; rotate: number; font: string; italic?: boolean }) {
  const common = {
    x,
    y,
    fontSize: size,
    fontFamily: font,
    fontWeight: 900,
    fontStyle: italic ? "italic" : "normal",
    textAnchor: "middle" as const,
    transform: `rotate(${rotate} ${x} ${y})`,
    strokeLinejoin: "round" as const,
  };
  return (
    <g>
      <text {...common} fill={INK} stroke={INK} strokeWidth={size * 0.36}>{text}</text>
      <text {...common} fill={INK} stroke={PAPER} strokeWidth={size * 0.2}>{text}</text>
      <text {...common} fill={INK}>{text}</text>
    </g>
  );
}

/**
 * A small panel showing what each sound effects mode does to the same effect: the original
 * Japanese lettering, the translation typeset over the erased art, or the translation redrawn in
 * the original's style. Drawn here (not a screenshot) so it reads in both themes.
 */
export function SoundEffectPreview({ mode, className }: { mode: SoundEffectMode; className?: string }) {
  const id = useId().replace(/:/g, "");
  const japanese = '"Noto Sans JP", "Yu Gothic UI", "Hiragino Sans", "Meiryo", sans-serif';
  const display = '"Arial Black", "Helvetica Neue", Impact, sans-serif';
  const comic = 'CCWildWords, "Comic Neue", "Comic Sans MS", sans-serif';
  return (
    <svg viewBox="0 0 120 76" className={className} role="img" aria-label={{ off: "ドン left in the art", typeset: "BOOM typeset in a comic font", redraw: "BOOM redrawn in the original style" }[mode]}>
      <defs>
        <pattern id={`tone-${id}`} width="3" height="3" patternUnits="userSpaceOnUse">
          <circle cx="1.5" cy="1.5" r="0.62" fill={INK} opacity="0.55" />
        </pattern>
        <clipPath id={`panel-${id}`}>
          <rect x="0" y="0" width="120" height="76" />
        </clipPath>
      </defs>
      <g clipPath={`url(#panel-${id})`}>
        <rect width="120" height="76" fill={PAPER} />
        {/* A door slamming: the wall in screentone, the door edge, and speed lines. */}
        <rect x="0" y="0" width="34" height="76" fill={`url(#tone-${id})`} />
        <path d="M34 0 L34 76 M40 6 L40 76" stroke={INK} strokeWidth="1.6" fill="none" />
        <path d="M40 6 L52 0" stroke={INK} strokeWidth="1.2" fill="none" />
        {[10, 22, 34, 46, 58, 70].map((y) => (
          <path key={y} d={`M${58 + (y % 3) * 6} ${y} L120 ${y + (y - 38) * 0.15}`} stroke={INK} strokeWidth="0.55" opacity="0.65" />
        ))}
        {mode === "off" && <Lettering text="ドン" x={82} y={50} size={28} rotate={-10} font={japanese} />}
        {mode === "typeset" && (
          <text x={82} y={44} fontSize={13} fontFamily={comic} fontWeight={700} textAnchor="middle" fill={INK} stroke={PAPER} strokeWidth={4} paintOrder="stroke">
            BOOM!
          </text>
        )}
        {mode === "redraw" && <Lettering text="BOOM" x={82} y={49} size={20} rotate={-10} font={display} italic />}
      </g>
      <rect x="0.5" y="0.5" width="119" height="75" fill="none" stroke={INK} strokeWidth="1" />
    </svg>
  );
}
