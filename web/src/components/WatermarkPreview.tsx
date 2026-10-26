import { useId } from "react";

export type WatermarkMode = "off" | "remove";

const INK = "#16171a";
const PAPER = "#ffffff";

/**
 * A small panel showing what each watermarks mode does to the same page: a translucent logo left
 * over the art, or lifted off with the art under it (the hair strands and the sleeve line run
 * on through where the logo was). Drawn here (not a screenshot) so it reads in both themes.
 */
export function WatermarkPreview({ mode, className }: { mode: WatermarkMode; className?: string }) {
  const id = useId().replace(/:/g, "");
  const display = '"Arial Black", "Helvetica Neue", Impact, sans-serif';
  return (
    <svg viewBox="0 0 120 76" className={className} role="img" aria-label={mode === "off" ? "A translucent logo over the art" : "The same art with the logo removed"}>
      <defs>
        <pattern id={`tone-${id}`} width="3" height="3" patternUnits="userSpaceOnUse">
          <circle cx="1.5" cy="1.5" r="0.62" fill={INK} opacity="0.5" />
        </pattern>
        <clipPath id={`panel-${id}`}>
          <rect x="0" y="0" width="120" height="76" />
        </clipPath>
      </defs>
      <g clipPath={`url(#panel-${id})`}>
        <rect width="120" height="76" fill={PAPER} />
        {/* A shoulder in screentone, its sleeve line, and hair strands falling across the corner. */}
        <path d="M0 76 L0 34 C18 30 34 36 50 52 L66 76 Z" fill={`url(#tone-${id})`} />
        <path d="M0 34 C18 30 34 36 50 52 L66 76" stroke={INK} strokeWidth="1.4" fill="none" />
        {[44, 56, 68, 80].map((x, i) => (
          <path key={x} d={`M${x} 0 C${x + 6} 22 ${x + 18 + i * 4} 44 ${x + 34 + i * 6} 76`} stroke={INK} strokeWidth="0.9" fill="none" />
        ))}
        {mode === "off" && (
          <g opacity="0.7">
            <rect x="60" y="42" width="56" height="21" rx="2" fill="#e4e4e8" stroke="#7c7c86" strokeWidth="1" />
            <text x="88" y="57.5" fontSize="13" fontFamily={display} fontWeight={900} textAnchor="middle" fill={PAPER} stroke="#5c5c66" strokeWidth="1.6" paintOrder="stroke">
              LOGO
            </text>
          </g>
        )}
      </g>
      <rect x="0.5" y="0.5" width="119" height="75" fill="none" stroke={INK} strokeWidth="1" />
    </svg>
  );
}
