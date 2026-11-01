import { useId } from "react";

export type ColorizationMode = "off" | "colorize";

const INK = "#16171a";
const PAPER = "#ffffff";

/**
 * A small panel showing what each colorization mode does to the same art: left in black and white
 * with its screentone, or painted in color with the lines kept. Drawn here (not a screenshot) so
 * it reads in both themes.
 */
export function ColorizationPreview({ mode, className }: { mode: ColorizationMode; className?: string }) {
  const id = useId().replace(/:/g, "");
  const color = mode === "colorize";
  return (
    <svg viewBox="0 0 120 76" className={className} role="img" aria-label={color ? "The art painted in color" : "The art in black and white"}>
      <defs>
        <pattern id={`tone-${id}`} width="3" height="3" patternUnits="userSpaceOnUse">
          <circle cx="1.5" cy="1.5" r="0.62" fill={INK} opacity="0.5" />
        </pattern>
        <clipPath id={`panel-${id}`}>
          <rect x="0" y="0" width="120" height="76" />
        </clipPath>
      </defs>
      <g clipPath={`url(#panel-${id})`}>
        <rect width="120" height="76" fill={color ? "#cfe6f5" : PAPER} />
        {/* The hair as one shape behind its strands, a sleeve over the shoulder, then the lines. */}
        {color && <path d="M44 0 L80 0 C86 22 110 44 132 76 L78 76 C62 44 50 22 44 0 Z" fill="#b5533c" />}
        <path d="M0 76 L0 34 C18 30 34 36 50 52 L66 76 Z" fill={color ? "#4f7fbf" : `url(#tone-${id})`} />
        <path d="M0 34 C18 30 34 36 50 52 L66 76" stroke={INK} strokeWidth="1.4" fill="none" />
        {[44, 56, 68, 80].map((x, i) => (
          <path key={x} d={`M${x} 0 C${x + 6} 22 ${x + 18 + i * 4} 44 ${x + 34 + i * 6} 76`} stroke={INK} strokeWidth="0.9" fill="none" />
        ))}
      </g>
      <rect x="0.5" y="0.5" width="119" height="75" fill="none" stroke={INK} strokeWidth="1" />
    </svg>
  );
}
