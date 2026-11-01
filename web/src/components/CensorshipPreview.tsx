import { useId } from "react";

export type CensorshipMode = "off" | "detect" | "remove";

const INK = "#16171a";
const PAPER = "#ffffff";
// The colour the canvas marks censor bars in.
const MARK = "#e5484d";
const BARS = [24, 36, 48];

/**
 * A small panel showing what each censorship mode does to the same art: bars left as drawn, bars
 * marked in the colour the canvas shows them in, or gone with the strands running on under them.
 * Drawn here (not a screenshot) so it reads in both themes.
 */
export function CensorshipPreview({ mode, className }: { mode: CensorshipMode; className?: string }) {
  const id = useId().replace(/:/g, "");
  const label = { off: "Black bars over the art", detect: "The bars marked in red", remove: "The same art with the bars gone" }[mode];
  return (
    <svg viewBox="0 0 120 76" className={className} role="img" aria-label={label}>
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
        {mode !== "remove" && (
          <g transform="rotate(-8 80 38)">
            {BARS.map((y) => (
              <rect key={y} x="50" y={y} width="60" height="7.5" fill={INK} />
            ))}
          </g>
        )}
        {mode === "detect" && (
          <g transform="rotate(-8 80 38)" opacity="0.5">
            {BARS.map((y) => (
              <rect key={y} x="47" y={y - 2.5} width="66" height="12.5" rx="1.5" fill={MARK} />
            ))}
          </g>
        )}
      </g>
      <rect x="0.5" y="0.5" width="119" height="75" fill="none" stroke={INK} strokeWidth="1" />
    </svg>
  );
}
