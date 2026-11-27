// Fonts in the browser, for the on-canvas text editor and the font lists: the service serves the faces
// the renderer draws with (a font from the bundled catalog is not installed on the system, so CSS alone would fall
// back to another face). Faces load when text first uses them.

import { useEffect } from "react";

const registered = new Set<string>();

const VARIANTS: [weight: string, style: string][] = [
  ["400", "normal"],
  ["700", "normal"],
  ["400", "italic"],
  ["700", "italic"],
];

function register(family: string): void {
  const key = family.toLowerCase();
  if (!family || registered.has(key) || typeof FontFace === "undefined") return;
  registered.add(key);
  for (const [weight, style] of VARIANTS) {
    const url = `/api/v1/fonts/${encodeURIComponent(family)}/face?weight=${weight}&italic=${style === "italic"}`;
    // local() covers what the service cannot send (a face in a collection), which the system may still have.
    document.fonts.add(new FontFace(family, `url("${url}"), local("${family.replace(/"/g, "")}")`, { weight, style }));
  }
}

/** Makes these families available to CSS. */
export function useFontFaces(families: readonly (string | undefined)[]): void {
  const key = families.filter(Boolean).join("\n");
  useEffect(() => {
    for (const family of key.split("\n")) register(family);
  }, [key]);
}
