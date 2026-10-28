// Phone widths get the mobile layer (components/mobile). The breakpoint matches Tailwind's
// `max-md:` variant, so a component can switch structure here and restyle with `max-md:`.

import { useSyncExternalStore } from "react";

const QUERY = "(width < 48rem)";
const media = typeof window === "undefined" ? undefined : window.matchMedia(QUERY);

function subscribe(onChange: () => void): () => void {
  media?.addEventListener("change", onChange);
  return () => media?.removeEventListener("change", onChange);
}

export function useMobile(): boolean {
  return useSyncExternalStore(subscribe, () => media?.matches ?? false);
}

/** iOS Safari in a tab owns the left-edge swipe (history back); a Home Screen app leaves it to the page. */
export function ownsEdgeSwipe(): boolean {
  const ios = /iP(hone|ad|od)/.test(navigator.userAgent);
  const standalone = window.matchMedia("(display-mode: standalone)").matches || (navigator as { standalone?: boolean }).standalone === true;
  return !ios || standalone;
}
