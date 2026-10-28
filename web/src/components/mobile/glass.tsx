// The phone layer's building blocks: glass bar buttons, a large-title navigation bar that
// collapses on scroll, a floating tab bar, bottom sheets with detents, grouped lists and the
// touch gestures (edge swipe back, long press). Glass comes from the `glass*` utilities in
// index.css, built on the tokens in styles/theme.css.

import { Dialog as DialogPrimitive } from "@base-ui/react/dialog";
import { ChevronLeft, ChevronRight, Search, X, type LucideIcon } from "lucide-react";
import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ButtonHTMLAttributes,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from "react";

import { cn } from "cn";

import { ownsEdgeSwipe } from "./useMobile";

// ---- bar buttons --------------------------------------------------------------------------------

type GlassButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  label: string;
  /** Prominent: the accent, for the one primary action on a bar. */
  tone?: "glass" | "prominent";
  /** A circle holds an icon; a capsule holds a word or an icon and a word. */
  shape?: "circle" | "capsule";
};

/** A floating glass button: 44pt, a circle for an icon or a capsule for a word. */
export function GlassButton({ label, tone = "glass", shape = "circle", className, children, type = "button", ...props }: GlassButtonProps) {
  return (
    <button
      type={type}
      aria-label={label}
      title={label}
      className={cn(
        "glass inline-flex h-touch flex-none cursor-pointer items-center justify-center gap-1.5 rounded-full text-body-lg font-medium text-fg outline-none transition-[scale,opacity] duration-150 ease-settle select-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand active:scale-[0.94] disabled:pointer-events-none disabled:opacity-40 [-webkit-tap-highlight-color:transparent]",
        shape === "circle" ? "w-touch" : "px-4",
        tone === "prominent" && "bg-brand! text-on-brand",
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
}

/** Several bar actions sharing one glass capsule, as Undo and Redo do. */
export function GlassGroup({ children, className, label }: { children: ReactNode; className?: string; label: string }) {
  return (
    <div className={cn("glass flex h-touch flex-none items-center rounded-full px-0.5", className)} role="group" aria-label={label}>
      {children}
    </div>
  );
}

/** A plain icon button inside a GlassGroup or a glass toolbar. */
export function BarIcon({ label, active, className, children, type = "button", ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { label: string; active?: boolean }) {
  return (
    <button
      type={type}
      aria-label={label}
      title={label}
      className={cn(
        "grid size-touch flex-none cursor-pointer place-items-center rounded-full text-fg outline-none transition-[background-color,color] duration-100 ease-out select-none focus-visible:outline-2 focus-visible:-outline-offset-4 focus-visible:outline-brand active:bg-glass-selected disabled:pointer-events-none disabled:opacity-35 [-webkit-tap-highlight-color:transparent]",
        active && "bg-glass-selected text-brand",
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
}

// ---- navigation bar and screens -----------------------------------------------------------------

/** How a screen arrives: pushed from the right, popped back from the left, or in place (a tab, a swipe). */
export type Enter = "push" | "pop" | "none";

const ENTER: Record<Enter, string> = {
  push: "animate-in slide-in-from-right-1/3 fade-in-0 duration-300 ease-sheet",
  pop: "animate-in slide-in-from-left-1/4 fade-in-0 duration-300 ease-sheet",
  none: "",
};

// The large title's height: past it the compact title shows in the bar.
const LARGE_TITLE = 48;

/**
 * A phone screen: a glass navigation bar over the page, a large title that hands over to the
 * bar's compact title as the page scrolls, and an edge swipe back when there is somewhere to go
 * back to. The page itself scrolls (the window), so the status bar tap and rubber banding work.
 */
export function Screen({
  title,
  back,
  actions,
  large = true,
  accessory,
  enter = "none",
  tabBar = true,
  children,
  className,
}: {
  title: string;
  /** `swiped` is true when an edge swipe already slid the screen away. */
  back?: { label: string; onBack: (swiped?: boolean) => void };
  actions?: ReactNode;
  large?: boolean;
  /** Under the large title and pinned with it: a search field or a segmented control. */
  accessory?: ReactNode;
  enter?: Enter;
  /** Leaves room for the floating tab bar at the bottom. */
  tabBar?: boolean;
  children: ReactNode;
  className?: string;
}) {
  const [collapsed, setCollapsed] = useState(!large);
  const header = useRef<HTMLElement>(null);
  const body = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    window.scrollTo(0, 0);
  }, [title]);

  useEffect(() => {
    // One Screen instance can go from a large-title page to a small one (the same slot, a new title).
    if (!large) return setCollapsed(true);
    const update = () => setCollapsed(window.scrollY > LARGE_TITLE - 6);
    update();
    window.addEventListener("scroll", update, { passive: true });
    return () => window.removeEventListener("scroll", update);
  }, [large, title]);

  useEdgeSwipe(back?.onBack, [header, body]);

  return (
    <>
      <header ref={header} className={cn("fixed inset-x-0 top-0 z-30 pt-safe-top", ENTER[enter])}>
        <div aria-hidden className={cn("glass-edge pointer-events-none absolute inset-x-0 top-0 -bottom-5 transition-opacity duration-200", collapsed ? "opacity-100" : "opacity-0")} />
        <div className="relative flex h-[52px] items-center gap-2 pr-[max(16px,var(--safe-right))] pl-[max(16px,var(--safe-left))]">
          {back && (
            <GlassButton label={back.label} onClick={() => back.onBack()}>
              <ChevronLeft size={24} strokeWidth={2.25} className="-ml-0.5" />
            </GlassButton>
          )}
          <h1
            className={cn(
              "pointer-events-none absolute inset-x-[112px] truncate text-center text-body-lg font-semibold transition-opacity duration-150",
              collapsed ? "opacity-100" : "opacity-0",
            )}
            aria-hidden={large && !collapsed}
          >
            {title}
          </h1>
          <span className="flex-1" />
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </div>
      </header>
      <div
        ref={body}
        className={cn(
          "min-h-dvh bg-grouped pt-[calc(var(--safe-top)+52px)] pr-[max(16px,var(--safe-right))] pl-[max(16px,var(--safe-left))]",
          tabBar ? "pb-[calc(var(--safe-bottom)+104px)]" : "pb-[calc(var(--safe-bottom)+24px)]",
          ENTER[enter],
          className,
        )}
      >
        {large && (
          <h1 className="truncate pt-1 pb-2 text-large-title font-bold" aria-hidden={collapsed}>
            {title}
          </h1>
        )}
        {accessory && <div className="pb-3">{accessory}</div>}
        {children}
      </div>
    </>
  );
}

/**
 * Swipe from the left edge to go back: the screen follows the finger and either settles back or
 * slides off. Only where the page owns the edge (not iOS Safari in a tab, whose own swipe goes back).
 */
function useEdgeSwipe(onBack: ((swiped: boolean) => void) | undefined, targets: { current: HTMLElement | null }[]) {
  const latest = useRef(onBack);
  latest.current = onBack;
  const enabled = !!onBack;
  useEffect(() => {
    if (!enabled || !ownsEdgeSwipe()) return;
    let start: { x: number; y: number; t: number } | null = null;
    let tracking = false;
    const set = (transform: string, transition: string) => {
      for (const target of targets) {
        if (!target.current) continue;
        target.current.style.transition = transition;
        target.current.style.transform = transform;
      }
    };
    const onStart = (event: TouchEvent) => {
      const touch = event.touches[0];
      start = event.touches.length === 1 && touch.clientX < 24 ? { x: touch.clientX, y: touch.clientY, t: performance.now() } : null;
      tracking = false;
    };
    const onMove = (event: TouchEvent) => {
      if (!start) return;
      const touch = event.touches[0];
      const dx = touch.clientX - start.x;
      const dy = touch.clientY - start.y;
      if (!tracking) {
        if (Math.abs(dy) > 10 && Math.abs(dy) > Math.abs(dx)) {
          start = null;
          return;
        }
        if (dx < 10) return;
        tracking = true;
      }
      event.preventDefault();
      set(`translateX(${Math.max(0, dx)}px)`, "none");
    };
    const onEnd = (event: TouchEvent) => {
      if (!start || !tracking) {
        start = null;
        return;
      }
      const dx = event.changedTouches[0].clientX - start.x;
      const speed = dx / Math.max(1, performance.now() - start.t);
      start = null;
      tracking = false;
      if (dx > window.innerWidth * 0.35 || speed > 0.5) {
        set(`translateX(${window.innerWidth}px)`, "transform 220ms var(--ease-sheet)");
        window.setTimeout(() => {
          latest.current?.(true);
          // The same Screen may show the page gone back to: put it back in place.
          set("", "none");
        }, 200);
      } else {
        set("", "transform 260ms var(--ease-sheet)");
      }
    };
    window.addEventListener("touchstart", onStart, { passive: true });
    window.addEventListener("touchmove", onMove, { passive: false });
    window.addEventListener("touchend", onEnd);
    window.addEventListener("touchcancel", onEnd);
    return () => {
      window.removeEventListener("touchstart", onStart);
      window.removeEventListener("touchmove", onMove);
      window.removeEventListener("touchend", onEnd);
      window.removeEventListener("touchcancel", onEnd);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled]);
}

/** The search field under a large title. */
export function SearchField({ value, onChange, placeholder }: { value: string; onChange: (value: string) => void; placeholder: string }) {
  return (
    <label className="relative block">
      <Search size={17} className="pointer-events-none absolute top-1/2 left-2.5 -translate-y-1/2 text-fg-3" />
      <input
        type="search"
        enterKeyHint="search"
        aria-label={placeholder}
        placeholder={placeholder}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => event.key === "Escape" && onChange("")}
        className="h-9 w-full rounded-[10px] bg-grouped-fill pr-3 pl-9 text-body-lg text-fg outline-none placeholder:text-fg-3 focus-visible:shadow-focus [&::-webkit-search-cancel-button]:hidden"
      />
    </label>
  );
}

// ---- tab bar ------------------------------------------------------------------------------------

export interface TabItem<T extends string> {
  id: T;
  label: string;
  icon: LucideIcon;
}

/** The floating glass tab bar for the top-level sections; the current one sits in a lighter lens. */
export function TabBar<T extends string>({ items, active, onSelect }: { items: TabItem<T>[]; active: T; onSelect: (id: T) => void }) {
  return (
    <nav
      className="pointer-events-none fixed inset-x-0 bottom-0 z-30 flex justify-center pr-[max(16px,var(--safe-right))] pb-[max(calc(var(--safe-bottom)_-_10px),12px)] pl-[max(16px,var(--safe-left))]"
      aria-label="Sections"
    >
      <div className="glass pointer-events-auto flex w-full max-w-[440px] items-center rounded-full p-1" role="tablist">
        {items.map(({ id, label, icon: Icon }) => {
          const current = id === active;
          return (
            <button
              key={id}
              type="button"
              role="tab"
              aria-selected={current}
              className={cn(
                "flex h-[54px] min-w-0 flex-1 cursor-pointer flex-col items-center justify-center gap-0.5 rounded-full text-[10px] leading-3 font-medium outline-none transition-[background-color,color] duration-150 ease-out select-none focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-brand [-webkit-tap-highlight-color:transparent]",
                current ? "bg-glass-selected text-brand" : "text-fg active:bg-glass-selected",
              )}
              onClick={() => onSelect(id)}
            >
              <Icon size={22} strokeWidth={current ? 2.25 : 1.75} />
              <span className="max-w-full truncate px-1">{label}</span>
            </button>
          );
        })}
      </div>
    </nav>
  );
}

// ---- sheets -------------------------------------------------------------------------------------

export type Detent = "medium" | "large";

const DETENT_HEIGHT: Record<Detent, string> = {
  medium: "min(54dvh, calc(100dvh - var(--safe-top) - 24px))",
  large: "calc(100dvh - var(--safe-top) - 24px)",
};

/**
 * A glass bottom sheet. With detents it rests at medium or large height and its grabber drags it
 * between them; "fit" sizes it to its content. Dragging down past the lowest height dismisses it.
 * It is a modal dialog underneath (focus trap, Escape, outside tap), from Base UI.
 */
export function Sheet({
  open,
  onOpenChange,
  title,
  detents = ["medium", "large"],
  leading,
  trailing,
  children,
  contentClassName,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  detents?: Detent[] | "fit";
  leading?: ReactNode;
  /** Replaces the close button. */
  trailing?: ReactNode;
  children: ReactNode;
  contentClassName?: string;
}) {
  const fit = detents === "fit";
  const list = fit ? [] : detents;
  const [index, setIndex] = useState(0);
  const popup = useRef<HTMLDivElement>(null);
  const probes = useRef<(HTMLDivElement | null)[]>([]);
  const drag = useRef<{ y: number; t: number; height: number; heights: number[]; samples: { y: number; t: number }[] } | null>(null);

  useEffect(() => {
    if (open) setIndex(0);
  }, [open]);

  const place = (height: number | null, offset: number, animate: boolean) => {
    const el = popup.current;
    if (!el) return;
    el.style.transition = animate ? "" : "none";
    el.style.height = height === null ? "" : `${height}px`;
    el.style.translate = offset ? `0 ${offset}px` : "";
  };

  const onPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || (event.target as HTMLElement).closest("button, a, input")) return;
    const el = popup.current;
    if (!el) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    const heights = list.map((_, i) => probes.current[i]?.offsetHeight ?? 0);
    drag.current = { y: event.clientY, t: performance.now(), height: el.offsetHeight, heights, samples: [] };
  };

  const onPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const state = drag.current;
    if (!state) return;
    const dy = event.clientY - state.y;
    state.samples = [...state.samples.slice(-4), { y: event.clientY, t: performance.now() }];
    if (fit) return place(null, Math.max(0, dy) + Math.min(0, dy) * 0.15, false);
    const min = Math.min(...state.heights);
    const max = Math.max(...state.heights);
    let height = state.height - dy;
    if (height > max) height = max + (height - max) * 0.15;
    place(Math.max(min, height), Math.max(0, min - height), false);
  };

  const onPointerUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    const state = drag.current;
    drag.current = null;
    if (!state) return;
    const dy = event.clientY - state.y;
    const first = state.samples[0];
    const speed = first ? (event.clientY - first.y) / Math.max(1, performance.now() - first.t) : 0;
    if (fit) {
      place(null, 0, true);
      if (dy > 90 || speed > 0.6) onOpenChange(false);
      return;
    }
    const min = Math.min(...state.heights);
    const projected = state.height - dy - speed * 160;
    if (projected < min * 0.7 || (state.height - dy < min && speed > 0.4)) {
      place(null, 0, true);
      onOpenChange(false);
      return;
    }
    let nearest = 0;
    state.heights.forEach((h, i) => {
      if (Math.abs(h - projected) < Math.abs(state.heights[nearest] - projected)) nearest = i;
    });
    place(null, 0, true);
    setIndex(nearest);
  };

  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Backdrop className="fixed inset-0 z-50 bg-glass-scrim transition-opacity duration-300 ease-sheet data-ending-style:opacity-0 data-starting-style:opacity-0" />
        <DialogPrimitive.Popup
          ref={popup}
          initialFocus={popup}
          className={cn(
            "glass-thick fixed right-2 bottom-2 left-2 z-[51] flex max-h-[calc(100dvh_-_var(--safe-top)_-_24px)] flex-col overflow-hidden rounded-sheet text-fg outline-none transition-[translate,height] duration-[320ms] ease-sheet data-ending-style:translate-y-[110%] data-starting-style:translate-y-[110%]",
            "mx-auto max-w-[560px]",
          )}
          style={fit ? undefined : { height: DETENT_HEIGHT[list[Math.min(index, list.length - 1)]] }}
        >
          {list.map((detent, i) => (
            <div key={detent} ref={(el) => void (probes.current[i] = el)} aria-hidden className="pointer-events-none invisible absolute w-0" style={{ height: DETENT_HEIGHT[detent] }} />
          ))}
          <div className="flex-none cursor-grab touch-none select-none" onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp}>
            <div className="mx-auto mt-1.5 h-[5px] w-9 rounded-full bg-line-strong" aria-hidden />
            <div className="grid h-[52px] grid-cols-[1fr_auto_1fr] items-center gap-2 px-3">
              <div className="flex justify-start">{leading}</div>
              <DialogPrimitive.Title className="max-w-[60vw] truncate text-body-lg font-semibold">{title}</DialogPrimitive.Title>
              <div className="flex justify-end gap-2">
                {trailing ?? (
                  <DialogPrimitive.Close render={<GlassButton label="Close" className="size-9 w-9" />}>
                    <X size={18} strokeWidth={2.25} />
                  </DialogPrimitive.Close>
                )}
              </div>
            </div>
          </div>
          <div className={cn("min-h-0 flex-1 overflow-y-auto overscroll-contain px-4 pb-[calc(var(--safe-bottom)+12px)]", contentClassName)}>{children}</div>
        </DialogPrimitive.Popup>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}

// ---- grouped lists ------------------------------------------------------------------------------

/** An inset grouped section: an optional caption above, rounded cells, an optional footnote below. */
export function ListSection({ header, footer, children, className }: { header?: ReactNode; footer?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={cn("mt-7 first:mt-2", className)}>
      {header && <h2 className="px-4 pb-1.5 text-[13px] leading-[18px] font-normal tracking-[0.02em] text-fg-2 uppercase">{header}</h2>}
      <div className="overflow-hidden rounded-group bg-grouped-cell" role="list">
        {children}
      </div>
      {footer && <p className="px-4 pt-1.5 text-[13px] leading-[18px] text-fg-2">{footer}</p>}
    </section>
  );
}

/** The accent tile in front of a settings row. */
export function ListIcon({ icon: Icon, tone = "brand" }: { icon: LucideIcon; tone?: "brand" | "neutral" | "danger" }) {
  return (
    <span
      className={cn(
        "grid size-[30px] flex-none place-items-center rounded-[8px]",
        tone === "brand" ? "bg-brand text-on-brand" : tone === "danger" ? "bg-danger text-on-brand" : "bg-surface-fill text-icon",
      )}
      aria-hidden
    >
      <Icon size={18} strokeWidth={2} />
    </span>
  );
}

/**
 * One cell: leading media, a title and subtitle, a trailing detail or control, a chevron for rows
 * that push a screen. Separators run from the text, not the edge, and stop at the last row.
 */
export function ListRow({
  leading,
  title,
  subtitle,
  detail,
  trailing,
  chevron,
  destructive,
  onClick,
  onPointerDown,
  disabled,
  className,
  ...rest
}: {
  leading?: ReactNode;
  title: ReactNode;
  subtitle?: ReactNode;
  detail?: ReactNode;
  trailing?: ReactNode;
  chevron?: boolean;
  destructive?: boolean;
  onClick?: () => void;
  onPointerDown?: (event: ReactPointerEvent<HTMLElement>) => void;
  disabled?: boolean;
  className?: string;
} & Record<`data-${string}` | `aria-${string}`, unknown>) {
  const inner = (
    <>
      {leading}
      <span className="flex min-h-touch min-w-0 flex-1 items-center gap-2 py-2.5 pr-4 shadow-[inset_0_-0.5px_0_var(--border)] group-last/row:shadow-none">
        <span className="min-w-0 flex-1">
          <span className={cn("block truncate text-body-lg", destructive ? "text-danger" : "text-fg")}>{title}</span>
          {subtitle && <span className="block truncate text-[13px] leading-[18px] text-fg-2">{subtitle}</span>}
        </span>
        {detail && <span className="max-w-[45%] flex-none truncate text-body-lg text-fg-2">{detail}</span>}
        {trailing}
        {chevron && <ChevronRight size={18} strokeWidth={2.25} className="flex-none text-fg-3" />}
      </span>
    </>
  );
  const base = cn("group/row flex w-full items-center gap-3 pl-4 text-left", className);
  if (!onClick) {
    return (
      <div className={base} role="listitem" {...rest}>
        {inner}
      </div>
    );
  }
  return (
    <button
      type="button"
      className={cn(base, "cursor-pointer outline-none transition-[background-color] duration-100 ease-out select-none focus-visible:bg-hover active:bg-hover disabled:pointer-events-none disabled:opacity-40 [-webkit-tap-highlight-color:transparent] [-webkit-touch-callout:none]")}
      onClick={onClick}
      onPointerDown={onPointerDown}
      disabled={disabled}
      role="listitem"
      {...rest}
    >
      {inner}
    </button>
  );
}

// ---- gestures -----------------------------------------------------------------------------------

/**
 * Press and hold (or a right click) runs `onLongPress`; the click that ends a long press is
 * swallowed, so a held cover does not also open. Spread the result on the pressed element.
 */
export function useLongPress(onLongPress: () => void, delay = 450) {
  const timer = useRef(0);
  const origin = useRef<{ x: number; y: number } | null>(null);
  const fired = useRef(false);
  const cancel = () => {
    window.clearTimeout(timer.current);
    origin.current = null;
  };
  return {
    onPointerDown: (event: ReactPointerEvent<HTMLElement>) => {
      fired.current = false;
      if (event.pointerType === "mouse") return;
      origin.current = { x: event.clientX, y: event.clientY };
      timer.current = window.setTimeout(() => {
        fired.current = true;
        origin.current = null;
        navigator.vibrate?.(8);
        onLongPress();
      }, delay);
    },
    onPointerMove: (event: ReactPointerEvent<HTMLElement>) => {
      if (origin.current && Math.hypot(event.clientX - origin.current.x, event.clientY - origin.current.y) > 8) cancel();
    },
    onPointerUp: cancel,
    onPointerCancel: cancel,
    onContextMenu: (event: { preventDefault: () => void }) => {
      event.preventDefault();
      if (!fired.current) {
        fired.current = true;
        onLongPress();
      }
    },
    onClickCapture: (event: { preventDefault: () => void; stopPropagation: () => void }) => {
      if (!fired.current) return;
      fired.current = false;
      event.preventDefault();
      event.stopPropagation();
    },
  };
}

/** The rows of an action sheet: plain actions, then the destructive one, each a full-width 50pt cell. */
export function ActionList({ actions }: { actions: { label: string; icon?: LucideIcon; destructive?: boolean; onSelect: () => void; disabled?: boolean }[] }) {
  return (
    <div className="overflow-hidden rounded-group bg-grouped-cell" role="list">
      {actions.map(({ label, icon: Icon, destructive, onSelect, disabled }) => (
        <button
          key={label}
          type="button"
          role="listitem"
          disabled={disabled}
          className={cn(
            "flex min-h-[50px] w-full cursor-pointer items-center gap-3 px-4 text-left text-body-lg shadow-[inset_0_-0.5px_0_var(--border)] outline-none select-none last:shadow-none focus-visible:bg-hover active:bg-hover disabled:pointer-events-none disabled:opacity-40 [-webkit-tap-highlight-color:transparent]",
            destructive ? "text-danger" : "text-fg",
          )}
          onClick={onSelect}
        >
          <span className="min-w-0 flex-1 truncate">{label}</span>
          {Icon && <Icon size={20} className={destructive ? "text-danger" : "text-icon"} />}
        </button>
      ))}
    </div>
  );
}
