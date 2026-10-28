// The page canvas: source or render image, raster layers composited on <canvas> elements,
// the remove-tool mask, and an SVG overlay for blocks, regions and gestures.
//
// Painting happens client-side on the layer's own canvas so strokes appear immediately;
// when a stroke ends the whole layer is sent to the server as one PNG (one undoable edit).
//
// Generative fill (Photoshop-style): the fill tool drags a rectangle or lasso selection, and a
// floating prompt under it sends the selection as a mask with the prompt; the fill model paints
// it into the Cleanup layer. The selection stays, so Generate again tries another result.
//
// Webtoon reader: in a webtoon project the chapter's pages are laid edge to edge as one strip,
// in fixed strip coordinates. The current page is drawn with its layers and overlays at its place
// in the strip, the others as plain images; scrolling makes the page under the middle of the view
// the current one, so nothing moves and every page is edited where it is read.

import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from "react";

import { CornerDownLeft, GalleryVertical, LoaderCircle, WandSparkles, X } from "lucide-react";

import { cn } from "cn";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { catalog, type InstalledModel } from "../api/catalog";
import { assetHref, assetUrl, assetVersion } from "../api/client";
import type { Page, RasterLayer, TextBlock } from "../gen/urakatsu/v1/urakatsu_pb";
import { matches } from "../shortcuts";
import { BRUSH_SIZES, DEFAULT_SHORTCUTS, MIN_BRUSH, NUDGES, TOOLS, chapterOf, currentPage, selectedPages, useWorkspace, type FillShape, type ShortcutAction, type Tool } from "../state/workspace";
import { Slider } from "./primitives";
import { useLayerFonts } from "../fontFaces";
import { cssGradient } from "../richText";
import { useDefaultFont } from "./FontPicker";
import { RichTextField } from "./RichTextField";
import { ToolBelt, isBrushTool, maxBrush } from "./ToolBelt";

type Pt = { x: number; y: number };
type Rect = { x: number; y: number; width: number; height: number };
type Gesture =
  // A move is not re-rendered per pointer move: the block is drawn on its own layer, shifted by a CSS transform.
  | { kind: "move"; id: string; start: Pt; origin: Pt[]; points: Pt[] }
  // Free transform, Photoshop style: `box` is the frame's oriented box when the drag started.
  | { kind: "transform"; id: string; mode: "resize" | "rotate"; handle: number; box: OBox; origin: Pt[]; start: Pt; points: Pt[]; readout: string }
  | { kind: "text"; start: Pt; rect: Rect }
  | { kind: "pan"; client: Pt; scroll: Pt }
  // A stroke is not re-rendered per pointer move either: segments go straight onto `target`.
  | { kind: "stroke"; tool: "draw" | "eraser" | "remove"; target: HTMLCanvasElement; layer?: RasterLayer }
  | { kind: "select"; shape: FillShape; start: Pt; points: Pt[] }
  | { kind: "compare"; bounds: Rect };
// A generative fill selection: a closed polygon in page pixels.
type Selection = { pageId: string; points: Pt[] };
// Before/after comparison of a selection's fills: the artwork under it before the first fill
// (`url`, cropped to `bounds`), shown left of the divider at `position` (0 to 1) once a fill succeeds.
type Comparison = { selection: Selection; bounds: Rect; url: string; ready: boolean; position: number };

const NUDGE_DIRECTIONS: Record<(typeof NUDGES)[number], [number, number]> = { nudge_left: [-1, 0], nudge_right: [1, 0], nudge_up: [0, -1], nudge_down: [0, 1] };

// Free transform handles in the box's own axes, clockwise from the top-left corner: u and v are
// -1, 0 or 1 across the width and height, so 0 marks an edge's midpoint.
const HANDLES = [
  [-1, -1],
  [0, -1],
  [1, -1],
  [1, 0],
  [1, 1],
  [0, 1],
  [-1, 1],
  [-1, 0],
] as const;
const CORNERS = [0, 2, 4, 6];
// Handle marks in screen pixels: square corners, bars along the edges, and the area that grabs them.
const HANDLE_CORNER = 8;
const HANDLE_BAR = 14;
const HANDLE_THICK = 5;
const HANDLE_HIT = 16;
const HANDLE_SHAPE =
  "pointer-events-none fill-white [stroke:var(--overlay-block)] [stroke-width:1] [vector-effect:non-scaling-stroke] [filter:drop-shadow(0_1px_1.5px_rgb(0_0_0/0.35))] transition-[fill] duration-[20ms] ease-in group-hover/handle:fill-[var(--overlay-block)]";
const MIN_FRAME = 4; // page pixels
const ROTATE_SNAP = Math.PI / 12; // 15 degrees, with Shift

/** A rotated rectangle: center, size and angle (radians, clockwise on screen). */
type OBox = { cx: number; cy: number; w: number; h: number; angle: number };

/**
 * The frame's oriented box: a rectangular quad keeps its own axes (the renderer turns its text the
 * same way, from the top edge); any other polygon uses its bounds, unrotated.
 */
function orientedBox(points: Pt[]): { box: OBox; quad: boolean } {
  if (points.length === 4) {
    const [tl, tr, br, bl] = points;
    const top = { x: tr.x - tl.x, y: tr.y - tl.y };
    const right = { x: br.x - tr.x, y: br.y - tr.y };
    const w = Math.hypot(top.x, top.y);
    const h = Math.hypot(right.x, right.y);
    const square = Math.abs(top.x * right.x + top.y * right.y) <= w * h * 1e-3;
    const parallel = Math.abs(tl.x + br.x - tr.x - bl.x) < 0.5 && Math.abs(tl.y + br.y - tr.y - bl.y) < 0.5;
    if (w > 0 && h > 0 && square && parallel) return { box: { cx: (tl.x + br.x) / 2, cy: (tl.y + br.y) / 2, w, h, angle: Math.atan2(top.y, top.x) }, quad: true };
  }
  const b = boundsOf(points);
  return { box: { cx: b.x + b.width / 2, cy: b.y + b.height / 2, w: b.width, h: b.height, angle: 0 }, quad: false };
}

function toLocal(p: Pt, box: OBox): Pt {
  const dx = p.x - box.cx;
  const dy = p.y - box.cy;
  const cos = Math.cos(box.angle);
  const sin = Math.sin(box.angle);
  return { x: dx * cos + dy * sin, y: -dx * sin + dy * cos };
}

function toWorld(p: Pt, box: OBox): Pt {
  const cos = Math.cos(box.angle);
  const sin = Math.sin(box.angle);
  return { x: box.cx + p.x * cos - p.y * sin, y: box.cy + p.x * sin + p.y * cos };
}

const boxCorners = (box: OBox): Pt[] => [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([u, v]) => toWorld({ x: (u * box.w) / 2, y: (v * box.h) / 2 }, box));

/**
 * Drags `handle` to `pointer` (page pixels): edges move one side, corners two. Shift keeps the
 * proportions (corners only); Alt scales about the center. Returns the new box.
 */
function resizeBox(box: OBox, handle: number, pointer: Pt, shift: boolean, alt: boolean): OBox {
  const [u, v] = HANDLES[handle];
  const p = toLocal(pointer, box);
  let [left, right, top, bottom] = [-box.w / 2, box.w / 2, -box.h / 2, box.h / 2];
  if (u === 1) right = alt ? Math.max(p.x, MIN_FRAME / 2) : Math.max(p.x, left + MIN_FRAME);
  if (u === -1) left = alt ? Math.min(p.x, -MIN_FRAME / 2) : Math.min(p.x, right - MIN_FRAME);
  if (v === 1) bottom = alt ? Math.max(p.y, MIN_FRAME / 2) : Math.max(p.y, top + MIN_FRAME);
  if (v === -1) top = alt ? Math.min(p.y, -MIN_FRAME / 2) : Math.min(p.y, bottom - MIN_FRAME);
  if (alt) {
    if (u === 1) left = -right;
    if (u === -1) right = -left;
    if (v === 1) top = -bottom;
    if (v === -1) bottom = -top;
  }
  if (shift && u !== 0 && v !== 0) {
    // Grow both sides by the larger stretch, from the anchor (the opposite corner, or the center).
    const scale = Math.max((right - left) / box.w, (bottom - top) / box.h);
    const width = box.w * scale;
    const height = box.h * scale;
    if (alt) [left, right, top, bottom] = [-width / 2, width / 2, -height / 2, height / 2];
    else {
      if (u === 1) right = left + width;
      else left = right - width;
      if (v === 1) bottom = top + height;
      else top = bottom - height;
    }
  }
  const center = toWorld({ x: (left + right) / 2, y: (top + bottom) / 2 }, box);
  return { cx: center.x, cy: center.y, w: right - left, h: bottom - top, angle: box.angle };
}

/** Maps points from one box onto another: a stretch in the box's axes, then the new placement. */
function mapPoints(points: Pt[], from: OBox, to: OBox): Pt[] {
  return points.map((p) => {
    const local = toLocal(p, from);
    return toWorld({ x: (local.x / from.w) * to.w, y: (local.y / from.h) * to.h }, to);
  });
}

const degrees = (radians: number) => {
  const value = Math.round((radians * 180) / Math.PI) % 360;
  return value > 180 ? value - 360 : value <= -180 ? value + 360 : value;
};

/** The resize cursor for a handle, turned with the box (the four CSS double arrows, by 45 degree sector). */
function resizeCursor(handle: number, angle: number): string {
  const [u, v] = HANDLES[handle];
  const direction = ((((Math.atan2(v, u) + angle) * 180) / Math.PI) % 180 + 180) % 180;
  const sector = Math.round(direction / 45) % 4;
  return ["ew-resize", "nwse-resize", "ns-resize", "nesw-resize"][sector];
}

// A curved double arrow, outlined so it shows on light and dark artwork.
const ROTATE_CURSOR = `url("data:image/svg+xml,${encodeURIComponent(
  '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><g fill="none" stroke-linecap="round" stroke-linejoin="round"><path d="M6 15a7 7 0 0 1 9-9M15 3.5l.6 2.7-2.6.8M6 15l-2.5-.6.7 2.7" stroke="white" stroke-width="4"/><path d="M6 15a7 7 0 0 1 9-9M15 3.5l.6 2.7-2.6.8M6 15l-2.5-.6.7 2.7" stroke="black" stroke-width="1.6"/></g></svg>',
)}") 12 12, crosshair`;

// Browsers drop cursor images larger than this (Chrome on Windows), so bigger brushes get a drawn ring.
const MAX_CURSOR = 128;

/**
 * The brush outline as the OS cursor, as Photoshop draws it: it moves with the pointer at no cost
 * and with no lag, where a drawn ring re-layerizes the page on every move. Small brushes get a
 * crosshair too. Null when the outline is too large to be a cursor.
 */
function ringCursor(screenDiameter: number): string | null {
  const size = 2 * Math.ceil(Math.max(screenDiameter + 4, 14) / 2);
  if (size > MAX_CURSOR) return null;
  const c = size / 2;
  const r = Math.max(screenDiameter / 2, 1);
  const outline = (stroke: string, width: number) =>
    `<circle cx="${c}" cy="${c}" r="${r}" fill="none" stroke="${stroke}" stroke-width="${width}"/>` +
    (screenDiameter < 10 ? `<path d="M${c - 5} ${c}h10M${c} ${c - 5}v10" stroke="${stroke}" stroke-width="${width}"/>` : "");
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}">${outline("white", 3)}${outline("black", 1)}</svg>`;
  return `url("data:image/svg+xml,${encodeURIComponent(svg)}") ${c} ${c}, crosshair`;
}

const REGION_KIND = {
  bubble: "dev.koharu.region.bubble",
  panel: "dev.koharu.region.panel",
};

const MIN_ZOOM = 0.05;
const MAX_ZOOM = 8;
// Camera clamp: the page can be scrolled until only this much of it is still on screen.
const MIN_OVERLAP = 100;
// Arrow keys move the selected block by 1 page pixel, 10 with Shift; one edit is saved once the keys rest.
const NUDGE = 1;
const NUDGE_FAST = 10;
const NUDGE_COMMIT_MS = 400;
// A selection smaller than this (page pixels) on either side is a click, which clears it.
const MIN_SELECTION = 4;

// Webtoon reader: room above a page scrolled to from the sidebar, and where the choice is kept.
const STRIP_MARGIN = 24;
const READER_KEY = "urakatsu.webtoon-reader";

type Strip = { key: string; items: { page: Page; top: number }[]; width: number; height: number };

/** A chapter's pages stacked edge to edge: each page's top in strip pixels, the widest page's width. */
function useStrip(pageIds: string[] | undefined, pages: Page[] | undefined): Strip | null {
  return useMemo(() => {
    if (!pageIds || pageIds.length < 2 || !pages) return null;
    const byId = new Map(pages.map((p) => [p.id, p]));
    let top = 0;
    const items = [];
    for (const id of pageIds) {
      const page = byId.get(id);
      if (!page) continue;
      items.push({ page, top });
      top += page.height;
    }
    return { key: pageIds.join(","), items, width: Math.max(...items.map((item) => item.page.width)), height: top };
  }, [pageIds, pages]);
}

function loadReader(): boolean {
  try {
    return window.localStorage.getItem(READER_KEY) !== "off";
  } catch {
    return true;
  }
}

function saveReader(on: boolean): void {
  try {
    window.localStorage.setItem(READER_KEY, on ? "on" : "off");
  } catch {
    // the choice lasts this session only
  }
}
const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

function boundsOf(points: Pt[]): Rect {
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  return { x: Math.min(...xs), y: Math.min(...ys), width: Math.max(...xs) - Math.min(...xs), height: Math.max(...ys) - Math.min(...ys) };
}

const rectPoints = (a: Pt, b: Pt): Pt[] => [a, { x: b.x, y: a.y }, b, { x: a.x, y: b.y }];

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = url;
  });
}

/** The canvas's pixels under `box`, grown to whole pixels and kept on the canvas, as a PNG patch and where it goes. */
async function patchOf(canvas: HTMLCanvasElement, box: Rect): Promise<{ png: Uint8Array; at: Pt }> {
  const x = Math.max(0, Math.floor(box.x));
  const y = Math.max(0, Math.floor(box.y));
  const width = Math.min(canvas.width, Math.ceil(box.x + box.width)) - x;
  const height = Math.min(canvas.height, Math.ceil(box.y + box.height)) - y;
  const out = document.createElement("canvas");
  out.width = Math.max(1, width);
  out.height = Math.max(1, height);
  out.getContext("2d")!.drawImage(canvas, x, y, out.width, out.height, 0, 0, out.width, out.height);
  return { png: await toPng(out), at: { x, y } };
}

function toPng(canvas: HTMLCanvasElement): Promise<Uint8Array> {
  return new Promise((resolve, reject) =>
    canvas.toBlob((blob) => (blob ? blob.arrayBuffer().then((buffer) => resolve(new Uint8Array(buffer)), reject) : reject(new Error("canvas export failed"))), "image/png"),
  );
}

/**
 * Where a raster layer's canvas stands against the server. A stroke paints the canvas first and is
 * saved when it ends, and every page revision fetches the layer again: a fetch must never paint an
 * older server image over strokes the server has not got yet. (Koharu repaints unconditionally, so a
 * stroke drawn while the previous one saves is wiped, and saved without its wiped part.)
 */
type LayerPixels = {
  /** Strokes begun on the canvas: a fetch that started before one is stale. */
  strokes: number;
  /** Strokes in progress on the canvas. */
  drawing: number;
  /** Strokes in progress or saving: the canvas is ahead of the server. */
  busy: number;
  /** The server image has been drawn once (or the layer has none); saves wait for it. */
  loaded: boolean;
  ready: Promise<void>;
  markLoaded: () => void;
  /**
   * The layer version (its blob's content address) the canvas shows: set when a fetch is drawn and
   * when a save of the canvas lands. Only the current one: an undo can bring back an older blob.
   */
  shown?: string;
};

const layerPixels = new WeakMap<HTMLCanvasElement, LayerPixels>();

function pixelsOf(canvas: HTMLCanvasElement): LayerPixels {
  let state = layerPixels.get(canvas);
  if (!state) {
    let resolve!: () => void;
    const ready = new Promise<void>((r) => (resolve = r));
    const s: LayerPixels = { strokes: 0, drawing: 0, busy: 0, loaded: false, ready, markLoaded: () => ((s.loaded = true), resolve()) };
    layerPixels.set(canvas, (state = s));
  }
  return state;
}

const layerCanvasClass = "absolute inset-0 h-full w-full pointer-events-none rounded-[inherit]";

// Dabs are stamped this fraction of the brush size apart: close enough that a hard edge stays smooth.
const DAB_SPACING = 0.12;

/** A point of a stroke and how hard the pen pressed there (0 to 1). */
type Sample = Pt & { pressure: number };

/** Pen pressure; a mouse or finger counts as pressing fully. A pen's first touch can read 0. */
function pressureOf(event: PointerEvent): number {
  return event.pointerType === "pen" ? clamp(event.pressure, 0.05, 1) : 1;
}

/**
 * The opacity of one dab for a stroke of opacity `target`: about 1 / DAB_SPACING dabs overlap at any
 * point of a line, so their opacities compound to the target (and going over it again builds it up).
 */
function dabAlpha(target: number): number {
  return target >= 1 ? 1 : 1 - Math.pow(1 - target, DAB_SPACING);
}

/** One brush size step (Alt + wheel, [ and ]): about a tenth of the size, at least a pixel. */
function stepBrush(diameter: number, direction: 1 | -1, max: number): number {
  return clamp(diameter + direction * Math.max(1, Math.round(diameter * 0.1)), MIN_BRUSH, max);
}

/** Centers the brush ring on `point` (page pixels); hidden while the pointer is off the page. */
function ringPlacement(point: Pt | null, diameter: number, zoom: number): { transform: string; visibility: "visible" | "hidden" } {
  if (!point) return { transform: "translate(0px, 0px)", visibility: "hidden" };
  const r = (diameter / 2) * zoom;
  return { transform: `translate(${point.x * zoom - r}px, ${point.y * zoom - r}px)`, visibility: "visible" };
}

// A block, balloon or panel outline. Hovering tints only a plain block: a selected, user-made or hidden one keeps its own fill.
function regionClass({ selected, user, hidden }: { selected: boolean; user: boolean; hidden: boolean }) {
  return cn(
    "cursor-move [transition:fill_20ms_ease-in] [vector-effect:non-scaling-stroke]",
    selected
      ? "fill-[color-mix(in_oklab,var(--overlay-block)_16%,transparent)] [stroke:var(--overlay-block)] [stroke-width:2]"
      : user
        ? "fill-[color-mix(in_oklab,var(--overlay-user)_8%,transparent)] [stroke:var(--overlay-user)] [stroke-width:1.5]"
        : cn(
            "fill-[color-mix(in_oklab,var(--overlay-block)_8%,transparent)] [stroke:var(--overlay-block)] [stroke-width:1.5]",
            !hidden && "hover:fill-[color-mix(in_oklab,var(--overlay-block)_14%,transparent)]",
          ),
    hidden && "fill-transparent [stroke-dasharray:4_3]",
  );
}

export function Canvas() {
  const ws = useWorkspace();
  const page = currentPage(ws);
  const webtoon = ws.project?.format?.kind === "webtoon";
  const [reader, setReaderState] = useState(loadReader);
  const setReader = useCallback((on: boolean) => {
    saveReader(on);
    setReaderState(on);
  }, []);
  const chapter = page ? chapterOf(ws, page.id) : undefined;
  const strip = useStrip(webtoon && reader ? chapter?.pageIds : undefined, ws.project?.pages);
  const [zoom, setZoom] = useState(1);
  const [fittedFor, setFittedFor] = useState<string | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  // The page `url` is for: in the strip a stale image would show at the new page's place.
  const [urlPage, setUrlPage] = useState<string | null>(null);
  const [missing, setMissing] = useState(false);
  const [gesture, setGesture] = useState<Gesture | null>(null);
  // The drawn brush ring follows the pointer by style writes, not renders; `cursor` keeps its spot across renders.
  const cursor = useRef<Pt | null>(null);
  const brushRing = useRef<HTMLDivElement>(null);
  const strokeLast = useRef<Sample>({ x: 0, y: 0, pressure: 1 });
  // How far along the next segment the next dab goes.
  const strokeCarry = useRef(0);
  // What the stroke in progress has painted over (page pixels): its save sends only that.
  const strokeBox = useRef<Rect | null>(null);
  const [spaceHeld, setSpaceHeld] = useState(false);
  const [maskDirty, setMaskDirty] = useState(false);
  const [selection, setSelection] = useState<Selection | null>(null);
  // Kept across selections, as Photoshop keeps the last prompt.
  const [fillPrompt, setFillPrompt] = useState("");
  const [filling, setFilling] = useState(false);
  const [comparison, setComparison] = useState<Comparison | null>(null);
  // The text block being edited on the canvas (Photoshop's text tool), and whether the next selected
  // block is one the text tool just drew, which opens straight into editing.
  const [editing, setEditing] = useState<string | null>(null);
  const editCreated = useRef(false);
  const lastPress = useRef<{ id: string; at: number } | null>(null);
  const scroll = useRef<HTMLDivElement>(null);
  const svg = useRef<SVGSVGElement>(null);
  const baseImage = useRef<HTMLImageElement>(null);
  const layerCanvases = useRef(new Map<string, HTMLCanvasElement>());
  const pendingPaint = useRef<HTMLCanvasElement>(null);
  const pendingDirty = useRef(false);
  // Raster saves go out one at a time, in stroke order: each sends the whole layer, so the last one sent wins.
  const saveQueue = useRef<Promise<unknown>>(Promise.resolve());
  const mask = useRef<HTMLCanvasElement>(null);
  const dragLayer = useRef<HTMLDivElement>(null);
  const dragDelta = useRef<Pt>({ x: 0, y: 0 });
  // In the render view the selected block is lifted off the render, so a move or a turn shows its text
  // where it goes rather than where the render drew it: `box` is the block's oriented box in that render,
  // and `quad` whether it was laid out in it (a free-form block is lifted as its box to be turned).
  const [lift, setLift] = useState<(Lift & { id: string; key: string; box: OBox; quad: boolean }) | null>(null);
  const liftText = useRef<HTMLImageElement>(null);
  // The version of the base image on screen, and of the one its URL is for.
  const [shown, setShown] = useState<string | null>(null);
  const urlVersion = useRef("");
  const anchor = useRef<{ point: Pt; client: Pt } | null>(null);
  // The scroll area is the page plus room to scroll it nearly out of view, so it depends on the viewport size.
  const [viewport, setViewport] = useState({ width: 0, height: 0 });
  const center = useRef(false);
  // An arrow-key nudge in progress: previewed like a drag, saved as one edit when the keys rest.
  // `saved` once the server has it: it is still shown until the render with it is on screen.
  const [nudge, setNudge] = useState<{ id: string; origin: Pt[]; points: Pt[]; saving?: boolean; saved?: boolean } | null>(null);
  const nudgeRef = useRef(nudge);
  nudgeRef.current = nudge;
  const nudgeTimer = useRef<number | undefined>(undefined);
  const updateBlock = useRef(ws.updateBlock);
  updateBlock.current = ws.updateBlock;

  const projectId = ws.project?.id;
  // Per-asset cache keys: a stroke changes one layer's version, so the base image is not fetched again.
  const sourceVersion = page ? assetVersion(page, "source") : "";
  const baseVersion = page ? assetVersion(page, ws.view === "render" ? "render" : "source") : "";
  const pageId = page?.id;
  const showLayers = ws.view === "cleanup";
  const rasters = page?.rasterLayers ?? [];
  const paintLayer = [...rasters].reverse().find((layer) => layer.kind === "paint");
  const paintLayerRef = useRef(paintLayer);
  paintLayerRef.current = paintLayer;

  // Base image: the source, or the server render.
  useEffect(() => {
    if (!projectId || !pageId) {
      setUrl(null);
      return;
    }
    let alive = true;
    setMissing(false);
    void (async () => {
      let found = await assetUrl(projectId, pageId, ws.view === "render" ? "render" : "source", baseVersion);
      let version = baseVersion;
      if (!found && ws.view === "render") {
        setMissing(true);
        found = await assetUrl(projectId, pageId, "source", sourceVersion);
        version = sourceVersion;
      }
      if (alive) {
        urlVersion.current = version;
        setUrl(found);
        setUrlPage(pageId);
      }
    })();
    return () => {
      alive = false;
    };
  }, [projectId, pageId, ws.view, baseVersion, sourceVersion]);

  // Brush tools and generative fill paint into layers, so make them visible.
  useEffect(() => {
    if ((isBrushTool(ws.tool) || ws.tool === "fill") && ws.view !== "cleanup") ws.setView("cleanup");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ws.tool]);

  useEffect(() => {
    if (editCreated.current && ws.blockId) {
      editCreated.current = false;
      setEditing(ws.blockId);
    }
  }, [ws.blockId]);
  const editingBlock = editing ? page?.blocks.find((block) => block.layerId === editing) : undefined;
  useEffect(() => {
    if (editing && !editingBlock) setEditing(null);
  }, [editing, editingBlock]);

  const movingId = gesture?.kind === "move" || gesture?.kind === "transform" ? gesture.id : nudge?.id;
  const movingRef = useRef(movingId);
  movingRef.current = movingId;
  const liftId = ws.view === "render" && !missing ? ws.blockId : undefined;
  useEffect(() => {
    const block = liftId ? page?.blocks.find((b) => b.layerId === liftId) : undefined;
    if (!projectId || !pageId || !liftId || !block?.frame.length) return;
    const key = `${pageId}/${liftId}@${baseVersion}`;
    // A moving block keeps the lift it started with: the render behind it changes once the move is saved.
    if (lift?.id === liftId && (lift.key === key || movingId === liftId)) return;
    const { box, quad } = orientedBox(block.frame);
    let alive = true;
    void fetchLift(projectId, pageId, liftId, baseVersion).then((found) => alive && found && setLift({ ...found, id: liftId, key, box, quad }));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, pageId, liftId, baseVersion, movingId]);
  // A free-form block lifted as its box would lay it out, for turning it (null when it needs none).
  const liftAsBox = (block: TextBlock) => {
    const { box, quad } = orientedBox(block.frame);
    return projectId && pageId && ws.view === "render" && !missing && !quad ? fetchLift(projectId, pageId, block.layerId, baseVersion, boxCorners(box)) : null;
  };
  // A block is lifted when the pointer comes onto it (or onto a rotate handle), ahead of a press that moves (or turns) it.
  const hoverBlock = (block: TextBlock, rotate?: boolean) => {
    if (rotate) void liftAsBox(block);
    else if (projectId && pageId && ws.view === "render" && !missing && ws.tool === "select") void fetchLift(projectId, pageId, block.layerId, baseVersion);
  };
  // Where the lifted block is shown, as its oriented box (a drag adds `dragDelta` as the pointer moves).
  const pose =
    !lift || lift.id !== movingId
      ? null
      : gesture?.kind === "move"
        ? orientedBox(gesture.origin)
        : gesture?.kind === "transform"
          ? orientedBox(gesture.points)
          : nudge && orientedBox(nudge.points);
  const shownBox = pose?.box;
  // Its text moves and turns with it, laid out as it is shown; a resize reflows the text, so that waits for the render.
  const rigid = lift && pose && pose.quad === lift.quad && Math.abs(pose.box.w - lift.box.w) < 0.5 && Math.abs(pose.box.h - lift.box.h) < 0.5;
  const lifted = rigid && ws.view === "render" && !missing ? lift : null;
  const liftTransform = (): string => {
    if (!lifted || !shownBox) return "none";
    const drag = gesture?.kind === "move" ? dragDelta.current : { x: 0, y: 0 };
    const dx = shownBox.cx + drag.x - lifted.box.cx;
    const dy = shownBox.cy + drag.y - lifted.box.cy;
    return `translate(${dx * zoom}px, ${dy * zoom}px) rotate(${shownBox.angle - lifted.box.angle}rad)`;
  };

  // A selection belongs to the fill tool and its page.
  useEffect(() => {
    if (ws.tool !== "fill" || (selection && selection.pageId !== page?.id)) setSelection(null);
  }, [ws.tool, page?.id, selection]);

  // A comparison belongs to its selection.
  useEffect(() => {
    if (comparison && comparison.selection !== selection) setComparison(null);
  }, [selection, comparison]);

  // The comparison opens once a fill of its selection has succeeded.
  useEffect(() => {
    if (!filling && comparison && !comparison.ready && ws.job?.status === "completed") setComparison({ ...comparison, ready: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filling]);

  // Clear the remove-tool mask once the inpainting run it started has finished.
  useEffect(() => {
    if (maskDirty && ws.job?.status !== "running") {
      const context = mask.current?.getContext("2d");
      if (context && mask.current) context.clearRect(0, 0, mask.current.width, mask.current.height);
      setMaskDirty(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ws.job?.status]);

  // A strip fits its width (it is read by scrolling down); a page fits whole.
  const fit = useCallback(() => {
    const el = scroll.current;
    if (!el || !page) return;
    center.current = true;
    setViewport({ width: el.clientWidth, height: el.clientHeight });
    // A phone has no room to spare beside the page, and its glass bars float over the top and bottom.
    const narrow = el.clientWidth < 768;
    const marginX = narrow ? 24 : 120;
    const marginY = narrow ? 200 : 80;
    const fitted = strip ? (el.clientWidth - marginX) / strip.width : Math.min((el.clientWidth - marginX) / page.width, (el.clientHeight - marginY) / page.height);
    setZoom(clamp(Math.min(fitted, 2), MIN_ZOOM, MAX_ZOOM));
  }, [page, strip]);

  const hasPage = !!page;
  useLayoutEffect(() => {
    const el = scroll.current;
    if (!el) return;
    const measure = () => setViewport((current) => (current.width === el.clientWidth && current.height === el.clientHeight ? current : { width: el.clientWidth, height: el.clientHeight }));
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [hasPage]);

  // Where the page sits: in the scroll area's padding, and in a strip at its place in the chapter.
  const padX = Math.max(viewport.width - MIN_OVERLAP, 40);
  const padY = Math.max(viewport.height - MIN_OVERLAP, 40);
  const columnWidth = strip?.width ?? page?.width ?? 0;
  const pageTop = (strip && page && strip.items.find((item) => item.page.id === page.id)?.top) || 0;
  const pageLeft = page ? (columnWidth - page.width) / 2 : 0;
  // The scroll position code set, which the strip's page tracking leaves alone.
  const placedScroll = useRef<number | null>(null);
  const placeStrip = (el: HTMLElement) => {
    // On a phone the page's top clears the glass bar floating over the canvas.
    el.scrollTop = padY + pageTop * zoom - (el.clientWidth < 768 ? STRIP_MARGIN + 96 : STRIP_MARGIN);
    placedScroll.current = el.scrollTop;
  };

  // After fitting, center the page in the scroll area; a strip shows the current page's top.
  useLayoutEffect(() => {
    const el = scroll.current;
    if (!center.current || !el) return;
    center.current = false;
    el.scrollLeft = (el.scrollWidth - el.clientWidth) / 2;
    if (strip) placeStrip(el);
    else el.scrollTop = (el.scrollHeight - el.clientHeight) / 2;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [zoom, viewport]);

  // A page fits when it opens; a strip when its chapter opens, not as it is scrolled through.
  const fitKey = strip && chapter ? `strip:${chapter.id}` : page?.id;
  useEffect(() => {
    if (page && fitKey && fittedFor !== fitKey) {
      fit();
      setFittedFor(fitKey);
    }
  }, [page, fitKey, fittedFor, fit]);

  // The page the strip's scrolling chose stays where it is; one chosen elsewhere (the sidebar) is scrolled to.
  const fromStrip = useRef<string | null>(null);
  const showPage = useCallback(
    (id: string) => {
      fromStrip.current = id;
      ws.selectPage(id);
    },
    [ws],
  );
  useLayoutEffect(() => {
    const el = scroll.current;
    if (!strip || !page || !el) return;
    if (fromStrip.current === page.id) {
      fromStrip.current = null;
      return;
    }
    if (!center.current) placeStrip(el);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page?.id, strip?.key]);

  // Scrolling the strip makes the page under the middle of the view the current one, unless
  // something is going on in the current page (a stroke, a move, text editing, a fill selection).
  const holdPage = (!!gesture && gesture.kind !== "pan") || !!editing || !!nudge || !!selection || selectedPages(ws).length > 1;
  const stripState = useRef({ holdPage, pageId: page?.id, showPage });
  stripState.current = { holdPage, pageId: page?.id, showPage };
  useEffect(() => {
    const el = scroll.current;
    if (!strip || !el) return;
    let frame = 0;
    const onScroll = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        if (placedScroll.current !== null && Math.abs(el.scrollTop - placedScroll.current) < 2) return;
        placedScroll.current = null;
        const state = stripState.current;
        if (state.holdPage) return;
        const middle = (el.scrollTop + el.clientHeight / 2 - padY) / zoom;
        const item = strip.items.find((entry) => middle < entry.top + entry.page.height) ?? strip.items[strip.items.length - 1];
        if (item && item.page.id !== state.pageId) state.showPage(item.page.id);
      });
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      cancelAnimationFrame(frame);
      el.removeEventListener("scroll", onScroll);
    };
  }, [strip, zoom, padY]);

  // Keep the page point under the cursor fixed while zooming.
  useLayoutEffect(() => {
    const pending = anchor.current;
    const bounds = svg.current?.getBoundingClientRect();
    if (!pending || !bounds || !scroll.current) return;
    anchor.current = null;
    scroll.current.scrollLeft += bounds.left + pending.point.x * zoom - pending.client.x;
    scroll.current.scrollTop += bounds.top + pending.point.y * zoom - pending.client.y;
  }, [zoom]);

  const toImage = useCallback(
    (event: { clientX: number; clientY: number }): Pt => {
      const bounds = svg.current?.getBoundingClientRect();
      if (!bounds) return { x: 0, y: 0 };
      return { x: (event.clientX - bounds.left) / zoom, y: (event.clientY - bounds.top) / zoom };
    },
    [zoom],
  );

  const clampPoint = useCallback((p: Pt): Pt => (page ? { x: clamp(p.x, 0, page.width), y: clamp(p.y, 0, page.height) } : p), [page]);

  // Shows a moved block where it was dropped until the saved page arrives, and its render with it.
  const saveFrame = useCallback((id: string, origin: Pt[], points: Pt[]) => {
    const saving = { id, origin, points, saving: true };
    nudgeRef.current = saving;
    setNudge(saving);
    void updateBlock.current(id, { frame: points }).finally(() => setNudge((current) => (current === saving ? { ...saving, saved: true } : current)));
  }, []);
  useEffect(() => {
    if (nudge?.saved && (shown === baseVersion || missing)) setNudge(null);
  }, [nudge, shown, baseVersion, missing]);

  const commitNudge = useCallback(() => {
    window.clearTimeout(nudgeTimer.current);
    const pending = nudgeRef.current;
    if (!pending || pending.saving) return;
    const moved = pending.points.some((p, i) => p.x !== pending.origin[i].x || p.y !== pending.origin[i].y);
    if (!moved) return setNudge(null);
    saveFrame(pending.id, pending.origin, pending.points);
  }, [saveFrame]);

  // Moves the selected block by whole page pixels, keeping all of it on the page.
  const nudgeBlock = useCallback(
    (dx: number, dy: number) => {
      if (!page || !ws.blockId) return false;
      const block = page.blocks.find((b) => b.layerId === ws.blockId);
      if (!block || block.frame.length < 3) return false;
      const pending = nudgeRef.current;
      // Keys pressed while a nudge is saving continue from where the block is shown.
      const base = pending && pending.id === block.layerId ? { ...pending, saving: false, saved: false } : { id: block.layerId, origin: block.frame.map((p) => ({ x: p.x, y: p.y })), points: block.frame.map((p) => ({ x: p.x, y: p.y })) };
      const xs = base.points.map((p) => p.x);
      const ys = base.points.map((p) => p.y);
      const stepX = clamp(dx, -Math.min(...xs), page.width - Math.max(...xs));
      const stepY = clamp(dy, -Math.min(...ys), page.height - Math.max(...ys));
      const next = { ...base, points: base.points.map((p) => ({ x: p.x + stepX, y: p.y + stepY })) };
      nudgeRef.current = next; // key repeat can outpace rendering
      setNudge(next);
      window.clearTimeout(nudgeTimer.current);
      nudgeTimer.current = window.setTimeout(commitNudge, NUDGE_COMMIT_MS);
      return true;
    },
    [page, ws.blockId, commitNudge],
  );

  // A nudge is saved before the selection or the page changes, and when the canvas goes away.
  useEffect(() => () => commitNudge(), [ws.blockId, page?.id, commitNudge]);
  useEffect(() => () => window.clearTimeout(nudgeTimer.current), []);

  // Keyboard: the shortcuts from Settings (tools, fit, undo/redo, nudge), space-to-pan, views, delete.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target instanceof Element ? event.target : null;
      if (target?.closest("input, textarea, [contenteditable], [role=listbox], [role=menu]")) return;
      const mod = event.ctrlKey || event.metaKey;
      if (event.type === "keyup") {
        if (event.code === "Space") setSpaceHeld(false);
        return;
      }
      if (event.code === "Space" && !mod) {
        event.preventDefault();
        setSpaceHeld(true);
        return;
      }
      const shortcuts = ws.shortcuts;
      if (matches(shortcuts.undo, event)) {
        event.preventDefault();
        void ws.undo();
        return;
      }
      // Ctrl+Y also redoes while redo keeps its default, as in most Windows apps.
      if (matches(shortcuts.redo, event) || (shortcuts.redo === DEFAULT_SHORTCUTS.redo && matches("mod+y", event))) {
        event.preventDefault();
        void ws.redo();
        return;
      }
      const nudge = NUDGES.find((action) => matches(shortcuts[action], event));
      if (nudge) {
        event.preventDefault();
        const [dx, dy] = NUDGE_DIRECTIONS[nudge];
        const step = event.shiftKey ? NUDGE_FAST : NUDGE;
        // With a block selected these move it; otherwise they scroll the canvas.
        if (!nudgeBlock(dx * step, dy * step) && scroll.current) {
          const el = scroll.current;
          el.scrollBy({ left: dx * el.clientWidth * (event.shiftKey ? 0.5 : 0.1), top: dy * el.clientHeight * (event.shiftKey ? 0.5 : 0.1) });
        }
        return;
      }
      const resize = BRUSH_SIZES.find((a) => matches(shortcuts[a], event));
      if (resize && isBrushTool(ws.tool) && page) {
        event.preventDefault();
        ws.setBrush({ ...ws.brush, diameter: stepBrush(ws.brush.diameter, resize === "brush_larger" ? 1 : -1, maxBrush(page.width, page.height)) });
        return;
      }
      const action = ([...TOOLS, "fit"] as ShortcutAction[]).find((a) => matches(shortcuts[a], event));
      if (action === "fit") {
        event.preventDefault();
        return fit();
      }
      if (action) {
        event.preventDefault();
        return ws.setTool(action as Tool);
      }
      if (mod || event.altKey) return;
      const key = event.key.toLowerCase();
      if ((event.key === "Delete" || event.key === "Backspace") && ws.blockId) void ws.deleteBlock(ws.blockId);
      if (event.key === "Enter" && ws.blockId) {
        event.preventDefault();
        setEditing(ws.blockId);
      }
      if (event.key === "Escape") {
        if (selection) setSelection(null);
        else ws.selectBlock(undefined);
      }
      if (key === "1") ws.setView("source");
      if (key === "2") ws.setView("cleanup");
      if (key === "3") ws.setView("render");
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("keyup", onKey);
    };
  }, [ws, page, fit, nudgeBlock, selection]);

  // Ctrl + wheel zooms at the cursor, Alt + wheel sizes the brush; plain wheel scrolls.
  useEffect(() => {
    const el = scroll.current;
    if (!el || !page) return;
    const onWheel = (event: WheelEvent) => {
      if (event.altKey && isBrushTool(ws.tool)) {
        event.preventDefault();
        ws.setBrush({ ...ws.brush, diameter: stepBrush(ws.brush.diameter, event.deltaY < 0 ? 1 : -1, maxBrush(page.width, page.height)) });
        return;
      }
      if (!event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      anchor.current = { point: toImage(event), client: { x: event.clientX, y: event.clientY } };
      setZoom((current) => clamp(current * Math.exp(-event.deltaY * 0.0015), MIN_ZOOM, MAX_ZOOM));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [page, ws, toImage]);

  // Touch: two fingers pinch to zoom and pan, whatever the tool (a stroke one finger began is
  // dropped); one finger on the margin around the page pans it. The browser's own touch panning
  // and zooming are off here, so they do not fight the tools.
  useEffect(() => {
    const el = scroll.current;
    if (!el || !hasPage) return;
    el.style.touchAction = "none";
    const fingers = new Map<number, Pt>();
    let pinch: { distance: number; zoom: number; point: Pt; mid: Pt } | null = null;
    let drag: { client: Pt; scroll: Pt } | null = null;
    const pair = () => {
      const [a, b] = [...fingers.values()];
      return { mid: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }, distance: Math.max(1, Math.hypot(a.x - b.x, a.y - b.y)) };
    };
    const down = (event: PointerEvent) => {
      if (event.pointerType !== "touch") return;
      fingers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      if (fingers.size === 1 && !svg.current?.contains(event.target as Node)) {
        drag = { client: { x: event.clientX, y: event.clientY }, scroll: { x: el.scrollLeft, y: el.scrollTop } };
        return;
      }
      if (fingers.size === 2) {
        touchState.current.cancelGesture();
        drag = null;
        const { mid, distance } = pair();
        pinch = { distance, zoom: touchState.current.zoom, point: touchState.current.toImage({ clientX: mid.x, clientY: mid.y }), mid };
      }
      if (fingers.size >= 2) {
        event.stopPropagation();
        event.preventDefault();
      }
    };
    const move = (event: PointerEvent) => {
      if (!fingers.has(event.pointerId)) return;
      fingers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      if (drag) {
        el.scrollLeft = drag.scroll.x - (event.clientX - drag.client.x);
        el.scrollTop = drag.scroll.y - (event.clientY - drag.client.y);
        return;
      }
      if (!pinch || fingers.size < 2) return;
      event.stopPropagation();
      const { mid, distance } = pair();
      const next = clamp(pinch.zoom * (distance / pinch.distance), MIN_ZOOM, MAX_ZOOM);
      if (Math.abs(next - touchState.current.zoom) > 1e-4) {
        // The page point under the fingers at the start stays under their midpoint.
        anchor.current = { point: pinch.point, client: mid };
        setZoom(next);
      } else {
        el.scrollLeft -= mid.x - pinch.mid.x;
        el.scrollTop -= mid.y - pinch.mid.y;
      }
      pinch.mid = mid;
    };
    const up = (event: PointerEvent) => {
      if (!fingers.delete(event.pointerId)) return;
      if (pinch) event.stopPropagation();
      if (fingers.size < 2) pinch = null;
      if (fingers.size === 0) drag = null;
    };
    el.addEventListener("pointerdown", down, { capture: true });
    el.addEventListener("pointermove", move, { capture: true });
    el.addEventListener("pointerup", up, { capture: true });
    el.addEventListener("pointercancel", up, { capture: true });
    return () => {
      el.style.touchAction = "";
      el.removeEventListener("pointerdown", down, { capture: true });
      el.removeEventListener("pointermove", move, { capture: true });
      el.removeEventListener("pointerup", up, { capture: true });
      el.removeEventListener("pointercancel", up, { capture: true });
    };
  }, [hasPage]);

  // Strokes are stamped as round dabs, so pen pressure can set each dab's size and opacity (Photoshop's
  // Shape Dynamics and Transfer). A mouse presses fully: solid strokes of the set size, as before.
  const strokeTo = (canvas: HTMLCanvasElement, tool: "draw" | "eraser" | "remove", from: Sample, to: Sample) => {
    const context = canvas.getContext("2d");
    if (!context) return;
    const { diameter, color, pressureSize, pressureOpacity } = ws.brush;
    // The remove mask marks where to inpaint, so pressure leaves it solid.
    const pen = tool !== "remove";
    const sizeAt = (pressure: number) => Math.max(1, diameter * (pen && pressureSize ? pressure : 1));
    const alphaAt = (pressure: number) => dabAlpha(pen && pressureOpacity ? pressure : 1);
    context.save();
    context.globalCompositeOperation = tool === "eraser" ? "destination-out" : "source-over";
    // The remove mask is drawn opaque red for display; finishStroke turns its coverage white.
    context.fillStyle = tool === "draw" ? color : tool === "remove" ? "#e5484d" : "#ffffff";
    let box = strokeBox.current;
    const dab = (x: number, y: number, pressure: number) => {
      const r = sizeAt(pressure) / 2;
      context.globalAlpha = alphaAt(pressure);
      context.beginPath();
      context.arc(x, y, r, 0, Math.PI * 2);
      context.fill();
      const edge = r + 2; // and its antialiased edge
      box = box
        ? { x: Math.min(box.x, x - edge), y: Math.min(box.y, y - edge), width: Math.max(box.x + box.width, x + edge) - Math.min(box.x, x - edge), height: Math.max(box.y + box.height, y + edge) - Math.min(box.y, y - edge) }
        : { x: x - edge, y: y - edge, width: 2 * edge, height: 2 * edge };
    };
    const length = Math.hypot(to.x - from.x, to.y - from.y);
    if (length === 0) {
      dab(to.x, to.y, to.pressure);
      strokeCarry.current = sizeAt(to.pressure) * DAB_SPACING;
    } else {
      // A dab every DAB_SPACING of the brush size along the segment; the rest carries into the next one.
      let at = strokeCarry.current;
      while (at <= length) {
        const t = at / length;
        const pressure = from.pressure + (to.pressure - from.pressure) * t;
        dab(from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t, pressure);
        at += Math.max(0.5, sizeAt(pressure) * DAB_SPACING);
      }
      strokeCarry.current = at - length;
    }
    context.restore();
    strokeBox.current = box;
  };

  const finishStroke = async (stroke: Extract<Gesture, { kind: "stroke" }>) => {
    if (!page) return;
    if (stroke.tool === "remove") {
      // White strokes on black, page-sized: the inpainter repaints only where the user painted.
      const out = document.createElement("canvas");
      out.width = page.width;
      out.height = page.height;
      const context = out.getContext("2d")!;
      context.drawImage(stroke.target, 0, 0);
      context.globalCompositeOperation = "source-in";
      context.fillStyle = "#ffffff";
      context.fillRect(0, 0, out.width, out.height);
      context.globalCompositeOperation = "destination-over";
      context.fillStyle = "#000000";
      context.fillRect(0, 0, out.width, out.height);
      setMaskDirty(true);
      void ws.run({ kind: "only", stage: "inpainting", scope: "page", mask: await toPng(out) });
      return;
    }
    const kind = stroke.layer?.kind === "cleanup" ? "cleanup" : "paint";
    const pixels = pixelsOf(stroke.target);
    pixels.drawing--;
    const box = strokeBox.current ?? { x: 0, y: 0, width: page.width, height: page.height };
    // Only what the stroke painted over is sent; a layer stroked before its first load is saved once
    // the server image is under it.
    const patch = pixels.ready.then(() => patchOf(stroke.target, box));
    const save = saveQueue.current.then(async () => {
      const { png, at } = await patch;
      const saved = await ws.setRasterLayer(kind, stroke.target === pendingPaint.current ? undefined : stroke.layer?.id, png, at);
      if (!saved) return;
      // The layer's canvas shows the version this save made, so it is not fetched back. First strokes
      // land in the page's paint layer, whose canvas takes them over from `pendingPaint`.
      const id = stroke.target === pendingPaint.current ? [...saved.rasterLayers].reverse().find((l) => l.kind === "paint")?.id : stroke.layer?.id;
      const canvas = id && layerCanvases.current.get(id);
      if (canvas) pixelsOf(canvas).shown = assetVersion(saved, `layer:${id}`);
    });
    saveQueue.current = save.catch(() => {});
    try {
      await save;
    } finally {
      pixels.busy--;
    }
    if (stroke.target === pendingPaint.current) adoptPending();
  };

  // The first strokes on a page go to `pendingPaint` until their save creates the paint layer; its
  // canvas takes them over when it appears, or when the stroke in progress ends (never just a clear,
  // which would drop a stroke begun meanwhile). Left on `pendingPaint`, they would outlive an undo.
  const adoptPending = () => {
    const pending = pendingPaint.current;
    const layer = paintLayerRef.current;
    const target = layer && layerCanvases.current.get(layer.id);
    if (!pending || !target || !pendingDirty.current || pixelsOf(pending).drawing > 0) return;
    target.getContext("2d")!.drawImage(pending, 0, 0);
    const pixels = pixelsOf(target);
    pixels.strokes++; // its first fetch predates these strokes
    if (!pixels.loaded) pixels.markLoaded();
    pending.getContext("2d")!.clearRect(0, 0, pending.width, pending.height);
    pendingDirty.current = false;
  };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useLayoutEffect(() => adoptPending(), [paintLayer?.id]);

  // The selection as a page-sized mask (white inside), sent with the prompt to the fill model.
  const generateFill = async () => {
    if (!page || !selection || filling) return;
    const out = document.createElement("canvas");
    out.width = page.width;
    out.height = page.height;
    const context = out.getContext("2d")!;
    context.fillStyle = "#000000";
    context.fillRect(0, 0, out.width, out.height);
    context.fillStyle = "#ffffff";
    context.beginPath();
    selection.points.forEach((p, i) => (i === 0 ? context.moveTo(p.x, p.y) : context.lineTo(p.x, p.y)));
    context.closePath();
    context.fill();
    // Generate again keeps comparing with the artwork from before the first fill.
    if (comparison?.selection !== selection) {
      const before = snapshot(boundsOf(selection.points));
      setComparison(before && { selection, ...before, ready: false, position: 0.5 });
    }
    setFilling(true);
    try {
      await ws.run({ kind: "only", stage: "inpainting", scope: "page", mask: await toPng(out), fillPrompt: fillPrompt.trim() });
    } finally {
      setFilling(false);
    }
  };

  const sampleColor = (point: Pt) => {
    if (!page || !baseImage.current) return;
    const out = document.createElement("canvas");
    out.width = page.width;
    out.height = page.height;
    const context = out.getContext("2d", { willReadFrequently: true })!;
    context.drawImage(baseImage.current, 0, 0, page.width, page.height);
    if (showLayers) for (const layer of rasters) if (layer.visible) {
      const canvas = layerCanvases.current.get(layer.id);
      if (!canvas) continue;
      context.globalAlpha = layer.opacity || 1;
      context.drawImage(canvas, 0, 0);
    }
    const [r, g, b] = context.getImageData(Math.floor(point.x), Math.floor(point.y), 1, 1).data;
    ws.setBrush({ ...ws.brush, color: `#${[r, g, b].map((v) => v.toString(16).padStart(2, "0")).join("")}` });
  };

  // The page as shown (base image and visible layers) under `bounds`, on whole pixels.
  const snapshot = (bounds: Rect): { bounds: Rect; url: string } | null => {
    if (!page || !baseImage.current) return null;
    const x = Math.max(0, Math.floor(bounds.x));
    const y = Math.max(0, Math.floor(bounds.y));
    const right = Math.min(page.width, Math.ceil(bounds.x + bounds.width));
    const bottom = Math.min(page.height, Math.ceil(bounds.y + bounds.height));
    if (right <= x || bottom <= y) return null;
    const out = document.createElement("canvas");
    out.width = right - x;
    out.height = bottom - y;
    const context = out.getContext("2d")!;
    context.drawImage(baseImage.current, -x, -y, page.width, page.height);
    if (showLayers) for (const layer of rasters) if (layer.visible) {
      const canvas = layerCanvases.current.get(layer.id);
      if (!canvas) continue;
      context.globalAlpha = layer.opacity || 1;
      context.drawImage(canvas, -x, -y);
    }
    return { bounds: { x, y, width: out.width, height: out.height }, url: out.toDataURL("image/png") };
  };

  // Dragging the comparison divider (or clicking along it) moves it under the pointer.
  const startCompare = (event: ReactPointerEvent) => {
    if (event.button !== 0 || !comparison) return;
    event.stopPropagation();
    svg.current?.setPointerCapture(event.pointerId);
    const bounds = comparison.bounds;
    setComparison({ ...comparison, position: clamp((toImage(event).x - bounds.x) / bounds.width, 0, 1) });
    setGesture({ kind: "compare", bounds });
  };

  const onPointerDown = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (!page) return;
    if (editing) setEditing(null);
    const point = toImage(event);
    // A finger with the Select tool pans the page (a tap on a text frame still selects it, in startMove).
    const fingerPan = event.pointerType === "touch" && ws.tool === "select";
    if (fingerPan) ws.selectBlock(undefined);
    const panning = event.button === 1 || ws.tool === "pan" || spaceHeld || fingerPan;
    if (event.button !== 0 && !panning) return;
    event.preventDefault();
    if (panning && scroll.current) {
      event.currentTarget.setPointerCapture(event.pointerId);
      setGesture({ kind: "pan", client: { x: event.clientX, y: event.clientY }, scroll: { x: scroll.current.scrollLeft, y: scroll.current.scrollTop } });
      return;
    }
    if (ws.tool === "select") {
      ws.selectBlock(undefined);
      return;
    }
    if (ws.tool === "text") {
      event.currentTarget.setPointerCapture(event.pointerId);
      setGesture({ kind: "text", start: point, rect: { ...point, width: 0, height: 0 } });
      return;
    }
    if (ws.tool === "color_picker") return sampleColor(point);
    if (ws.tool === "fill") {
      if (filling) return; // the running fill keeps its selection
      const start = clampPoint(point);
      event.currentTarget.setPointerCapture(event.pointerId);
      setSelection(null);
      setGesture({ kind: "select", shape: ws.fillShape, start, points: [start] });
      return;
    }
    if (ws.tool === "draw" || ws.tool === "eraser" || ws.tool === "remove") {
      let target: HTMLCanvasElement | null | undefined;
      let layer: RasterLayer | undefined;
      if (ws.tool !== "remove") adoptPending();
      const selectedRaster = rasters.find((r) => r.id === ws.rasterId);
      if (ws.tool === "remove") target = mask.current;
      else if (ws.tool === "draw" && !selectedRaster) {
        // No layer selected: the newest paint layer, created by the first stroke if there is none.
        layer = paintLayer;
        // Until the paint layer's canvas has taken them over, strokes keep going where the first ones are.
        target = layer && !pendingDirty.current ? layerCanvases.current.get(layer.id) : pendingPaint.current;
      } else {
        // As in Photoshop, the brush and eraser work on the selected layer.
        layer = selectedRaster ?? paintLayer;
        if (!layer) {
          ws.reportError("Select a paint or cleanup layer in Layers before using the eraser.");
          return;
        }
        target = layerCanvases.current.get(layer.id);
      }
      if (layer && !layer.visible) {
        ws.reportError(`${layer.name} is hidden. Show it in Layers to paint on it.`);
        return;
      }
      if (!target) return;
      event.currentTarget.setPointerCapture(event.pointerId);
      if (ws.tool !== "remove") {
        const pixels = pixelsOf(target);
        pixels.strokes++;
        pixels.drawing++;
        pixels.busy++;
        if (target === pendingPaint.current) {
          pixels.markLoaded(); // nothing on the server to wait for
          pendingDirty.current = true;
        }
      }
      strokeBox.current = null;
      const first = { ...point, pressure: pressureOf(event.nativeEvent) };
      strokeTo(target, ws.tool, first, first);
      strokeLast.current = first;
      setGesture({ kind: "stroke", tool: ws.tool, target, layer });
    }
  };

  const startMove = (event: ReactPointerEvent, block: TextBlock) => {
    if (event.button !== 0 || ws.tool !== "select" || spaceHeld || block.frame.length < 3) return;
    event.stopPropagation();
    // A second press on the same block soon after the first edits its text on the canvas. (The first press
    // captures the pointer, so the browser's dblclick goes to the canvas, not the block.)
    const now = performance.now();
    const previous = lastPress.current;
    lastPress.current = { id: block.layerId, at: now };
    if (previous && previous.id === block.layerId && now - previous.at < DOUBLE_PRESS_MS) {
      lastPress.current = null;
      ws.selectBlock(block.layerId);
      setEditing(block.layerId);
      return;
    }
    ws.selectBlock(block.layerId);
    svg.current?.setPointerCapture(event.pointerId);
    const origin = block.frame.map((p) => ({ x: p.x, y: p.y }));
    dragDelta.current = { x: 0, y: 0 };
    setGesture({ kind: "move", id: block.layerId, start: toImage(event), origin, points: origin });
  };

  const startTransform = (event: ReactPointerEvent, block: TextBlock, mode: "resize" | "rotate", handle: number) => {
    if (event.button !== 0 || block.frame.length < 3) return;
    event.stopPropagation();
    svg.current?.setPointerCapture(event.pointerId);
    const { box, quad } = orientedBox(block.frame);
    // Rotating a free-form outline turns it into its box: only a rectangle carries an angle to the renderer.
    const origin = mode === "rotate" && !quad ? boxCorners(box) : block.frame.map((p) => ({ x: p.x, y: p.y }));
    // So its text is lifted as the box would lay it out, to be turned while it rotates.
    const turning = mode === "rotate" ? liftAsBox(block) : null;
    if (turning) {
      movingRef.current = block.layerId; // a prefetched lift arrives before the gesture renders
      const key = `${pageId}/${block.layerId}@${baseVersion}#box`;
      void turning.then((found) => found && movingRef.current === block.layerId && setLift({ ...found, id: block.layerId, key, box, quad: true }));
    }
    setGesture({ kind: "transform", id: block.layerId, mode, handle, box, origin, start: toImage(event), points: origin, readout: "" });
  };

  // Esc drops a transform in progress, as in Photoshop.
  const transforming = gesture?.kind === "transform";
  useEffect(() => {
    if (!transforming) return;
    const cancel = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      event.stopImmediatePropagation(); // Esc here only cancels; the selection stays
      setGesture(null);
    };
    window.addEventListener("keydown", cancel, true);
    return () => window.removeEventListener("keydown", cancel, true);
  }, [transforming]);

  const panning = ws.tool === "pan" || spaceHeld || gesture?.kind === "pan";
  const brushCursor = isBrushTool(ws.tool) && !panning;
  const osRing = brushCursor ? ringCursor(ws.brush.diameter * zoom) : null;
  const moveRing = (point: Pt | null) => {
    cursor.current = point;
    const ring = brushRing.current;
    if (ring) Object.assign(ring.style, ringPlacement(point, ws.brush.diameter, zoom));
  };

  const onPointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
    const point = toImage(event);
    if (brushCursor) moveRing(point);
    if (!gesture) return;
    // The pointer is captured for the gesture, so keep its moves from the document listeners
    // (the panel resizer hit-tests every panel on each document pointermove, forcing a layout).
    event.stopPropagation();
    if (gesture.kind === "pan" && scroll.current) {
      scroll.current.scrollLeft = gesture.scroll.x - (event.clientX - gesture.client.x);
      scroll.current.scrollTop = gesture.scroll.y - (event.clientY - gesture.client.y);
    } else if (gesture.kind === "stroke") {
      const events = event.nativeEvent.getCoalescedEvents?.() ?? [event.nativeEvent];
      for (const sample of events.length ? events : [event.nativeEvent]) {
        const next = { ...toImage(sample), pressure: pressureOf(sample) };
        strokeTo(gesture.target, gesture.tool, strokeLast.current, next);
        strokeLast.current = next;
      }
    } else if (gesture.kind === "move" && page) {
      // Keep the whole block on the page, as the arrow keys do.
      const xs = gesture.origin.map((p) => p.x);
      const ys = gesture.origin.map((p) => p.y);
      const dx = clamp(point.x - gesture.start.x, -Math.min(...xs), page.width - Math.max(...xs));
      const dy = clamp(point.y - gesture.start.y, -Math.min(...ys), page.height - Math.max(...ys));
      dragDelta.current = { x: dx, y: dy };
      if (dragLayer.current) dragLayer.current.style.transform = `translate(${dx * zoom}px, ${dy * zoom}px)`;
      if (liftText.current && lifted) liftText.current.style.transform = liftTransform();
    } else if (gesture.kind === "transform") {
      const { box } = gesture;
      if (gesture.mode === "rotate") {
        // Turn about the center by the pointer's sweep; Shift snaps the result to 15 degrees.
        const sweep = Math.atan2(point.y - box.cy, point.x - box.cx) - Math.atan2(gesture.start.y - box.cy, gesture.start.x - box.cx);
        let angle = box.angle + sweep;
        if (event.shiftKey) angle = Math.round(angle / ROTATE_SNAP) * ROTATE_SNAP;
        const turned = { ...box, angle };
        setGesture({ ...gesture, points: mapPoints(gesture.origin, box, turned), readout: `${degrees(angle)}°` });
      } else {
        const next = resizeBox(box, gesture.handle, clampPoint(point), event.shiftKey, event.altKey);
        setGesture({ ...gesture, points: mapPoints(gesture.origin, box, next), readout: `${Math.round(next.w)} × ${Math.round(next.h)}` });
      }
    } else if (gesture.kind === "text") {
      const x = Math.min(point.x, gesture.start.x);
      const y = Math.min(point.y, gesture.start.y);
      setGesture({ ...gesture, rect: { x, y, width: Math.abs(point.x - gesture.start.x), height: Math.abs(point.y - gesture.start.y) } });
    } else if (gesture.kind === "compare") {
      const position = clamp((point.x - gesture.bounds.x) / gesture.bounds.width, 0, 1);
      setComparison((current) => current && { ...current, position });
    } else if (gesture.kind === "select") {
      const target = clampPoint(point);
      if (gesture.shape === "rect") return setGesture({ ...gesture, points: rectPoints(gesture.start, target) });
      // The lasso keeps a point per screen pixel or two of travel.
      const last = gesture.points[gesture.points.length - 1];
      if (Math.hypot(target.x - last.x, target.y - last.y) * zoom >= 2) setGesture({ ...gesture, points: [...gesture.points, target] });
    }
  };

  const cancelGesture = () => {
    // A cancelled stroke is not saved; its canvas stops counting it.
    if (gesture?.kind === "stroke" && gesture.tool !== "remove") {
      const pixels = pixelsOf(gesture.target);
      pixels.drawing--;
      pixels.busy--;
    }
    setGesture(null);
  };
  const touchState = useRef({ cancelGesture, toImage, zoom });
  touchState.current = { cancelGesture, toImage, zoom };

  const onPointerUp = () => {
    if (!gesture) return;
    setGesture(null);
    if (gesture.kind === "pan" || gesture.kind === "compare") return;
    if (gesture.kind === "stroke") return void finishStroke(gesture);
    if (gesture.kind === "text") {
      if (gesture.rect.width > 6 && gesture.rect.height > 6) {
        editCreated.current = true;
        void ws.createBlock(gesture.rect);
      }
      return;
    }
    if (gesture.kind === "select") {
      const bounds = boundsOf(gesture.points);
      const closed = gesture.points.length >= (gesture.shape === "rect" ? 4 : 3);
      if (page && closed && bounds.width >= MIN_SELECTION && bounds.height >= MIN_SELECTION) setSelection({ pageId: page.id, points: gesture.points });
      return;
    }
    if (gesture.kind === "move") {
      const { x: dx, y: dy } = dragDelta.current;
      if (Math.abs(dx) > 0.01 || Math.abs(dy) > 0.01) saveFrame(gesture.id, gesture.origin, gesture.origin.map((p) => ({ x: p.x + dx, y: p.y + dy })));
      return;
    }
    const moved = gesture.points.some((p, i) => Math.abs(p.x - gesture.origin[i].x) > 0.01 || Math.abs(p.y - gesture.origin[i].y) > 0.01);
    if (moved) saveFrame(gesture.id, gesture.origin, gesture.points);
  };

  if (!ws.project) return null;
  // The comparison shows between fills of the current selection, not while one runs.
  const comparing = comparison?.ready && comparison.selection === selection && !filling ? comparison : null;
  const liftAt = liftTransform();
  const svgCursor = panning ? (gesture?.kind === "pan" ? "grabbing" : "grab") : brushCursor ? (osRing ?? "none") : ws.tool === "text" || ws.tool === "fill" ? "crosshair" : ws.tool === "color_picker" ? "copy" : "default";

  return (
    <div className="relative flex min-h-0 flex-1 flex-col">
      {page && <ToolBelt />}
      {!page ? (
        <EmptyCanvas />
      ) : (
        <ScrollArea className="flex-1 bg-canvas" scrollbars="both" viewportRef={scroll}>
          <div
            className="relative"
            style={{
              width: columnWidth * zoom + 2 * padX,
              height: (strip?.height ?? page.height) * zoom + 2 * padY,
            }}
          >
          {strip && (
            <StripPages
              strip={strip}
              projectId={ws.project!.id}
              view={ws.view}
              zoom={zoom}
              left={padX}
              top={padY}
              onPointerDown={(event, id) => {
                if (event.button === 1 || ws.tool === "pan" || spaceHeld) {
                  // Dragging the strip pans it, from any page.
                  if (!scroll.current || !svg.current) return;
                  event.preventDefault();
                  svg.current.setPointerCapture(event.pointerId);
                  setGesture({ kind: "pan", client: { x: event.clientX, y: event.clientY }, scroll: { x: scroll.current.scrollLeft, y: scroll.current.scrollTop } });
                } else if (event.button === 0 && !holdPage) {
                  showPage(id);
                }
              }}
            />
          )}
          <div
            className={cn("absolute", strip ? "" : "rounded-pill bg-surface shadow-floating")}
            style={{ width: page.width * zoom, height: page.height * zoom, left: padX + pageLeft * zoom, top: padY + pageTop * zoom }}
          >
            {url && (!strip || urlPage === page.id) && (
              <img
                ref={baseImage}
                className="block rounded-[inherit] select-none"
                src={url}
                width={page.width * zoom}
                height={page.height * zoom}
                alt={page.name}
                draggable={false}
                onLoad={() => setShown(urlVersion.current)}
              />
            )}
            {showLayers &&
              rasters.map((layer) => (
                <LayerCanvas
                  key={layer.id}
                  page={page}
                  layer={layer}
                  projectId={ws.project!.id}
                  version={assetVersion(page, `layer:${layer.id}`)}
                  register={(canvas) => {
                    if (canvas) layerCanvases.current.set(layer.id, canvas);
                    else layerCanvases.current.delete(layer.id);
                  }}
                />
              ))}
            <canvas ref={pendingPaint} className={layerCanvasClass} width={page.width} height={page.height} />
            <canvas ref={mask} className={cn(layerCanvasClass, "opacity-[0.45]")} width={page.width} height={page.height} />
            {comparing && (
              // The artwork before the fill, cut off at the divider: before on its left, the result on its right.
              <img
                className="pointer-events-none absolute select-none"
                src={comparing.url}
                alt=""
                draggable={false}
                style={{
                  left: comparing.bounds.x * zoom,
                  top: comparing.bounds.y * zoom,
                  width: comparing.bounds.width * zoom,
                  height: comparing.bounds.height * zoom,
                  clipPath: `inset(0 ${(1 - comparing.position) * 100}% 0 0)`,
                }}
              />
            )}
            {ws.job?.previews
              ?.filter((preview) => preview.page === page.id)
              .map((preview, index, shown) => (
                // A diffusion model's result so far, laid over its crop until the stage commits.
                <div
                  key={`${preview.box.x},${preview.box.y},${preview.box.width},${preview.box.height}`}
                  className="pointer-events-none absolute"
                  style={{ left: preview.box.x * zoom, top: preview.box.y * zoom, width: preview.box.width * zoom, height: preview.box.height * zoom }}
                >
                  <img className="h-full w-full select-none" src={preview.url} alt="" draggable={false} />
                  {index === shown.length - 1 && preview.steps > 0 && (
                    <span className="absolute left-1.5 top-1.5 rounded-pill bg-surface-over px-1.5 py-0.5 text-[11px] font-medium tabular-nums text-fg-2 shadow-floating">
                      {preview.step}/{preview.steps}
                    </span>
                  )}
                </div>
              ))}
            {lifted && (
              // The moving or turning block's text lifted off the render, over the render as it is without it.
              <div className="pointer-events-none absolute inset-0 overflow-hidden rounded-[inherit]" aria-hidden>
                <img className="absolute max-w-none select-none" src={lifted.under} alt="" draggable={false} style={{ left: lifted.x * zoom, top: lifted.y * zoom, width: lifted.width * zoom, height: lifted.height * zoom }} />
                <img
                  ref={liftText}
                  className="absolute max-w-none select-none will-change-transform"
                  src={lifted.text}
                  alt=""
                  draggable={false}
                  style={{
                    left: lifted.x * zoom,
                    top: lifted.y * zoom,
                    width: lifted.width * zoom,
                    height: lifted.height * zoom,
                    // Turned about the block's center, where a rotation turns it.
                    transformOrigin: `${(lifted.box.cx - lifted.x) * zoom}px ${(lifted.box.cy - lifted.y) * zoom}px`,
                    transform: liftAt,
                  }}
                />
              </div>
            )}
            <svg
              ref={svg}
              // touch-none: a pen or finger drag draws instead of scrolling the page (which would cancel the stroke).
              className="absolute inset-0 h-full w-full touch-none"
              viewBox={`0 0 ${page.width} ${page.height}`}
              style={{ cursor: svgCursor }}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerLeave={() => !gesture && moveRing(null)}
              onPointerCancel={cancelGesture}
            >
              <Overlays
                page={page}
                zoom={zoom}
                gesture={gesture ?? (nudge ? { kind: "move", start: { x: 0, y: 0 }, ...nudge } : null)}
                dragging={gesture?.kind === "move" ? gesture.id : undefined}
                startMove={startMove}
                startTransform={startTransform}
                hoverBlock={hoverBlock}
              />
              {gesture?.kind === "select" ? <Marquee points={gesture.points} /> : selection && <Marquee points={selection.points} busy={filling} />}
              {comparing && <CompareDivider bounds={comparing.bounds} position={comparing.position} zoom={zoom} onPointerDown={startCompare} />}
            </svg>
            {brushCursor && !osRing && (
              // Too big for a cursor image: a ring on its own compositor layer, moved by transform (inside the SVG, every move repainted all the overlays).
              <div
                ref={brushRing}
                className="pointer-events-none absolute left-0 top-0 rounded-full border-[1.5px] border-[var(--n-950)] shadow-[0_0_1px_white,inset_0_0_1px_white] will-change-transform"
                style={{ width: ws.brush.diameter * zoom, height: ws.brush.diameter * zoom, ...ringPlacement(cursor.current, ws.brush.diameter, zoom) }}
                aria-hidden
              />
            )}
            {gesture?.kind === "move" && (
              // The transform goes on a wrapper: on the <svg> itself it would re-lay-out the SVG text on every move.
              <div ref={dragLayer} className="pointer-events-none absolute inset-0 will-change-transform" aria-hidden>
                <svg className="h-full w-full" viewBox={`0 0 ${page.width} ${page.height}`}>
                  {page.blocks.map((block, index) =>
                    block.layerId === gesture.id ? <BlockOverlay key={block.layerId} block={block} index={index} points={gesture.origin} selected interactive={ws.tool === "select"} zoom={zoom} /> : null,
                  )}
                </svg>
              </div>
            )}
            {editingBlock && (
              <CanvasTextEditor key={editingBlock.layerId} block={editingBlock} page={page} projectId={ws.project.id} zoom={zoom} onDone={() => setEditing(null)} />
            )}
            {selection && (!gesture || gesture.kind === "compare") && (
              <FillPrompt
                bounds={boundsOf(selection.points)}
                page={page}
                zoom={zoom}
                prompt={fillPrompt}
                setPrompt={setFillPrompt}
                filling={filling}
                compare={comparing ? { position: comparing.position, set: (position) => setComparison((current) => current && { ...current, position }) } : undefined}
                onGenerate={() => void generateFill()}
                onCancel={() => (filling ? void ws.stop() : setSelection(null))}
              />
            )}
          </div>
          </div>
        </ScrollArea>
      )}
      {page && (
        <StatusBar
          zoom={zoom}
          setZoom={setZoom}
          fit={fit}
          fitKey={ws.shortcuts.fit}
          missing={missing}
          width={page.width}
          height={page.height}
          reader={webtoon ? { on: reader, set: setReader } : undefined}
        />
      )}
    </div>
  );
}

// Memoized like the tool belt: a drag re-renders the canvas on every pointer move, not this bar.
const StatusBar = memo(function StatusBar({
  zoom,
  setZoom,
  fit,
  fitKey,
  missing,
  width,
  height,
  reader,
}: {
  zoom: number;
  setZoom: (zoom: number) => void;
  fit: () => void;
  fitKey: string;
  missing: boolean;
  width: number;
  height: number;
  /** Webtoon projects: the strip reader, on or off. */
  reader?: { on: boolean; set: (on: boolean) => void };
}) {
  return (
    // A phone pinches to zoom; its glass toolbar takes the bottom edge.
    <footer className="flex h-[30px] items-center gap-3 px-3 text-xs tabular-nums text-fg-3 shadow-[inset_0_1px_0_var(--border-soft)] max-md:hidden">
      <div className="flex w-[180px] items-center gap-1">
        <Slider label="Zoom" min={10} max={800} step={5} value={clamp(Math.round(zoom * 100), 10, 800)} onChange={(value) => setZoom(value / 100)} />
        <button className="h-[22px] w-12 cursor-pointer rounded-pill bg-transparent text-xs tabular-nums text-fg hover:bg-hover" title={`Fit (${fitKey.toUpperCase()})`} onClick={fit}>
          {Math.round(zoom * 100)}%
        </button>
      </div>
      {missing && <span className="text-warning">Render not ready: showing the source.</span>}
      <span style={{ flex: 1 }} />
      {reader && (
        <button
          className="inline-flex h-[22px] cursor-pointer items-center gap-1 rounded-pill bg-transparent px-1.5 text-xs text-fg-3 hover:bg-hover hover:text-fg aria-pressed:text-fg aria-pressed:[&_svg]:text-brand"
          aria-pressed={reader.on}
          title={reader.on ? "Strip: the chapter's pages joined, read by scrolling. Click for one page at a time." : "One page at a time. Click to join the chapter's pages into one strip."}
          onClick={() => reader.set(!reader.on)}
        >
          <GalleryVertical size={14} /> Strip
        </button>
      )}
      <span>
        {width} × {height} px
      </span>
    </footer>
  );
});

/** The strip's pages as plain images, edge to edge. Tops are floored and bottoms ceiled to whole
 * screen pixels, so neighbours overlap by under a pixel rather than leave a hairline between them;
 * the current page is drawn over its own image once its layers are ready. */
const StripPages = memo(function StripPages({
  strip,
  projectId,
  view,
  zoom,
  left,
  top,
  onPointerDown,
}: {
  strip: Strip;
  projectId: string;
  view: string;
  zoom: number;
  left: number;
  top: number;
  onPointerDown: (event: ReactPointerEvent, pageId: string) => void;
}) {
  return (
    <>
      <div className="pointer-events-none absolute bg-surface shadow-floating" style={{ left, top, width: strip.width * zoom, height: strip.height * zoom }} />
      {strip.items.map(({ page, top: y }) => {
        // The render once the page has one worth showing; until then the source, without a render per page.
        const kind = view === "render" && page.stages.some((stage) => stage.done) ? "render" : "source";
        const from = Math.floor(y * zoom);
        return (
          <img
            key={page.id}
            className="absolute max-w-none select-none"
            src={assetHref(projectId, page.id, kind, assetVersion(page, kind))}
            alt={page.name}
            loading="lazy"
            decoding="async"
            draggable={false}
            style={{ left: left + ((strip.width - page.width) / 2) * zoom, top: top + from, width: page.width * zoom, height: Math.ceil((y + page.height) * zoom) - from }}
            onPointerDown={(event) => onPointerDown(event, page.id)}
          />
        );
      })}
    </>
  );
});

// Marching ants: black dashes over white, legible on any artwork, in screen pixels at any zoom.
function Marquee({ points, busy }: { points: Pt[]; busy?: boolean }) {
  const shape = points.map((p) => `${p.x},${p.y}`).join(" ");
  return (
    <g className="pointer-events-none" aria-hidden>
      <polygon
        className={cn(
          "[stroke:white] [stroke-width:1.5] [vector-effect:non-scaling-stroke] [transition:fill_100ms_ease-out]",
          busy ? "fill-[color-mix(in_oklab,var(--brand)_16%,transparent)]" : "fill-[color-mix(in_oklab,var(--brand)_6%,transparent)]",
        )}
        points={shape}
      />
      <polygon className="fill-none [stroke:black] [stroke-width:1.5] [stroke-dasharray:5_4] [vector-effect:non-scaling-stroke]" points={shape}>
        <animate attributeName="stroke-dashoffset" from="0" to="-9" dur={busy ? "0.45s" : "0.9s"} repeatCount="indefinite" />
      </polygon>
    </g>
  );
}

// The before/after divider across a filled selection, with a handle to drag it.
function CompareDivider({ bounds, position, zoom, onPointerDown }: { bounds: Rect; position: number; zoom: number; onPointerDown: (event: ReactPointerEvent) => void }) {
  const x = bounds.x + bounds.width * position;
  const cy = bounds.y + bounds.height / 2;
  const r = 9 / zoom;
  const labels = bounds.width * zoom >= 140 && bounds.height * zoom >= 40;
  const label = "fill-white [paint-order:stroke] [stroke:rgb(0_0_0/0.7)] [stroke-width:3] [stroke-linejoin:round] [font-family:var(--font)] font-medium pointer-events-none";
  return (
    <g>
      {labels && (
        <>
          <text className={label} x={bounds.x + 6 / zoom} y={bounds.y + 15 / zoom} style={{ fontSize: 11 / zoom }}>
            Before
          </text>
          <text className={label} x={bounds.x + bounds.width - 6 / zoom} y={bounds.y + 15 / zoom} textAnchor="end" style={{ fontSize: 11 / zoom }}>
            After
          </text>
        </>
      )}
      <line className="pointer-events-none [stroke:white] [stroke-width:2] [vector-effect:non-scaling-stroke] [filter:drop-shadow(0_0_1px_rgb(0_0_0/0.8))]" x1={x} x2={x} y1={bounds.y} y2={bounds.y + bounds.height} />
      {/* A wide invisible stroke, so the divider can be grabbed anywhere along it. */}
      <line className="cursor-ew-resize [stroke:transparent] [stroke-width:14] [vector-effect:non-scaling-stroke]" x1={x} x2={x} y1={bounds.y} y2={bounds.y + bounds.height} onPointerDown={onPointerDown} />
      <g className="cursor-ew-resize" onPointerDown={onPointerDown}>
        <circle className="fill-surface [stroke:var(--brand)] [stroke-width:1.5] [vector-effect:non-scaling-stroke] [filter:drop-shadow(0_1px_2px_rgb(0_0_0/0.35))]" cx={x} cy={cy} r={r} />
        <path className="fill-none [stroke:var(--brand)] [stroke-width:1.5] [stroke-linecap:round] [stroke-linejoin:round] [vector-effect:non-scaling-stroke]" d={`M ${x - 2.5 / zoom} ${cy - 3 / zoom} l ${-2.5 / zoom} ${3 / zoom} l ${2.5 / zoom} ${3 / zoom} M ${x + 2.5 / zoom} ${cy - 3 / zoom} l ${2.5 / zoom} ${3 / zoom} l ${-2.5 / zoom} ${3 / zoom}`} />
      </g>
    </g>
  );
}

const FILL_PROMPT_WIDTH = 400;
// Room the prompt needs below the selection before it moves above it.
const FILL_PROMPT_HEIGHT = 112;

// The floating prompt under a generative fill selection. It lives in the page box, so it
// scrolls and zooms with the page; near the bottom of the page it sits above the selection.
const DOUBLE_PRESS_MS = 450;

const CJK = /[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]/;

/** Text size for editing when the layer fits its text automatically: what roughly fills the box. */
function fittedSize(text: string, width: number, height: number): number {
  const characters = Math.max(Array.from(text).length, 4);
  return clamp(Math.sqrt((width * height) / (characters * 0.6 * 1.2)), 10, 72);
}

/** How the render laid out a text block: the size auto-fit chose and where its lines (or columns) start. */
type BlockLayout = { font_size: number; line_starts: number[]; writing_mode: string };

const textLayouts = new Map<string, Promise<Record<string, BlockLayout>>>();

/** The page's text layout as rendered at its current revision (the service renders it if need be). */
function useTextLayout(projectId: string, page: Page): Record<string, BlockLayout> | null {
  const key = `${projectId}/${page.id}@${page.revision}`;
  const [layouts, setLayouts] = useState<{ key: string; value: Record<string, BlockLayout> } | null>(null);
  useEffect(() => {
    let alive = true;
    if (!textLayouts.has(key)) {
      const url = `/api/v1/projects/${encodeURIComponent(projectId)}/pages/${encodeURIComponent(page.id)}/text-layout?v=${page.revision}`;
      textLayouts.set(key, fetch(url).then((response) => (response.ok ? response.json() : {})).catch(() => ({})));
    }
    void textLayouts.get(key)!.then((value) => alive && setLayouts({ key, value }));
    return () => {
      alive = false;
    };
  }, [key, projectId, page.id, page.revision]);
  return layouts?.key === key ? layouts.value : (layouts?.value ?? null);
}

/** A text block lifted off its page as rendered (the service's lift): its pixels and the page under them, as data URLs, and where they go. */
type Lift = { x: number; y: number; width: number; height: number; text: string; under: string };

const lifts = new Map<string, Promise<Lift | null>>();

/**
 * A text block lifted off the render at `version`, or as it would render with the frame `outline`; the
 * last few are kept, for moving a block again.
 */
function fetchLift(projectId: string, pageId: string, layerId: string, version: string, outline?: Pt[]): Promise<Lift | null> {
  const frame = outline?.map((p) => `${p.x},${p.y}`).join(",");
  const key = `${projectId}/${pageId}/${layerId}@${version}${frame ? `#${frame}` : ""}`;
  let found = lifts.get(key);
  if (!found) {
    const url = `/api/v1/projects/${encodeURIComponent(projectId)}/pages/${encodeURIComponent(pageId)}/blocks/${encodeURIComponent(layerId)}/lift${frame ? `?frame=${frame}` : ""}`;
    found = fetch(url).then((response) => (response.ok ? response.json() : null)).catch(() => null);
    lifts.set(key, found);
    if (lifts.size > 8) lifts.delete(lifts.keys().next().value!);
  }
  return found;
}

/**
 * Editing a text block on the page, as Photoshop's text tool does: the text shows as the render laid it
 * out (the size auto-fit chose, the same line breaks, the layer's font, colour and border, centred), over
 * the art without the rendered text, so it is the text itself being edited. Typing reflows in the browser
 * until it is saved and rendered again.
 */
function CanvasTextEditor({ block, page, projectId, zoom, onDone }: { block: TextBlock; page: Page; projectId: string; zoom: number; onDone: () => void }) {
  const ws = useWorkspace();
  const defaultFont = useDefaultFont();
  const layouts = useTextLayout(projectId, page);
  const layout = layouts?.[block.layerId];
  // The layout is the render's, usually already made for the page; past a moment, open with an estimate.
  const [waited, setWaited] = useState(false);
  useEffect(() => {
    const timer = window.setTimeout(() => setWaited(true), 800);
    return () => window.clearTimeout(timer);
  }, []);
  const { box } = orientedBox(block.frame);
  const typography = block.typography;
  const text = block.translation ?? "";
  const vertical = layout ? layout.writing_mode === "vertical-rl" : typography?.writingMode === "Vertical" || (!typography?.writingMode && block.ocrDirection === "Vertical" && CJK.test(text));
  const size = (layout?.font_size ?? typography?.size ?? fittedSize(text, vertical ? box.h : box.w, vertical ? box.w : box.h)) * zoom;
  const align = { Start: "left", Center: "center", End: "right", Justify: "justify" }[typography?.alignment ?? ""] ?? "center";
  const family = typography?.preferredFont || defaultFont;
  useLayerFonts([family, ...block.translationStyles.map((style) => style.font)]);
  if (!layouts && !waited) return null;

  const strokeWidth = typography?.strokeWidth ?? 0;
  const strokeColor = typography?.strokeColor ?? "#ffffffff";
  const stroke = strokeWidth > 0 && (strokeColor.length !== 9 || strokeColor.slice(7) !== "00" || !!typography?.strokeGradient);
  const fill = typography?.fillGradient && typography.fillGradient.stops.length >= 2 ? typography.fillGradient : undefined;
  const color = typography?.color ?? "#000000ff";
  // The art without its rendered text (the cleanup layer over the source), cropped to the box; a turned
  // box keeps the page under it.
  const left = (box.cx - box.w / 2) * zoom;
  const top = (box.cy - box.h / 2) * zoom;
  const art = box.angle
    ? undefined
    : [
        page.assets.includes("cleanup") && `url("${assetHref(projectId, page.id, "cleanup", assetVersion(page, "cleanup"))}")`,
        `url("${assetHref(projectId, page.id, "source", assetVersion(page, "source"))}")`,
      ]
        .filter(Boolean)
        .join(", ");
  return (
    <div
      className="absolute z-20 rounded-[2px] shadow-[0_0_0_1.5px_var(--overlay-block)]"
      style={{
        left,
        top,
        width: box.w * zoom,
        height: box.h * zoom,
        transform: box.angle ? `rotate(${box.angle}rad)` : undefined,
        backgroundImage: art,
        backgroundSize: `${page.width * zoom}px ${page.height * zoom}px`,
        backgroundPosition: `${-left}px ${-top}px`,
      }}
      onPointerDown={(event) => event.stopPropagation()}
      onDoubleClick={(event) => event.stopPropagation()}
    >
      <RichTextField
        label="Text on the canvas"
        text={text}
        styles={block.translationStyles}
        onCommit={(next, styles) => void ws.updateBlock(block.layerId, { translation: next, translationStyles: styles })}
        onDone={onDone}
        overlay={{
          boxClassName: "flex items-center justify-center",
          className: vertical ? "h-full max-w-full" : "w-full max-h-full",
          vertical,
          softBreaks: layout?.line_starts,
          style: {
            padding: 4 * zoom,
            fontFamily: family ? `"${family}", var(--font), sans-serif` : undefined,
            fontSize: size,
            lineHeight: 1.2,
            fontWeight: typography?.fontWeight ?? undefined,
            fontStyle: typography?.fontStyle === "italic" ? "italic" : undefined,
            textAlign: align as CSSProperties["textAlign"],
            writingMode: vertical ? "vertical-rl" : undefined,
          },
          fillStyle: {
            caretColor: fill ? fill.stops[0].color : color,
            ...(fill ? { backgroundImage: cssGradient(fill), WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent" } : { color }),
          },
          // The border behind the fill, outward by its width (a centred stroke twice as wide); a gradient
          // border is the gradient clipped to the text and its stroke.
          underlayStyle: stroke
            ? typography?.strokeGradient && typography.strokeGradient.stops.length >= 2
              ? { backgroundImage: cssGradient(typography.strokeGradient), WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent", WebkitTextStroke: `${2 * strokeWidth * zoom}px transparent` }
              : { color: strokeColor, WebkitTextStroke: `${2 * strokeWidth * zoom}px ${strokeColor}` }
            : undefined,
        }}
      />
    </div>
  );
}

function FillPrompt({
  bounds,
  page,
  zoom,
  prompt,
  setPrompt,
  filling,
  compare,
  onGenerate,
  onCancel,
}: {
  bounds: Rect;
  page: Page;
  zoom: number;
  prompt: string;
  setPrompt: (prompt: string) => void;
  filling: boolean;
  /** The before/after divider, once a fill of the selection has succeeded. */
  compare?: { position: number; set: (position: number) => void };
  onGenerate: () => void;
  onCancel: () => void;
}) {
  const ws = useWorkspace();
  const input = useRef<HTMLInputElement>(null);
  const modelId = ws.config?.fill || ws.config?.inpainting;
  const model = ws.capabilities?.providers.find((p) => p.id === modelId);
  const installed = useInstalledModel(modelId ? ws.config?.processorJson[modelId] : undefined);
  const name = installed ? installedName(installed.id) : (model?.name ?? modelId ?? "The inpainting model");
  const prompted = !!model?.prompted;
  // An installed image model edits by instruction when its recipe does; built-in models say so themselves.
  const instructed = prompted && (installed?.model?.recipe ? !!installed.model.edits : !!model?.instructed);
  const busy = filling || ws.job?.status === "running";
  const loading = filling && ws.job?.current?.kind === "loading";
  const removes = !prompted || !prompt.trim();

  useEffect(() => input.current?.focus({ preventScroll: true }), []);

  const width = page.width * zoom;
  const left = clamp((bounds.x + bounds.width / 2) * zoom - FILL_PROMPT_WIDTH / 2, Math.min(0, width - FILL_PROMPT_WIDTH), Math.max(0, width - FILL_PROMPT_WIDTH));
  const bottom = (bounds.y + bounds.height) * zoom;
  const above = bottom + FILL_PROMPT_HEIGHT > page.height * zoom && bounds.y * zoom > FILL_PROMPT_HEIGHT;
  const place = above ? { bottom: page.height * zoom - bounds.y * zoom + 8 } : { top: bottom + 8 };

  return (
    <div
      className="absolute z-10 flex flex-col gap-0.5 rounded-[12px] bg-surface-over p-1 shadow-floating"
      style={{ left, width: FILL_PROMPT_WIDTH, ...place }}
      role="dialog"
      aria-label="Generative fill"
      onPointerDown={(event) => event.stopPropagation()}
    >
      <div className="flex items-center gap-1">
        <WandSparkles size={14} className="ml-1.5 flex-none text-brand" aria-hidden />
        <Input
          ref={input}
          aria-label="Prompt"
          className="flex-1 bg-transparent shadow-none focus-visible:shadow-none"
          placeholder={
            instructed ? "Replace the SFX with “BOOM!”" : prompted ? "Describe what belongs here, or leave empty" : "Removes what is selected"
          }
          value={prompted ? prompt : ""}
          disabled={!prompted || busy}
          onChange={(event) => setPrompt(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.nativeEvent.isComposing) {
              event.preventDefault();
              if (!busy) onGenerate();
            } else if (event.key === "Escape") {
              event.preventDefault();
              onCancel();
            }
          }}
        />
        <Button size="sm" disabled={busy} onClick={onGenerate}>
          {busy ? <LoaderCircle size={14} className="animate-spin" /> : <CornerDownLeft size={14} />}
          {removes ? "Remove" : "Generate"}
        </Button>
        <Button size="icon-sm" variant="ghost" aria-label={filling ? "Stop" : "Clear selection"} title={filling ? "Stop" : "Clear selection (Esc)"} onClick={onCancel}>
          <X size={14} />
        </Button>
      </div>
      {compare && (
        <div className="flex items-center gap-2 px-2 pt-0.5 text-xs tracking-[0.005em] text-fg-2">
          <span>Before</span>
          <div className="flex-1">
            <Slider label="Before and after" min={0} max={100} value={Math.round(compare.position * 100)} onChange={(value) => compare.set(value / 100)} />
          </div>
          <span>After</span>
        </div>
      )}
      <p className="flex items-center gap-1 truncate px-2 pb-0.5 text-xs tracking-[0.005em] text-fg-3">
        {filling ? (
          <span className="truncate">{loading ? `Loading ${name}…` : `Painting with ${name}…`}</span>
        ) : (
          <>
            <span className="truncate" title={name}>
              {/* The hint comes first: installed model names are long and the line truncates. */}
              {instructed ? `Say what to change · ${name}` : prompted ? `Describe what belongs here · ${name}` : `${name} cannot follow prompts.`}
            </span>
            <Button variant="link" size="xs" className="text-xs" onClick={() => ws.setSettingsOpen(true, "pipeline")}>
              {prompted ? "Change model" : "Choose a fill model"}
            </Button>
          </>
        )}
      </p>
    </div>
  );
}

// "abenzerps/Qwen-Image-2.1-Uncensored-Quantized:Q4_0" -> "Qwen-Image-2.1-Uncensored-Quantized Q4_0", as Settings lists it.
function installedName(id: string): string {
  const [repo, variant] = id.split(":");
  return `${repo.split("/").pop()}${variant ? ` ${variant}` : ""}`;
}

/** The installed catalog model a provider's settings select (Settings, Pipeline), if any. */
function useInstalledModel(settingsJson: string | undefined): { id: string; model?: InstalledModel } | undefined {
  let id: string | undefined;
  try {
    const value = settingsJson ? JSON.parse(settingsJson).model : undefined;
    id = typeof value === "string" && value ? value : undefined;
  } catch {
    id = undefined;
  }
  const [model, setModel] = useState<InstalledModel>();
  useEffect(() => {
    setModel(undefined);
    if (!id) return;
    let alive = true;
    catalog.installed().then((models) => alive && setModel(models.find((m) => m.id === id)), () => undefined);
    return () => {
      alive = false;
    };
  }, [id]);
  return id ? { id, model } : undefined;
}

function LayerCanvas({
  page,
  layer,
  projectId,
  version,
  register,
}: {
  page: Page;
  layer: RasterLayer;
  projectId: string;
  version: string;
  register: (canvas: HTMLCanvasElement | null) => void;
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    let alive = true;
    const strokes = ref.current ? pixelsOf(ref.current).strokes : 0;
    // The canvas's own save made this version: nothing to fetch.
    if (ref.current && pixelsOf(ref.current).loaded && pixelsOf(ref.current).shown === version) return;
    void assetUrl(projectId, page.id, `layer:${layer.id}`, version).then(async (url) => {
      const canvas = ref.current;
      if (!alive || !canvas) return;
      const pixels = pixelsOf(canvas);
      if (!url) return pixels.markLoaded();
      const image = await loadImage(url);
      if (!alive) return;
      // Painted on since this fetch began, or still saving: the canvas is newer than this image.
      const ahead = pixels.busy > 0 || pixels.strokes !== strokes;
      if (pixels.loaded && (ahead || pixels.shown === version)) return;
      const context = canvas.getContext("2d")!;
      context.save();
      // Strokes made before the layer first loaded stay on top of it.
      if (ahead) context.globalCompositeOperation = "destination-over";
      else context.clearRect(0, 0, canvas.width, canvas.height);
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      context.restore();
      pixels.shown = version;
      pixels.markLoaded();
    });
    return () => {
      alive = false;
    };
  }, [projectId, page.id, layer.id, version]);
  return (
    <canvas
      ref={(canvas) => {
        ref.current = canvas;
        register(canvas);
      }}
      className={layerCanvasClass}
      width={page.width}
      height={page.height}
      style={{ opacity: layer.visible ? layer.opacity || 1 : 0 }}
    />
  );
}

function Overlays({
  page,
  zoom,
  gesture,
  dragging,
  startMove,
  startTransform,
  hoverBlock,
}: {
  page: Page;
  zoom: number;
  gesture: Gesture | null;
  dragging?: string;
  startMove: (event: ReactPointerEvent, block: TextBlock) => void;
  startTransform: TransformStart;
  hoverBlock: (block: TextBlock, rotate?: boolean) => void;
}) {
  const ws = useWorkspace();
  return (
    <>
      {ws.showRegions &&
        page.regions
          .filter((r) => r.kind === REGION_KIND.bubble || r.kind === REGION_KIND.panel)
          .map((region) => (
            <polygon
              key={region.id}
              className={cn(
                "pointer-events-none fill-none [vector-effect:non-scaling-stroke] stroke-1",
                region.kind === REGION_KIND.bubble ? "[stroke:var(--overlay-balloon)] [stroke-dasharray:5_4]" : "[stroke:var(--overlay-panel)] [stroke-dasharray:2_4]",
              )}
              points={region.polygon.map((p) => `${p.x},${p.y}`).join(" ")}
            />
          ))}
      {ws.showBlocks &&
        page.blocks.map((block, index) =>
          // The block being dragged is drawn on the drag layer instead.
          block.layerId === dragging ? null : (
            <BlockOverlay
              key={block.layerId}
              block={block}
              index={index}
              points={gesture && "id" in gesture && gesture.id === block.layerId ? gesture.points : block.frame}
              selected={block.layerId === ws.blockId}
              interactive={ws.tool === "select"}
              zoom={zoom}
              startMove={startMove}
              startTransform={startTransform}
              hoverBlock={hoverBlock}
              readout={gesture?.kind === "transform" && gesture.id === block.layerId ? gesture.readout : undefined}
            />
          ),
        )}
      {gesture?.kind === "text" && <rect className={regionClass({ selected: true, user: false, hidden: false })} x={gesture.rect.x} y={gesture.rect.y} width={gesture.rect.width} height={gesture.rect.height} />}
    </>
  );
}

type TransformStart = (event: ReactPointerEvent, block: TextBlock, mode: "resize" | "rotate", handle: number) => void;

function BlockOverlay({
  block,
  index,
  points,
  selected,
  interactive,
  zoom,
  startMove,
  startTransform,
  hoverBlock,
  readout,
}: {
  block: TextBlock;
  index: number;
  points: Pt[];
  selected: boolean;
  interactive: boolean;
  zoom: number;
  startMove?: (event: ReactPointerEvent, block: TextBlock) => void;
  startTransform?: TransformStart;
  hoverBlock?: (block: TextBlock, rotate?: boolean) => void;
  /** "120 × 80" or "15°" while this block is being transformed. */
  readout?: string;
}) {
  if (block.frame.length < 3) return null;
  const rect = boundsOf(points);
  const tag = `${index + 1}`;
  const manual = block.translationOrigin === "user" || block.sourceOrigin === "user" || !block.regionId;
  const { box, quad } = orientedBox(points);
  const turn = (box.angle * 180) / Math.PI;
  return (
    <g className={interactive ? undefined : "pointer-events-none"}>
      <polygon
        className={regionClass({ selected, user: manual, hidden: !block.visible })}
        points={points.map((p) => `${p.x},${p.y}`).join(" ")}
        onPointerDown={startMove && ((event) => startMove(event, block))}
        onPointerEnter={hoverBlock && (() => hoverBlock(block))}
      />
      <rect className="fill-[var(--overlay-block)] [vector-effect:non-scaling-stroke]" x={rect.x} y={rect.y - 16 / zoom} width={(10 + 7 * tag.length) / zoom} height={16 / zoom} rx={3 / zoom} />
      <text className="fill-on-brand [font-family:var(--font)] font-semibold pointer-events-none tabular-nums" x={rect.x + 5 / zoom} y={rect.y - 4 / zoom} style={{ fontSize: 11 / zoom }}>
        {tag}
      </text>
      {selected && interactive && (
        <>
          {/* A free-form outline transforms by its box, so show the box the handles sit on. */}
          {!quad && (
            <polygon
              className="pointer-events-none fill-none [stroke:var(--overlay-block)] [stroke-dasharray:3_3] [vector-effect:non-scaling-stroke]"
              points={boxCorners(box).map((p) => `${p.x},${p.y}`).join(" ")}
            />
          )}
          {/* Just outside each corner, the pointer turns into a rotate cursor (Photoshop's free transform). */}
          {CORNERS.map((handle) => {
            const [u, v] = HANDLES[handle];
            const at = toWorld({ x: (u * box.w) / 2 + (u * 12) / zoom, y: (v * box.h) / 2 + (v * 12) / zoom }, box);
            return (
              <circle
                key={`rotate-${handle}`}
                className="fill-transparent"
                cx={at.x}
                cy={at.y}
                r={12 / zoom}
                style={{ cursor: ROTATE_CURSOR }}
                onPointerDown={startTransform && ((event) => startTransform(event, block, "rotate", handle))}
                onPointerEnter={hoverBlock && (() => hoverBlock(block, true))}
              />
            );
          })}
          {HANDLES.map(([u, v], handle) => {
            const at = toWorld({ x: (u * box.w) / 2, y: (v * box.h) / 2 }, box);
            // Corners are small squares; edge handles are bars lying along their edge.
            const corner = u !== 0 && v !== 0;
            const [w, h] = corner ? [HANDLE_CORNER, HANDLE_CORNER] : u === 0 ? [HANDLE_BAR, HANDLE_THICK] : [HANDLE_THICK, HANDLE_BAR];
            return (
              <g key={handle} className="group/handle" transform={turn ? `rotate(${turn} ${at.x} ${at.y})` : undefined}>
                <rect
                  className={HANDLE_SHAPE}
                  x={at.x - w / zoom / 2}
                  y={at.y - h / zoom / 2}
                  width={w / zoom}
                  height={h / zoom}
                  rx={(corner ? 2 : HANDLE_THICK / 2) / zoom}
                />
                {/* A target about twice the handle's size, so the small mark is easy to grab. */}
                <rect
                  className="fill-transparent"
                  x={at.x - HANDLE_HIT / zoom / 2}
                  y={at.y - HANDLE_HIT / zoom / 2}
                  width={HANDLE_HIT / zoom}
                  height={HANDLE_HIT / zoom}
                  style={{ cursor: resizeCursor(handle, box.angle) }}
                  onPointerDown={startTransform && ((event) => startTransform(event, block, "resize", handle))}
                />
              </g>
            );
          })}
        </>
      )}
      {readout && <TransformReadout text={readout} x={box.cx} y={rect.y + rect.height + 10 / zoom} zoom={zoom} />}
    </g>
  );
}

/** A small chip under the box with the size or angle, as Photoshop shows beside the cursor. */
function TransformReadout({ text, x, y, zoom }: { text: string; x: number; y: number; zoom: number }) {
  const width = (12 + 6.5 * text.length) / zoom;
  return (
    <g className="pointer-events-none">
      <rect className="fill-[var(--overlay-block)]" x={x - width / 2} y={y} width={width} height={18 / zoom} rx={4 / zoom} />
      <text className="fill-on-brand [font-family:var(--font)] tabular-nums" x={x} y={y + 13 / zoom} textAnchor="middle" style={{ fontSize: 11 / zoom }}>
        {text}
      </text>
    </g>
  );
}

function EmptyCanvas() {
  const ws = useWorkspace();
  const input = useRef<HTMLInputElement>(null);
  const project = ws.project!;
  const chapter = project.chapters[project.chapters.length - 1];
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-1 bg-canvas p-8 text-center">
      <p className="text-base leading-6 text-fg">{project.pages.length ? "No page selected." : "No pages yet."}</p>
      <p className="max-w-[42ch] text-fg-2">{project.pages.length ? "Choose a page in the sidebar to edit it." : "Pages you add to a chapter appear here."}</p>
      {project.pages.length === 0 && chapter && (
        <>
          <Button variant="outline" className="mt-2" onClick={() => input.current?.click()}>
            Add pages
          </Button>
          <input
            ref={input}
            type="file"
            accept="image/*"
            multiple
            hidden
            onChange={(event) => {
              if (event.target.files) void ws.uploadPages(event.target.files, chapter.id);
              event.target.value = "";
            }}
          />
        </>
      )}
    </div>
  );
}
