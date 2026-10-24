// Workspace state: one React context owning projects, selection, jobs, and config.

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { client, describeError, forgetAssets } from "../api/client";
import { challengeOf, type Challenge } from "../api/catalog";
import { sources, type LinkView } from "../api/sources";
import type { RichStyle } from "../richText";
import { applyPalette, loadPalette, type Palette } from "../theme";
import { initialRoute, useHistorySync, type Route } from "./history";
import type {
  Capabilities,
  Chapter,
  Config,
  Page,
  ProgressEvent,
  Project,
  ProjectSummary,
  RasterLayer,
  TextBlock,
  TranslationModel,
  Typography,
} from "../gen/urakatsu/v1/urakatsu_pb";

export type Theme = "light" | "dark" | "system";
export type ViewMode = "source" | "cleanup" | "render";
// Canvas tools, in tool-belt order, plus generative fill (Urakatsu's own).
export const TOOLS = ["select", "text", "draw", "eraser", "color_picker", "remove", "fill", "pan"] as const;
export type Tool = (typeof TOOLS)[number];
// Keyboard shortcuts (src/shortcuts.ts has the format: "v", "arrowleft", "mod+shift+z").
export const NUDGES = ["nudge_left", "nudge_right", "nudge_up", "nudge_down"] as const;
export const BRUSH_SIZES = ["brush_smaller", "brush_larger"] as const;
export type ShortcutAction = Tool | "fit" | "undo" | "redo" | (typeof NUDGES)[number] | (typeof BRUSH_SIZES)[number];
export const DEFAULT_SHORTCUTS: Record<ShortcutAction, string> = {
  select: "v",
  text: "t",
  draw: "b",
  eraser: "e",
  color_picker: "i",
  remove: "j",
  fill: "g",
  pan: "h",
  fit: "0",
  undo: "mod+z",
  redo: "mod+shift+z",
  nudge_left: "arrowleft",
  nudge_right: "arrowright",
  nudge_up: "arrowup",
  nudge_down: "arrowdown",
  // Photoshop's keys.
  brush_smaller: "[",
  brush_larger: "]",
};
export const MIN_BRUSH = 1;
export interface Brush {
  color: string;
  diameter: number;
  /** A drawing tablet's pen pressure scales the size, and the opacity (Photoshop's pen pressure dynamics). */
  pressureSize: boolean;
  pressureOpacity: boolean;
}
/** How the generative fill tool selects: a dragged rectangle or a freehand lasso. */
export type FillShape = "rect" | "lasso";
export type RunScope = "page" | "chapter" | "project";
/** How a sidebar click changes the page selection: a plain click, Ctrl/Cmd, Shift, or Ctrl+Shift. */
export type SelectMode = "replace" | "toggle" | "range" | "add-range";
// The four pipeline stages; rendering happens on demand from the scene.
export const STAGES = ["detection", "ocr", "translation", "inpainting"] as const;
export type StageName = (typeof STAGES)[number];

export interface JobState {
  id: string;
  status: "running" | "completed" | "stopped" | "failed";
  events: ProgressEvent[];
  completed: number;
  total: number;
  current?: { page: string; stage: string; model: string; kind: string };
  message?: string;
}

export interface RunOptions {
  kind: "full" | "through" | "only" | "stages";
  stage?: StageName;
  stages?: StageName[];
  /** "selection" runs the sidebar's multi-selection (the Run button, when one is active). */
  scope: RunScope | "selection";
  /** Remove-tool strokes: a page-sized PNG, white where the inpainter should repaint. */
  mask?: Uint8Array;
  /** Generative fill of `mask` by the fill model; empty removes what is selected. */
  fillPrompt?: string;
}

export interface BlockPatch {
  sourceText?: string;
  translation?: string;
  /** Rich text over `translation`, sent with it; a translation without styles is plain. */
  translationStyles?: RichStyle[];
  typography?: Typography;
  /** The typography properties the user changed (proto names); only these merge into the layer's style. */
  typographyFields?: string[];
  frame?: { x: number; y: number }[];
  visible?: boolean;
}

interface State {
  ready: boolean;
  capabilities?: Capabilities;
  config?: Config;
  translationModels: TranslationModel[];
  projects: ProjectSummary[];
  project?: Project;
  pageId?: string;
  /** Pages picked in the sidebar with Shift/Ctrl, in project order; see ``selectedPages``. */
  pageSelection: string[];
  /** Where a Shift-click range starts: the last page clicked without Shift. */
  pageAnchor?: string;
  blockId?: string;
  rasterId?: string;
  view: ViewMode;
  tool: Tool;
  brush: Brush;
  fillShape: FillShape;
  shortcuts: Record<ShortcutAction, string>;
  settingsOpen: boolean;
  settingsTab?: string;
  /** Settings was opened from the Models page, so closing it goes back there. */
  settingsFromModels?: boolean;
  /** Settings was opened from the source browser, so closing it goes back there. */
  settingsFromSources?: boolean;
  modelsOpen: boolean;
  /** The source browser, linking a series to the open project or to a new one. */
  sourcesOpen: boolean;
  /** The open project's source link; null when it has none, undefined while loading. */
  link?: LinkView | null;
  /** The built-in browser, open on a site check (or any page); `resume` runs once it closes. */
  browser?: Challenge & { resume?: () => void };
  runScope: RunScope;
  runStages: StageName[];
  theme: Theme;
  /** Color theme (neutral temperature and accent), independent of light/dark. */
  palette: Palette;
  showBlocks: boolean;
  showRegions: boolean;
  job?: JobState;
  error?: string;
}

/** A chapter's place in the project: its id and its pages in order. */
export interface ChapterLayout {
  id: string;
  pageIds: string[];
}

interface Actions {
  refresh(): Promise<void>;
  refreshModels(): Promise<void>;
  /** Re-reads a project after the server changed it (an agent's edits). */
  reloadProject(projectId: string): Promise<void>;
  /** Opens a project at ``pageId`` when it has that page, else at its first page. */
  openProject(id: string, pageId?: string): Promise<void>;
  closeProject(): void;
  createProject(name: string): Promise<void>;
  renameProject(name: string): Promise<void>;
  deleteProject(id: string): Promise<void>;
  undo(): Promise<void>;
  redo(): Promise<void>;
  uploadPages(files: FileList | File[], chapterId?: string): Promise<void>;
  removePage(pageId: string): Promise<void>;
  createChapter(name?: string): Promise<void>;
  renameChapter(chapterId: string, name: string): Promise<void>;
  deleteChapter(chapterId: string): Promise<void>;
  arrange(layout: ChapterLayout[]): Promise<void>;
  selectPage(pageId: string, mode?: SelectMode): void;
  selectBlock(blockId?: string): void;
  selectRaster(rasterId?: string): void;
  setView(view: ViewMode): void;
  setTool(tool: Tool): void;
  setBrush(brush: Brush): void;
  setFillShape(shape: FillShape): void;
  setShortcut(action: ShortcutAction, key: string): void;
  setSettingsOpen(open: boolean, tab?: string): void;
  setModelsOpen(open: boolean): void;
  setSourcesOpen(open: boolean): void;
  openBrowser(challenge: Challenge, resume?: () => void): void;
  closeBrowser(resumed: boolean): void;
  linkSeries(system: string, source: string, series: string): Promise<void>;
  unlinkSource(): Promise<void>;
  refreshSource(): Promise<void>;
  downloadChapters(chapterIds: string[]): Promise<void>;
  cancelDownload(chapterId: string): Promise<void>;
  setRunScope(scope: RunScope): void;
  setRunStages(stages: StageName[]): void;
  setLayerVisibility(layerIds: string[], visible?: boolean, opacity?: number): Promise<void>;
  moveLayer(layerId: string, delta: number): Promise<void>;
  /**
   * Replaces a raster layer's pixels; with `at`, `png` is a patch replacing only its rectangle there.
   * Resolves to the saved page, or undefined when the save failed (the error is reported).
   */
  setRasterLayer(kind: "paint" | "cleanup", layerId: string | undefined, png: Uint8Array, at?: { x: number; y: number }): Promise<Page | undefined>;
  deleteLayer(layerId: string): Promise<void>;
  /** A new, empty paint layer above the selected raster layer (or on top), which becomes the selected one. */
  addRasterLayer(): Promise<void>;
  setTheme(theme: Theme): void;
  setPalette(palette: Palette): void;
  toggleOverlay(kind: "blocks" | "regions"): void;
  run(options: RunOptions): Promise<void>;
  stop(): Promise<void>;
  updateBlock(layerId: string, patch: BlockPatch): Promise<void>;
  createBlock(bounds: { x: number; y: number; width: number; height: number }): Promise<void>;
  deleteBlock(layerId: string): Promise<void>;
  saveConfig(config: Config): Promise<void>;
  dismissError(): void;
  reportError(message: string): void;
}

const Ctx = createContext<(State & Actions) | null>(null);
const THEME_KEY = "urakatsu.theme";
const SHORTCUTS_KEY = "urakatsu.shortcuts";
const BRUSH_KEY = "urakatsu.brush";
const FILL_KEY = "urakatsu.fill";
const RUN_KEY = "urakatsu.run";

function loadJson<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? { ...fallback, ...JSON.parse(raw) } : fallback;
  } catch {
    return fallback;
  }
}

function saveJson(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable */
  }
}

function loadTheme(): Theme {
  try {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored === "light" || stored === "dark" || stored === "system") return stored;
    if (stored === "dark-blue") return "dark";
    if (stored === "light-blue" || stored === "yellow") return "light";
  } catch {
    /* storage unavailable */
  }
  return "system";
}

function applyTheme(theme: Theme): void {
  const dark = theme === "dark" || (theme === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.classList.toggle("dark", dark);
  document.documentElement.style.colorScheme = dark ? "dark" : "light";
}

export function WorkspaceProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<State>({
    ready: false,
    projects: [],
    translationModels: [],
    view: "source",
    tool: "select",
    brush: loadJson<Brush>(BRUSH_KEY, { color: "#000000", diameter: 24, pressureSize: true, pressureOpacity: true }),
    fillShape: loadJson<{ shape: FillShape }>(FILL_KEY, { shape: "rect" }).shape,
    shortcuts: loadJson(SHORTCUTS_KEY, DEFAULT_SHORTCUTS),
    // The view the URL names (/settings, /models, /sources); a project opens once loaded.
    settingsOpen: initialRoute.view === "settings",
    settingsTab: initialRoute.tab,
    modelsOpen: initialRoute.view === "models",
    sourcesOpen: initialRoute.view === "sources",
    pageSelection: [],
    runScope: loadJson<{ scope: RunScope }>(RUN_KEY, { scope: "page" }).scope,
    runStages: loadJson<{ stages: StageName[] }>(RUN_KEY, { stages: [...STAGES] }).stages,
    theme: loadTheme(),
    palette: loadPalette(),
    showBlocks: true,
    showRegions: true,
  });
  const jobRef = useRef<JobState | undefined>(undefined);
  const linkRef = useRef<LinkView | null | undefined>(undefined);
  linkRef.current = state.link;

  useEffect(() => {
    applyTheme(state.theme);
    try {
      localStorage.setItem(THEME_KEY, state.theme);
    } catch {
      /* ignore */
    }
    if (state.theme !== "system") return;
    const query = window.matchMedia("(prefers-color-scheme: dark)");
    const follow = () => applyTheme("system");
    query.addEventListener("change", follow);
    return () => query.removeEventListener("change", follow);
  }, [state.theme]);

  useEffect(() => applyPalette(state.palette), [state.palette]);

  const fail = useCallback((error: unknown) => setState((s) => ({ ...s, error: describeError(error) })), []);

  const replacePage = useCallback((page: Page) => {
    setState((s) => {
      if (!s.project) return s;
      const pages = s.project.pages.map((p) => (p.id === page.id ? page : p));
      return withLiveSelection({ ...s, project: { ...s.project, pages } as Project });
    });
  }, []);

  const reloadProject = useCallback(async (projectId: string) => {
    const project = await client.getProject({ projectId });
    setState((s) => withLiveSelection({ ...s, project }));
  }, []);

  const refreshModels = useCallback(async () => {
    try {
      const list = await client.listTranslationModels({});
      setState((s) => ({ ...s, translationModels: list.models }));
    } catch (error) {
      fail(error);
    }
  }, [fail]);

  const refresh = useCallback(async () => {
    try {
      const [capabilities, config, list] = await Promise.all([client.getCapabilities({}), client.getConfig({}), client.listProjects({})]);
      setState((s) => ({ ...s, ready: true, capabilities, config, projects: list.projects }));
      void refreshModels();
    } catch (error) {
      setState((s) => ({ ...s, ready: true }));
      fail(error);
    }
  }, [fail, refreshModels]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const openProject = useCallback(
    async (id: string, pageId?: string) => {
      try {
        const project = await client.getProject({ projectId: id });
        // Start on the wanted page: selecting the first one first would load its image for nothing.
        const page = project.pages.some((p) => p.id === pageId) ? pageId : project.pages[0]?.id;
        setState((s) => ({ ...s, project, link: undefined, pageId: page, blockId: undefined, job: undefined, view: "source" }));
        const link = await sources.link(id);
        setState((s) => (s.project?.id === id ? { ...s, link } : s));
      } catch (error) {
        fail(error);
      }
    },
    [fail],
  );

  const closeProject = useCallback(() => {
    setState((s) => ({ ...s, project: undefined, link: undefined, pageId: undefined, blockId: undefined, job: undefined }));
    void refresh();
  }, [refresh]);

  // Browser history: Back, Forward and reloads restore the view their URL names (history.ts).
  const openProjectId = useRef<string | undefined>(undefined);
  openProjectId.current = state.project?.id;
  const applyRoute = useCallback(
    async (route: Route) => {
      setState((s) => ({
        ...s,
        settingsOpen: route.view === "settings",
        settingsTab: route.view === "settings" ? route.tab : s.settingsTab,
        settingsFromModels: false,
        settingsFromSources: false,
        modelsOpen: route.view === "models",
        sourcesOpen: route.view === "sources",
      }));
      if (route.view === "home") {
        if (openProjectId.current) closeProject();
        return;
      }
      if (route.view === "project" && route.projectId) {
        if (openProjectId.current !== route.projectId) await openProject(route.projectId, route.pageId);
        const pageId = route.pageId;
        if (pageId) {
          setState((s) => (s.project?.pages.some((p) => p.id === pageId) ? { ...s, pageId, blockId: undefined, rasterId: undefined } : s));
        }
      }
    },
    [openProject, closeProject],
  );
  useHistorySync(state, applyRoute);

  const createProject = useCallback(
    async (name: string) => {
      try {
        const project = await client.createProject({ name });
        const list = await client.listProjects({});
        setState((s) => ({ ...s, project, link: null, projects: list.projects, pageId: undefined, blockId: undefined, job: undefined }));
      } catch (error) {
        fail(error);
      }
    },
    [fail],
  );

  const renameProject = useCallback(
    async (name: string) => {
      if (!state.project) return;
      try {
        const project = await client.renameProject({ projectId: state.project.id, name });
        const list = await client.listProjects({});
        setState((s) => ({ ...s, project, projects: list.projects }));
      } catch (error) {
        fail(error);
      }
    },
    [state.project, fail],
  );

  const deleteProject = useCallback(
    async (id: string) => {
      try {
        await client.deleteProject({ projectId: id });
        forgetAssets(id);
        setState((s) => ({
          ...s,
          project: s.project?.id === id ? undefined : s.project,
          pageId: s.project?.id === id ? undefined : s.pageId,
          projects: s.projects.filter((p) => p.id !== id),
        }));
      } catch (error) {
        fail(error);
      }
    },
    [fail],
  );

  const history = useCallback(
    async (direction: "undo" | "redo") => {
      const project = state.project;
      if (!project || jobRef.current?.status === "running") return;
      try {
        const next = direction === "undo" ? await client.undo({ projectId: project.id }) : await client.redo({ projectId: project.id });
        setState((s) => ({
          ...s,
          project: next,
          pageId: next.pages.some((p) => p.id === s.pageId) ? s.pageId : next.pages[0]?.id,
          blockId: undefined,
        }));
      } catch (error) {
        fail(error);
      }
    },
    [state.project, fail],
  );

  const uploadPages = useCallback(
    async (files: FileList | File[], chapterId?: string) => {
      const project = state.project;
      if (!project) return;
      const list = Array.from(files).filter((f) => f.type.startsWith("image/") || /\.(png|jpe?g|webp|bmp)$/i.test(f.name));
      list.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
      try {
        let first: string | undefined;
        for (const file of list) {
          const page = await client.addPage({
            projectId: project.id,
            name: file.name,
            data: new Uint8Array(await file.arrayBuffer()),
            chapterId: chapterId ?? "",
          });
          first ??= page.id;
        }
        await reloadProject(project.id);
        setState((s) => ({ ...s, pageId: s.pageId ?? first }));
      } catch (error) {
        fail(error);
      }
    },
    [state.project, fail, reloadProject],
  );

  const removePage = useCallback(
    async (pageId: string) => {
      if (!state.project) return;
      try {
        const project = await client.removePage({ projectId: state.project.id, pageId });
        forgetAssets(state.project.id, pageId);
        setState((s) => ({
          ...s,
          project,
          pageId: s.pageId === pageId ? project.pages[0]?.id : s.pageId,
          blockId: s.pageId === pageId ? undefined : s.blockId,
        }));
      } catch (error) {
        fail(error);
      }
    },
    [state.project, fail],
  );

  const restructure = useCallback(
    async (call: (projectId: string) => Promise<Project>) => {
      const project = state.project;
      if (!project) return;
      try {
        const next = await call(project.id);
        setState((s) => ({
          ...s,
          project: next,
          pageId: next.pages.some((p) => p.id === s.pageId) ? s.pageId : next.pages[0]?.id,
        }));
      } catch (error) {
        fail(error);
        await reloadProject(project.id).catch(() => undefined);
      }
    },
    [state.project, fail, reloadProject],
  );

  const createChapter = useCallback(
    (name?: string) => restructure((projectId) => client.createChapter({ projectId, name: name ?? "" })),
    [restructure],
  );

  const renameChapter = useCallback(
    (chapterId: string, name: string) => restructure((projectId) => client.renameChapter({ projectId, chapterId, name })),
    [restructure],
  );

  const deleteChapter = useCallback(
    async (chapterId: string) => {
      const project = state.project;
      const doomed = project?.chapters.find((c) => c.id === chapterId)?.pageIds ?? [];
      await restructure((projectId) => client.deleteChapter({ projectId, chapterId }));
      if (project) for (const pageId of doomed) forgetAssets(project.id, pageId);
    },
    [state.project, restructure],
  );

  const arrange = useCallback(
    async (layout: ChapterLayout[]) => {
      // Optimistic: the sidebar already shows the new order, so apply it locally first.
      setState((s) => {
        if (!s.project) return s;
        const byId = new Map(s.project.pages.map((p) => [p.id, p]));
        const known = new Map(s.project.chapters.map((c) => [c.id, c]));
        const chapters = layout.map((entry) => ({ ...known.get(entry.id)!, pageIds: entry.pageIds }) as Chapter);
        const pages = layout.flatMap((entry) => entry.pageIds.map((id) => byId.get(id))).filter((p): p is Page => !!p);
        return { ...s, project: { ...s.project, chapters, pages } as Project };
      });
      await restructure((projectId) =>
        client.arrangeProject({ projectId, chapters: layout.map((entry) => ({ chapterId: entry.id, pageIds: entry.pageIds })) }),
      );
    },
    [restructure],
  );

  const run = useCallback(
    async (options: RunOptions) => {
      const project = state.project;
      if (!project || jobRef.current?.status === "running") return;
      const chapterPages = project.chapters.find((c) => c.pageIds.includes(state.pageId ?? ""))?.pageIds ?? [];
      const pageIds =
        options.scope === "selection" ? selectedPages(state) : options.scope === "page" && state.pageId ? [state.pageId] : options.scope === "chapter" ? chapterPages : [];
      if ((options.scope === "chapter" || options.scope === "selection") && pageIds.length === 0) return;
      const job: JobState = { id: "", status: "running", events: [], completed: 0, total: 0 };
      jobRef.current = job;
      setState((s) => ({ ...s, job: { ...job } }));
      try {
        const stream = client.run({
          projectId: project.id,
          pageIds,
          operation: { kind: options.kind, stage: options.stage ?? "", stages: options.stages ?? [] },
          inpaintingMask: options.mask,
          fillPrompt: options.fillPrompt,
        });
        for await (const event of stream) {
          job.id = event.jobId;
          job.events.push(event);
          if (event.total) job.total = event.total;
          if (event.completed) job.completed = event.completed;
          if (event.kind === "loading" || event.kind === "running") {
            job.current = { page: event.pageId, stage: event.stage, model: event.model, kind: event.kind };
          }
          if (event.page) replacePage(event.page);
          if (!event.pageId && (event.kind === "completed" || event.kind === "stopped" || event.kind === "failed")) {
            job.status = event.kind === "completed" ? "completed" : event.kind === "stopped" ? "stopped" : "failed";
            job.message = event.message || undefined;
            job.current = undefined;
          }
          setState((s) => ({ ...s, job: { ...job, events: [...job.events] } }));
        }
        if (job.status === "running") job.status = "completed";
        job.current = undefined;
        const fresh = await client.getProject({ projectId: project.id });
        setState((s) => ({
          ...s,
          job: { ...job, events: [...job.events] },
          project: fresh,
          view: s.view === "source" && job.status === "completed" ? "render" : s.view,
        }));
      } catch (error) {
        job.status = "failed";
        job.message = describeError(error);
        job.current = undefined;
        setState((s) => ({ ...s, job: { ...job, events: [...job.events] } }));
        fail(error);
      }
    },
    [state.project, state.pageId, replacePage, fail],
  );

  const stop = useCallback(async () => {
    const job = jobRef.current;
    if (!job?.id) return;
    try {
      await client.stop({ jobId: job.id });
    } catch (error) {
      fail(error);
    }
  }, [fail]);

  const updateBlock = useCallback(
    async (layerId: string, patch: BlockPatch) => {
      const project = state.project;
      const pageId = state.pageId;
      if (!project || !pageId) return;
      const fields: string[] = [];
      if (patch.sourceText !== undefined) fields.push("source_text");
      if (patch.translation !== undefined) fields.push("translation");
      if (patch.typography !== undefined) fields.push("typography", ...(patch.typographyFields ?? []).map((key) => `typography.${key}`));
      if (patch.frame !== undefined) fields.push("frame");
      if (patch.visible !== undefined) fields.push("visible");
      try {
        const page = await client.updateTextBlock({
          projectId: project.id,
          pageId,
          layerId,
          sourceText: patch.sourceText,
          translation: patch.translation,
          translationStyles: patch.translationStyles ?? [],
          typography: patch.typography,
          frame: patch.frame ?? [],
          visible: patch.visible,
          fields,
        });
        replacePage(page);
        setState((s) => (s.project ? { ...s, project: { ...s.project, canUndo: true, canRedo: false } as Project } : s));
      } catch (error) {
        fail(error);
      }
    },
    [state.project, state.pageId, replacePage, fail],
  );

  const createBlock = useCallback(
    async (bounds: { x: number; y: number; width: number; height: number }) => {
      const project = state.project;
      const pageId = state.pageId;
      if (!project || !pageId) return;
      try {
        const before = new Set(project.pages.find((p) => p.id === pageId)?.blocks.map((b) => b.layerId));
        const page = await client.createTextBlock({ projectId: project.id, pageId, bounds });
        replacePage(page);
        const created = page.blocks.find((b) => !before.has(b.layerId));
        setState((s) => ({ ...s, blockId: created?.layerId, tool: "select" }));
      } catch (error) {
        fail(error);
      }
    },
    [state.project, state.pageId, replacePage, fail],
  );

  const deleteBlock = useCallback(
    async (layerId: string) => {
      const project = state.project;
      const pageId = state.pageId;
      if (!project || !pageId) return;
      try {
        replacePage(await client.deleteTextBlock({ projectId: project.id, pageId, layerId }));
        setState((s) => ({ ...s, blockId: s.blockId === layerId ? undefined : s.blockId }));
      } catch (error) {
        fail(error);
      }
    },
    [state.project, state.pageId, replacePage, fail],
  );

  const pageCall = useCallback(
    async (call: (projectId: string, pageId: string) => Promise<Page>) => {
      const project = state.project;
      const pageId = state.pageId;
      if (!project || !pageId) return;
      try {
        replacePage(await call(project.id, pageId));
        setState((s) => (s.project ? { ...s, project: { ...s.project, canUndo: true, canRedo: false } as Project } : s));
      } catch (error) {
        fail(error);
      }
    },
    [state.project, state.pageId, replacePage, fail],
  );

  const setLayerVisibility = useCallback(
    (layerIds: string[], visible?: boolean, opacity?: number) =>
      pageCall((projectId, pageId) => client.setLayerVisibility({ projectId, pageId, layerIds, visible, opacity })),
    [pageCall],
  );

  const moveLayer = useCallback(
    (layerId: string, delta: number) => pageCall((projectId, pageId) => client.moveLayer({ projectId, pageId, layerId, delta })),
    [pageCall],
  );

  // Like pageCall, but hands back the saved page: the canvas keeps its revision to know the layer pixels it already shows.
  const setRasterLayer = useCallback(
    async (kind: "paint" | "cleanup", layerId: string | undefined, png: Uint8Array, at?: { x: number; y: number }) => {
      let saved: Page | undefined;
      const patch = at ? { patch: true, x: at.x, y: at.y } : {};
      await pageCall(async (projectId, pageId) => (saved = await client.setRasterLayer({ projectId, pageId, kind, layerId: layerId ?? "", png, ...patch })));
      return saved;
    },
    [pageCall],
  );

  const addRasterLayer = useCallback(async () => {
    const before = new Set(currentPage(state)?.rasterLayers.map((layer) => layer.id));
    const above = state.rasterId && before.has(state.rasterId) ? state.rasterId : "";
    let saved: Page | undefined;
    await pageCall(async (projectId, pageId) => (saved = await client.addRasterLayer({ projectId, pageId, aboveLayerId: above })));
    const created = saved?.rasterLayers.find((layer) => !before.has(layer.id));
    if (created) setState((s) => ({ ...s, rasterId: created.id, blockId: undefined }));
  }, [state, pageCall]);

  const deleteLayer = useCallback(
    async (layerId: string) => {
      await pageCall((projectId, pageId) => client.deleteLayer({ projectId, pageId, layerId }));
      setState((s) => ({
        ...s,
        blockId: s.blockId === layerId ? undefined : s.blockId,
        rasterId: s.rasterId === layerId ? undefined : s.rasterId,
      }));
    },
    [pageCall],
  );

  const linkSeries = useCallback(
    async (system: string, source: string, series: string) => {
      try {
        const project = state.project ?? (await client.createProject({ name: "Untitled" }));
        let link: LinkView;
        try {
          link = await sources.setLink(project.id, { system, source, series });
        } catch (error) {
          const challenge = challengeOf(error);
          if (!challenge) throw error;
          setState((s) => ({ ...s, browser: { ...challenge, resume: () => void linkSeries(system, source, series) } }));
          return;
        }
        const [fresh, list] = await Promise.all([client.getProject({ projectId: project.id }), client.listProjects({})]);
        setState((s) => ({
          ...s,
          project: fresh,
          link,
          projects: list.projects,
          sourcesOpen: false,
          pageId: fresh.pages.some((p) => p.id === s.pageId) ? s.pageId : fresh.pages[0]?.id,
        }));
      } catch (error) {
        fail(error);
      }
    },
    [state.project, fail],
  );

  const linkCall = useCallback(
    async (call: (projectId: string) => Promise<LinkView | null | void>) => {
      const project = state.project;
      if (!project) return;
      try {
        const link = await call(project.id);
        setState((s) => (s.project?.id === project.id ? { ...s, link: link ?? null } : s));
      } catch (error) {
        const challenge = challengeOf(error);
        if (challenge) setState((s) => ({ ...s, browser: { ...challenge, resume: () => void linkCall(call) } }));
        else fail(error);
      }
    },
    [state.project, fail],
  );

  const unlinkSource = useCallback(() => linkCall((projectId) => sources.unlink(projectId)), [linkCall]);
  const refreshSource = useCallback(() => linkCall((projectId) => sources.refresh(projectId)), [linkCall]);
  const downloadChapters = useCallback((ids: string[]) => linkCall((projectId) => sources.download(projectId, ids)), [linkCall]);
  const cancelDownload = useCallback((id: string) => linkCall((projectId) => sources.cancel(projectId, id)), [linkCall]);

  // While chapters download, follow their progress; a finished chapter means new pages to load.
  const downloading = !!state.link?.chapters.some((c) => c.download && c.download.state !== "failed");
  const linkedProject = state.project?.id;
  useEffect(() => {
    if (!downloading || !linkedProject) return;
    const timer = window.setInterval(async () => {
      try {
        const link = await sources.link(linkedProject);
        const landed = (current?: LinkView | null) => {
          const before = new Set(current?.chapters.filter((c) => c.chapter_id).map((c) => c.id));
          return !!link?.chapters.some((c) => c.chapter_id && !before.has(c.id));
        };
        let fresh: Project | undefined;
        if (landed(linkRef.current)) fresh = await client.getProject({ projectId: linkedProject });
        setState((s) =>
          s.project?.id !== linkedProject ? s : fresh ? { ...s, link, project: fresh, pageId: s.pageId ?? fresh.pages[0]?.id } : { ...s, link },
        );
      } catch {
        /* the next tick retries */
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [downloading, linkedProject]);

  const saveConfig = useCallback(
    async (config: Config) => {
      try {
        const saved = await client.setConfig(config);
        const capabilities = await client.getCapabilities({});
        setState((s) => ({ ...s, config: saved, capabilities }));
        void refreshModels();
      } catch (error) {
        fail(error);
      }
    },
    [fail, refreshModels],
  );

  const actions: Actions = useMemo(
    () => ({
      refresh,
      refreshModels,
      reloadProject,
      openProject,
      closeProject,
      createProject,
      renameProject,
      deleteProject,
      undo: () => history("undo"),
      redo: () => history("redo"),
      uploadPages,
      removePage,
      createChapter,
      renameChapter,
      deleteChapter,
      arrange,
      selectPage: (pageId, mode = "replace") => setState((s) => ({ ...s, ...nextSelection(s, pageId, mode), blockId: undefined, rasterId: undefined })),
      selectBlock: (blockId) => setState((s) => ({ ...s, blockId, rasterId: blockId ? undefined : s.rasterId })),
      selectRaster: (rasterId) => setState((s) => ({ ...s, rasterId, blockId: rasterId ? undefined : s.blockId })),
      setView: (view) => setState((s) => ({ ...s, view })),
      setTool: (tool) => setState((s) => ({ ...s, tool })),
      setBrush: (brush) => {
        saveJson(BRUSH_KEY, brush);
        setState((s) => ({ ...s, brush }));
      },
      setFillShape: (fillShape) => {
        saveJson(FILL_KEY, { shape: fillShape });
        setState((s) => ({ ...s, fillShape }));
      },
      setShortcut: (action, key) =>
        setState((s) => {
          const shortcuts = { ...s.shortcuts, [action]: key.toLowerCase() };
          saveJson(SHORTCUTS_KEY, shortcuts);
          return { ...s, shortcuts };
        }),
      setSettingsOpen: (settingsOpen, settingsTab) =>
        setState((s) =>
          settingsOpen
            ? { ...s, settingsOpen, settingsTab, settingsFromModels: s.modelsOpen, modelsOpen: false, settingsFromSources: s.sourcesOpen, sourcesOpen: false }
            : { ...s, settingsOpen, modelsOpen: !!s.settingsFromModels, settingsFromModels: false, sourcesOpen: !!s.settingsFromSources, settingsFromSources: false },
        ),
      setModelsOpen: (modelsOpen) => setState((s) => ({ ...s, modelsOpen, settingsOpen: false })),
      setSourcesOpen: (sourcesOpen) => setState((s) => ({ ...s, sourcesOpen, settingsOpen: false, modelsOpen: false })),
      openBrowser: (challenge, resume) => setState((s) => ({ ...s, browser: { ...challenge, resume } })),
      closeBrowser: (resumed) =>
        setState((s) => {
          if (resumed) window.setTimeout(() => s.browser?.resume?.(), 0);
          return { ...s, browser: undefined };
        }),
      linkSeries,
      unlinkSource,
      refreshSource,
      downloadChapters,
      cancelDownload,
      setRunScope: (runScope) =>
        setState((s) => {
          saveJson(RUN_KEY, { scope: runScope, stages: s.runStages });
          return { ...s, runScope };
        }),
      setRunStages: (runStages) =>
        setState((s) => {
          saveJson(RUN_KEY, { scope: s.runScope, stages: runStages });
          return { ...s, runStages };
        }),
      setLayerVisibility,
      moveLayer,
      setRasterLayer,
      deleteLayer,
      addRasterLayer,
      setTheme: (theme) => setState((s) => ({ ...s, theme })),
      setPalette: (palette) => setState((s) => ({ ...s, palette })),
      toggleOverlay: (kind) => setState((s) => (kind === "blocks" ? { ...s, showBlocks: !s.showBlocks } : { ...s, showRegions: !s.showRegions })),
      run,
      stop,
      updateBlock,
      createBlock,
      deleteBlock,
      saveConfig,
      dismissError: () => setState((s) => ({ ...s, error: undefined })),
      reportError: (message) => setState((s) => ({ ...s, error: message })),
    }),
    [
      refresh,
      refreshModels,
      reloadProject,
      openProject,
      closeProject,
      createProject,
      renameProject,
      deleteProject,
      history,
      uploadPages,
      removePage,
      createChapter,
      renameChapter,
      deleteChapter,
      arrange,
      run,
      stop,
      updateBlock,
      createBlock,
      deleteBlock,
      saveConfig,
      setLayerVisibility,
      moveLayer,
      setRasterLayer,
      deleteLayer,
      addRasterLayer,
      linkSeries,
      unlinkSource,
      refreshSource,
      downloadChapters,
      cancelDownload,
    ],
  );

  const value = useMemo(() => ({ ...state, ...actions }), [state, actions]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useWorkspace(): State & Actions {
  const value = useContext(Ctx);
  if (!value) throw new Error("useWorkspace outside WorkspaceProvider");
  return value;
}

/** Drops a selected block or raster layer the page no longer has: a re-run, an undo or the agent
 * replaced the page's layers, and an edit sent for the old id would fail ("has no text layer"). */
function withLiveSelection(state: State): State {
  const page = currentPage(state);
  const blockId = state.blockId && page?.blocks.some((b) => b.layerId === state.blockId) ? state.blockId : undefined;
  const rasterId = state.rasterId && page?.rasterLayers.some((r) => r.id === state.rasterId) ? state.rasterId : undefined;
  return blockId === state.blockId && rasterId === state.rasterId ? state : { ...state, blockId, rasterId };
}

export function currentPage(state: State): Page | undefined {
  return state.project?.pages.find((p) => p.id === state.pageId);
}

export function currentRaster(state: State): RasterLayer | undefined {
  return currentPage(state)?.rasterLayers.find((layer) => layer.id === state.rasterId);
}

export function currentBlock(state: State): TextBlock | undefined {
  return currentPage(state)?.blocks.find((b) => b.layerId === state.blockId);
}

export function stageDone(page: Page, stage: StageName): boolean {
  return page.stages.some((s) => s.stage === stage && s.done);
}

/** Pages finished through every stage. */
export function pageComplete(page: Page): boolean {
  return STAGES.every((stage) => stageDone(page, stage));
}

/** The pages a sidebar multi-selection holds, in project order; empty unless 2+ pages including the current one. */
export function selectedPages(state: Pick<State, "project" | "pageId" | "pageSelection">): string[] {
  const order = state.project?.chapters.flatMap((c) => c.pageIds) ?? [];
  const picked = order.filter((id) => state.pageSelection.includes(id));
  return picked.length > 1 && state.pageId && picked.includes(state.pageId) ? picked : [];
}

/**
 * Applies a sidebar click, as file managers do: a plain click selects one page, Ctrl/Cmd adds or
 * removes one, Shift selects the run from the anchor (the last page clicked without Shift), and
 * Ctrl+Shift adds that run to the selection. The clicked page becomes the current one.
 */
function nextSelection(s: State, pageId: string, mode: SelectMode): Pick<State, "pageId" | "pageSelection" | "pageAnchor"> {
  const order = s.project?.chapters.flatMap((c) => c.pageIds) ?? [];
  const current = selectedPages(s);
  const base = current.length ? current : s.pageId ? [s.pageId] : [];
  let selection: string[];
  let anchor = s.pageAnchor && order.includes(s.pageAnchor) ? s.pageAnchor : s.pageId;
  if (mode === "toggle") {
    selection = base.includes(pageId) ? base.filter((id) => id !== pageId) : [...base, pageId];
    anchor = pageId;
  } else if (mode === "range" || mode === "add-range") {
    const from = order.indexOf(anchor ?? pageId);
    const to = order.indexOf(pageId);
    const run = from < 0 || to < 0 ? [pageId] : order.slice(Math.min(from, to), Math.max(from, to) + 1);
    selection = mode === "add-range" ? [...base, ...run] : run;
  } else {
    selection = [pageId];
    anchor = pageId;
  }
  selection = order.filter((id) => selection.includes(id));
  const page = selection.includes(pageId) ? pageId : (selection[selection.length - 1] ?? pageId);
  return { pageId: page, pageSelection: selection.length ? selection : [page], pageAnchor: anchor };
}

export function chapterOf(state: State, pageId: string | undefined): Chapter | undefined {
  if (!pageId) return undefined;
  return state.project?.chapters.find((c) => c.pageIds.includes(pageId));
}
