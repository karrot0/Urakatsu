// Browser history for the workspace: each view gets a URL, so Back, Forward, reloads and
// links work. The workspace keeps its own state (which view is open, which project and
// page); this module mirrors that state into the URL and applies URL changes back to it.
//
//   /                          projects
//   /projects/<id>?page=<id>   a project (page changes replace the entry instead of adding one)
//   /models                    the model catalog
//   /settings[/<tab>]          settings
//   /sources                   the source browser

import { useEffect, useRef, useState } from "react";

export type RouteView = "home" | "project" | "models" | "settings" | "sources";

export interface Route {
  view: RouteView;
  projectId?: string;
  pageId?: string;
  tab?: string;
}

/** The part of the workspace state a URL describes. */
export interface RoutedState {
  ready: boolean;
  settingsOpen: boolean;
  settingsTab?: string;
  modelsOpen: boolean;
  sourcesOpen: boolean;
  project?: { id: string; pages: { id: string }[] };
  pageId?: string;
}

export function parseRoute(pathname: string, search: string): Route {
  const parts = pathname.split("/").filter(Boolean).map(decodeURIComponent);
  const page = new URLSearchParams(search).get("page") ?? undefined;
  switch (parts[0]) {
    case "projects":
      return parts[1] ? { view: "project", projectId: parts[1], pageId: page } : { view: "home" };
    case "models":
      return { view: "models" };
    case "settings":
      return { view: "settings", tab: parts[1] };
    case "sources":
      return { view: "sources" };
    default:
      return { view: "home" };
  }
}

export function pathFor(state: RoutedState): string {
  if (state.settingsOpen) return `/settings${state.settingsTab ? `/${encodeURIComponent(state.settingsTab)}` : ""}`;
  if (state.modelsOpen) return "/models";
  if (state.sourcesOpen) return "/sources";
  if (state.project) {
    const page = state.pageId ? `?page=${encodeURIComponent(state.pageId)}` : "";
    return `/projects/${encodeURIComponent(state.project.id)}${page}`;
  }
  return "/";
}

/** The route the app opened on, for the workspace's initial state. */
export const initialRoute: Route = parseRoute(window.location.pathname, window.location.search);

// The paths of this tab's history entries, so the app can tell when a view change returns to
// the previous entry (an in-app Back button) and step back rather than add an entry.
const STACK_KEY = "urakatsu.history";

function loadStack(): string[] {
  try {
    const stored = JSON.parse(sessionStorage.getItem(STACK_KEY) ?? "[]");
    return Array.isArray(stored) ? stored : [];
  } catch {
    return [];
  }
}

function saveStack(stack: string[]): void {
  try {
    sessionStorage.setItem(STACK_KEY, JSON.stringify(stack));
  } catch {
    /* private mode: the stack only lives in memory */
  }
}

function entryIndex(): number {
  const value = (window.history.state as { urakatsu?: number } | null)?.urakatsu;
  return typeof value === "number" ? value : 0;
}

function currentPath(): string {
  return window.location.pathname + window.location.search;
}

/**
 * Keeps the URL and the workspace in step.
 *
 * `apply` receives a route from Back/Forward and updates the workspace; it returns a promise
 * while it opens a project, during which the URL is left alone.
 */
export function useHistorySync(state: RoutedState, apply: (route: Route) => Promise<void> | void): void {
  const stack = useRef<string[]>(loadStack());
  // While a route is being applied (a project loading), the state does not describe the URL yet.
  const pending = useRef(initialRoute.view === "project");
  const applyRef = useRef(apply);
  applyRef.current = apply;
  // After a route was applied, the URL is corrected in place if the app could not follow it
  // (a deleted project): `settled` re-runs the sync once the resulting state has rendered.
  const correcting = useRef(false);
  const [settled, setSettled] = useState(0);
  const finish = () => {
    pending.current = false;
    correcting.current = true;
    setSettled((n) => n + 1);
  };

  // The first entry: this tab's existing history, or the URL it opened on.
  useEffect(() => {
    const index = entryIndex();
    stack.current[index] = currentPath();
    stack.current.length = index + 1;
    saveStack(stack.current);
    window.history.replaceState({ ...(window.history.state ?? {}), urakatsu: index }, "", currentPath());
  }, []);

  // The route the app opened on, once the workspace has loaded.
  useEffect(() => {
    if (!state.ready || initialRoute.view !== "project") return;
    void Promise.resolve(applyRef.current(initialRoute)).finally(finish);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.ready]);

  // Back and Forward.
  useEffect(() => {
    const onPop = () => {
      const index = entryIndex();
      stack.current[index] = currentPath();
      saveStack(stack.current);
      const route = parseRoute(window.location.pathname, window.location.search);
      const result = applyRef.current(route);
      if (result instanceof Promise) {
        pending.current = true;
        void result.finally(finish);
      }
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  // State changes made in the app: a new entry, the previous one again, or a quiet update.
  const path = pathFor(state);
  useEffect(() => {
    if (!state.ready || pending.current) return;
    const here = currentPath();
    const correction = correcting.current;
    correcting.current = false;
    if (path === here) return;
    const index = entryIndex();
    const samePlace = path.split("?")[0] === window.location.pathname;
    if (samePlace || correction) {
      // Another page of the same project, or a URL the app could not follow: update the URL
      // without adding an entry.
      window.history.replaceState({ urakatsu: index }, "", path);
      stack.current[index] = path;
    } else if (index > 0 && stack.current[index - 1] === path) {
      // An in-app Back button returned to the previous view: behave like the browser's Back.
      window.history.back();
      return;
    } else {
      stack.current.length = index + 1;
      stack.current.push(path);
      window.history.pushState({ urakatsu: index + 1 }, "", path);
    }
    saveStack(stack.current);
  }, [path, state.ready, settled]);
}
