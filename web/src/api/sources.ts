// JSON client for source systems and project links (/api/v1/sources, /api/v1/projects/{id}/source).
// Source system: a scraper runtime (Suwayomi Server, Mana). Source: one site inside it.
// Series: one title on a source.

import { call } from "./catalog";

export type SystemState = "not_installed" | "stopped" | "starting" | "running" | "error";

export interface SystemStatus {
  id: string;
  name: string;
  description: string;
  homepage: string;
  state: SystemState;
  runtime: string | null;
  version: string | null;
  message: string | null;
  enabled: boolean;
  /** The built-in browser, for systems that have one. */
  browser: string | null;
}

export interface Repository {
  url: string;
  name: string;
  extensions: number;
}

export interface Extension {
  id: string;
  name: string;
  version: string;
  language: string;
  icon: string | null;
  nsfw: boolean;
  installed: boolean;
  has_update: boolean;
  repository: string | null;
  description: string;
}

export interface SourceInfo {
  id: string;
  name: string;
  language: string;
  icon: string | null;
  nsfw: boolean;
  extension: string | null;
  website: string | null;
}

export interface SystemSources {
  system: SystemStatus;
  sources: SourceInfo[];
  error: string | null;
}

export interface Series {
  id: string;
  title: string;
  cover: string | null;
  url: string | null;
  summary: string;
  authors: string[];
  status: string | null;
  tags: string[];
}

export interface SearchPage {
  results: Series[];
  has_more: boolean;
}

export interface RemoteChapter {
  id: string;
  name: string;
  number: number | null;
  volume: number | null;
  scanlator: string | null;
  language: string | null;
  date: number | null;
}

export interface DownloadState {
  state: "queued" | "downloading" | "importing" | "stitching" | "failed";
  done: number;
  total: number;
  error: string | null;
  /** A page to clear in the system's browser before retrying. */
  challenge: string | null;
}

export interface BrowserFrame {
  /** Base64 JPEG of the page. */
  image: string;
  url: string;
  title: string;
  loading: boolean;
  /** The page still shows a site check. */
  challenge: boolean;
  width: number;
  height: number;
}

export type BrowserInput =
  | { action: "click"; x: number; y: number }
  | { action: "type"; text: string }
  | { action: "press"; key: string }
  | { action: "scroll"; dx: number; dy: number }
  | { action: "navigate"; url: string }
  | { action: "back" | "forward" | "reload" };

export interface LinkedChapter extends RemoteChapter {
  chapter_id: string | null;
  download: DownloadState | null;
}

export interface LinkView {
  system: string;
  system_name: string;
  source: string;
  source_name: string;
  source_language: string;
  series: Series;
  refreshed_at: number;
  chapters: LinkedChapter[];
}

const q = (params: Record<string, string | number | boolean>) =>
  new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)])).toString();

const system = (id: string) => `/sources/systems/${encodeURIComponent(id)}`;
const source = (systemId: string, sourceId: string) => `${system(systemId)}/sources/${encodeURIComponent(sourceId)}`;
const json = (method: string, body: unknown): RequestInit => ({ method, body: JSON.stringify(body) });

export const sources = {
  systems: () => call<SystemStatus[]>("/sources/systems"),
  setEnabled: (id: string, enabled: boolean) => call<SystemStatus>(`${system(id)}/enabled`, json("PUT", { enabled })),
  start: (id: string) => call<SystemStatus>(`${system(id)}/start`, { method: "POST" }),
  stop: (id: string) => call<SystemStatus>(`${system(id)}/stop`, { method: "POST" }),
  repositories: (id: string) => call<Repository[]>(`${system(id)}/repositories`),
  addRepository: (id: string, url: string) => call<Repository[]>(`${system(id)}/repositories`, json("POST", { url })),
  removeRepository: (id: string, url: string) => call<Repository[]>(`${system(id)}/repositories?${q({ url })}`, { method: "DELETE" }),
  extensions: (id: string, refresh = false) => call<Extension[]>(`${system(id)}/extensions?${q({ refresh })}`),
  install: (id: string, ext: string) => call<Extension[]>(`${system(id)}/extensions/${encodeURIComponent(ext)}/install`, { method: "POST" }),
  update: (id: string, ext: string) => call<Extension[]>(`${system(id)}/extensions/${encodeURIComponent(ext)}/update`, { method: "POST" }),
  uninstall: (id: string, ext: string) => call<Extension[]>(`${system(id)}/extensions/${encodeURIComponent(ext)}`, { method: "DELETE" }),

  browserOpen: (id: string, url: string) => call<{ id: string }>(`${system(id)}/browser`, json("POST", { url })),
  browserFrame: (id: string, session: string) => call<BrowserFrame>(`${system(id)}/browser/${session}/frame`),
  browserInput: (id: string, session: string, input: BrowserInput) => call<void>(`${system(id)}/browser/${session}/input`, json("POST", input)),
  browserClose: (id: string, session: string) => call<void>(`${system(id)}/browser/${session}`, { method: "DELETE" }),

  all: () => call<SystemSources[]>("/sources"),
  search: (systemId: string, sourceId: string, query: string, page = 1) =>
    call<SearchPage>(`${source(systemId, sourceId)}/search?${q({ q: query, page })}`),
  series: (systemId: string, sourceId: string, id: string) => call<Series>(`${source(systemId, sourceId)}/series?${q({ id })}`),
  chapters: (systemId: string, sourceId: string, series: string) =>
    call<RemoteChapter[]>(`${source(systemId, sourceId)}/chapters?${q({ series })}`),
  coverUrl: (systemId: string, sourceId: string, url: string) => `/api/v1${source(systemId, sourceId)}/cover?${q({ url })}`,

  link: (projectId: string) => call<LinkView | null>(`/projects/${projectId}/source`),
  setLink: (projectId: string, body: { system: string; source: string; series: string }) =>
    call<LinkView>(`/projects/${projectId}/source`, json("PUT", body)),
  unlink: (projectId: string) => call<void>(`/projects/${projectId}/source`, { method: "DELETE" }),
  refresh: (projectId: string) => call<LinkView>(`/projects/${projectId}/source/refresh`, { method: "POST" }),
  download: (projectId: string, chapterIds: string[]) =>
    call<LinkView>(`/projects/${projectId}/source/downloads`, json("POST", { chapter_ids: chapterIds })),
  cancel: (projectId: string, chapterId: string) =>
    call<LinkView>(`/projects/${projectId}/source/downloads?${q({ chapter: chapterId })}`, { method: "DELETE" }),
  projectCoverUrl: (projectId: string, version: string | number) => `/api/v1/projects/${projectId}/cover?v=${version}`,
};

/** "Ch. 12" style label from a chapter's number, or null when the source gives none. */
export function chapterNumber(chapter: RemoteChapter): string | null {
  if (chapter.number === null) return null;
  return `Ch. ${Number.isInteger(chapter.number) ? chapter.number : chapter.number.toFixed(1)}`;
}

export const LANGUAGE_NAMES = new Intl.DisplayNames(undefined, { type: "language" });

export function languageName(tag: string): string {
  if (!tag || tag === "all") return "Multiple languages";
  try {
    return LANGUAGE_NAMES.of(tag.replace("_", "-")) ?? tag;
  } catch {
    return tag;
  }
}
