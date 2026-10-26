// JSON client for Komga (/api/v1/komga, /api/v1/projects/{id}/komga): the connection to a Komga
// server, and each project's series on it, whose books are the project's chapters as rendered.

import { call } from "./catalog";

export interface KomgaLibrary {
  id: string;
  name: string;
  /** The library's folder as Komga sees it. */
  root: string;
  /** Where Urakatsu writes this library's books; null when it is not known yet. */
  folder: string | null;
  /** "set" here, Komga's own path ("komga", when Komga runs on this computer), or "none". */
  folder_source: "set" | "komga" | "none";
  problem: string | null;
}

export interface KomgaStatus {
  configured: boolean;
  connected: boolean;
  url: string;
  user: string | null;
  version: string | null;
  error: string | null;
  libraries: KomgaLibrary[];
}

export interface KomgaSeries {
  id: string;
  library_id: string;
  title: string;
  /** The series' folder as Komga sees it. */
  url: string;
  books: number;
}

export interface ExportReport {
  started_at: number;
  finished_at: number | null;
  /** Chapters written. */
  books: number;
  error: string | null;
  /** Komga had not listed the new series yet; the next export links it. */
  pending: boolean;
}

export interface ExportState {
  state: "queued" | "rendering" | "scanning";
  done: number;
  total: number;
  chapter: string;
}

export interface KomgaLink {
  library_id: string;
  library_name: string;
  /** The series' folder, relative to the library's. */
  folder: string;
  title: string;
  series_id: string | null;
  /** The series in Komga's web reader, once Komga lists it. */
  series_url: string | null;
  /** Urakatsu made the series (and gave it its metadata and cover). */
  created: boolean;
  /** Project chapter id -> when its book was written (unix seconds). */
  exported: Record<string, number>;
  last_export: ExportReport | null;
  /** The export on its way, if any. */
  export: ExportState | null;
}

const json = (method: string, body: unknown): RequestInit => ({ method, body: JSON.stringify(body) });
const project = (projectId: string) => `/projects/${encodeURIComponent(projectId)}/komga`;

export const komga = {
  status: () => call<KomgaStatus>("/komga"),
  /** Saves the server; an `api_key` of null keeps the saved key (the service never sends it back). */
  connect: (url: string, apiKey: string | null) => call<KomgaStatus>("/komga", json("PUT", { url, api_key: apiKey })),
  /** Where the library's folder is on this computer; null goes back to Komga's own path. */
  setFolder: (libraryId: string, folder: string | null) => call<KomgaStatus>(`/komga/libraries/${encodeURIComponent(libraryId)}/folder`, json("PUT", { folder })),
  search: (libraryId: string, query: string) =>
    call<{ series: KomgaSeries[] }>(`/komga/libraries/${encodeURIComponent(libraryId)}/series?q=${encodeURIComponent(query)}`),
  link: (projectId: string) => call<KomgaLink | null>(project(projectId)),
  /** Links the project to a series Komga already has. */
  linkSeries: (projectId: string, seriesId: string) => call<KomgaLink>(project(projectId), json("PUT", { series_id: seriesId })),
  /** Adds the project to a library as a new series. */
  add: (projectId: string, libraryId: string, title: string) => call<KomgaLink>(project(projectId), json("POST", { library_id: libraryId, title })),
  unlink: (projectId: string) => call<void>(project(projectId), { method: "DELETE" }),
  /** Exports in the background; null exports every chapter with pages. */
  export: (projectId: string, chapterIds: string[] | null) => call<KomgaLink>(`${project(projectId)}/export`, json("POST", { chapter_ids: chapterIds })),
};
