// JSON client for the model catalog and runtime versions (/api/v1/catalog, /api/v1/runtimes).
// The rest of the UI speaks gRPC-web; these routes are plain JSON because they mostly relay
// Hugging Face and GitHub.

export type CatalogRuntime = "llama" | "diffusion" | "torch" | "onnx";
export type FitLevel = "gpu" | "partial" | "offload" | "cpu" | "too_large";
export type SupportStatus = "supported" | "needs_update" | "unsupported" | "not_installed" | "unknown";

export interface Fit {
  level: FitLevel;
  required: number;
  weights: number;
  context: number;
  gpu_share: number;
  notes: string[];
}

export interface Usage {
  stage: string;
  provider: string;
  name: string;
  runtime: string;
}

export interface SearchResult {
  repo: string;
  author: string;
  downloads: number;
  likes: number;
  updated: string;
  pipeline_tag: string | null;
  architecture: string | null;
  model_type: string | null;
  family: string | null;
  parameters: number | null;
  gated: boolean;
  usable_by: Usage[];
  estimate: Fit | null;
  installed: boolean;
}

export interface Variant {
  id: string;
  label: string;
  files: string[];
  size: number;
  fit: Fit;
  recommended: boolean;
}

export interface Compatibility {
  status: SupportStatus;
  message: string;
  runtime?: string;
  installed?: string | null;
  latest?: string | null;
  pinned?: string | null;
}

export interface InstalledModel {
  id: string;
  repo: string;
  revision: string;
  runtime: CatalogRuntime;
  variant: string;
  files: { path: string; role: string; size: number }[];
  architecture: string | null;
  family: string | null;
  size: number;
  installed_at: number;
  usable_by: Usage[];
  compatibility?: Compatibility;
  complete?: boolean;
  /** stable-diffusion.cpp recipe id, for image models. */
  recipe?: string | null;
  /** The recipe edits by instruction (reference image), so prompts say what to change. */
  edits?: boolean;
}

export interface Hardware {
  devices: {
    name: string;
    description: string;
    backend: string; // cuda | rocm | vulkan | dml | metal
    device_type: string; // gpu | integrated_gpu
    memory_total: number;
    memory_free: number | null;
    driver_version?: string;
    target?: string; // AMD ISA, as gfx1100
  }[];
  system: { ram_total: number; ram_available: number; cpu_name: string; cpu_threads: number; cpu_features: string[]; unified_memory: boolean };
  usable_vram: number;
}

export interface ModelDetails {
  repo: string;
  revision: string;
  author: string;
  downloads: number;
  likes: number;
  updated: string;
  pipeline_tag: string | null;
  library: string | null;
  tags: string[];
  gated: boolean;
  architecture: string | null;
  model_type: string | null;
  family: string | null;
  parameters: number | null;
  context_length: number | null;
  vision: boolean;
  reasoning: boolean;
  usable_by: Usage[];
  compatibility: Compatibility;
  variants: Variant[];
  recommended: string | null;
  installed: InstalledModel[];
  readme: string;
  hardware: Hardware;
  /** stable-diffusion.cpp models: the other files the model needs, downloaded with it. */
  components: { slot: string; label: string; path: string; repo: string; size: number; here: boolean }[];
  /** Components no file was found for (the model cannot be installed). */
  missing: string[];
  /** Repos with a loadable copy, when this one has none. */
  alternatives: string[];
  recipe: string | null;
  edits: boolean;
}

export interface Download {
  id: string;
  model: InstalledModel;
  status: "queued" | "downloading" | "paused" | "completed" | "failed" | "cancelled";
  position: number | null;
  done: number;
  total: number;
  current: string;
  error: string | null;
}

export interface NativeRuntime {
  id: string;
  name: string;
  repository: string;
  installed: string | null;
  variant: string | null;
  checked_at: number | null;
  pinned: string | null;
  latest: string | null;
  kept: { tag: string; variant: string }[];
}

export interface Runtimes {
  native: NativeRuntime[];
  python: { id: string; name: string; installed: string }[];
  offline: boolean;
}

export interface Release {
  tag: string;
  published: string;
  variant: string | null;
}

/** A site check a person has to clear in a source system's browser before the call can work. */
export interface Challenge {
  system: string;
  url: string;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly challenge?: Challenge,
  ) {
    super(message);
  }
}

export function challengeOf(error: unknown): Challenge | undefined {
  return error instanceof ApiError ? error.challenge : undefined;
}

const OUTDATED = "The Urakatsu server is older than this app and does not know this feature yet. Restart Urakatsu to finish updating.";

export async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api/v1${path}`, {
    ...init,
    headers: init?.body ? { "Content-Type": "application/json" } : undefined,
  });
  // A server from before a route existed answers it with the app page (or a bare 404); say so plainly.
  const json = (response.headers.get("content-type") ?? "").includes("application/json");
  if (response.status !== 204 && !json && (response.ok || response.status === 404)) throw new ApiError(OUTDATED, response.status);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    let challenge: Challenge | undefined;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
      else if (body.detail && typeof body.detail.message === "string") {
        message = body.detail.message;
        challenge = body.detail.challenge ?? undefined;
      }
    } catch {
      /* not JSON */
    }
    throw new ApiError(message, response.status, challenge);
  }
  return (response.status === 204 ? undefined : await response.json()) as T;
}

const query = (params: Record<string, string | number | boolean>) =>
  new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)])).toString();

/** One memory pool; ``app`` is Urakatsu plus the runtimes it started, null when the OS does not say. */
export type MemoryPool = { name: string; total: number; used: number; app: number | null };
/**
 * Live GPU and system memory. RAM includes model layers offloaded to the CPU; ``spilled`` is
 * Urakatsu's GPU memory the driver moved into shared system memory (Windows only).
 */
export type MemoryUsage = { device: string; gpu: (MemoryPool & { spilled: number | null }) | null; ram: MemoryPool };

export type FolderMove = { source: string; target: string; status: "moving" | "completed" | "failed"; done: number; total: number; current: string; error: string | null; left_behind: number };
/** The models folder: where it is, what the models take, its drive's space, and a move in progress. */
export type Storage = { models_dir: string; default_dir: string; size: number; items: StorageItem[]; disk: { total: number; used: number; free: number }; move: FolderMove | null };
/**
 * One thing taking space in the models folder, largest first. ``detail`` is the variant of an
 * installed model, or the providers using built-in weights (null when none does now).
 */
export type StorageItem = { id: string | null; kind: "installed" | "builtin" | "partial" | "other"; name: string; detail: string | null; size: number };

export const catalog = {
  hardware: () => call<Hardware>("/catalog/hardware"),
  memory: () => call<MemoryUsage>("/memory"),
  storage: () => call<Storage>("/storage"),
  /** ``null`` returns to the default folder; ``move`` carries the installed models along. */
  setModelsFolder: (models_dir: string | null, move: boolean) => call<Storage>("/storage", { method: "PUT", body: JSON.stringify({ models_dir, move }) }),
  browseFolder: () => call<{ path: string | null }>("/storage/browse", { method: "POST" }),
  search: (runtime: CatalogRuntime, q: string, sort: string, usable: boolean) =>
    call<SearchResult[]>(`/catalog/search?${query({ runtime, q, sort, usable, limit: 40 })}`),
  details: (runtime: CatalogRuntime, repo: string) => call<ModelDetails>(`/catalog/models/${repo}?${query({ runtime })}`),
  install: (runtime: CatalogRuntime, repo: string, variant: string) =>
    call<Download>("/catalog/install", { method: "POST", body: JSON.stringify({ runtime, repo, variant }) }),
  downloads: () => call<Download[]>("/catalog/downloads"),
  cancel: (id: string) => call<void>(`/catalog/downloads/${id}/cancel`, { method: "POST" }),
  pause: (id: string) => call<void>(`/catalog/downloads/${id}/pause`, { method: "POST" }),
  resume: (id: string) => call<void>(`/catalog/downloads/${id}/resume`, { method: "POST" }),
  move: (id: string, position: number) =>
    call<Download[]>(`/catalog/downloads/${id}/move`, { method: "POST", body: JSON.stringify({ position }) }),
  installed: () => call<InstalledModel[]>("/catalog/installed"),
  remove: (id: string) => call<void>(`/catalog/installed/${id}`, { method: "DELETE" }),
  runtimes: () => call<Runtimes>("/runtimes"),
  releases: (id: string) => call<Release[]>(`/runtimes/${id}/releases`),
  pin: (id: string, tag: string | null) => call<Runtimes>(`/runtimes/${id}/pin`, { method: "PUT", body: JSON.stringify({ tag }) }),
  update: (id: string) => call<Runtimes>(`/runtimes/${id}/update`, { method: "POST" }),
};

// ---- formatting ---------------------------------------------------------------------------------

export function formatBytes(bytes: number): string {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const exponent = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  const value = bytes / 1024 ** exponent;
  return `${value >= 100 || exponent === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[exponent]}`;
}

export function formatCount(count: number): string {
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M`;
  if (count >= 1_000) return `${(count / 1_000).toFixed(1)}k`;
  return String(count);
}

export function formatParameters(count: number | null): string | null {
  if (!count) return null;
  return count >= 1e9 ? `${(count / 1e9).toFixed(count >= 1e10 ? 0 : 1)}B` : `${Math.round(count / 1e6)}M`;
}

export const FIT_LABEL: Record<FitLevel, string> = {
  gpu: "Full GPU",
  partial: "Partial offload",
  offload: "GPU offload",
  cpu: "CPU only",
  too_large: "Too large",
};

export const FIT_TONE: Record<FitLevel, "success" | "info" | "warning" | "danger"> = {
  gpu: "success",
  partial: "info",
  offload: "info",
  cpu: "warning",
  too_large: "danger",
};

export const STAGE_LABEL: Record<string, string> = {
  detection: "Detection",
  ocr: "Text recognition",
  inpainting: "Inpainting",
  translation: "Translation",
  sound_effects: "Sound effects",
};

export const RUNTIME_LABEL: Record<CatalogRuntime, string> = {
  llama: "llama.cpp",
  diffusion: "stable-diffusion.cpp",
  torch: "PyTorch",
  onnx: "ONNX",
};
