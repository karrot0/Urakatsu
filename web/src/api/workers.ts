// JSON client for the app's workers (/api/v1/workers, urakatsu/service/workers_api.py): notebooks
// and machines that run detection, OCR and inpainting models for the remote runtime.

import { call } from "./catalog";

export type WorkerKind = "kaggle" | "colab" | "manual";
export type WorkerState = "online" | "starting" | "offline" | "outdated";

export interface WorkerStatus {
  state: WorkerState;
  device: string | null;
  url: string | null;
  detail: string | null;
  stoppable?: boolean;
}

export interface Worker {
  id: string;
  name: string;
  kind: WorkerKind;
  /** Where a Colab or other-machine worker answers (a Kaggle worker's comes from its session log). */
  url: string | null;
  token_set: boolean;
  auto_start: boolean;
  kaggle_username: string | null;
  kaggle_token_set: boolean;
  status: WorkerStatus | null;
  /** The models set to run here (their runtime menu in Settings, Pipeline). */
  runs: { role: string; provider: string; name: string }[];
}

export interface WorkerList {
  workers: Worker[];
  colab_url: string;
  kaggle_settings_url: string;
}

export interface NewWorker {
  kind: WorkerKind;
  name?: string;
  /** The line the worker printed: "Connection: https://<address>#<token>". */
  connection?: string;
  url?: string;
  token?: string;
  kaggle_token?: string;
  kaggle_username?: string;
}

export interface Quota {
  used_hours: number;
  reserved_hours: number;
  allowed_hours: number;
}

const json = (method: string, body: unknown): RequestInit => ({ method, body: JSON.stringify(body) });

export const workers = {
  /** Without `status`, skips the live checks (for the runtime menu). */
  list: (status = true) => call<WorkerList>(status ? "/workers" : "/workers?status=false"),
  add: (body: NewWorker) => call<Worker>("/workers", json("POST", body)),
  change: (id: string, patch: Partial<NewWorker> & { auto_start?: boolean }) => call<Worker>(`/workers/${id}`, json("PATCH", patch)),
  remove: (id: string) => call<{ ok: boolean }>(`/workers/${id}`, { method: "DELETE" }),
  order: (ids: string[]) => call<{ ok: boolean }>("/workers/order", json("PUT", { ids })),
  start: (id: string) => call<Worker>(`/workers/${id}/start`, { method: "POST" }),
  stop: (id: string) => call<Worker>(`/workers/${id}/stop`, { method: "POST" }),
  quota: (id: string) => call<Quota>(`/workers/${id}/quota`),
};
