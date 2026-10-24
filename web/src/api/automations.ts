// JSON client for automations (/api/v1/automations, /api/v1/projects/{id}/automations):
// unattended batches that run a project's chapters one stage at a time, by hand or on a schedule.

import type { StageName } from "../state/workspace";
import { call } from "./catalog";

export type ScheduleKind = "manual" | "once" | "daily";

export interface Schedule {
  kind: ScheduleKind;
  /** Local "HH:MM". */
  time: string;
  /** "YYYY-MM-DD", for "once". */
  date: string | null;
  /** Weekdays for "daily", Monday = 0. */
  days: number[];
}

export interface Target {
  kind: "chapters" | "pending";
  chapter_ids: string[];
  /** "pending": at most this many chapters per run. */
  limit: number | null;
  /** "pending": first download the source's chapters the project lacks. */
  download: boolean;
}

export interface AutomationBody {
  name: string;
  enabled: boolean;
  schedule: Schedule;
  target: Target;
  stages: StageName[];
  /** A model profile id; null runs the current settings. */
  profile: string | null;
  wake: boolean;
}

export interface Automation extends AutomationBody {
  id: string;
  next_run: number | null;
  created_at: number;
  completed_chapters: string[];
}

export interface TranslationChoice {
  provider: string;
  model: string | null;
  quantization: string | null;
  vision: boolean;
  reasoning: boolean;
}

export interface ModelProfile {
  id: string;
  name: string;
  detection: string | null;
  ocr: string | null;
  inpainting: string | null;
  translation: TranslationChoice | null;
}

export type RunState = "queued" | "waiting" | "downloading" | "running" | "completed" | "stopped" | "failed";

export interface ChapterResult {
  id: string;
  name: string;
  pages: number;
  failed: number;
  errors: string[];
  status: "done" | "partial" | "failed" | "stopped";
}

export interface RunProgress {
  phase: number;
  phases: number;
  stage: StageName | null;
  model: string;
  state: string;
  chapter: string;
  done: number;
  total: number;
}

export interface RunRecord {
  id: string;
  automation_id: string;
  automation_name: string;
  project_id: string;
  project_name: string;
  trigger: "schedule" | "manual";
  status: RunState;
  queued_at: number;
  started_at: number | null;
  finished_at: number | null;
  profile: string | null;
  stages: StageName[];
  models: string[];
  downloaded: number;
  chapters: ChapterResult[];
  progress: RunProgress | null;
  error: string | null;
}

export interface ProjectAutomations {
  automations: Automation[];
  runs: RunRecord[];
}

export interface AutomationStatus {
  active: RunRecord | null;
  queued: RunRecord[];
  wake_supported: boolean;
}

const base = (projectId: string) => `/projects/${encodeURIComponent(projectId)}/automations`;
const one = (projectId: string, id: string) => `${base(projectId)}/${encodeURIComponent(id)}`;
const json = (method: string, body: unknown): RequestInit => ({ method, body: JSON.stringify(body) });

export const automations = {
  status: () => call<AutomationStatus>("/automations/status"),
  stop: (runId: string) => call<AutomationStatus>(`/automations/runs/${encodeURIComponent(runId)}/stop`, { method: "POST" }),
  profiles: () => call<{ profiles: ModelProfile[] }>("/automations/profiles"),
  createProfile: (profile: Omit<ModelProfile, "id">) => call<ModelProfile>("/automations/profiles", json("POST", profile)),
  updateProfile: (profile: ModelProfile) => call<ModelProfile>(`/automations/profiles/${encodeURIComponent(profile.id)}`, json("PUT", profile)),
  deleteProfile: (id: string) => call<void>(`/automations/profiles/${encodeURIComponent(id)}`, { method: "DELETE" }),
  list: (projectId: string) => call<ProjectAutomations>(base(projectId)),
  create: (projectId: string, body: AutomationBody) => call<ProjectAutomations>(base(projectId), json("POST", body)),
  update: (projectId: string, id: string, body: AutomationBody) => call<ProjectAutomations>(one(projectId, id), json("PUT", body)),
  remove: (projectId: string, id: string) => call<ProjectAutomations>(one(projectId, id), { method: "DELETE" }),
  run: (projectId: string, id: string) => call<ProjectAutomations>(`${one(projectId, id)}/run`, { method: "POST" }),
  reset: (projectId: string, id: string) => call<ProjectAutomations>(`${one(projectId, id)}/reset`, { method: "POST" }),
};

/** A run that has not finished yet. */
export function live(run: RunRecord): boolean {
  return run.status === "queued" || run.status === "waiting" || run.status === "downloading" || run.status === "running";
}

const DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

export function weekdayName(day: number): string {
  return DAY_NAMES[day] ?? "";
}

/** "Daily at 01:00", "Weekdays at 02:30", "Mon, Wed at 01:00", "Once on 25 Oct 2026 at 02:00", "By hand". */
export function describeSchedule(schedule: Schedule, formatDay: (iso: string) => string): string {
  if (schedule.kind === "manual") return "By hand";
  if (schedule.kind === "once") return `Once on ${schedule.date ? formatDay(schedule.date) : "—"} at ${schedule.time}`;
  const days = [...schedule.days].sort();
  const key = days.join(",");
  const when = key === "0,1,2,3,4,5,6" ? "Daily" : key === "0,1,2,3,4" ? "Weekdays" : key === "5,6" ? "Weekends" : days.map(weekdayName).join(", ");
  return `${when} at ${schedule.time}`;
}
