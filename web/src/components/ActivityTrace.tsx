// Activity tab: a run as a trace. The "started" event names every page and stage the run will
// do, so the trace shows the whole plan up front; each page folds its stages' lifecycle
// (loading, running, then finished, skipped or failed) into one row per stage.

import { ChevronRight, CircleCheck, CircleDashed, CircleMinus, CircleX, LoaderCircle } from "lucide-react";
import { useMemo, useState } from "react";
import { cn } from "cn";

import { STAGE_LABEL } from "../api/catalog";
import type { ProgressEvent } from "../gen/urakatsu/v1/urakatsu_pb";
import { STAGES, useWorkspace, type JobState } from "../state/workspace";
import { Badge } from "./ui";
import { Button } from "@/components/ui/button";

type StepState = "queued" | "loading" | "running" | "finished" | "skipped" | "failed";
type Step = { state: StepState; model?: string; ms?: number; message?: string };
type Trace = { pages: string[]; stages: string[]; steps: Map<string, Step>; elapsedMs?: number };

const STEP_STATES = new Set<string>(["loading", "running", "finished", "skipped", "failed"]);
const key = (page: string, stage: string) => `${page}\u0000${stage}`;

function buildTrace(events: ProgressEvent[]): Trace {
  const started = events.find((e) => e.kind === "started");
  const pages = [...(started?.pages ?? [])];
  const stages = [...(started?.stages ?? [])];
  const steps = new Map<string, Step>();
  let elapsedMs: number | undefined;
  for (const event of events) {
    if (!event.pageId && (event.kind === "completed" || event.kind === "stopped") && event.elapsedMs) elapsedMs = event.elapsedMs;
    if (!event.pageId || !event.stage) continue;
    if (!pages.includes(event.pageId)) pages.push(event.pageId);
    if (!stages.includes(event.stage)) stages.push(event.stage);
    const step = steps.get(key(event.pageId, event.stage)) ?? { state: "queued" };
    if (STEP_STATES.has(event.kind)) step.state = event.kind as StepState;
    if (event.model) step.model = event.model;
    if (event.elapsedMs) step.ms = event.elapsedMs;
    if (event.message) step.message = event.message;
    steps.set(key(event.pageId, event.stage), step);
  }
  const order = (stage: string) => {
    const index = (STAGES as readonly string[]).indexOf(stage);
    return index < 0 ? STAGES.length : index;
  };
  stages.sort((a, b) => order(a) - order(b));
  return { pages, stages, steps, elapsedMs };
}

export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(ms < 10_000 ? 1 : 0)} s`;
  const minutes = Math.floor(ms / 60_000);
  return `${minutes}m ${Math.round((ms % 60_000) / 1000)}s`;
}

const JOB_TONE = { running: "info", completed: "success", failed: "danger", stopped: "warning" } as const;

export function ActivityPanel() {
  const ws = useWorkspace();
  const job = ws.job;
  const trace = useMemo(() => buildTrace(job?.events ?? []), [job?.events]);
  if (!job) return <p className="text-fg-2">No runs yet. Runs you start appear here.</p>;
  const pageName = (id: string) => ws.project?.pages.find((p) => p.id === id)?.name ?? id.slice(0, 8);
  const share = job.total ? Math.min(100, (job.completed / job.total) * 100) : 0;
  const running = job.status === "running";

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between gap-2">
          <span className="flex min-w-0 items-center gap-2">
            <Badge tone={JOB_TONE[job.status]}>{job.status[0].toUpperCase() + job.status.slice(1)}</Badge>
            <span className="truncate text-xs leading-4 text-fg-2 tabular-nums">
              {job.completed} of {job.total || "?"} stages
              {trace.elapsedMs ? ` in ${formatDuration(trace.elapsedMs)}` : ""}
            </span>
          </span>
          {running && (
            <Button variant="outline" size="sm" onClick={() => void ws.stop()}>
              Stop
            </Button>
          )}
        </div>
        <div className="relative h-1 overflow-hidden rounded-full bg-surface-fill" role="progressbar" aria-valuemin={0} aria-valuemax={job.total} aria-valuenow={job.completed} aria-label="Run progress">
          <span
            className={cn("absolute inset-y-0 left-0 rounded-[inherit] transition-[width] duration-200 ease-settle", job.status === "failed" ? "bg-danger" : job.status === "stopped" ? "bg-warning" : running ? "bg-brand" : "bg-success")}
            style={{ width: `${share}%` }}
          />
        </div>
      </div>

      {job.message && <div className="rounded-control bg-danger-weak px-3 py-2 text-xs leading-4 text-danger shadow-[inset_0_0_0_1px_var(--danger-line)] [word-break:break-word]">{job.message}</div>}

      {trace.pages.length > 0 && (
        <ol className="flex flex-col" aria-label="Pages in this run">
          {trace.pages.map((page) => (
            <PageTrace key={page} name={pageName(page)} stages={trace.stages} steps={trace.stages.map((stage) => trace.steps.get(key(page, stage)) ?? { state: "queued" })} job={job} />
          ))}
        </ol>
      )}
    </div>
  );
}

/** One page: a summary line (stage dots, count, time) that opens to its stages. Open by default while it is active or has failed. */
function PageTrace({ name, stages, steps, job }: { name: string; stages: string[]; steps: Step[]; job: JobState }) {
  const active = steps.some((s) => s.state === "loading" || s.state === "running");
  const failed = steps.some((s) => s.state === "failed");
  const [open, setOpen] = useState<boolean>();
  const expanded = open ?? (active || failed);
  const done = steps.filter((s) => s.state === "finished" || s.state === "skipped").length;
  const ms = steps.reduce((sum, s) => sum + (s.ms ?? 0), 0);
  return (
    <li className="shadow-[inset_0_-1px_0_var(--border-soft)] last:shadow-none">
      <button
        className="flex w-full cursor-pointer items-center gap-2 rounded-control px-1 py-1.5 text-left transition-colors duration-[20ms] ease-in hover:bg-hover"
        aria-expanded={expanded}
        onClick={() => setOpen(!expanded)}
      >
        <ChevronRight size={14} className={cn("text-icon-2 transition-transform duration-150 ease-settle", expanded && "rotate-90")} />
        <span className={cn("min-w-0 flex-1 truncate font-medium", !active && done === 0 && !failed && "text-fg-2")}>{name}</span>
        <span className="flex flex-none items-center gap-0.5" aria-hidden>
          {steps.map((step, i) => (
            <span key={i} className={cn("size-1.5 rounded-full", DOT[step.state], step.state === "queued" && job.status !== "running" && "opacity-50")} />
          ))}
        </span>
        <span className="w-14 flex-none text-right text-xs leading-4 text-fg-3 tabular-nums">{ms ? formatDuration(ms) : `${done}/${steps.length}`}</span>
      </button>
      {expanded && (
        <ol className="mb-1.5 ml-[22px] flex flex-col border-l border-line-soft pl-2">
          {stages.map((stage, i) => (
            <StepRow key={stage} stage={stage} step={steps[i]} runOver={job.status !== "running"} />
          ))}
        </ol>
      )}
    </li>
  );
}

const DOT: Record<StepState, string> = {
  queued: "bg-line-strong",
  loading: "bg-brand animate-[soft-pulse_1.4s_ease-in-out_infinite]",
  running: "bg-brand animate-[soft-pulse_1.4s_ease-in-out_infinite]",
  finished: "bg-success",
  skipped: "bg-fg-3",
  failed: "bg-danger",
};

function StepRow({ stage, step, runOver }: { stage: string; step: Step; runOver: boolean }) {
  const label = STAGE_LABEL[stage] ?? stage;
  const status =
    step.state === "finished"
      ? step.ms
        ? formatDuration(step.ms)
        : "Done"
      : step.state === "skipped"
        ? "Skipped"
        : step.state === "failed"
          ? "Failed"
          : step.state === "loading"
            ? "Loading model"
            : step.state === "running"
              ? "Running"
              : runOver
                ? "Not run"
                : "Queued";
  return (
    <li className="flex flex-col gap-0.5 py-1">
      <div className="flex min-w-0 items-center gap-2">
        <StepIcon state={step.state} />
        <span className={cn("flex-none", step.state === "queued" || step.state === "skipped" ? "text-fg-2" : "text-fg")}>{label}</span>
        <span className="min-w-0 flex-1 truncate font-mono text-xs leading-4 text-fg-3" title={step.model}>
          {step.model}
        </span>
        <span
          className={cn(
            "flex-none text-xs leading-4 tabular-nums",
            step.state === "failed" ? "text-danger" : step.state === "loading" || step.state === "running" ? "text-brand" : "text-fg-3",
          )}
        >
          {status}
        </span>
      </div>
      {step.message && step.state === "failed" && <p className="ml-6 text-xs leading-4 text-danger [word-break:break-word]">{step.message}</p>}
    </li>
  );
}

function StepIcon({ state }: { state: StepState }) {
  const size = 16;
  if (state === "finished") return <CircleCheck size={size} className="text-success" aria-hidden />;
  if (state === "failed") return <CircleX size={size} className="text-danger" aria-hidden />;
  if (state === "skipped") return <CircleMinus size={size} className="text-fg-3" aria-hidden />;
  if (state === "loading" || state === "running") return <LoaderCircle size={size} className="animate-spin text-brand motion-reduce:animate-none" aria-hidden />;
  return <CircleDashed size={size} className="text-fg-disabled" aria-hidden />;
}
