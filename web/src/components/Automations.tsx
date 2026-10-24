// Automations: a project's unattended batches. Each one picks chapters, stages, a model profile
// and a schedule; a run takes every chosen page through one stage at a time, unloading between
// stages, so the heavy models get the GPU to themselves. Opened from the project header.

import { CalendarClock, Check, ChevronDown, ChevronRight, MoreHorizontal, Play, Plus, Square } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { cn } from "cn";
import {
  automations,
  describeSchedule,
  discordUserId,
  isDiscordWebhook,
  live,
  weekdayName,
  type Automation,
  type AutomationBody,
  type AutomationStatus,
  type ModelProfile,
  type ProjectAutomations,
  type RunRecord,
} from "../api/automations";
import { describeError } from "../api/client";
import { formatDate, formatTimestamp, plural } from "../format";
import { STAGES, useWorkspace, type StageName } from "../state/workspace";
import { ActionMenu, NumberField, Select, Switch, Tooltip, type Option } from "./primitives";
import { Badge, Segmented, type Tone } from "./ui";

const STAGE_LABEL: Record<StageName, string> = { detection: "Detection", ocr: "OCR", translation: "Translation", inpainting: "Inpainting" };
const CURRENT = "__current";

/** How often the project's runs are read while one is on its way, and otherwise. */
const LIVE_POLL_MS = 2000;
const IDLE_POLL_MS = 15000;
/** A running batch on the open project re-reads it at most this often, so results appear. */
const RELOAD_MS = 5000;

function emptyBody(): AutomationBody {
  return {
    name: "Overnight",
    enabled: true,
    schedule: { kind: "daily", time: "01:00", date: null, days: [0, 1, 2, 3, 4, 5, 6] },
    target: { kind: "pending", chapter_ids: [], limit: null, download: false },
    stages: [...STAGES],
    profile: null,
    wake: false,
    notify: { enabled: false, webhook_url: "", mention: "" },
  };
}

function bodyOf(automation: Automation): AutomationBody {
  const { name, enabled, schedule, target, stages, profile, wake, notify } = automation;
  return structuredClone({ name, enabled, schedule, target, stages, profile, wake, notify: notify ?? emptyBody().notify });
}

// ---- header button ---------------------------------------------------------------------------

/** The project header's Automations button: a dot while one of this project's runs is on its way. */
export function AutomationsButton() {
  const ws = useWorkspace();
  const projectId = ws.project!.id;
  const [open, setOpen] = useState(false);
  const status = useAutomationStatus(projectId);
  const busy = !!status && [status.active, ...status.queued].some((run) => run && run.project_id === projectId);
  return (
    <>
      <Tooltip content="Automations" side="bottom">
        <Button variant="ghost" size="icon-sm" className="relative" aria-label="Automations" onClick={() => setOpen(true)}>
          <CalendarClock size={16} />
          {busy && <span className="absolute top-1 right-1 size-1.5 rounded-full bg-brand" aria-hidden />}
        </Button>
      </Tooltip>
      <AutomationsDialog open={open} onOpenChange={setOpen} status={status} />
    </>
  );
}

/**
 * The automation queue, read every few seconds while this project has a run on its way (and
 * re-reading the project so its pages show the results), otherwise now and then.
 */
function useAutomationStatus(projectId: string): AutomationStatus | undefined {
  const ws = useWorkspace();
  const [status, setStatus] = useState<AutomationStatus>();
  const lastReload = useRef(0);
  const lastDone = useRef<string>("");
  const reload = useRef(ws.reloadProject);
  reload.current = ws.reloadProject;

  useEffect(() => {
    let timer: number | undefined;
    let stopped = false;
    let wasRunning = false;
    const tick = async () => {
      let next: AutomationStatus | undefined;
      try {
        next = await automations.status();
        setStatus(next);
      } catch {
        /* an older server without automations: try again later */
      }
      const active = next?.active?.project_id === projectId ? next.active : null;
      const running = active?.status === "running";
      const done = active?.progress ? `${active.progress.phase}:${active.progress.done}` : "";
      const now = Date.now();
      if ((running && done !== lastDone.current && now - lastReload.current > RELOAD_MS) || (wasRunning && !running)) {
        lastReload.current = now;
        void reload.current(projectId).catch(() => undefined);
      }
      lastDone.current = done;
      wasRunning = running;
      const ours = !!next && [next.active, ...next.queued].some((run) => run && run.project_id === projectId);
      if (!stopped) timer = window.setTimeout(tick, ours ? LIVE_POLL_MS : IDLE_POLL_MS);
    };
    void tick();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [projectId]);
  return status;
}

// ---- dialog --------------------------------------------------------------------------------------

type View = { kind: "list" } | { kind: "edit"; id?: string; body: AutomationBody } | { kind: "profiles" };

function AutomationsDialog({ open, onOpenChange, status }: { open: boolean; onOpenChange: (open: boolean) => void; status?: AutomationStatus }) {
  const ws = useWorkspace();
  const projectId = ws.project!.id;
  const [data, setData] = useState<ProjectAutomations>();
  const [profiles, setProfiles] = useState<ModelProfile[]>([]);
  const [view, setView] = useState<View>({ kind: "list" });
  const [error, setError] = useState<string>();

  const load = useCallback(async () => {
    try {
      const [list, saved] = await Promise.all([automations.list(projectId), automations.profiles()]);
      setData(list);
      setProfiles(saved.profiles);
      setError(undefined);
    } catch (failure) {
      setError(describeError(failure));
    }
  }, [projectId]);

  // While open: the list now, then again on the queue's pace.
  const hasLive = !!data?.runs.some(live);
  useEffect(() => {
    if (!open) return;
    void load();
    const timer = window.setInterval(() => void load(), hasLive ? LIVE_POLL_MS : IDLE_POLL_MS);
    return () => window.clearInterval(timer);
  }, [open, load, hasLive]);
  useEffect(() => {
    if (!open) setView({ kind: "list" });
  }, [open]);

  const act = async (work: () => Promise<ProjectAutomations | void>) => {
    try {
      const next = await work();
      if (next) setData(next);
      setError(undefined);
      return true;
    } catch (failure) {
      setError(describeError(failure));
      return false;
    }
  };

  const otherActive = status?.active && status.active.project_id !== projectId ? status.active : null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[min(760px,calc(100vh-48px))] gap-0 p-0 sm:max-w-2xl">
        <DialogHeader className="px-5 pt-5 pb-3">
          <DialogTitle>{view.kind === "profiles" ? "Model profiles" : view.kind === "edit" ? (view.id ? "Edit automation" : "New automation") : "Automations"}</DialogTitle>
          <DialogDescription>
            {view.kind === "profiles"
              ? "Models an automation runs instead of your current settings. Each stage runs alone on the GPU, so a profile can name models too large to load together."
              : "Translate chapters while you're away. A run takes every page through one stage at a time and unloads each model before loading the next."}
          </DialogDescription>
        </DialogHeader>
        {error && <div className="mx-5 mb-3 rounded-control bg-danger-weak px-3 py-2 text-xs text-danger">{error}</div>}
        {view.kind === "list" && (
          <ScrollArea className="min-h-0 flex-1" contentClassName="px-5 pb-5">
            {otherActive && (
              <p className="mb-3 text-xs text-fg-3">
                {otherActive.project_name || "Another project"} is running “{otherActive.automation_name}”. Runs here wait for it.
              </p>
            )}
            <AutomationList
              data={data}
              profiles={profiles}
              onNew={() => setView({ kind: "edit", body: emptyBody() })}
              onEdit={(automation) => setView({ kind: "edit", id: automation.id, body: bodyOf(automation) })}
              onProfiles={() => setView({ kind: "profiles" })}
              act={act}
            />
            <RunHistory runs={data?.runs ?? []} onStop={(run) => void act(async () => void (await automations.stop(run.id), await load()))} />
          </ScrollArea>
        )}
        {view.kind === "edit" && (
          <AutomationEditor
            key={view.id ?? "new"}
            initial={view.body}
            profiles={profiles}
            wakeSupported={status?.wake_supported ?? false}
            onCancel={() => setView({ kind: "list" })}
            onProfiles={() => setView({ kind: "profiles" })}
            onSave={async (body) => {
              const saved = await act(() => (view.id ? automations.update(projectId, view.id, body) : automations.create(projectId, body)));
              if (saved) setView({ kind: "list" });
            }}
          />
        )}
        {view.kind === "profiles" && (
          <ProfileEditor
            profiles={profiles}
            onChange={setProfiles}
            onError={setError}
            onDone={() => {
              setView({ kind: "list" });
              void load();
            }}
          />
        )}
      </DialogContent>
    </Dialog>
  );
}

// ---- list ------------------------------------------------------------------------------------------

function AutomationList({
  data,
  profiles,
  onNew,
  onEdit,
  onProfiles,
  act,
}: {
  data?: ProjectAutomations;
  profiles: ModelProfile[];
  onNew: () => void;
  onEdit: (automation: Automation) => void;
  onProfiles: () => void;
  act: (work: () => Promise<ProjectAutomations | void>) => Promise<boolean>;
}) {
  const ws = useWorkspace();
  const projectId = ws.project!.id;
  const chapters = ws.project!.chapters;
  const list = data?.automations ?? [];
  const running = new Map((data?.runs ?? []).filter(live).map((run) => [run.automation_id, run.status === "queued" ? "Queued" : "Running"]));

  const targetText = (automation: Automation) => {
    const { target } = automation;
    if (target.kind === "chapters") {
      const names = chapters.filter((c) => target.chapter_ids.includes(c.id)).map((c) => c.name);
      return names.length <= 2 ? names.join(", ") || "No chapters" : `${names.length} chapters`;
    }
    const limit = target.limit ? `, at most ${target.limit} a run` : "";
    return `Unprocessed chapters${limit}${target.download ? ", downloading new ones first" : ""}`;
  };

  return (
    <section>
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-fg-2">This project</span>
        <span className="flex gap-1">
          <Button variant="ghost" size="sm" onClick={onProfiles}>
            Model profiles
          </Button>
          <Button variant="outline" size="sm" onClick={onNew}>
            <Plus size={14} />
            New automation
          </Button>
        </span>
      </div>
      {data && list.length === 0 && (
        <div className="rounded-card px-4 py-6 text-center text-sm text-fg-3 shadow-[inset_0_0_0_1px_var(--border-soft)]">
          No automations yet. Create one to translate chapters overnight.
        </div>
      )}
      <ul className="flex flex-col rounded-card shadow-[inset_0_0_0_1px_var(--border-soft)] empty:hidden">
        {list.map((automation) => {
          const profile = profiles.find((p) => p.id === automation.profile);
          return (
            <li key={automation.id} className="flex items-center gap-3 px-3 py-2.5 [&+&]:shadow-[inset_0_1px_0_var(--border-soft)]">
              <Switch
                label={automation.enabled ? `Turn off ${automation.name}` : `Turn on ${automation.name}`}
                checked={automation.enabled}
                onChange={(enabled) => void act(() => automations.update(projectId, automation.id, { ...bodyOf(automation), enabled }))}
              />
              <div className="flex min-w-0 flex-1 flex-col">
                <span className="flex items-center gap-2 truncate font-medium">
                  {automation.name}
                  {automation.wake && automation.enabled && automation.schedule.kind !== "manual" && <Badge tone="info">Wakes PC</Badge>}
                </span>
                <span className="truncate text-xs text-fg-3">
                  {describeSchedule(automation.schedule, formatDate)} · {targetText(automation)} · {profile ? profile.name : "Current settings"}
                </span>
                {automation.enabled && automation.next_run && <span className="text-xs text-fg-3">Next: {formatTimestamp(automation.next_run)}</span>}
              </div>
              <Button variant="outline" size="sm" disabled={running.has(automation.id)} onClick={() => void act(() => automations.run(projectId, automation.id))}>
                <Play size={14} />
                {running.get(automation.id) ?? "Run now"}
              </Button>
              <ActionMenu
                trigger={
                  <Button variant="ghost" size="icon-sm" aria-label={`More for ${automation.name}`}>
                    <MoreHorizontal size={16} />
                  </Button>
                }
                actions={[
                  { label: "Edit", onSelect: () => onEdit(automation) },
                  ...(automation.target.kind === "pending" && automation.completed_chapters.length
                    ? [{ label: `Forget ${plural(automation.completed_chapters.length, "finished chapter")}`, onSelect: () => void act(() => automations.reset(projectId, automation.id)) }]
                    : []),
                  { label: "Delete", danger: true, onSelect: () => void act(() => automations.remove(projectId, automation.id)) },
                ]}
              />
            </li>
          );
        })}
      </ul>
    </section>
  );
}

// ---- runs ------------------------------------------------------------------------------------------

const RUN_LABEL: Record<RunRecord["status"], string> = {
  queued: "Queued",
  waiting: "Waiting",
  downloading: "Downloading",
  running: "Running",
  completed: "Done",
  stopped: "Stopped",
  failed: "Failed",
};

function runTone(run: RunRecord): Tone {
  if (run.status === "failed") return "danger";
  if (live(run)) return "info";
  if (run.chapters.some((c) => c.status !== "done")) return "warning";
  return run.status === "completed" ? "success" : "neutral";
}

function RunHistory({ runs, onStop }: { runs: RunRecord[]; onStop: (run: RunRecord) => void }) {
  if (runs.length === 0) return null;
  return (
    <section className="mt-5">
      <div className="mb-2 text-xs font-medium text-fg-2">Runs</div>
      <ul className="flex flex-col rounded-card shadow-[inset_0_0_0_1px_var(--border-soft)]">
        {runs.map((run) => (
          <RunRow key={run.id} run={run} onStop={() => onStop(run)} />
        ))}
      </ul>
    </section>
  );
}

function RunRow({ run, onStop }: { run: RunRecord; onStop: () => void }) {
  const [open, setOpen] = useState(false);
  const progress = run.progress;
  const failed = run.chapters.filter((c) => c.status === "failed" || c.status === "partial").length;
  const summary = live(run)
    ? progress && run.status === "running"
      ? `Stage ${progress.phase} of ${progress.phases}: ${STAGE_LABEL[progress.stage ?? "detection"]} with ${progress.model} · page ${Math.min(progress.done + 1, progress.total)} of ${progress.total}${progress.chapter ? ` · ${progress.chapter}` : ""}`
      : run.status === "downloading"
        ? "Downloading new chapters"
        : run.status === "waiting"
          ? "Waiting for the current run to finish"
          : "Waiting for another automation"
    : run.error
      ? run.error
      : run.chapters.length === 0
        ? "Nothing left to process"
        : `${plural(run.chapters.length, "chapter")}${failed ? `, ${failed} with errors` : ""}${run.downloaded ? ` · ${run.downloaded} downloaded` : ""}`;
  const when = run.finished_at ?? run.started_at ?? run.queued_at;
  const expandable = !live(run) && (run.chapters.length > 0 || run.models.length > 0);
  return (
    <li className="flex flex-col px-3 py-2.5 [&+&]:shadow-[inset_0_1px_0_var(--border-soft)]">
      <div className="flex items-center gap-2">
        <button
          className={cn("flex min-w-0 flex-1 items-center gap-2 text-left", expandable ? "cursor-pointer" : "cursor-default")}
          aria-expanded={expandable ? open : undefined}
          onClick={() => expandable && setOpen(!open)}
        >
          {expandable ? open ? <ChevronDown size={14} className="text-icon-2" /> : <ChevronRight size={14} className="text-icon-2" /> : <span className="w-3.5" />}
          <Badge tone={runTone(run)}>{RUN_LABEL[run.status]}</Badge>
          <span className="truncate font-medium">{run.automation_name}</span>
        </button>
        <span className="shrink-0 text-xs text-fg-3 tabular-nums">{formatTimestamp(when)}</span>
        {live(run) && (
          <Button variant="ghost" size="icon-xs" aria-label={`Stop ${run.automation_name}`} title="Stop" onClick={onStop}>
            <Square size={12} />
          </Button>
        )}
      </div>
      <span className={cn("truncate pl-5.5 text-xs", run.error ? "text-danger" : "text-fg-3")} title={summary}>
        {summary}
      </span>
      {run.notified && run.notified !== "sent" && (
        <span className="truncate pl-5.5 text-xs text-[var(--warning)]" title={run.notified}>
          Discord message {run.notified}
        </span>
      )}
      {run.status === "running" && progress && progress.total > 0 && (
        <Progress className="mt-2 pl-5.5" value={((progress.phase - 1 + progress.done / progress.total) / Math.max(progress.phases, 1)) * 100} aria-label="Run progress" />
      )}
      {open && (
        <div className="mt-2 flex flex-col gap-1 pl-5.5 text-xs">
          {run.models.length > 0 && <span className="text-fg-3">{run.models.join(" · ")}</span>}
          {run.chapters.map((chapter) => (
            <div key={chapter.id} className="flex flex-col">
              <span className="flex items-center gap-2">
                <span className={cn("size-1.5 rounded-full", chapter.status === "done" ? "bg-success" : chapter.status === "stopped" ? "bg-fg-3" : chapter.status === "partial" ? "bg-warning" : "bg-danger")} />
                <span className="text-fg-2">{chapter.name}</span>
                <span className="text-fg-3">
                  {chapter.failed ? `${chapter.failed} of ${plural(chapter.pages, "page")} failed` : chapter.status === "stopped" ? "Stopped" : plural(chapter.pages, "page")}
                </span>
              </span>
              {chapter.errors.map((message) => (
                <span key={message} className="truncate pl-3.5 text-fg-3" title={message}>
                  {message}
                </span>
              ))}
            </div>
          ))}
        </div>
      )}
    </li>
  );
}

// ---- editor -----------------------------------------------------------------------------------------

function Row({ title, description, children }: { title: string; description?: string; children: ReactNode }) {
  return (
    <div className="flex items-start gap-4 py-3 [&+&]:shadow-[inset_0_1px_0_var(--border-soft)]">
      <div className="flex w-44 shrink-0 flex-col pt-1.5">
        <span className="text-sm font-medium">{title}</span>
        {description && <span className="text-xs text-fg-3">{description}</span>}
      </div>
      <div className="flex min-w-0 flex-1 flex-col gap-2">{children}</div>
    </div>
  );
}

function CheckRow({ checked, label, detail, onChange }: { checked: boolean; label: string; detail?: string; onChange: (checked: boolean) => void }) {
  return (
    <button
      role="checkbox"
      aria-checked={checked}
      className="flex w-full cursor-pointer items-center gap-2 rounded-control px-2 py-1 text-left hover:bg-hover"
      onClick={() => onChange(!checked)}
    >
      <span
        className={cn(
          "flex size-4 shrink-0 items-center justify-center rounded-[4px]",
          checked ? "bg-brand text-on-brand" : "bg-surface shadow-[inset_0_0_0_1px_var(--border-strong)]",
        )}
      >
        {checked && <Check size={12} strokeWidth={3} />}
      </span>
      <span className="truncate">{label}</span>
      {detail && <span className="ml-auto shrink-0 text-xs text-fg-3">{detail}</span>}
    </button>
  );
}

function AutomationEditor({
  initial,
  profiles,
  wakeSupported,
  onCancel,
  onSave,
  onProfiles,
}: {
  initial: AutomationBody;
  profiles: ModelProfile[];
  wakeSupported: boolean;
  onCancel: () => void;
  onSave: (body: AutomationBody) => Promise<void>;
  onProfiles: () => void;
}) {
  const ws = useWorkspace();
  const project = ws.project!;
  const [body, setBody] = useState<AutomationBody>(initial);
  const [saving, setSaving] = useState(false);
  const set = (patch: Partial<AutomationBody>) => setBody((b) => ({ ...b, ...patch }));
  const setSchedule = (patch: Partial<AutomationBody["schedule"]>) => setBody((b) => ({ ...b, schedule: { ...b.schedule, ...patch } }));
  const setTarget = (patch: Partial<AutomationBody["target"]>) => setBody((b) => ({ ...b, target: { ...b.target, ...patch } }));
  const setNotify = (patch: Partial<AutomationBody["notify"]>) => {
    setTested(null);
    setBody((b) => ({ ...b, notify: { ...b.notify, ...patch } }));
  };
  // The test message's outcome, shown beside the button.
  const [tested, setTested] = useState<{ ok: boolean; text: string } | null>(null);
  const [testing, setTesting] = useState(false);
  const linked = !!ws.link;
  const chapters = project.chapters.filter((c) => c.pageIds.length > 0);
  const profileOptions: Option[] = [{ value: CURRENT, label: "Current settings" }, ...profiles.map((p) => ({ value: p.id, label: p.name }))];
  const { schedule, target, notify } = body;
  const webhookOk = isDiscordWebhook(notify.webhook_url);
  const mentionId = discordUserId(notify.mention);
  const invalid =
    (notify.enabled && (!webhookOk || mentionId === null)) ||
    !body.name.trim() ||
    body.stages.length === 0 ||
    (target.kind === "chapters" && target.chapter_ids.length === 0) ||
    (schedule.kind === "daily" && schedule.days.length === 0) ||
    (schedule.kind === "once" && !schedule.date);

  return (
    <>
      <ScrollArea className="min-h-0 flex-1" contentClassName="px-5">
        <Row title="Name">
          <Input aria-label="Automation name" value={body.name} onChange={(event) => set({ name: event.target.value })} />
        </Row>
        <Row title="Chapters" description={target.kind === "pending" ? "Chapters with pages a chosen stage has not processed, in project order." : undefined}>
          <Segmented
            label="Chapters to run"
            value={target.kind}
            options={[
              { value: "pending", label: "Unprocessed" },
              { value: "chapters", label: "Chosen" },
            ]}
            onChange={(kind) => setTarget({ kind })}
          />
          {target.kind === "pending" ? (
            <>
              <div className="flex items-center gap-2 text-sm text-fg-2">
                <span>At most</span>
                <NumberField label="Chapters per run" value={target.limit} placeholder="All" min={1} max={999} step={1} onCommit={(limit) => setTarget({ limit: limit || null })} />
                <span>chapters per run</span>
              </div>
              {linked && (
                <label className="flex items-center gap-2 text-sm">
                  <Switch label="Download new chapters first" checked={target.download} onChange={(download) => setTarget({ download })} />
                  Download chapters from the linked source first
                </label>
              )}
            </>
          ) : (
            <div className="flex max-h-48 flex-col overflow-y-auto rounded-control py-1 shadow-[inset_0_0_0_1px_var(--border-soft)]">
              {chapters.length === 0 && <span className="px-2 py-1 text-xs text-fg-3">This project has no pages yet.</span>}
              {chapters.length > 1 && (
                <CheckRow
                  checked={target.chapter_ids.length === chapters.length}
                  label="All chapters"
                  onChange={(all) => setTarget({ chapter_ids: all ? chapters.map((c) => c.id) : [] })}
                />
              )}
              {chapters.map((chapter) => (
                <CheckRow
                  key={chapter.id}
                  checked={target.chapter_ids.includes(chapter.id)}
                  label={chapter.name}
                  detail={plural(chapter.pageIds.length, "page")}
                  onChange={(on) => setTarget({ chapter_ids: on ? [...target.chapter_ids, chapter.id] : target.chapter_ids.filter((id) => id !== chapter.id) })}
                />
              ))}
            </div>
          )}
        </Row>
        <Row title="Stages" description="Run in this order, each model alone.">
          <ToggleGroup
            multiple
            aria-label="Stages"
            value={body.stages}
            onValueChange={(next) => set({ stages: STAGES.filter((stage) => (next as string[]).includes(stage)) })}
          >
            {STAGES.map((stage) => (
              <ToggleGroupItem key={stage} value={stage}>
                {STAGE_LABEL[stage]}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        </Row>
        <Row title="Models">
          <div className="flex items-center gap-2">
            <Select label="Model profile" className="min-w-52" value={body.profile ?? CURRENT} options={profileOptions} onChange={(id) => set({ profile: id === CURRENT ? null : id })} />
            <Button variant="link" size="sm" onClick={onProfiles}>
              Edit profiles
            </Button>
          </div>
        </Row>
        <Row title="When">
          <Segmented
            label="Schedule"
            value={schedule.kind}
            options={[
              { value: "daily", label: "Repeat" },
              { value: "once", label: "Once" },
              { value: "manual", label: "By hand" },
            ]}
            onChange={(kind) => setSchedule({ kind, date: kind === "once" && !schedule.date ? tomorrow() : schedule.date })}
          />
          {schedule.kind !== "manual" && (
            <div className="flex flex-wrap items-center gap-2">
              {schedule.kind === "once" && (
                <Input type="date" aria-label="Date" className="w-40" value={schedule.date ?? ""} onChange={(event) => setSchedule({ date: event.target.value || null })} />
              )}
              <span className="text-sm text-fg-2">at</span>
              <Input type="time" aria-label="Time" className="w-28" value={schedule.time} onChange={(event) => event.target.value && setSchedule({ time: event.target.value })} />
            </div>
          )}
          {schedule.kind === "daily" && (
            <ToggleGroup
              multiple
              aria-label="Days"
              value={schedule.days.map(String)}
              onValueChange={(next) => setSchedule({ days: (next as string[]).map(Number).sort() })}
            >
              {[0, 1, 2, 3, 4, 5, 6].map((day) => (
                <ToggleGroupItem key={day} value={String(day)}>
                  {weekdayName(day)}
                </ToggleGroupItem>
              ))}
            </ToggleGroup>
          )}
        </Row>
        {schedule.kind !== "manual" && (
          <Row title="Wake the computer" description={wakeSupported ? "Wakes it from sleep and starts Urakatsu if it is closed." : "Available on Windows."}>
            <div className="pt-1">
              <Switch label="Wake the computer" checked={body.wake && wakeSupported} disabled={!wakeSupported} onChange={(wake) => set({ wake })} />
            </div>
          </Row>
        )}
        <Row title="Discord" description="Post to a channel when a run finishes. Runs that find nothing new stay quiet.">
          <label className="flex items-center gap-2 text-sm">
            <Switch label="Post to Discord when a run finishes" checked={notify.enabled} onChange={(enabled) => setNotify({ enabled })} />
            Post to Discord when a run finishes
          </label>
          {notify.enabled && (
            <>
              <Input
                aria-label="Discord webhook URL"
                placeholder="https://discord.com/api/webhooks/…"
                value={notify.webhook_url}
                aria-invalid={notify.webhook_url !== "" && !webhookOk}
                onChange={(event) => setNotify({ webhook_url: event.target.value })}
              />
              <span className="text-xs text-fg-3">In Discord: the channel's settings, Integrations, Webhooks, New Webhook, then Copy Webhook URL.</span>
              <div className="flex flex-wrap items-center gap-2">
                <Input
                  aria-label="Discord user ID to mention"
                  className="w-60"
                  placeholder="User ID to mention (optional)"
                  value={notify.mention}
                  aria-invalid={mentionId === null}
                  onChange={(event) => setNotify({ mention: event.target.value })}
                />
                <Button
                  variant="outline"
                  size="sm"
                  disabled={!webhookOk || mentionId === null || testing}
                  onClick={async () => {
                    setTesting(true);
                    try {
                      await automations.testNotify({ ...notify, mention: mentionId ?? "" });
                      setTested({ ok: true, text: "Sent: check the channel." });
                    } catch (error) {
                      setTested({ ok: false, text: describeError(error) });
                    } finally {
                      setTesting(false);
                    }
                  }}
                >
                  Send a test
                </Button>
                {tested && <span className={cn("text-xs", tested.ok ? "text-fg-2" : "text-danger")}>{tested.text}</span>}
              </div>
              <span className="text-xs text-fg-3">To get a user ID: turn on Developer Mode (Settings, Advanced), then right-click the user and Copy User ID.</span>
            </>
          )}
        </Row>
      </ScrollArea>
      <DialogFooter className="px-5 py-4 shadow-[inset_0_1px_0_var(--border-soft)]">
        <Button variant="ghost" onClick={onCancel}>
          Cancel
        </Button>
        <Button
          disabled={invalid || saving}
          onClick={async () => {
            setSaving(true);
            await onSave({ ...body, name: body.name.trim(), notify: { ...notify, webhook_url: notify.webhook_url.trim(), mention: mentionId ?? "" } });
            setSaving(false);
          }}
        >
          Save
        </Button>
      </DialogFooter>
    </>
  );
}

function tomorrow(): string {
  const day = new Date(Date.now() + 86_400_000);
  return `${day.getFullYear()}-${String(day.getMonth() + 1).padStart(2, "0")}-${String(day.getDate()).padStart(2, "0")}`;
}

// ---- profiles -------------------------------------------------------------------------------------

function emptyProfile(): ModelProfile {
  return { id: "", name: "Best quality", detection: null, ocr: null, inpainting: null, translation: null };
}

function ProfileEditor({
  profiles,
  onChange,
  onError,
  onDone,
}: {
  profiles: ModelProfile[];
  onChange: (profiles: ModelProfile[]) => void;
  onError: (message?: string) => void;
  onDone: () => void;
}) {
  const ws = useWorkspace();
  const [selected, setSelected] = useState<string>(profiles[0]?.id ?? "");
  const [draft, setDraft] = useState<ModelProfile>(() => structuredClone(profiles[0] ?? emptyProfile()));
  const [saving, setSaving] = useState(false);
  const { refreshModels, translationModels } = ws;
  useEffect(() => {
    if (translationModels.length === 0) void refreshModels();
  }, [refreshModels, translationModels.length]);

  const pick = (id: string) => {
    setSelected(id);
    setDraft(structuredClone(profiles.find((p) => p.id === id) ?? emptyProfile()));
  };
  const providers = ws.capabilities?.providers ?? [];
  const stageOptions = (stage: StageName): Option[] => [
    { value: CURRENT, label: "Current setting" },
    ...providers.filter((p) => p.stage === stage).map((p) => ({ value: p.id, label: p.name, disabled: !p.available, detail: p.available ? undefined : "Unavailable" })),
  ];
  const translation = draft.translation;
  const models = translationModels.filter((m) => m.provider === translation?.provider);
  const model = models.find((m) => m.model === translation?.model);
  const chooseTranslation = (provider: string, modelId?: string) => {
    if (provider === CURRENT) return setDraft((d) => ({ ...d, translation: null }));
    const choice = translationModels.find((m) => m.provider === provider && (modelId === undefined || m.model === modelId));
    setDraft((d) => ({
      ...d,
      translation: {
        provider,
        model: choice?.model ?? modelId ?? null,
        quantization: choice?.quantizations[0]?.id ?? null,
        vision: !!choice?.vision,
        reasoning: !!choice?.reasoning,
      },
    }));
  };

  const save = async () => {
    setSaving(true);
    try {
      const saved = draft.id ? await automations.updateProfile(draft) : await automations.createProfile(draft);
      onChange(draft.id ? profiles.map((p) => (p.id === saved.id ? saved : p)) : [...profiles, saved]);
      setSelected(saved.id);
      setDraft(structuredClone(saved));
      onError(undefined);
    } catch (failure) {
      onError(describeError(failure));
    } finally {
      setSaving(false);
    }
  };
  const remove = async () => {
    if (!draft.id) return pick("");
    try {
      await automations.deleteProfile(draft.id);
      const rest = profiles.filter((p) => p.id !== draft.id);
      onChange(rest);
      setSelected(rest[0]?.id ?? "");
      setDraft(structuredClone(rest[0] ?? emptyProfile()));
    } catch (failure) {
      onError(describeError(failure));
    }
  };

  const listOptions: Option[] = useMemo(() => [...profiles.map((p) => ({ value: p.id, label: p.name })), { value: "", label: "New profile" }], [profiles]);

  return (
    <>
      <ScrollArea className="min-h-0 flex-1" contentClassName="px-5">
        <Row title="Profile">
          <div className="flex items-center gap-2">
            <Select label="Profile" className="min-w-52" value={selected} options={listOptions} onChange={pick} />
            <Input aria-label="Profile name" value={draft.name} onChange={(event) => setDraft({ ...draft, name: event.target.value })} />
          </div>
        </Row>
        {(["detection", "ocr", "inpainting"] as const).map((stage) => (
          <Row key={stage} title={STAGE_LABEL[stage]}>
            <Select label={`${STAGE_LABEL[stage]} model`} value={draft[stage] ?? CURRENT} options={stageOptions(stage)} onChange={(id) => setDraft({ ...draft, [stage]: id === CURRENT ? null : id })} />
          </Row>
        ))}
        <Row title="Translation">
          <Select label="Translation provider" value={translation?.provider ?? CURRENT} options={stageOptions("translation")} onChange={(id) => chooseTranslation(id)} />
          {translation && models.length > 0 && (
            <Select
              label="Translation model"
              value={translation.model ?? ""}
              options={models.map((m) => ({ value: m.model, label: m.name || m.model, detail: [m.vision && "Vision", m.reasoning && "Reasoning"].filter(Boolean).join(" · ") || undefined }))}
              onChange={(id) => chooseTranslation(translation.provider, id)}
            />
          )}
          {translation && models.length === 0 && (
            <Input
              aria-label="Translation model ID"
              placeholder="Model ID"
              value={translation.model ?? ""}
              onChange={(event) => setDraft({ ...draft, translation: { ...translation, model: event.target.value || null } })}
            />
          )}
          {translation && model && model.quantizations.length > 0 && (
            <Select
              label="Quantization"
              value={translation.quantization ?? ""}
              options={model.quantizations.map((q) => ({ value: q.id, label: q.name, detail: q.downloaded ? "Downloaded" : undefined }))}
              onChange={(quantization) => setDraft({ ...draft, translation: { ...translation, quantization } })}
            />
          )}
        </Row>
      </ScrollArea>
      <DialogFooter className="px-5 py-4 shadow-[inset_0_1px_0_var(--border-soft)]">
        <Button variant="ghost" className="mr-auto" onClick={() => void remove()}>
          {draft.id ? "Delete profile" : "Discard"}
        </Button>
        <Button variant="ghost" onClick={onDone}>
          Done
        </Button>
        <Button disabled={!draft.name.trim() || saving} onClick={() => void save()}>
          {draft.id ? "Save profile" : "Create profile"}
        </Button>
      </DialogFooter>
    </>
  );
}
