// Settings > Workers: notebooks and machines that run models for this app (urakatsu/remote/workers.py).
// Kaggle workers the app starts by itself and finds through the session's log; Colab opens in one
// click and prints a connection line to paste here; another machine runs `urakatsu worker` at a
// fixed address.

import { ExternalLink, MoreHorizontal } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { describeError } from "../api/client";
import { workers, type Quota, type Worker, type WorkerList, type WorkerState } from "../api/workers";
import { useWorkspace } from "../state/workspace";
import { ActionMenu, Switch } from "./primitives";
import { PreferencePage, PreferenceRow, PreferenceSection } from "./SettingsPage";
import { Badge, type Tone } from "./ui";
import { Button, buttonVariants } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const STATE: Record<WorkerState, [string, Tone]> = {
  online: ["Online", "success"],
  starting: ["Starting", "info"],
  offline: ["Offline", "neutral"],
  outdated: ["Update needed", "warning"],
};

const KIND: Record<Worker["kind"], string> = { kaggle: "Kaggle", colab: "Colab", manual: "Another machine" };

// Seconds between status checks: often while a worker starts, rarely otherwise.
const POLL_STARTING = 5_000;
const POLL_IDLE = 30_000;

export function WorkersSettings() {
  const ws = useWorkspace();
  const [list, setList] = useState<WorkerList>();
  const [failed, setFailed] = useState(false);
  const [busy, setBusy] = useState<string>();

  const refresh = () =>
    workers.list().then(
      (next) => {
        setList(next);
        setFailed(false);
      },
      (error) => {
        setFailed(true);
        ws.reportError(describeError(error));
      },
    );
  useEffect(() => {
    void refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const starting = !!list?.workers.some((w) => w.status?.state === "starting");
  useEffect(() => {
    const timer = window.setInterval(() => workers.list().then(setList, () => undefined), starting ? POLL_STARTING : POLL_IDLE);
    return () => window.clearInterval(timer);
  }, [starting]);

  /** Runs `task` for one worker (or "add"), then reloads the list; errors go to the error bar. */
  const act = async (key: string, task: () => Promise<unknown>) => {
    setBusy(key);
    try {
      await task();
      await refresh();
      return true;
    } catch (error) {
      ws.reportError(describeError(error));
      return false;
    } finally {
      setBusy(undefined);
    }
  };

  const move = (index: number, by: number) => {
    if (!list) return;
    const ids = list.workers.map((w) => w.id);
    const [id] = ids.splice(index, 1);
    ids.splice(index + by, 0, id);
    void act(id, () => workers.order(ids));
  };

  return (
    <PreferencePage
      title="Workers"
      description="GPUs elsewhere that run detection, OCR and inpainting for this app. Pick a worker as a model's runtime in Settings, Pipeline."
    >
      {!list && <p className="text-fg-2">{failed ? "Could not load workers. Restart Urakatsu if it was just updated." : "Loading workers."}</p>}
      {list?.workers.map((worker, index) => (
        <WorkerSection
          key={worker.id}
          worker={worker}
          list={list}
          busy={busy === worker.id}
          first={index === 0}
          last={index === list.workers.length - 1}
          act={(task) => act(worker.id, task)}
          move={(by) => move(index, by)}
        />
      ))}
      {list && <AddWorker list={list} busy={busy === "add"} act={(task) => act("add", task)} />}
    </PreferencePage>
  );
}

function WorkerSection({
  worker,
  list,
  busy,
  first,
  last,
  act,
  move,
}: {
  worker: Worker;
  list: WorkerList;
  busy: boolean;
  first: boolean;
  last: boolean;
  act: (task: () => Promise<unknown>) => Promise<boolean>;
  move: (by: number) => void;
}) {
  const status = worker.status;
  const [label, tone] = STATE[status?.state ?? "offline"];
  const online = status?.state === "online" || status?.state === "outdated";
  const statusText = online ? (status?.device ?? "Answering") : status?.state === "starting" ? "Starting" : "Not running";
  return (
    <PreferenceSection
      title={worker.name}
      description={worker.name !== KIND[worker.kind] ? KIND[worker.kind] : undefined}
      action={
        <div className="flex items-center gap-2">
          <Badge tone={tone}>{label}</Badge>
          <ActionMenu
            trigger={
              <Button variant="ghost" size="icon-sm" aria-label={`More for ${worker.name}`} disabled={busy}>
                <MoreHorizontal size={14} />
              </Button>
            }
            actions={[
              { label: "Move up", disabled: first, onSelect: () => move(-1) },
              { label: "Move down", disabled: last, onSelect: () => move(1) },
              {
                label: "Remove worker",
                danger: true,
                onSelect: () => window.confirm(`Remove ${worker.name}? A running notebook keeps running until it stops itself.`) && void act(() => workers.remove(worker.id)),
              },
            ]}
          />
        </div>
      }
    >
      <PreferenceRow title="Status" description={status?.detail ?? (online ? status?.url ?? undefined : undefined)}>
        <div className="flex items-center justify-between gap-2">
          <span className="truncate">{statusText}</span>
          {online && status?.stoppable && (
            <Button variant="outline" size="sm" disabled={busy} onClick={() => void act(() => workers.stop(worker.id))}>
              Stop
            </Button>
          )}
          {!online && worker.kind === "kaggle" && (
            <Button variant="outline" size="sm" disabled={busy || status?.state === "starting"} onClick={() => void act(() => workers.start(worker.id))}>
              {status?.state === "starting" ? "Starting" : "Start"}
            </Button>
          )}
          {!online && worker.kind === "colab" && <OpenColab url={list.colab_url} />}
        </div>
      </PreferenceRow>
      <PreferenceRow title="Runs" description={worker.runs.length ? "Loaded ahead of use, so the first page does not wait for weights." : "Pick this worker as a model's runtime in Settings, Pipeline."}>
        <span className="block truncate" title={worker.runs.map((run) => `${run.role}: ${run.name}`).join(", ")}>
          {worker.runs.length ? worker.runs.map((run) => `${run.role}: ${run.name}`).join(", ") : "Nothing yet"}
        </span>
      </PreferenceRow>
      {worker.kind === "kaggle" && (
        <>
          <PreferenceRow title="Start when needed" description="Starts the notebook when a run needs one of its models. It stops itself after 30 minutes without requests.">
            <Switch label="Start when needed" checked={worker.auto_start} disabled={busy} onChange={(on) => void act(() => workers.change(worker.id, { auto_start: on }))} />
          </PreferenceRow>
          <KaggleQuota worker={worker} />
          <PreferenceRow title="Kaggle account" description="The API token stays on this computer.">
            <span className="truncate">{worker.kaggle_username ?? "Not set"}</span>
          </PreferenceRow>
        </>
      )}
      {worker.kind === "colab" && <ConnectionRow worker={worker} busy={busy} act={act} />}
      {worker.kind === "manual" && (
        <PreferenceRow title="Address" description={worker.token_set ? "Sends its token with every request." : "Sends no token."}>
          <span className="truncate">{worker.url}</span>
        </PreferenceRow>
      )}
    </PreferenceSection>
  );
}

/** Opens the worker notebook in Colab, where Run all starts it. */
function OpenColab({ url }: { url: string }) {
  return (
    <a className={buttonVariants({ variant: "outline", size: "sm" })} href={url} target="_blank" rel="noreferrer">
      Open in Colab <ExternalLink size={14} />
    </a>
  );
}

/** The line a Colab notebook prints once its tunnel is open; pasted again after a restart. */
function ConnectionRow({ worker, busy, act }: { worker: Worker; busy: boolean; act: (task: () => Promise<unknown>) => Promise<boolean> }) {
  const [line, setLine] = useState("");
  return (
    <PreferenceRow
      title="Connection"
      description={worker.url ? `Answers at ${worker.url}. If the notebook restarts, paste its new line.` : "In Colab, choose Runtime, Run all, then paste the line that starts with Connection:."}
      align="start"
    >
      <form
        className="flex gap-1"
        onSubmit={async (event) => {
          event.preventDefault();
          if (await act(() => workers.change(worker.id, { connection: line.trim() }))) setLine("");
        }}
      >
        <Input className="min-w-0 flex-1" aria-label="Connection line" placeholder="Connection: https://" value={line} onChange={(e) => setLine(e.target.value)} />
        <Button variant="outline" size="sm" type="submit" disabled={busy || !line.trim()}>
          Connect
        </Button>
      </form>
    </PreferenceRow>
  );
}

function KaggleQuota({ worker }: { worker: Worker }) {
  const [quota, setQuota] = useState<Quota | null>();
  const state = worker.status?.state;
  const asked = useRef<string | undefined | null>(null); // null: not asked yet
  useEffect(() => {
    // Read again when the session starts or ends, since that is when the hours move.
    if (asked.current === state) return;
    asked.current = state;
    workers.quota(worker.id).then(setQuota, () => setQuota(null));
  }, [worker.id, state]);
  const hours = (value: number) => value.toFixed(value < 10 ? 1 : 0);
  return (
    <PreferenceRow title="GPU hours this week" description="Kaggle's free weekly allowance, shared with your other notebooks.">
      <span className="tabular-nums">
        {quota === undefined ? "Checking" : quota === null ? "Unavailable" : `${hours(quota.used_hours + quota.reserved_hours)} of ${hours(quota.allowed_hours)} used`}
      </span>
    </PreferenceRow>
  );
}

function AddWorker({ list, busy, act }: { list: WorkerList; busy: boolean; act: (task: () => Promise<unknown>) => Promise<boolean> }) {
  const [kaggleToken, setKaggleToken] = useState("");
  const [kaggleUser, setKaggleUser] = useState("");
  const [url, setUrl] = useState("");
  const [token, setToken] = useState("");
  const legacy = kaggleToken.trim() !== "" && !kaggleToken.trim().startsWith("KGAT_") && !kaggleToken.trim().startsWith("{");
  return (
    <PreferenceSection title={list.workers.length ? "Add a worker" : "No workers yet."} description="Kaggle and Colab workers are reached through a Cloudflare tunnel: your pages travel encrypted, but Cloudflare relays them and could read them.">
      <PreferenceRow
        title="Kaggle"
        description="Urakatsu starts and stops it for you on a free T4 GPU. Needs a phone-verified Kaggle account and an API token from your Kaggle settings."
        align="start"
      >
        <form
          className="flex flex-col gap-1"
          onSubmit={async (event) => {
            event.preventDefault();
            if (await act(() => workers.add({ kind: "kaggle", kaggle_token: kaggleToken.trim(), kaggle_username: kaggleUser.trim() }))) {
              setKaggleToken("");
              setKaggleUser("");
            }
          }}
        >
          <Input type="password" aria-label="Kaggle API token" placeholder="API token, or kaggle.json" value={kaggleToken} onChange={(e) => setKaggleToken(e.target.value)} />
          {legacy && <Input aria-label="Kaggle username" placeholder="Username (for a legacy key)" value={kaggleUser} onChange={(e) => setKaggleUser(e.target.value)} />}
          <div className="flex justify-between gap-1">
            <a className={buttonVariants({ variant: "ghost", size: "sm" })} href={list.kaggle_settings_url} target="_blank" rel="noreferrer">
              Get a token
            </a>
            <Button variant="outline" size="sm" type="submit" disabled={busy || !kaggleToken.trim() || (legacy && !kaggleUser.trim())}>
              Add Kaggle worker
            </Button>
          </div>
        </form>
      </PreferenceRow>
      <PreferenceRow title="Colab" description="Opens the worker notebook in Colab. Choose Run all there, then paste the connection line it prints into this worker.">
        <div className="flex justify-end">
          <Button
            variant="outline"
            size="sm"
            disabled={busy}
            onClick={() => {
              // Opened in the click itself: a window opened after a request would be blocked.
              window.open(list.colab_url, "_blank", "noreferrer");
              void act(() => workers.add({ kind: "colab" }));
            }}
          >
            Add Colab worker <ExternalLink size={14} />
          </Button>
        </div>
      </PreferenceRow>
      <PreferenceRow title="Another machine" description="Any computer running urakatsu worker, at an address this one can reach." align="start">
        <form
          className="flex flex-col gap-1"
          onSubmit={async (event) => {
            event.preventDefault();
            if (await act(() => workers.add({ kind: "manual", url: url.trim(), token: token.trim() }))) {
              setUrl("");
              setToken("");
            }
          }}
        >
          <Input aria-label="Worker address" placeholder="https://" value={url} onChange={(e) => setUrl(e.target.value)} />
          <Input type="password" aria-label="Worker token" placeholder="Token it printed" value={token} onChange={(e) => setToken(e.target.value)} />
          <div className="flex justify-end">
            <Button variant="outline" size="sm" type="submit" disabled={busy || !url.trim()}>
              Add worker
            </Button>
          </div>
        </form>
      </PreferenceRow>
    </PreferenceSection>
  );
}
