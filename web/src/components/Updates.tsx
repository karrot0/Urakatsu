// App updates: the popup that offers a new release, and Settings > About (version and channel).
// The server checks GitHub every hour; the UI reads its status every few minutes and while an
// update downloads. Both views share one store, so dismissing in one hides it in the other.

import { ArrowUpCircle, ExternalLink, RefreshCw, X } from "lucide-react";
import { useEffect, useSyncExternalStore } from "react";

import { formatBytes } from "../api/catalog";
import { describeError } from "../api/client";
import { releaseName, updates, type Channel, type UpdateStatus } from "../api/updates";
import { formatDate, formatTimestamp } from "../format";
import { NetworkServerSettings } from "./NetworkServer";
import { useWorkspace } from "../state/workspace";
import { PreferencePage, PreferenceRow, PreferenceSection } from "./SettingsPage";
import { Segmented } from "./ui";
import { Button, buttonVariants } from "@/components/ui/button";

const POLL_MS = 5 * 60_000;
const INSTALL_POLL_MS = 700;

// ---- the shared status ---------------------------------------------------------------------------

let current: UpdateStatus | undefined;
const listeners = new Set<() => void>();
const publish = (next: UpdateStatus) => {
  current = next;
  listeners.forEach((listener) => listener());
};
const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};

/** The update status, refreshed every few minutes (and quickly while an update downloads). */
export function useUpdates() {
  const ws = useWorkspace();
  const status = useSyncExternalStore(subscribe, () => current);
  const busy = status?.install.status === "downloading" || status?.install.status === "installing" || status?.checking;

  useEffect(() => {
    let alive = true;
    const load = () => updates.status().then((next) => alive && publish(next), () => undefined);
    void load();
    const timer = window.setInterval(load, busy ? INSTALL_POLL_MS : POLL_MS);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [busy]);

  const act = async (task: () => Promise<UpdateStatus>) => {
    try {
      publish(await task());
    } catch (error) {
      ws.reportError(describeError(error));
    }
  };
  return {
    status,
    check: () => act(updates.check),
    setChannel: (channel: Channel) => act(() => updates.setChannel(channel)),
    dismiss: () => act(updates.dismiss),
    install: () => act(updates.install),
  };
}

// ---- the popup -----------------------------------------------------------------------------------

/** Bottom-right card when a release the user has not put off is available. */
export function UpdatePopup() {
  const { status, dismiss, install } = useUpdates();
  if (!status?.available || !status.latest) return null;
  const { latest, install: progress } = status;
  const working = progress.status === "downloading" || progress.status === "installing";
  if (status.dismissed && !working && progress.status !== "failed") return null;
  const share = progress.total ? Math.min(100, (progress.done / progress.total) * 100) : 0;

  return (
    <aside
      className="fixed right-4 bottom-4 z-40 flex w-[340px] animate-[fade-in_var(--t-in)] flex-col gap-3 rounded-card bg-surface-over p-4 shadow-overlay"
      role="dialog"
      aria-label="Update available"
    >
      <div className="flex items-start gap-3">
        <ArrowUpCircle size={20} className="mt-0.5 flex-none text-brand" />
        <div className="flex min-w-0 flex-1 flex-col gap-0.5">
          <p className="font-medium">{status.switching ? `Switch to ${latest.channel === "nightly" ? "nightly" : "stable"} builds` : `${releaseName(latest)} is available`}</p>
          <p className="text-xs leading-4 text-fg-2">
            {status.switching
              ? `${releaseName(latest)}, published ${formatDate(latest.published)}.`
              : latest.channel === "nightly"
                ? `Built ${formatDate(latest.published)}. Nightly builds have the newest changes and may break.`
                : `You have ${status.current.version}. Published ${formatDate(latest.published)}.`}
          </p>
        </div>
        {!working && (
          <Button variant="ghost" size="icon-xs" className="-mt-0.5 -mr-1" aria-label="Remind me later" title="Later" onClick={() => void dismiss()}>
            <X size={14} />
          </Button>
        )}
      </div>

      {working && (
        <div className="flex flex-col gap-1" role="status">
          <div className="relative h-1 overflow-hidden rounded-full bg-surface-fill">
            <span className="absolute inset-y-0 left-0 rounded-[inherit] bg-brand transition-[width] duration-200 ease-settle" style={{ width: `${progress.status === "installing" ? 100 : share}%` }} />
          </div>
          <span className="text-xs leading-4 text-fg-3 tabular-nums">
            {progress.status === "installing" ? "Restarting to install the update." : `Downloading ${formatBytes(progress.done)} of ${formatBytes(progress.total)}`}
          </span>
        </div>
      )}
      {progress.status === "failed" && <p className="text-xs leading-4 text-danger">{progress.error}</p>}

      {!working && (
        <div className="flex items-center justify-end gap-2">
          <a className={buttonVariants({ variant: "ghost", size: "sm" })} href={latest.url} target="_blank" rel="noreferrer">
            Release notes
          </a>
          {status.can_install ? (
            <Button size="sm" onClick={() => void install()}>
              {progress.status === "failed" ? "Try again" : "Update now"}
            </Button>
          ) : (
            <a className={buttonVariants({ size: "sm" })} href={latest.asset?.url ?? latest.url} target="_blank" rel="noreferrer">
              Download
            </a>
          )}
        </div>
      )}
    </aside>
  );
}

// ---- Settings > About --------------------------------------------------------------------------

const CHANNEL_DETAIL: Record<Channel, string> = {
  stable: "Tested releases. Recommended.",
  nightly: "Built from the newest code every night. May break.",
};

export function AboutSettings() {
  const { status, check, setChannel, install } = useUpdates();
  if (!status) return <p className="text-fg-2">Loading.</p>;
  const { current, latest } = status;
  const build = current.channel === "dev" ? "Development build (running from source)" : current.channel === "nightly" ? "Nightly build" : "Stable release";
  const working = status.install.status === "downloading" || status.install.status === "installing";

  return (
    <PreferencePage title="About" description="Which Urakatsu this is, how it updates, and who else can reach it.">
      <PreferenceSection title="This version">
        <PreferenceRow title="Version" description={build}>
          <span className="tabular-nums">{current.version}</span>
        </PreferenceRow>
        {current.commit && (
          <PreferenceRow title="Commit" description={current.built ? `Built ${formatDate(current.built)}` : undefined}>
            <a className="font-mono text-xs text-fg-2 hover:text-fg hover:underline" href={`https://github.com/karrot0/Urakatsu/commit/${current.commit}`} target="_blank" rel="noreferrer">
              {current.commit.slice(0, 7)}
            </a>
          </PreferenceRow>
        )}
      </PreferenceSection>

      <PreferenceSection title="Updates" description="Urakatsu checks GitHub for a new release every hour.">
        <PreferenceRow title="Update channel" description={CHANNEL_DETAIL[status.channel]}>
          <Segmented<Channel>
            label="Update channel"
            value={status.channel}
            onChange={(channel) => void setChannel(channel)}
            options={[
              { value: "stable", label: "Stable" },
              { value: "nightly", label: "Nightly" },
            ]}
          />
        </PreferenceRow>
        <PreferenceRow
          title={status.available && latest ? (status.switching ? `${releaseName(latest)} (switch channel)` : `${releaseName(latest)} is available`) : "Up to date"}
          description={
            status.error
              ? status.error
              : status.checked_at
                ? `Checked ${formatTimestamp(status.checked_at)}.${latest ? "" : ` No ${status.channel} release yet.`}`
                : "Not checked yet."
          }
        >
          <div className="flex items-center justify-end gap-2">
            {status.available && latest && !working && (
              status.can_install ? (
                <Button size="sm" onClick={() => void install()}>
                  Update now
                </Button>
              ) : (
                <a className={buttonVariants({ size: "sm" })} href={latest.asset?.url ?? latest.url} target="_blank" rel="noreferrer">
                  Download
                </a>
              )
            )}
            {working && <span className="text-xs text-fg-3 tabular-nums">{status.install.status === "installing" ? "Restarting" : `${formatBytes(status.install.done)} of ${formatBytes(status.install.total)}`}</span>}
            <Button variant="outline" size="sm" disabled={status.checking || working} onClick={() => void check()}>
              <RefreshCw size={14} className={status.checking ? "animate-spin motion-reduce:animate-none" : undefined} /> Check now
            </Button>
          </div>
        </PreferenceRow>
        {latest && (
          <PreferenceRow title="Release notes" description={`${releaseName(latest)}, published ${formatDate(latest.published)}.`}>
            <a className={buttonVariants({ variant: "ghost", size: "sm" })} href={latest.url} target="_blank" rel="noreferrer">
              Open on GitHub <ExternalLink size={14} />
            </a>
          </PreferenceRow>
        )}
      </PreferenceSection>

      <NetworkServerSettings />
    </PreferencePage>
  );
}
