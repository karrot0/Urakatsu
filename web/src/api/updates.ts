// JSON client for app updates (/api/v1/updates): the running build, the chosen channel, the newest release.

import { call } from "./catalog";

export type Channel = "stable" | "nightly";

export interface BuildInfo {
  version: string;
  channel: string;
  commit: string;
  built: string;
}

export interface UpdateRelease {
  /** "v0.2.0", or "nightly@<commit>": what "Later" remembers. */
  id: string;
  channel: Channel;
  /** "0.2.0"; a nightly's short commit. */
  version: string;
  commit: string;
  published: string;
  notes: string;
  url: string;
  asset: { name: string; url: string; size: number; sha256: string | null } | null;
}

export interface UpdateStatus {
  current: BuildInfo;
  channel: Channel;
  checked_at: number | null;
  checking: boolean;
  error: string | null;
  latest: UpdateRelease | null;
  available: boolean;
  switching: boolean;
  dismissed: boolean;
  can_install: boolean;
  install: { status: "idle" | "downloading" | "ready" | "installing" | "failed"; done: number; total: number; error: string | null };
}

export const updates = {
  status: () => call<UpdateStatus>("/updates"),
  check: () => call<UpdateStatus>("/updates/check", { method: "POST" }),
  setChannel: (channel: Channel) => call<UpdateStatus>("/updates/channel", { method: "PUT", body: JSON.stringify({ channel }) }),
  dismiss: () => call<UpdateStatus>("/updates/dismiss", { method: "POST" }),
  install: () => call<UpdateStatus>("/updates/install", { method: "POST" }),
};

/** "Urakatsu 0.2.0", or "Nightly a1b2c3d". */
export function releaseName(release: UpdateRelease): string {
  return release.channel === "nightly" ? `Nightly ${release.version}` : `Urakatsu ${release.version}`;
}
