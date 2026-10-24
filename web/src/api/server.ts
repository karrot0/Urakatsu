// JSON client for the network server (/api/v1/server): the workspace shared on another address.

import { call } from "./catalog";

export interface NetworkSettings {
  autostart: boolean;
  host: string;
  port: number;
  cors_origins: string[];
  require_key: boolean;
  api_key: string;
  serve_ui: boolean;
}

export interface NetworkStatus {
  settings: NetworkSettings;
  running: boolean;
  error: string | null;
  /** Where it can be reached while running: 127.0.0.1 and each LAN address for 0.0.0.0. */
  urls: string[];
}

export const networkServer = {
  status: () => call<NetworkStatus>("/server"),
  update: (settings: NetworkSettings) => call<NetworkStatus>("/server", { method: "PUT", body: JSON.stringify(settings) }),
  start: () => call<NetworkStatus>("/server/start", { method: "POST" }),
  stop: () => call<NetworkStatus>("/server/stop", { method: "POST" }),
  regenerateKey: () => call<NetworkStatus>("/server/key", { method: "POST" }),
};
