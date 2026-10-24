// Settings > About > Network server: share the workspace on another address, like LM Studio's
// local server (urakatsu/service/network.py). The app's own window keeps its private server; this
// adds a listener with its own host, port, CORS origins and API key.

import { Check, Copy, Eye, EyeOff, RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";

import { describeError } from "../api/client";
import { networkServer, type NetworkSettings, type NetworkStatus } from "../api/server";
import { useWorkspace } from "../state/workspace";
import { NumberField, Switch } from "./primitives";
import { PreferenceRow, PreferenceSection } from "./SettingsPage";
import { Badge, Segmented } from "./ui";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

const LOCAL = "127.0.0.1";
const NETWORK = "0.0.0.0";

function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (!copied) return;
    const timer = window.setTimeout(() => setCopied(false), 1200);
    return () => window.clearTimeout(timer);
  }, [copied]);
  return (
    <Button
      variant="ghost"
      size="icon-sm"
      aria-label={label}
      title={label}
      onClick={() => void navigator.clipboard.writeText(text).then(() => setCopied(true), () => undefined)}
    >
      {copied ? <Check size={14} /> : <Copy size={14} />}
    </Button>
  );
}

export function NetworkServerSettings() {
  const ws = useWorkspace();
  const [status, setStatus] = useState<NetworkStatus>();
  const [busy, setBusy] = useState(false);
  const [shown, setShown] = useState(false);
  const [origins, setOrigins] = useState("");

  useEffect(() => {
    networkServer.status().then(setStatus, () => undefined);
  }, []);
  useEffect(() => setOrigins(status?.settings.cors_origins.join("\n") ?? ""), [status?.settings.cors_origins]);

  const act = async (task: () => Promise<NetworkStatus>) => {
    setBusy(true);
    try {
      setStatus(await task());
    } catch (error) {
      ws.reportError(describeError(error));
      networkServer.status().then(setStatus, () => undefined);
    } finally {
      setBusy(false);
    }
  };
  const save = (patch: Partial<NetworkSettings>) => status && act(() => networkServer.update({ ...status.settings, ...patch }));

  if (!status) return null;
  const { settings, running } = status;
  const hosts = [
    { value: LOCAL, label: "This computer" },
    { value: NETWORK, label: "Local network" },
    ...(settings.host === LOCAL || settings.host === NETWORK ? [] : [{ value: settings.host, label: settings.host }]),
  ];
  const exposed = settings.host !== LOCAL && settings.host !== "localhost" && settings.host !== "::1";
  const ui = status.urls.find((url) => !url.includes(LOCAL)) ?? status.urls[0];

  return (
    <PreferenceSection
      title="Network server"
      description="Share this workspace on another address, for other devices and tools. The app window does not need it."
      action={running ? <Badge tone="success">Running</Badge> : <Badge>Stopped</Badge>}
    >
      <PreferenceRow
        title="Server"
        description={status.error ?? (running ? undefined : `Starts on ${settings.host}:${settings.port}.`)}
      >
        <div className="flex items-center justify-end gap-3">
          <span className="text-xs text-fg-3">Start with Urakatsu</span>
          <Switch label="Start with Urakatsu" checked={settings.autostart} onChange={(autostart) => void save({ autostart })} />
          <Button size="sm" variant={running ? "outline" : "default"} disabled={busy} onClick={() => void act(running ? networkServer.stop : networkServer.start)}>
            {running ? "Stop" : "Start"}
          </Button>
        </div>
      </PreferenceRow>

      {running && status.urls.length > 0 && (
        <PreferenceRow title="Addresses" description={settings.serve_ui ? "The workspace and its API (/api/v1, docs at /api/docs)." : "The API (/api/v1, docs at /api/docs)."} align="start">
          <ul className="flex flex-col gap-0.5">
            {status.urls.map((url) => (
              <li key={url} className="flex items-center justify-between gap-2">
                <span className="truncate font-mono text-xs">{url}</span>
                <CopyButton text={url} label={`Copy ${url}`} />
              </li>
            ))}
          </ul>
        </PreferenceRow>
      )}

      <PreferenceRow
        title="Reachable from"
        description={exposed ? "Other devices on your network can connect. Turn on the API key unless you trust them all." : "Only programs on this computer can connect."}
      >
        <div className="flex justify-end">
          <Segmented<string> label="Reachable from" value={settings.host} options={hosts} onChange={(host) => void save({ host })} />
        </div>
      </PreferenceRow>

      <PreferenceRow title="Port">
        <NumberField label="Port" stepper={false} min={1} max={65535} format={{ useGrouping: false }} value={settings.port} onCommit={(port) => port && port !== settings.port && void save({ port })} />
      </PreferenceRow>

      <PreferenceRow title="Share the workspace UI" description="Off shares the API only.">
        <div className="flex justify-end">
          <Switch label="Share the workspace UI" checked={settings.serve_ui} onChange={(serve_ui) => void save({ serve_ui })} />
        </div>
      </PreferenceRow>

      <PreferenceRow
        title="Require an API key"
        description="Clients send it as Authorization: Bearer <key> or X-API-Key. In a browser, open the address with ?key=<key> once."
      >
        <div className="flex justify-end">
          <Switch label="Require an API key" checked={settings.require_key} onChange={(require_key) => void save({ require_key })} />
        </div>
      </PreferenceRow>

      {settings.require_key && settings.api_key && (
        <PreferenceRow title="API key">
          <div className="flex items-center gap-0.5">
            <Input readOnly aria-label="API key" className="min-w-0 flex-1 font-mono text-xs" type={shown ? "text" : "password"} value={settings.api_key} />
            <Button variant="ghost" size="icon-sm" aria-label={shown ? "Hide API key" : "Show API key"} onClick={() => setShown((s) => !s)}>
              {shown ? <EyeOff size={14} /> : <Eye size={14} />}
            </Button>
            <CopyButton text={settings.api_key} label="Copy API key" />
            <Button variant="ghost" size="icon-sm" aria-label="New API key" title="New API key (the old one stops working)" disabled={busy} onClick={() => void act(networkServer.regenerateKey)}>
              <RefreshCw size={14} />
            </Button>
          </div>
          {running && settings.serve_ui && ui && (
            <Button variant="link" size="sm" className="mt-1 text-xs" onClick={() => void navigator.clipboard.writeText(`${ui}/?key=${encodeURIComponent(settings.api_key)}`)}>
              Copy a sign-in link for a browser
            </Button>
          )}
        </PreferenceRow>
      )}

      <PreferenceRow title="CORS origins" description="Web pages allowed to call the server, one per line (https://example.com, or * for any). Empty allows none." align="start">
        <Textarea
          aria-label="CORS origins"
          className="font-mono text-xs"
          placeholder="http://localhost:3000"
          value={origins}
          onChange={(event) => setOrigins(event.target.value)}
          onBlur={() => {
            const next = origins.split(/[\s,]+/).filter(Boolean);
            if (next.join("\n") !== settings.cors_origins.join("\n")) void save({ cors_origins: next });
          }}
        />
      </PreferenceRow>
    </PreferenceSection>
  );
}
