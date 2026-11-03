// Settings > Sources: the installer. Turn a source system on (Urakatsu installs and starts its
// runtime, or connects to a Suwayomi Server you already run), add extension repositories, and install the sources you read. Below them, Komga, where
// projects are exported to (KomgaSettings).

import { ChevronDown, ChevronUp, Eye, EyeOff, Globe, Plus, RefreshCw, Search, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { describeError } from "../api/client";
import { languageName, sources, type Extension, type Repository, type SystemStatus } from "../api/sources";
import { useWorkspace } from "../state/workspace";
import { confirm } from "./ConfirmDialog";
import { KomgaSettings } from "./KomgaSettings";
import { Switch } from "./primitives";
import { PreferencePage, PreferenceRow, PreferenceSection } from "./SettingsPage";
import { SourceIcon } from "./SourcesPage";
import { Badge, type Tone } from "./ui";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";

const PAGE = 50;

const STATE: Record<SystemStatus["state"], [string, Tone]> = {
  running: ["Running", "success"],
  starting: ["Starting", "info"],
  stopped: ["Stopped", "neutral"],
  not_installed: ["Not installed", "neutral"],
  error: ["Error", "danger"],
};

const FIRST_START: Record<string, string> = {
  suwayomi: "The first start downloads Suwayomi Server with its own Java runtime (about 350 MB) and can take a few minutes.",
  mana: "The first start downloads Bun (about 40 MB) to run Mana bundles.",
};

const SERVER_HINT: Record<string, string> = {
  suwayomi: "http://192.168.1.10:4567",
};

const REPOSITORY_HINT: Record<string, string> = {
  suwayomi: "A Tachiyomi or Mihon extension repository, the index.min.json URL.",
  mana: "A Mana repository URL, the folder that holds sources.json.",
};

export function SourcesSettings() {
  const ws = useWorkspace();
  const [systems, setSystems] = useState<SystemStatus[]>();

  useEffect(() => {
    sources.systems().then(setSystems, (error) => ws.reportError(describeError(error)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const replace = (status: SystemStatus) => setSystems((current) => current?.map((s) => (s.id === status.id ? status : s)));

  // A system starting with the service (or from another window) reports in when it is up.
  const starting = !!systems?.some((s) => s.state === "starting");
  useEffect(() => {
    if (!starting) return;
    const timer = window.setInterval(() => sources.systems().then(setSystems, () => undefined), 2000);
    return () => window.clearInterval(timer);
  }, [starting]);

  return (
    <PreferencePage
      title="Sources"
      description="Source systems run the scrapers that link a project to a series and download its chapters. Each keeps its runtime on the newest release. Komga is where finished chapters go."
    >
      {!systems && <p className="text-fg-2">Loading source systems.</p>}
      {systems?.map((system) => <SystemSection key={system.id} system={system} onStatus={replace} />)}
      <KomgaSettings />
    </PreferencePage>
  );
}

function SystemSection({ system, onStatus }: { system: SystemStatus; onStatus: (status: SystemStatus) => void }) {
  const ws = useWorkspace();
  const [busy, setBusy] = useState(false);
  const running = system.state === "running";
  const starting = busy || system.state === "starting";
  const [label, tone] = STATE[starting && !running ? "starting" : system.state];

  const act = async (task: () => Promise<SystemStatus>) => {
    setBusy(true);
    try {
      onStatus(await task());
    } catch (error) {
      ws.reportError(describeError(error));
      sources.systems().then((all) => all.forEach(onStatus), () => undefined);
    } finally {
      setBusy(false);
    }
  };

  // A server already running elsewhere is connected to, never installed, started or stopped.
  const remote = !!system.server;
  const save = async (url: string | null, username: string, password: string | null) => {
    setBusy(true);
    try {
      const status = await sources.connect(system.id, url, username, password);
      onStatus(status);
      return true;
    } catch (error) {
      ws.reportError(describeError(error));
      return false;
    } finally {
      setBusy(false);
    }
  };

  return (
    <PreferenceSection title={system.name} description={system.description} action={<Badge tone={tone}>{label}</Badge>}>
      <PreferenceRow
        title="Turned on"
        description={system.enabled ? (remote ? "Connects when Urakatsu starts." : "Starts with Urakatsu.") : remote ? "Connects to your server; nothing is downloaded." : FIRST_START[system.id]}
      >
        <Switch label={`Turn on ${system.name}`} checked={system.enabled} disabled={busy} onChange={(on) => act(() => sources.setEnabled(system.id, on))} />
      </PreferenceRow>
      {system.connectable && <ServerForm system={system} busy={busy} onSave={save} />}
      <PreferenceRow
        title={remote ? "Server" : "Runtime"}
        description={system.message ?? (starting && !running ? (remote ? "Connecting." : "Installing and starting. This can take a few minutes the first time.") : (system.server ?? undefined))}
      >
        <div className="flex items-center justify-between gap-2">
          <span className="tabular-nums">{system.version ?? (remote ? "Not connected" : "Not installed")}</span>
          {running ? (
            <Button variant="outline" size="sm" disabled={busy} onClick={() => act(() => sources.stop(system.id))}>
              {remote ? "Disconnect" : "Stop"}
            </Button>
          ) : (
            <Button variant="outline" size="sm" disabled={starting || !system.enabled} onClick={() => act(() => sources.start(system.id))}>
              {starting ? (remote ? "Connecting" : "Starting") : remote ? "Connect" : "Start"}
            </Button>
          )}
        </div>
      </PreferenceRow>
      {system.browser && <BrowserRow system={system} />}
      {running && <Repositories system={system} />}
    </PreferenceSection>
  );
}

/** Where the system's server runs: Urakatsu's own, or one already running elsewhere (always on, on a NAS or in Docker). */
function ServerForm({ system, busy, onSave }: { system: SystemStatus; busy: boolean; onSave: (url: string | null, username: string, password: string | null) => Promise<boolean> }) {
  const [elsewhere, setElsewhere] = useState(!!system.server);
  const [url, setUrl] = useState(system.server ?? "");
  const [username, setUsername] = useState(system.username ?? "");
  const [password, setPassword] = useState("");
  const [shown, setShown] = useState(false);
  useEffect(() => {
    setElsewhere(!!system.server);
    setUrl(system.server ?? "");
    setUsername(system.username ?? "");
  }, [system.server, system.username]);
  // The saved password is only ever sent to the address and user it was saved for.
  const moved = url.trim().replace(/\/+$/, "") !== (system.server ?? "") || username.trim() !== (system.username ?? "");
  const saved = !!system.username && !moved;

  return (
    <>
      <PreferenceRow title="Use a server already running" description={`Connect to a ${system.name} you run yourself instead of running one here.`}>
        <Switch
          label="Use a server already running"
          checked={elsewhere}
          disabled={busy}
          onChange={async (on) => {
            setElsewhere(on);
            if (!on && system.server && !(await onSave(null, "", null))) setElsewhere(true);
          }}
        />
      </PreferenceRow>
      {elsewhere && (
        <form
          onSubmit={async (event) => {
            event.preventDefault();
            if (await onSave(url.trim(), username.trim(), password || (saved ? null : ""))) setPassword("");
          }}
        >
          <PreferenceRow title="Address" description={`Such as ${SERVER_HINT[system.id] ?? "http://192.168.1.10:4567"}.`}>
            <Input aria-label={`${system.name} address`} placeholder={SERVER_HINT[system.id] ?? "http://"} value={url} onChange={(e) => setUrl(e.target.value)} />
          </PreferenceRow>
          <PreferenceRow
            title="Sign-in"
            description={saved ? "Saved. Enter a new password to replace it." : "Only when the server asks for one (basic authentication). Kept on this computer; the password is never shown again."}
          >
            <div className="flex gap-1">
              <Input className="min-w-0 flex-1" aria-label="User name" autoComplete="off" placeholder="User name" value={username} onChange={(e) => setUsername(e.target.value)} />
              <Input
                className="min-w-0 flex-1"
                aria-label="Password"
                type={shown ? "text" : "password"}
                autoComplete="off"
                placeholder={saved ? "Saved" : "Password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
              <Button type="button" variant="ghost" size="icon-sm" aria-label={shown ? "Hide the password" : "Show the password"} onClick={() => setShown((s) => !s)}>
                {shown ? <EyeOff size={14} /> : <Eye size={14} />}
              </Button>
            </div>
          </PreferenceRow>
          <div className="flex justify-end py-3">
            <Button type="submit" size="sm" disabled={busy || !url.trim() || (!moved && !password && system.state === "running")}>
              {system.server ? "Save and connect" : "Connect"}
            </Button>
          </div>
        </form>
      )}
    </>
  );
}

/** The system's built-in browser: what runs it, and a way to open any page (to sign in, say). */
function BrowserRow({ system }: { system: SystemStatus }) {
  const ws = useWorkspace();
  const [url, setUrl] = useState("");
  return (
    <PreferenceRow title="Browser" description={`${system.browser}. Sites that check visitors open here; sign in to a site here too.`} align="start">
      <form
        className="flex gap-1"
        onSubmit={(event) => {
          event.preventDefault();
          const target = url.trim();
          if (target) ws.openBrowser({ system: system.id, url: /^https?:\/\//.test(target) ? target : `https://${target}` });
        }}
      >
        <Input className="min-w-0 flex-1" aria-label="Page to open" placeholder="https://" value={url} onChange={(e) => setUrl(e.target.value)} />
        <Button variant="outline" size="sm" type="submit" disabled={!url.trim()}>
          <Globe size={14} /> Open
        </Button>
      </form>
    </PreferenceRow>
  );
}

function Repositories({ system }: { system: SystemStatus }) {
  const ws = useWorkspace();
  const [repositories, setRepositories] = useState<Repository[]>();
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [revision, setRevision] = useState(0);

  useEffect(() => {
    sources.repositories(system.id).then(setRepositories, (error) => ws.reportError(describeError(error)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [system.id]);

  const change = async (task: () => Promise<Repository[]>) => {
    setBusy(true);
    try {
      setRepositories(await task());
      setRevision((n) => n + 1);
      return true;
    } catch (error) {
      ws.reportError(describeError(error));
      return false;
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <PreferenceRow title="Repositories" description={REPOSITORY_HINT[system.id]} align="start">
        <div className="flex min-w-0 flex-col gap-1">
          {repositories?.map((repo) => (
            <div key={repo.url} className="flex min-h-7 items-center justify-between gap-2">
              <span className="truncate" title={repo.url}>
                {repo.name}
                <span className="text-xs leading-4 tracking-[0.005em] text-fg-3"> {repo.extensions} extensions</span>
              </span>
              <Button
                variant="ghost"
                size="icon-xs"
                className="shrink"
                aria-label={`Remove ${repo.name}`}
                title="Remove repository"
                disabled={busy}
                onClick={async () => {
                  if (await confirm({ title: `Remove ${repo.name}?`, description: "Installed sources stay installed.", action: "Remove", danger: true })) void change(() => sources.removeRepository(system.id, repo.url));
                }}
              >
                <Trash2 size={14} />
              </Button>
            </div>
          ))}
          {repositories?.length === 0 && <p className="text-xs leading-4 tracking-[0.005em] text-fg-2">No repositories yet.</p>}
          <form
            className="flex gap-1"
            onSubmit={async (event) => {
              event.preventDefault();
              if (url.trim() && (await change(() => sources.addRepository(system.id, url.trim())))) setUrl("");
            }}
          >
            <Input className="min-w-0 flex-1" aria-label="Repository URL" placeholder="https://" value={url} onChange={(e) => setUrl(e.target.value)} />
            <Button variant="outline" size="sm" type="submit" disabled={busy || !url.trim()}>
              <Plus size={14} /> Add
            </Button>
          </form>
        </div>
      </PreferenceRow>
      {repositories && repositories.length > 0 && <Extensions key={revision} system={system} />}
    </>
  );
}

function Extensions({ system }: { system: SystemStatus }) {
  const ws = useWorkspace();
  const [extensions, setExtensions] = useState<Extension[]>();
  const [query, setQuery] = useState("");
  const [shown, setShown] = useState(PAGE);
  const [expanded, setExpanded] = useState(false);
  const [busy, setBusy] = useState<string>();

  const load = (refresh = false) => {
    setBusy(refresh ? "refresh" : "load");
    sources
      .extensions(system.id, refresh)
      .then(setExtensions, (error) => ws.reportError(describeError(error)))
      .finally(() => setBusy(undefined));
  };

  useEffect(() => load(), [system.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const matches = useMemo(() => {
    const text = query.trim().toLowerCase();
    return (extensions ?? []).filter((e) => !text || e.name.toLowerCase().includes(text) || languageName(e.language).toLowerCase().includes(text));
  }, [extensions, query]);

  const act = async (id: string, task: () => Promise<Extension[]>) => {
    setBusy(id);
    try {
      setExtensions(await task());
    } catch (error) {
      ws.reportError(describeError(error));
    } finally {
      setBusy(undefined);
    }
  };

  const installed = extensions?.filter((e) => e.installed).length ?? 0;
  // Collapsed shows what is installed; a search always covers every extension, since that is how you find one to install.
  const searching = !!query.trim();
  const all = expanded || searching || installed === 0;
  const listed = all ? matches : matches.filter((e) => e.installed);

  return (
    <PreferenceRow title="Extensions" description={extensions ? `${installed} of ${extensions.length} installed.` : "Loading extensions."} align="start">
      <div className="flex min-w-0 flex-col gap-1">
        <div className="flex gap-1">
          <label className="relative block flex-1">
            <Search className="pointer-events-none absolute top-1/2 left-2.5 -translate-y-1/2 text-icon-2" size={14} />
            <Input className="pl-[30px]" aria-label="Find extensions" placeholder="Find a site" value={query} onChange={(e) => setQuery(e.target.value)} />
          </label>
          <Button variant="ghost" size="icon-sm" title="Check repositories for updates" aria-label="Check for updates" disabled={!!busy} onClick={() => load(true)}>
            <RefreshCw size={14} />
          </Button>
        </div>
        <ScrollArea viewportClassName={all ? "max-h-[480px]" : undefined} contentClassName={all ? "pr-2.5" : undefined}>
        {listed.slice(0, shown).map((extension) => (
          <div key={extension.id} className="flex min-h-10 items-center gap-2 shadow-[inset_0_-1px_0_var(--border-soft)]">
            <SourceIcon url={extension.icon} />
            <span className="flex min-w-0 flex-1 flex-col">
              <span className="truncate">{extension.name}</span>
              <span className="truncate text-xs leading-4 tracking-[0.005em] text-fg-3">
                {languageName(extension.language)}
                {extension.version && ` · ${extension.version}`}
              </span>
            </span>
            {extension.nsfw && <Badge tone="warning">18+</Badge>}
            {extension.has_update && (
              <Button variant="outline" size="sm" disabled={!!busy} onClick={() => act(extension.id, () => sources.update(system.id, extension.id))}>
                Update
              </Button>
            )}
            {extension.installed ? (
              <Button variant="ghost" size="sm" disabled={!!busy} onClick={() => act(extension.id, () => sources.uninstall(system.id, extension.id))}>
                {busy === extension.id ? "Removing" : "Remove"}
              </Button>
            ) : (
              <Button variant="outline" size="sm" disabled={!!busy} onClick={() => act(extension.id, () => sources.install(system.id, extension.id))}>
                {busy === extension.id ? "Installing" : "Install"}
              </Button>
            )}
          </div>
        ))}
        {extensions && listed.length === 0 && <p className="py-2 text-xs leading-4 tracking-[0.005em] text-fg-2">No extensions match.</p>}
        {listed.length > shown && (
          <Button variant="ghost" size="sm" className="w-full" onClick={() => setShown((n) => n + PAGE)}>
            Show {Math.min(PAGE, listed.length - shown)} more of {listed.length - shown}
          </Button>
        )}
        </ScrollArea>
        {extensions && installed > 0 && !searching && (
          <Button
            variant="ghost"
            size="sm"
            className="justify-start text-fg-2"
            aria-expanded={expanded}
            onClick={() => {
              setExpanded((open) => !open);
              setShown(PAGE);
            }}
          >
            {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            {expanded ? "Show installed only" : `Show all ${extensions.length.toLocaleString()} extensions`}
          </Button>
        )}
      </div>
    </PreferenceRow>
  );
}
