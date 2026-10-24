// Settings > Sources: the installer. Turn a source system on (Urakatsu installs and starts its
// runtime), add extension repositories, and install the sources you read.

import { ChevronDown, ChevronUp, Globe, Plus, RefreshCw, Search, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { describeError } from "../api/client";
import { languageName, sources, type Extension, type Repository, type SystemStatus } from "../api/sources";
import { useWorkspace } from "../state/workspace";
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
      description="Source systems run the scrapers that link a project to a series and download its chapters. Each keeps its runtime on the newest release."
    >
      {!systems && <p className="text-fg-2">Loading source systems.</p>}
      {systems?.map((system) => <SystemSection key={system.id} system={system} onStatus={replace} />)}
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

  return (
    <PreferenceSection title={system.name} description={system.description} action={<Badge tone={tone}>{label}</Badge>}>
      <PreferenceRow title="Turned on" description={system.enabled ? "Starts with Urakatsu." : FIRST_START[system.id]}>
        <Switch label={`Turn on ${system.name}`} checked={system.enabled} disabled={busy} onChange={(on) => act(() => sources.setEnabled(system.id, on))} />
      </PreferenceRow>
      <PreferenceRow title="Runtime" description={system.message ?? (starting && !running ? "Installing and starting. This can take a few minutes the first time." : undefined)}>
        <div className="flex items-center justify-between gap-2">
          <span className="tabular-nums">{system.version ?? "Not installed"}</span>
          {running ? (
            <Button variant="outline" size="sm" disabled={busy} onClick={() => act(() => sources.stop(system.id))}>
              Stop
            </Button>
          ) : (
            <Button variant="outline" size="sm" disabled={starting || !system.enabled} onClick={() => act(() => sources.start(system.id))}>
              {starting ? "Starting" : "Start"}
            </Button>
          )}
        </div>
      </PreferenceRow>
      {system.browser && <BrowserRow system={system} />}
      {running && <Repositories system={system} />}
    </PreferenceSection>
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
                onClick={() => window.confirm(`Remove ${repo.name}? Installed sources stay installed.`) && void change(() => sources.removeRepository(system.id, repo.url))}
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
