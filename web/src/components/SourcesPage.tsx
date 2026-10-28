// Source browser: pick a source (grouped by source system), search it, and link a series to the
// open project, or to a new project when opened from the project list.

import { ArrowLeft, BookOpen, ChevronRight, ExternalLink, Link2, Library, MousePointerClick, Search, Settings2, ShieldCheck } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { challengeOf, type Challenge } from "../api/catalog";
import { describeError } from "../api/client";
import { languageName, sources, type RemoteChapter, type SearchPage, type Series, type SourceInfo, type SystemSources } from "../api/sources";
import { useWorkspace } from "../state/workspace";
import { Button, buttonVariants } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "cn";
import { Select, type Option } from "./primitives";
import { Badge } from "./ui";
import { ScrollArea } from "@/components/ui/scroll-area";
import { TitleBarInset } from "./WindowControls";
import { GlassButton, ListRow, ListSection, Screen, SearchField, type Enter } from "./mobile/glass";
import { useMobile } from "./mobile/useMobile";

const SEARCH_DELAY = 350;
const ALL = "all";
const LAST_SOURCE = "urakatsu.source";
const COLLAPSED = "urakatsu.sources.collapsed";
const SITE_LANGUAGE = "urakatsu.sources.language"; // extension -> the language last picked for it

function stored<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function store(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable */
  }
}

// Remembered for the session, so coming back to this page or to a source shows them at once:
// the source list (refreshed on every visit) and each source's first page of results per query
// (refreshed when older than RESULTS_FRESH_MS).
let recentSources: SystemSources[] | undefined;
const recentResults = new Map<string, { at: number; page: SearchPage }>();
const RESULTS_FRESH_MS = 5 * 60 * 1000;

/**
 * A site: one extension's sources. Multi-language extensions (MangaDex registers one source per
 * language, ~60) show as one row with a language picker instead of one row per language.
 */
interface Site {
  key: string;
  name: string;
  sources: SourceInfo[];
}

function sitesOf(list: SourceInfo[]): Site[] {
  const sites = new Map<string, Site>();
  for (const source of list) {
    const key = source.extension ?? `source:${source.id}`;
    const site = sites.get(key) ?? { key, name: siteName(source), sources: [] };
    site.sources.push(source);
    sites.set(key, site);
  }
  return [...sites.values()];
}

/** "MangaDex (EN)" -> "MangaDex": the language is picked separately. */
const siteName = (source: SourceInfo) => source.name.replace(/\s*\([^)]*\)$/, "") || source.name;

/** The source of ``site`` to open: the language picked for it last, else the browser's, else English, else the first. */
function siteSource(site: Site): SourceInfo {
  const picked = stored<Record<string, string>>(SITE_LANGUAGE, {})[site.key];
  const browser = navigator.language.toLowerCase();
  const by = (match: (tag: string) => boolean) => site.sources.find((s) => match(s.language.toLowerCase()));
  return by((tag) => tag === picked) ?? by((tag) => tag === browser) ?? by((tag) => tag === browser.split("-")[0]) ?? by((tag) => tag === "en") ?? site.sources[0];
}

function rememberLanguage(source: SourceInfo) {
  if (source.extension) store(SITE_LANGUAGE, { ...stored<Record<string, string>>(SITE_LANGUAGE, {}), [source.extension]: source.language.toLowerCase() });
}

/** The source used last time, if it is still listed: the page reopens it; otherwise it waits for a choice. */
function lastSelected(systems: SystemSources[] | undefined): Selected | undefined {
  const last = stored<{ system: string; source: string } | null>(LAST_SOURCE, null);
  const entry = systems?.find((e) => e.system.id === last?.system);
  const source = entry?.sources.find((s) => s.id === last?.source);
  return entry && source ? { system: entry.system.id, source } : undefined;
}

const byName = (a: SourceInfo, b: SourceInfo) => a.name.localeCompare(b.name, undefined, { numeric: true }) || a.language.localeCompare(b.language);

interface Selected {
  system: string;
  source: SourceInfo;
}

export function SourcesPage() {
  const ws = useWorkspace();
  // A phone shows one level at a time (sources, a source's search, a series), so it starts at the list.
  const mobile = useMobile();
  const [enter, setEnter] = useState<Enter>("none");
  const [systems, setSystems] = useState<SystemSources[] | undefined>(recentSources);
  const [error, setError] = useState<string>();
  const [selected, setSelected] = useState<Selected | undefined>(() => (mobile ? undefined : lastSelected(recentSources)));
  const [series, setSeries] = useState<Series>();
  const [filter, setFilter] = useState("");
  const [language, setLanguage] = useState(ALL);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>(() => stored(COLLAPSED, {}));

  const [reload, setReload] = useState(0);
  const starting = !!systems?.some((entry) => entry.system.enabled && entry.system.state === "starting");
  useEffect(() => {
    if (!starting) return;
    const timer = window.setTimeout(() => setReload((n) => n + 1), 3000);
    return () => window.clearTimeout(timer);
  }, [starting, systems]);

  useEffect(() => {
    sources.all().then(
      (found) => {
        const sorted = found.map((entry) => ({ ...entry, sources: [...entry.sources].sort(byName) }));
        recentSources = sorted;
        setSystems(sorted);
        const last = lastSelected(sorted);
        if (last && !mobile) setSelected((current) => current ?? last);
      },
      (e) => setError(describeError(e)),
    );
  }, [reload]);

  const choose = (system: string, source: SourceInfo) => {
    setSelected({ system, source });
    setSeries(undefined);
    store(LAST_SOURCE, { system, source: source.id });
  };

  const toggleGroup = (id: string) =>
    setCollapsed((current) => {
      const next = { ...current, [id]: !current[id] };
      store(COLLAPSED, next);
      return next;
    });

  const languages = useMemo(() => {
    const tags = new Set(systems?.flatMap((entry) => entry.sources.map((s) => s.language)) ?? []);
    const options: Option[] = [{ value: ALL, label: "All languages" }];
    for (const tag of [...tags].sort((a, b) => languageName(a).localeCompare(languageName(b)))) options.push({ value: tag, label: languageName(tag) });
    return options;
  }, [systems]);

  const visible = (source: SourceInfo) =>
    (language === ALL || source.language === language) && source.name.toLowerCase().includes(filter.trim().toLowerCase());
  const total = systems?.reduce((sum, entry) => sum + entry.sources.length, 0) ?? 0;

  if (mobile) {
    const back = (step: () => void) => (swiped?: boolean) => {
      setEnter(swiped ? "none" : "pop");
      step();
    };
    if (selected && series) {
      return (
        <Screen title={series.title} large={false} enter={enter} back={{ label: `Back to ${siteName(selected.source)}`, onBack: back(() => setSeries(undefined)) }}>
          <div className="-mx-4 flex flex-col">
            <SeriesDetail key={series.id} selected={selected} series={series} />
          </div>
        </Screen>
      );
    }
    if (selected) {
      return (
        <Screen title={siteName(selected.source)} large={false} enter={enter} back={{ label: "Back to sources", onBack: back(() => setSelected(undefined)) }}>
          <div className="-mx-4 flex flex-col">
            <SeriesSearch
              key={`${selected.system}:${selected.source.id}`}
              selected={selected}
              languages={selected.source.extension ? (systems?.find((e) => e.system.id === selected.system)?.sources.filter((s) => s.extension === selected.source.extension) ?? []) : []}
              onLanguage={(source) => {
                rememberLanguage(source);
                choose(selected.system, source);
              }}
              active={series?.id}
              onSelect={(next) => {
                setEnter("push");
                setSeries(next);
              }}
            />
          </div>
        </Screen>
      );
    }
    return (
      <Screen
        title="Browse"
        enter={enter}
        actions={
          <GlassButton label="Manage sources" onClick={() => ws.setSettingsOpen(true, "sources")}>
            <Settings2 size={20} />
          </GlassButton>
        }
        accessory={total > 0 && <SearchField value={filter} onChange={setFilter} placeholder="Filter sources" />}
      >
        <p className="px-4 pb-1 text-[13px] leading-[18px] text-fg-2">{ws.project ? `Link a series to ${ws.project.name}.` : "Pick a series to follow. It becomes a new project."}</p>
        {!systems && !error && <p className="px-4 py-10 text-center text-[15px] text-fg-2">Loading sources.</p>}
        {error && (
          <div className="px-4 py-10 text-center">
            <p className="text-body-lg font-semibold">Sources did not load.</p>
            <p className="mt-1 text-[15px] leading-5 text-fg-2">{error}</p>
          </div>
        )}
        {systems && total === 0 && (
          <div className="flex flex-col items-center gap-1 px-6 py-14 text-center">
            <p className="text-body-lg font-semibold">No sources installed.</p>
            <p className="max-w-[34ch] text-[15px] leading-5 text-fg-2">Turn on a source system and install the sites you read in Settings.</p>
            <GlassButton label="Set up sources" shape="capsule" tone="prominent" className="mt-4" onClick={() => ws.setSettingsOpen(true, "sources")}>
              Set up sources
            </GlassButton>
          </div>
        )}
        {systems?.map((entry) => {
          const sites = sitesOf(entry.sources.filter(visible));
          const note = !entry.system.enabled
            ? "Turned off. Turn it on in Settings, Sources."
            : entry.error
              ? entry.error
              : entry.system.state === "starting"
                ? "Starting. Its sources appear in a moment."
                : entry.sources.length === 0
                  ? "No sources installed."
                  : undefined;
          if (sites.length === 0 && !note) return null;
          return (
            <ListSection key={entry.system.id} header={entry.system.name} footer={note}>
              {sites.map((site) => {
                const source = site.sources.length > 1 ? siteSource(site) : site.sources[0];
                return (
                  <ListRow
                    key={site.key}
                    leading={<SourceIcon url={source.icon} size={30} />}
                    title={site.name}
                    detail={site.sources.length > 1 ? `${site.sources.length} languages` : source.language.toUpperCase()}
                    trailing={source.nsfw ? <Badge tone="warning">18+</Badge> : undefined}
                    chevron
                    onClick={() => {
                      setEnter("push");
                      choose(entry.system.id, source);
                    }}
                  />
                );
              })}
            </ListSection>
          );
        })}
      </Screen>
    );
  }

  return (
    <div className="flex h-screen flex-col bg-surface">
      <header className="flex h-12 flex-none items-center gap-3 pr-4 pl-2 shadow-[inset_0_-1px_0_var(--border-soft)]" data-titlebar>
        <TitleBarInset side="left" />
        <Button variant="ghost" size="sm" onClick={() => ws.setSourcesOpen(false)}>
          <ArrowLeft size={14} /> {ws.project ? "Back to editor" : "Back to projects"}
        </Button>
        <span className="inline-flex items-center gap-2 font-semibold">
          <Link2 size={16} className="text-icon-2" /> Link a series
        </span>
        <span className="truncate text-xs tracking-[0.005em] text-fg-2">{ws.project ? `to ${ws.project.name}` : "Creates a new project"}</span>
        <span className="flex-1" />
        <Button variant="ghost" size="sm" onClick={() => ws.setSettingsOpen(true, "sources")}>
          <Settings2 size={14} /> Manage sources
        </Button>
        <TitleBarInset side="right" />
      </header>
      <div className="grid flex-1 min-h-0 grid-cols-[280px_minmax(0,1fr)_380px]">
        <aside className="flex min-h-0 flex-col bg-surface-sunk shadow-[inset_-1px_0_0_var(--border-soft)]" aria-label="Sources">
          <div className="flex flex-col gap-2 p-3 shadow-[inset_0_-1px_0_var(--border-soft)]">
            <label className="relative block">
              <Search size={14} className="pointer-events-none absolute top-1/2 left-2.5 -translate-y-1/2 text-icon-2" />
              <Input aria-label="Filter sources" placeholder="Filter sources" value={filter} onChange={(e) => setFilter(e.target.value)} className="pl-[30px]" />
            </label>
            {languages.length > 2 && <Select size="sm" label="Language" value={language} options={languages} onChange={setLanguage} />}
          </div>
          <ScrollArea className="flex-1" contentClassName="relative px-2 pt-1 pb-8">
            {!systems && !error && (
              <div className="absolute inset-x-0 top-0 h-0.5 w-auto overflow-hidden rounded-none bg-surface-fill" aria-label="Loading sources">
                <span className="absolute inset-y-0 left-0 w-[30%]! rounded-[inherit] bg-brand animate-[slide_1.1s_ease-in-out_infinite]" />
              </div>
            )}
            {error && (
              <div className="px-3 py-8 text-center">
                <p className="text-fg">Sources did not load.</p>
                <p className="mt-0.5 text-xs leading-5 text-fg-2">{error}</p>
              </div>
            )}
            {systems?.map((entry) => (
              <SystemGroup
                key={entry.system.id}
                entry={entry}
                sources={entry.sources.filter(visible)}
                selected={selected}
                open={!collapsed[entry.system.id] || !!filter.trim()}
                onToggle={() => toggleGroup(entry.system.id)}
                onSelect={(source) => choose(entry.system.id, source)}
              />
            ))}
          </ScrollArea>
        </aside>
        {selected ? (
          <SeriesSearch
            key={`${selected.system}:${selected.source.id}`}
            selected={selected}
            languages={
              selected.source.extension
                ? (systems?.find((e) => e.system.id === selected.system)?.sources.filter((s) => s.extension === selected.source.extension) ?? [])
                : []
            }
            onLanguage={(source) => {
              rememberLanguage(source);
              choose(selected.system, source);
            }}
            active={series?.id}
            onSelect={setSeries}
          />
        ) : (
          <section className="flex min-h-0 min-w-0 flex-col">
            {systems && total === 0 && (
              <div className="flex flex-1 flex-col items-center justify-center gap-1 bg-transparent p-8 text-center [&_svg]:text-icon-2">
                <Library size={24} />
                <p className="text-base text-fg">No sources installed.</p>
                <p className="max-w-[42ch] text-fg-2">Turn on a source system and install the sites you read in Settings.</p>
                <Button className="mt-2" onClick={() => ws.setSettingsOpen(true, "sources")}>
                  Set up sources
                </Button>
              </div>
            )}
            {systems && total > 0 && (
              <div className="flex flex-1 flex-col items-center justify-center gap-1 bg-transparent p-8 text-center [&_svg]:text-icon-2">
                <MousePointerClick size={24} />
                <p className="text-base text-fg">No source selected.</p>
                <p className="max-w-[42ch] text-fg-2">Choose the site that carries the series, then search it here.</p>
              </div>
            )}
          </section>
        )}
        <section className="flex min-h-0 min-w-0 flex-col shadow-[inset_1px_0_0_var(--border-soft)]" aria-label="Series">
          {selected && series ? (
            <SeriesDetail key={series.id} selected={selected} series={series} />
          ) : (
            <div className="flex flex-1 flex-col items-center justify-center gap-1 bg-transparent p-8 text-center [&_svg]:text-icon-2">
              <BookOpen size={24} />
              <p className="text-base text-fg">No series selected.</p>
              <p className="max-w-[42ch] text-fg-2">Search a source, then choose a series to see its chapters.</p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

function SystemGroup({
  entry,
  sources: list,
  selected,
  open,
  onToggle,
  onSelect,
}: {
  entry: SystemSources;
  sources: SourceInfo[];
  selected?: Selected;
  open: boolean;
  onToggle: () => void;
  onSelect: (source: SourceInfo) => void;
}) {
  const ws = useWorkspace();
  const { system } = entry;
  return (
    <div className="px-1 pt-1 pb-3">
      <div className="flex h-7 items-center justify-between px-1 text-xs text-fg-3">
        <button className="inline-flex cursor-pointer items-center gap-1 p-0 focus-visible:rounded-pill focus-visible:shadow-focus focus-visible:outline-none" aria-expanded={open} onClick={onToggle}>
          <ChevronRight size={12} className={cn("transition-[rotate] duration-150 ease-settle", open && "rotate-90")} />
          {system.name}
        </button>
        {!system.enabled ? (
          <Badge>Off</Badge>
        ) : system.state === "starting" ? (
          <Badge tone="info">Starting</Badge>
        ) : entry.error ? (
          <Badge tone="danger">Error</Badge>
        ) : (
          <span className="tabular-nums" title={`${entry.sources.length} sources`}>
            {sitesOf(entry.sources).length}
          </span>
        )}
      </div>
      {!system.enabled && (
        <p className="px-1 pt-0.5 pb-1 text-xs text-fg-2">
          Turned off. <button className="cursor-pointer p-0 text-brand" onClick={() => ws.setSettingsOpen(true, "sources")}>Turn on in Settings</button>
        </p>
      )}
      {entry.error && <p className="px-1 pt-0.5 pb-1 text-xs text-danger">{entry.error}</p>}
      {system.enabled && system.state === "starting" && <p className="px-1 pt-0.5 pb-1 text-xs text-fg-2">Starting. Its sources appear in a moment.</p>}
      {system.enabled && system.state !== "starting" && !entry.error && entry.sources.length === 0 && (
        <p className="px-1 pt-0.5 pb-1 text-xs text-fg-2">
          No sources installed. <button className="cursor-pointer p-0 text-brand" onClick={() => ws.setSettingsOpen(true, "sources")}>Install sources</button>
        </p>
      )}
      {open &&
        sitesOf(list).map((site) => {
          const languages = site.sources.length;
          const source = languages > 1 ? siteSource(site) : site.sources[0];
          const active = selected?.system === system.id && site.sources.some((s) => s.id === selected.source.id);
          return (
            <button
              key={site.key}
              className={cn(
                "flex min-h-8 w-full cursor-pointer items-center gap-2 rounded-control px-1 text-left text-fg transition-colors duration-[20ms] ease-in focus-visible:shadow-focus focus-visible:outline-none",
                active ? "bg-selected" : "hover:bg-hover",
              )}
              aria-current={active || undefined}
              onClick={() => onSelect(source)}
            >
              <SourceIcon url={source.icon} />
              <span className="min-w-0 flex-1 truncate">{site.name}</span>
              {source.nsfw && <Badge tone="warning">18+</Badge>}
              <span className="text-xs leading-5 tracking-[0.005em] text-fg-3">{languages > 1 ? `${languages} languages` : source.language.toUpperCase()}</span>
            </button>
          );
        })}
    </div>
  );
}

export function SourceIcon({ url, size }: { url: string | null; size?: number }) {
  const [broken, setBroken] = useState(false);
  // A list row on a phone takes a larger, rounder tile.
  const style = size ? { width: size, height: size, borderRadius: 7 } : undefined;
  if (!url || broken) return <span className="size-5 flex-none rounded-pill bg-surface-fill object-cover" style={style} aria-hidden />;
  return <img className="size-5 flex-none rounded-pill bg-surface-fill object-cover" style={style} src={url} alt="" loading="lazy" onError={() => setBroken(true)} />;
}

/** A site check stopped the request: open the browser on it, then try again. */
function CheckNotice({ challenge, onRetry }: { challenge: Challenge; onRetry: () => void }) {
  const ws = useWorkspace();
  const host = new URL(challenge.url).host;
  return (
    <div className="px-3 py-8 text-center">
      <ShieldCheck size={20} className="text-fg-3" />
      <p className="text-fg">{host} wants a quick check.</p>
      <p className="mt-0.5 text-xs leading-5 text-fg-2">The site checks that a person is visiting. Complete it in the browser and the search continues.</p>
      <Button size="sm" className="mt-3" onClick={() => ws.openBrowser(challenge, onRetry)}>
        Open browser
      </Button>
    </div>
  );
}

function SeriesSearch({
  selected,
  languages,
  onLanguage,
  active,
  onSelect,
}: {
  selected: Selected;
  /** The other languages of the selected site; a picker shows when there is more than one. */
  languages: SourceInfo[];
  onLanguage: (source: SourceInfo) => void;
  active?: string;
  onSelect: (series: Series) => void;
}) {
  const [query, setQuery] = useState("");
  const [challenge, setChallenge] = useState<Challenge>();
  const [results, setResults] = useState<Series[]>([]);
  const [page, setPage] = useState(1);
  const [more, setMore] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();

  const load = async (text: string, number: number, cancelled: () => boolean = () => false) => {
    const key = `${selected.system}/${selected.source.id}/${text}`;
    const recent = number === 1 ? recentResults.get(key) : undefined;
    if (recent) {
      // Shown at once; a fresh one is kept as is, an older one is refreshed below.
      setResults(recent.page.results);
      setMore(recent.page.has_more);
      setPage(1);
      setError(undefined);
      setChallenge(undefined);
      if (Date.now() - recent.at < RESULTS_FRESH_MS) return;
    }
    setBusy(true);
    setError(undefined);
    setChallenge(undefined);
    try {
      const found = await sources.search(selected.system, selected.source.id, text, number);
      if (number === 1) recentResults.set(key, { at: Date.now(), page: found });
      if (cancelled()) return;
      setResults((current) => (number === 1 ? found.results : [...current, ...found.results.filter((r) => !current.some((c) => c.id === r.id))]));
      setMore(found.has_more);
      setPage(number);
    } catch (e) {
      if (cancelled()) return;
      setError(describeError(e));
      setChallenge(challengeOf(e));
    } finally {
      if (!cancelled()) setBusy(false);
    }
  };

  // Typing waits SEARCH_DELAY for a pause; opening the source (nothing typed yet) loads at once.
  const typed = useRef(false);
  useEffect(() => {
    let cancelled = false;
    const remembered = recentResults.has(`${selected.system}/${selected.source.id}/${query.trim()}`);
    const delay = typed.current && !remembered ? SEARCH_DELAY : 0;
    typed.current = true;
    const timer = window.setTimeout(() => void load(query.trim(), 1, () => cancelled), delay);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query]);

  return (
    <section className="flex min-h-0 min-w-0 flex-col" aria-label={`Search ${selected.source.name}`}>
      <div className="flex items-center gap-3 py-3 px-4 shadow-[inset_0_-1px_0_var(--border-soft)] max-md:flex-wrap max-md:gap-2 max-md:shadow-none">
        <label className="relative block max-w-[480px] flex-1 max-md:max-w-none max-md:basis-full">
          <Search size={14} className="pointer-events-none absolute top-1/2 left-2.5 -translate-y-1/2 text-icon-2" />
          <Input
            autoFocus
            aria-label={`Search ${selected.source.name}`}
            placeholder={`Search ${selected.source.name}`}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="pl-[30px]"
          />
        </label>
        {languages.length > 1 && (
          <Select
            size="sm"
            className="w-[180px]"
            label={`${siteName(selected.source)} language`}
            value={selected.source.id}
            options={[...languages]
              .sort((a, b) => languageName(a.language).localeCompare(languageName(b.language)))
              .map((s) => ({ value: s.id, label: languageName(s.language), detail: s.language.toUpperCase() }))}
            onChange={(id) => {
              const source = languages.find((s) => s.id === id);
              if (source) onLanguage(source);
            }}
          />
        )}
        <span className="text-xs tracking-[0.005em] text-fg-2">{query.trim() ? "Search results" : "Popular"}</span>
      </div>
      <ScrollArea className="flex-1" contentClassName="relative">
        {busy && page === 1 && (
          <div className="absolute inset-x-0 top-0 h-0.5 w-auto overflow-hidden rounded-none bg-surface-fill" aria-label="Searching">
            <span className="absolute inset-y-0 left-0 w-[30%]! rounded-[inherit] bg-brand animate-[slide_1.1s_ease-in-out_infinite]" />
          </div>
        )}
        {challenge && <CheckNotice challenge={challenge} onRetry={() => void load(query.trim(), 1)} />}
        {error && !challenge && (
          <div className="px-3 py-8 text-center">
            <p className="text-fg">The source did not answer.</p>
            <p className="mt-0.5 text-xs leading-5 text-fg-2">{error}</p>
            <Button variant="outline" size="sm" className="mt-3" onClick={() => void load(query.trim(), 1)}>
              Try again
            </Button>
          </div>
        )}
        {!error && !busy && results.length === 0 && (
          <div className="px-3 py-8 text-center">
            <p className="text-fg">No series found.</p>
            <p className="mt-0.5 text-xs leading-5 text-fg-2">Try another title, or the title in the source's language.</p>
          </div>
        )}
        <div className="grid grid-cols-[repeat(auto-fill,minmax(128px,1fr))] gap-x-3 gap-y-4 p-4 max-md:grid-cols-3 max-md:gap-x-2.5">
          {results.map((series) => (
            <button
              key={series.id}
              className={cn(
                "flex min-w-0 cursor-pointer flex-col gap-1 rounded-card p-1 text-left text-fg transition-colors duration-[20ms] ease-in focus-visible:shadow-focus focus-visible:outline-none",
                active === series.id ? "bg-selected" : "hover:bg-hover",
              )}
              aria-current={active === series.id || undefined}
              onClick={() => onSelect(series)}
            >
              <Cover system={selected.system} source={selected.source.id} url={series.cover} />
              <span className="line-clamp-2 text-xs">{series.title}</span>
            </button>
          ))}
        </div>
        {more && !error && (
          <div className="flex justify-center px-4 pb-8">
            <Button variant="outline" size="sm" disabled={busy} onClick={() => void load(query.trim(), page + 1)}>
              {busy ? "Loading" : "Load more"}
            </Button>
          </div>
        )}
      </ScrollArea>
    </section>
  );
}

function Cover({ system, source, url, large }: { system: string; source: string; url: string | null; large?: boolean }) {
  const [broken, setBroken] = useState(false);
  return (
    <span
      className={cn(
        "block aspect-[2/3] overflow-hidden bg-surface-fill shadow-[0_0_0_1px_var(--border-soft)]",
        large ? "w-[160px] rounded-card" : "w-full rounded-control",
      )}
    >
      {url && !broken && <img className="block size-full object-cover" src={sources.coverUrl(system, source, url)} alt="" loading="lazy" onError={() => setBroken(true)} />}
    </span>
  );
}

function SeriesDetail({ selected, series: summary }: { selected: Selected; series: Series }) {
  const ws = useWorkspace();
  const [series, setSeries] = useState<Series>(summary);
  const [chapters, setChapters] = useState<RemoteChapter[]>();
  const [error, setError] = useState<string>();
  const [challenge, setChallenge] = useState<Challenge>();
  const [linking, setLinking] = useState(false);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let alive = true;
    setError(undefined);
    setChallenge(undefined);
    sources.series(selected.system, selected.source.id, summary.id).then(
      (found) => alive && setSeries(found),
      () => undefined,
    );
    sources.chapters(selected.system, selected.source.id, summary.id).then(
      (found) => alive && setChapters(found),
      (e) => {
        if (!alive) return;
        setError(describeError(e));
        setChallenge(challengeOf(e));
      },
    );
    return () => {
      alive = false;
    };
  }, [selected.system, selected.source.id, summary.id, attempt]);

  const link = async () => {
    setLinking(true);
    await ws.linkSeries(selected.system, selected.source.id, series.id);
    setLinking(false);
  };

  const replacing = ws.link && ws.project ? `${ws.project.name} follows ${ws.link.series.title} now. Linking replaces it; downloaded chapters stay.` : null;

  return (
    <ScrollArea className="flex-1">
      <div className="flex flex-col gap-2 px-5 pt-6 pb-12">
        <Cover system={selected.system} source={selected.source.id} url={series.cover ?? summary.cover} large />
        <h2 className="mt-2 text-xl font-semibold [overflow-wrap:anywhere]">{series.title}</h2>
        {series.authors.length > 0 && <p className="text-fg-2">{series.authors.join(", ")}</p>}
        <div className="flex flex-wrap items-center gap-1">
          <Badge>{selected.source.name}</Badge>
          {series.status && <Badge>{series.status}</Badge>}
          <span className="text-xs tracking-[0.005em] text-fg-2 tabular-nums">
            {chapters ? `${chapters.length} chapter${chapters.length === 1 ? "" : "s"}` : error ? "Chapters unavailable" : "Loading chapters"}
          </span>
        </div>
        {replacing && <p className="text-xs tracking-[0.005em] text-fg-2">{replacing}</p>}
        <div className="mt-1 flex gap-2">
          <Button disabled={linking || !chapters} onClick={() => void link()}>
            <Link2 size={14} /> {linking ? "Linking" : ws.project ? "Link series" : "Create project"}
          </Button>
          {series.url && (
            <a className={buttonVariants({ variant: "ghost" })} href={series.url} target="_blank" rel="noreferrer">
              <ExternalLink size={14} /> Open site
            </a>
          )}
        </div>
        {challenge ? (
          <p className="text-xs tracking-[0.005em] text-fg-2">
            {new URL(challenge.url).host} wants a quick check.{" "}
            <button className="cursor-pointer p-0 text-brand" onClick={() => ws.openBrowser(challenge, () => setAttempt((n) => n + 1))}>
              Open browser
            </button>
          </p>
        ) : (
          error && <p className="text-xs tracking-[0.005em] text-danger">{error}</p>
        )}
        {series.summary && <p className="whitespace-pre-line text-fg-2">{series.summary}</p>}
        {series.tags.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {series.tags.slice(0, 16).map((tag) => (
              <span key={tag} className="rounded-pill bg-surface-fill px-2 py-0.5 text-xs text-fg-2">
                {tag}
              </span>
            ))}
          </div>
        )}
        {chapters && chapters.length > 0 && (
          <div>
            <h3 className="mt-2 mb-1 text-xs leading-5 font-normal text-fg-3">Chapters</h3>
            <ol>
              {[...chapters].reverse().slice(0, 30).map((chapter) => (
                <li key={chapter.id} className="flex justify-between gap-2 py-1 shadow-[inset_0_-1px_0_var(--border-soft)]">
                  <span className="min-w-0 truncate">{chapter.name}</span>
                  {chapter.scanlator && <span className="truncate text-xs tracking-[0.005em] text-fg-3">{chapter.scanlator}</span>}
                </li>
              ))}
            </ol>
            {chapters.length > 30 && <p className="text-xs tracking-[0.005em] text-fg-3">And {chapters.length - 30} earlier chapters.</p>}
          </div>
        )}
      </div>
    </ScrollArea>
  );
}
