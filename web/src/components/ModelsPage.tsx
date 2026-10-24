// The Models page, after LM Studio's Discover: search Hugging Face per runtime, see how each
// quantization fits this machine, install one, and pick installed models for a stage.

import {
  ArrowLeft,
  Boxes,
  ChevronDown,
  ChevronUp,
  CircleAlert,
  CircleCheck,
  Cpu,
  Download as DownloadIcon,
  ExternalLink,
  Eye,
  Heart,
  Info,
  Lock,
  MemoryStick,
  Pause,
  Play,
  Search,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import {
  catalog,
  FIT_LABEL,
  FIT_TONE,
  formatBytes,
  formatCount,
  formatParameters,
  RUNTIME_LABEL,
  STAGE_LABEL,
  type CatalogRuntime,
  type Compatibility,
  type Download,
  type Fit,
  type Hardware,
  type InstalledModel,
  type ModelDetails,
  type SearchResult,
  type Usage,
  type Variant,
} from "../api/catalog";
import { describeError } from "../api/client";
import { formatDate, formatTimestamp } from "../format";
import { useWorkspace } from "../state/workspace";
import { ModelsFolder, StorageBar, useModelsStorage } from "./ModelsStorage";
import { Select, Switch, Tooltip } from "./primitives";
import { Badge, Segmented } from "./ui";
import { Button, buttonVariants } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "cn";
import { ScrollArea } from "@/components/ui/scroll-area";
import { TitleBarInset } from "./WindowControls";

const RUNTIMES: { value: CatalogRuntime; label: string; title: string }[] = [
  { value: "llama", label: "LLM", title: "GGUF language and vision models for llama.cpp" },
  { value: "diffusion", label: "Diffusion", title: "Image models for stable-diffusion.cpp" },
  { value: "torch", label: "PyTorch", title: "Transformers and safetensors checkpoints" },
  { value: "onnx", label: "ONNX", title: "ONNX Runtime models" },
];

const SORTS = [
  { value: "downloads", label: "Most downloads" },
  { value: "likes", label: "Most likes" },
  { value: "trendingScore", label: "Trending" },
  { value: "lastModified", label: "Recently updated" },
];

const SEARCH_DELAY = 300;
// Remembered for the session, so reopening the page shows the last list and model at once while
// they refresh: search results per runtime, query, sort and filter, and model details.
const recentSearches = new Map<string, SearchResult[]>();
const recentDetails = new Map<string, ModelDetails>();
const SPINNER_DELAY = 250;

const PROGRESS_TRACK = "relative h-1 overflow-hidden rounded-full bg-surface-fill";
const PROGRESS_FILL = "absolute inset-y-0 left-0 rounded-[inherit] bg-brand transition-[width] duration-200 ease-settle";
const PROGRESS_SLIDE = "w-[30%]! animate-[slide_1.1s_ease-in-out_infinite]";

const CALLOUT = "mt-5 flex items-center gap-2 rounded-card px-3 py-2 [&>svg]:shrink-0";
const CALLOUT_TONE = {
  info: "bg-info-weak shadow-[inset_0_0_0_1px_var(--info-line)] [&>svg]:text-info",
  success: "bg-success-weak shadow-[inset_0_0_0_1px_var(--success-line)] [&>svg]:text-success",
  warning: "bg-warning-weak shadow-[inset_0_0_0_1px_var(--warning-line)] [&>svg]:text-warning",
  danger: "bg-danger-weak shadow-[inset_0_0_0_1px_var(--danger-line)] [&>svg]:text-danger",
} as const;

/** True once ``busy`` has lasted a moment, so quick loads never flash a spinner. */
function useDelayed(busy: boolean): boolean {
  const [shown, setShown] = useState(false);
  useEffect(() => {
    if (!busy) {
      setShown(false);
      return;
    }
    const timer = window.setTimeout(() => setShown(true), SPINNER_DELAY);
    return () => window.clearTimeout(timer);
  }, [busy]);
  return shown;
}

// ---- downloads and installed models, shared by both tabs ----------------------------------------

function useCatalogState() {
  const ws = useWorkspace();
  const [downloads, setDownloads] = useState<Download[]>([]);
  const [installed, setInstalled] = useState<InstalledModel[]>([]);
  const [loaded, setLoaded] = useState(false);
  const finished = useRef(new Set<string>());

  const loadInstalled = useCallback(async () => {
    try {
      setInstalled(await catalog.installed());
      setLoaded(true);
    } catch (error) {
      ws.reportError(describeError(error));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const poll = useCallback(async () => {
    try {
      const list = await catalog.downloads();
      setDownloads(list);
      const done = list.filter((d) => d.status === "completed" && !finished.current.has(d.id));
      if (done.length) {
        done.forEach((d) => finished.current.add(d.id));
        void loadInstalled();
        void ws.refreshModels();
      }
      return list.some((d) => d.status === "queued" || d.status === "downloading");
    } catch {
      return false;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadInstalled]);

  const [polling, setPolling] = useState(true);
  useEffect(() => {
    if (!polling) return;
    let stop = false;
    let timer = 0;
    const tick = async () => {
      const active = await poll();
      if (stop) return;
      if (active) timer = window.setTimeout(tick, 700);
      else setPolling(false);
    };
    void tick();
    return () => {
      stop = true;
      window.clearTimeout(timer);
    };
  }, [polling, poll]);

  useEffect(() => {
    void loadInstalled();
  }, [loadInstalled]);

  return {
    downloads,
    installed,
    loaded,
    reloadInstalled: loadInstalled,
    install: async (runtime: CatalogRuntime, repo: string, variant: string) => {
      try {
        await catalog.install(runtime, repo, variant);
        setPolling(true);
        void poll();
      } catch (error) {
        ws.reportError(describeError(error));
      }
    },
    cancel: async (id: string) => {
      await catalog.cancel(id).catch((error) => ws.reportError(describeError(error)));
      void poll();
    },
    pause: async (id: string) => {
      await catalog.pause(id).catch((error) => ws.reportError(describeError(error)));
      setPolling(true);
      void poll();
    },
    resume: async (id: string) => {
      await catalog.resume(id).catch((error) => ws.reportError(describeError(error)));
      setPolling(true);
      void poll();
    },
    move: async (id: string, position: number) => {
      try {
        setDownloads(await catalog.move(id, position));
      } catch (error) {
        ws.reportError(describeError(error));
      }
    },
  };
}

type CatalogState = ReturnType<typeof useCatalogState>;

// ---- page ---------------------------------------------------------------------------------------

export function ModelsPage() {
  const ws = useWorkspace();
  const [tab, setTab] = useState<"discover" | "installed">("discover");
  const [hardware, setHardware] = useState<Hardware>();
  const state = useCatalogState();

  useEffect(() => {
    catalog.hardware().then(setHardware, () => undefined);
  }, []);

  const active = state.downloads.filter((d) => d.status === "queued" || d.status === "downloading" || d.status === "paused");

  return (
    <div className="flex h-screen flex-col bg-surface">
      <header className="flex h-12 flex-none items-center gap-3 pr-4 pl-2 shadow-[inset_0_-1px_0_var(--border-soft)]" data-titlebar>
        <TitleBarInset side="left" />
        <Button variant="ghost" size="sm" onClick={() => ws.setModelsOpen(false)}>
          <ArrowLeft size={14} /> {ws.project ? "Back to editor" : "Back to projects"}
        </Button>
        <span className="inline-flex items-center gap-2 font-semibold">
          <Boxes size={16} className="text-icon-2" /> Models
        </span>
        <Segmented
          label="Models view"
          value={tab}
          onChange={setTab}
          options={[
            { value: "discover", label: "Discover" },
            { value: "installed", label: `Installed${state.installed.length ? ` ${state.installed.length}` : ""}` },
          ]}
        />
        <span className="flex-1" />
        {hardware && <HardwareSummary hardware={hardware} />}
        <Button variant="ghost" size="sm" onClick={() => ws.setSettingsOpen(true, "runtimes")}>
          <Cpu size={14} /> Runtimes
        </Button>
        <TitleBarInset side="right" />
      </header>
      {tab === "discover" ? <Discover state={state} /> : <Installed state={state} onDiscover={() => setTab("discover")} />}
      {active.length > 0 && <DownloadTray downloads={active} state={state} />}
    </div>
  );
}

const BACKENDS: Record<string, string> = { cuda: "CUDA", rocm: "ROCm", vulkan: "Vulkan", dml: "DirectML", metal: "Metal" };

/** "12 GB VRAM", or "16 GB shared" for an integrated GPU, which works in system memory. */
function gpuMemory(gpu: Hardware["devices"][number]) {
  return `${formatBytes(gpu.memory_total)} ${gpu.device_type === "integrated_gpu" ? "shared" : "VRAM"}`;
}

function HardwareSummary({ hardware }: { hardware: Hardware }) {
  const gpu = hardware.devices[0];
  const system = hardware.system;
  const backend = gpu && [BACKENDS[gpu.backend] ?? gpu.backend, gpu.target].filter(Boolean).join(" ");
  const detail = [
    gpu && `${gpu.name}, ${gpuMemory(gpu)}, ${backend}${gpu.driver_version ? `, driver ${gpu.driver_version}` : ""}`,
    `${formatBytes(system.ram_total)} RAM (${formatBytes(system.ram_available)} free)`,
    `${system.cpu_name || "CPU"}, ${system.cpu_threads} threads${system.cpu_features.length ? `, ${system.cpu_features.join(" ").toUpperCase()}` : ""}`,
  ]
    .filter(Boolean)
    .join("\n");
  return (
    <Tooltip content={<span className="whitespace-pre-line">{detail}</span>} side="bottom">
      <span
        className="inline-flex h-6 items-center gap-2 rounded-control bg-surface-sunk px-2 text-[length:var(--t-12)] tabular-nums text-fg-2 shadow-[inset_0_0_0_1px_var(--border-soft)] outline-none focus-visible:shadow-focus max-[900px]:hidden"
        tabIndex={0}
      >
        <MemoryStick size={14} className="text-icon-2" />
        {gpu ? gpuMemory(gpu) : "No GPU"}
        <span className="h-3 w-px bg-line" />
        {formatBytes(system.ram_total)} RAM
      </span>
    </Tooltip>
  );
}

// ---- discover -----------------------------------------------------------------------------------

function Discover({ state }: { state: CatalogState }) {
  const [runtime, setRuntime] = useState<CatalogRuntime>("llama");
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState("downloads");
  const [usable, setUsable] = useState(true);
  const [results, setResults] = useState<SearchResult[]>();
  const [error, setError] = useState<string>();
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState<string>();
  const spinner = useDelayed(busy);

  // Typing waits SEARCH_DELAY for a pause; the first load and a remembered list do not.
  const typed = useRef(false);
  useEffect(() => {
    let cancelled = false;
    const key = `${runtime}\n${query.trim()}\n${sort}\n${usable}`;
    const recent = recentSearches.get(key);
    if (recent) {
      setResults(recent);
      setSelected((current) => (current && recent.some((r) => r.repo === current) ? current : recent[0]?.repo));
    }
    const delay = typed.current && !recent ? SEARCH_DELAY : 0;
    typed.current = true;
    const timer = window.setTimeout(async () => {
      setBusy(!recent);
      setError(undefined);
      try {
        const found = await catalog.search(runtime, query.trim(), sort, usable);
        recentSearches.set(key, found);
        if (cancelled) return;
        setResults(found);
        setSelected((current) => (current && found.some((r) => r.repo === current) ? current : found[0]?.repo));
      } catch (e) {
        if (!cancelled && !recent) setError(describeError(e));
      } finally {
        if (!cancelled) setBusy(false);
      }
    }, delay);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [runtime, query, sort, usable]);

  return (
    <div className="grid min-h-0 flex-1 grid-cols-[380px_minmax(0,1fr)] max-[900px]:grid-cols-1 max-[900px]:grid-rows-[minmax(240px,40vh)_minmax(0,1fr)]">
      <aside
        className="flex min-h-0 flex-col bg-surface-sunk shadow-[inset_-1px_0_0_var(--border-soft)] max-[900px]:shadow-[inset_0_-1px_0_var(--border-soft)]"
        aria-label="Search results"
      >
        <div className="flex flex-col gap-2 p-3 shadow-[inset_0_-1px_0_var(--border-soft)]">
          <label className="relative block">
            <Search size={14} className="pointer-events-none absolute top-1/2 left-[10px] -translate-y-1/2 text-icon-2" />
            <Input
              className="w-full pl-[30px]"
              aria-label="Search Hugging Face"
              placeholder="Search Hugging Face"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </label>
          <Segmented label="Runtime" fill value={runtime} onChange={setRuntime} options={RUNTIMES} />
          <div className="flex items-center justify-between gap-2">
            <Select size="sm" label="Sort" value={sort} options={SORTS} onChange={setSort} />
            <label className="inline-flex items-center gap-2 text-fg-2">
              <Switch label="Only models Urakatsu can use" checked={usable} onChange={setUsable} />
              <span className="text-xs tracking-[0.005em]">Usable only</span>
            </label>
          </div>
        </div>
        <ScrollArea className="flex-1" contentClassName="relative px-2 pt-1 pb-8">
          {spinner && (
            <div className="absolute inset-x-0 top-0 h-0.5 w-auto overflow-hidden rounded-none bg-surface-fill" aria-label="Searching">
              <span className={cn(PROGRESS_FILL, PROGRESS_SLIDE)} />
            </div>
          )}
          {error && (
            <div className="px-3 py-8 text-center">
              <p className="text-fg">Search failed.</p>
              <p className="mt-0.5 text-[length:var(--t-12)] text-fg-2">{error}</p>
            </div>
          )}
          {!error && results && results.length === 0 && !busy && (
            <div className="px-3 py-8 text-center">
              <p className="text-fg">No matching models.</p>
              <p className="mt-0.5 text-[length:var(--t-12)] text-fg-2">{usable ? "Turn off Usable only to see every repository." : "Try another search."}</p>
            </div>
          )}
          {results?.map((result) => (
            <ResultRow key={result.repo} result={result} active={result.repo === selected} onSelect={() => setSelected(result.repo)} />
          ))}
        </ScrollArea>
      </aside>
      <section className="flex min-h-0 min-w-0 flex-col">
        {selected ? <Details key={`${runtime}:${selected}`} runtime={runtime} repo={selected} state={state} onOpen={setSelected} /> : <EmptyDetail />}
      </section>
    </div>
  );
}

function ResultRow({ result, active, onSelect }: { result: SearchResult; active: boolean; onSelect: () => void }) {
  const [author, ...rest] = result.repo.split("/");
  const params = formatParameters(result.parameters);
  return (
    <button
      className={cn(
        "flex w-full cursor-pointer flex-col gap-1 rounded-control p-2 text-left text-fg transition-colors duration-[20ms] ease-in focus-visible:shadow-focus focus-visible:outline-none",
        active ? "bg-selected" : "bg-transparent hover:bg-hover",
      )}
      aria-current={active || undefined}
      onClick={onSelect}
    >
      <span className="max-w-full truncate font-medium">
        <span className="text-fg-3">{author}/</span>
        <span>{rest.join("/")}</span>
      </span>
      <span className="flex items-center gap-3 text-xs text-fg-2 [&>span]:inline-flex [&>span]:items-center [&>span]:gap-1 [&_svg]:text-icon-2">
        <span className="tabular-nums">
          <DownloadIcon size={12} /> {formatCount(result.downloads)}
        </span>
        <span className="tabular-nums">
          <Heart size={12} /> {formatCount(result.likes)}
        </span>
        {params && <span className="tabular-nums">{params}</span>}
        {(result.architecture || result.family || result.model_type) && <span>{result.architecture || result.family || result.model_type}</span>}
        {result.gated && <Lock size={12} aria-label="Gated" />}
      </span>
      <span className="flex flex-wrap gap-1 empty:hidden">
        {result.installed && <Badge tone="success">Installed</Badge>}
        {uniqueStages(result.usable_by).map((stage) => (
          <Badge key={stage}>{STAGE_LABEL[stage] ?? stage}</Badge>
        ))}
        {result.estimate && (
          <Badge tone={FIT_TONE[result.estimate.level]} title="Estimate for a typical 4-bit quantization">
            {FIT_LABEL[result.estimate.level]}
          </Badge>
        )}
      </span>
    </button>
  );
}

function uniqueStages(usage: Usage[]): string[] {
  return [...new Set(usage.map((u) => u.stage))];
}

function EmptyDetail() {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-1 bg-canvas p-8 text-center">
      <p className="text-base text-fg">No model selected.</p>
      <p className="max-w-[42ch] text-fg-2">Pick a result to see its files and how they fit this machine.</p>
    </div>
  );
}

// ---- details ------------------------------------------------------------------------------------

function Details({ runtime, repo, state, onOpen }: { runtime: CatalogRuntime; repo: string; state: CatalogState; onOpen: (repo: string) => void }) {
  const ws = useWorkspace();
  const [details, setDetails] = useState<ModelDetails | undefined>(() => recentDetails.get(`${runtime}\n${repo}`));
  const [error, setError] = useState<string>();
  const [readme, setReadme] = useState(false);
  const spinner = useDelayed(!details && !error);

  useEffect(() => {
    let cancelled = false;
    const key = `${runtime}\n${repo}`;
    const recent = recentDetails.get(key);
    setDetails(recent);
    catalog.details(runtime, repo).then(
      (d) => {
        recentDetails.set(key, d);
        if (!cancelled) setDetails(d);
      },
      (e) => !cancelled && !recent && setError(describeError(e)),
    );
    return () => {
      cancelled = true;
    };
  }, [runtime, repo]);

  const installedIds = useMemo(() => new Set(state.installed.filter((m) => m.repo === repo).map((m) => m.variant || "snapshot")), [state.installed, repo]);

  if (error) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-1 bg-canvas p-8 text-center">
        <CircleAlert size={20} className="text-danger" />
        <p className="text-base text-fg">Could not load {repo}.</p>
        <p className="max-w-[42ch] text-fg-2">{error}</p>
      </div>
    );
  }
  if (!details) {
    return (
      <div className="flex flex-1 items-center justify-center">
        {spinner && (
          <div className={cn(PROGRESS_TRACK, "w-40")} aria-label="Loading model">
            <span className={cn(PROGRESS_FILL, PROGRESS_SLIDE)} />
          </div>
        )}
      </div>
    );
  }

  const [author, ...rest] = details.repo.split("/");
  const params = formatParameters(details.parameters);
  const gpu = details.hardware.devices[0];
  const blocked = details.compatibility.status === "unsupported" || details.usable_by.length === 0 || details.missing.length > 0;
  const recommended = details.variants.find((v) => v.recommended);

  const action = (variant: Variant) => {
    const download = state.downloads.find(
      (d) =>
        d.model.repo === details.repo &&
        (d.model.variant || "snapshot") === variant.id &&
        (d.status === "queued" || d.status === "downloading" || d.status === "paused"),
    );
    if (installedIds.has(variant.id)) {
      return (
        <Badge tone="success">
          <CircleCheck size={12} /> Installed
        </Badge>
      );
    }
    if (download)
      return (
        <span className="inline-flex items-center gap-2">
          <DownloadProgress download={download} className="w-24" />
          <QueueControls download={download} state={state} compact />
        </span>
      );
    return (
      <Button
        size="sm"
        variant={variant.recommended && !blocked ? "default" : "outline"}
        disabled={blocked || variant.fit.level === "too_large"}
        onClick={() => state.install(runtime, details.repo, variant.id)}
      >
        <DownloadIcon size={14} /> Download
      </Button>
    );
  };

  return (
    <ScrollArea className="flex-1">
      <div className="mx-auto w-full max-w-[880px] px-8 pt-8 pb-16 max-[900px]:px-4 max-[900px]:pt-5 max-[900px]:pb-12">
        <header>
          <p className="text-xs tracking-[0.005em] text-fg-3">{author}</p>
          <h2 className="text-2xl font-semibold tracking-[-0.01em] [overflow-wrap:anywhere]">{rest.join("/")}</h2>
          <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-fg-2 [&>span]:inline-flex [&>span]:items-center [&>span]:gap-1 [&_svg]:text-icon-2">
            <span className="tabular-nums">
              <DownloadIcon size={14} /> {formatCount(details.downloads)}
            </span>
            <span className="tabular-nums">
              <Heart size={14} /> {formatCount(details.likes)}
            </span>
            {details.updated && <span>Updated {formatDate(details.updated)}</span>}
            {params && <span className="tabular-nums">{params} parameters</span>}
            {details.context_length && <span className="tabular-nums">{formatCount(details.context_length)} context</span>}
            <a className={buttonVariants({ variant: "ghost", size: "sm" })} href={`https://huggingface.co/${details.repo}`} target="_blank" rel="noreferrer">
              <ExternalLink size={14} /> Hugging Face
            </a>
          </div>
          <div className="mt-3 flex flex-wrap gap-1">
            <Badge>{RUNTIME_LABEL[runtime]}</Badge>
            {(details.architecture || details.family || details.model_type) && <Badge>{details.architecture || details.family || details.model_type}</Badge>}
            {details.vision && (
              <Badge tone="info">
                <Eye size={12} /> Vision
              </Badge>
            )}
            {details.reasoning && (
              <Badge tone="info">
                <Sparkles size={12} /> Reasoning
              </Badge>
            )}
            {details.gated && (
              <Badge tone="warning">
                <Lock size={12} /> Gated
              </Badge>
            )}
          </div>
        </header>

        <CompatibilityNote compatibility={details.compatibility} onRuntimes={() => ws.setSettingsOpen(true, "runtimes")} />

        <section className="mt-8">
          <h3 className="mb-2 text-sm font-semibold">Works with</h3>
          {details.usable_by.length ? (
            <ul className="flex flex-col rounded-card px-3 shadow-raised">
              {details.usable_by.map((u) => (
                <li
                  key={`${u.stage}:${u.provider}`}
                  className="flex items-center justify-between gap-3 py-2 shadow-[inset_0_-1px_0_var(--border-soft)] last:shadow-none"
                >
                  <span className="font-medium">{STAGE_LABEL[u.stage] ?? u.stage}</span>
                  <span className="text-fg-2">{u.name}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-fg-2">No pipeline stage in Urakatsu can load this model. You can still browse its files.</p>
          )}
        </section>

        <section className="mt-8">
          <div className="mb-2 flex items-baseline justify-between gap-3">
            <h3 className="text-sm font-semibold">Files</h3>
            <p className="text-xs tracking-[0.005em] text-fg-2">
              Rated for {gpu ? `${gpu.name} (${formatBytes(details.hardware.usable_vram)} usable)` : "CPU only"} and {formatBytes(details.hardware.system.ram_available)} free RAM.
            </p>
          </div>
          {recommended && details.variants.length > 1 && (
            <div className="mb-3 flex items-center gap-2 rounded-card bg-success-weak px-3 py-2 shadow-[inset_0_0_0_1px_var(--success-line)]">
              <Sparkles size={16} className="shrink-0 text-success" />
              <p className="flex flex-1 flex-wrap items-baseline gap-x-2 gap-y-1">
                Best fit for this machine: <span className="font-mono text-[length:var(--t-12)] font-medium">{recommended.label}</span>
                <span className="tabular-nums text-fg-2">{formatBytes(recommended.size)}</span>
              </p>
              <FitCell fit={recommended.fit} />
              {action(recommended)}
            </div>
          )}
          {details.components.length > 0 && (
            <div className="mb-3">
              <p className="mb-1 text-xs tracking-[0.005em] text-fg-2">
                Every option below also downloads what {details.recipe ?? "the model"} needs{details.edits ? " (it edits images from an instruction)" : ""}:
              </p>
              <ul className="flex flex-col rounded-card px-3 shadow-raised">
                {details.components.map((c) => (
                  <li
                    key={c.slot}
                    className="flex items-center justify-between gap-3 py-2 shadow-[inset_0_-1px_0_var(--border-soft)] last:shadow-none"
                  >
                    <span className="font-medium">{c.label}</span>
                    <span className="min-w-0 truncate text-right text-xs tracking-[0.005em] text-fg-2">
                      {!c.here && <span>{c.repo} </span>}
                      <span className="font-mono">{c.path.split("/").pop()}</span> <span className="tabular-nums">{formatBytes(c.size)}</span>
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {details.missing.length > 0 && (
            <div className={cn(CALLOUT, CALLOUT_TONE.danger)} role="alert">
              <CircleAlert size={16} />
              <p className="flex-1 text-fg">No file was found for: {details.missing.join(", ")}. This model cannot be installed yet.</p>
            </div>
          )}
          {details.variants.length === 0 ? (
            details.alternatives.length > 0 ? (
              <div>
                <p className="mb-2 text-fg-2">stable-diffusion.cpp cannot load this repository's files as they are. These copies of the model work:</p>
                <div className="flex flex-wrap gap-1">
                  {details.alternatives.map((alt) => (
                    <Button key={alt} variant="outline" size="sm" onClick={() => onOpen(alt)}>
                      {alt}
                    </Button>
                  ))}
                </div>
              </div>
            ) : (
              <p className="text-fg-2">This repository has no files Urakatsu can install for {RUNTIME_LABEL[runtime]}.</p>
            )
          ) : (
            <Table className="overflow-hidden rounded-card shadow-raised">
              <TableHeader>
                <TableRow>
                  <TableHead className="px-3 whitespace-normal">
                    {runtime === "llama" || runtime === "diffusion" || details.variants.some((v) => v.id !== v.label) ? "Variant" : "File"}
                  </TableHead>
                  <TableHead className="px-3 text-right whitespace-nowrap">Size</TableHead>
                  <TableHead className="px-3 whitespace-normal">On this machine</TableHead>
                  <TableHead aria-label="Action" className="px-3" />
                </TableRow>
              </TableHeader>
              <TableBody className="[&>tr:last-child>td]:shadow-none">
                {details.variants.map((variant) => {
                  return (
                    <TableRow
                      key={variant.id}
                      className={cn(
                        "[tbody_&]:hover:bg-transparent",
                        variant.recommended && "[&>td]:bg-[color-mix(in_oklab,var(--success-weak)_55%,transparent)]",
                      )}
                    >
                      <TableCell className="h-auto px-3 py-2">
                        <span className="mr-2 font-mono text-[length:var(--t-12)]" title={variant.files.join("\n")}>
                          {variant.label}
                        </span>
                        {variant.recommended && <Badge tone="success">Recommended</Badge>}
                      </TableCell>
                      <TableCell className="h-auto px-3 py-2 text-right whitespace-nowrap tabular-nums">{formatBytes(variant.size)}</TableCell>
                      <TableCell className="h-auto px-3 py-2">
                        <FitCell fit={variant.fit} />
                      </TableCell>
                      <TableCell className="h-auto w-[1%] px-3 py-2 text-right whitespace-nowrap">{action(variant)}</TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          )}
        </section>

        {details.readme && (
          <section className="mt-8">
            <div className="mb-2 flex items-baseline justify-between gap-3">
              <h3 className="text-sm font-semibold">Model card</h3>
              <Button variant="ghost" size="sm" onClick={() => setReadme((r) => !r)}>
                {readme ? "Hide" : "Show"}
              </Button>
            </div>
            {readme && (
              <pre className="max-h-[520px] overflow-auto rounded-card bg-surface-sunk px-4 py-3 font-mono text-[length:var(--t-12)] leading-[1.6] whitespace-pre-wrap text-fg-2 shadow-[inset_0_0_0_1px_var(--border-soft)] [overflow-wrap:anywhere]">
                {details.readme}
              </pre>
            )}
          </section>
        )}
      </div>
    </ScrollArea>
  );
}

function FitCell({ fit }: { fit: Fit }) {
  const share = fit.level === "partial" ? ` ${Math.round(fit.gpu_share * 100)}% GPU` : "";
  const detail = [
    `Needs about ${formatBytes(fit.required)}${fit.context ? `, including ${formatBytes(fit.context)} of context cache` : ""}.`,
    ...fit.notes,
  ].join(" ");
  return (
    <Tooltip content={detail} side="top">
      <span className="inline-flex outline-none focus-visible:rounded-control focus-visible:shadow-focus" tabIndex={0}>
        <Badge tone={FIT_TONE[fit.level]}>
          {FIT_LABEL[fit.level]}
          {share}
        </Badge>
      </span>
    </Tooltip>
  );
}

function CompatibilityNote({ compatibility, onRuntimes }: { compatibility: Compatibility; onRuntimes: () => void }) {
  if (!compatibility.message) return null;
  const tone = compatibility.status === "supported" ? "success" : compatibility.status === "needs_update" ? "warning" : compatibility.status === "unsupported" ? "danger" : "info";
  const Icon = tone === "success" ? CircleCheck : tone === "info" ? Info : CircleAlert;
  return (
    <div className={cn(CALLOUT, CALLOUT_TONE[tone])} role={tone === "warning" || tone === "danger" ? "alert" : "status"}>
      <Icon size={16} />
      <p className="flex-1 text-fg">{compatibility.message}</p>
      {compatibility.status === "needs_update" && (
        <Button variant="outline" size="sm" onClick={onRuntimes}>
          Runtime settings
        </Button>
      )}
    </div>
  );
}

/** Where a download stands: its share while running, its place in the queue, or paused. */
function downloadLabel(download: Download): string {
  if (download.status === "paused") return "Paused";
  if (download.status === "queued") return download.position ? `#${download.position + 1} in queue` : "Up next";
  return `${Math.round((download.total ? download.done / download.total : 0) * 100)}%`;
}

function DownloadProgress({ download, className }: { download: Download; className?: string }) {
  const share = download.total ? download.done / download.total : 0;
  const running = download.status === "downloading";
  return (
    <span className="inline-flex min-w-0 items-center gap-2">
      <span className={cn(PROGRESS_TRACK, className)} aria-label={`${downloadLabel(download)}, ${Math.round(share * 100)}% downloaded`}>
        <span className={cn(PROGRESS_FILL, !running && "bg-fg-3")} style={{ width: `${share * 100}%` }} />
      </span>
      <span className="text-xs tracking-[0.005em] whitespace-nowrap tabular-nums text-fg-2">{downloadLabel(download)}</span>
    </span>
  );
}

/** Pause or resume, move within the queue (not when ``compact``), and cancel. */
function QueueControls({ download, state, queued = 0, compact = false }: { download: Download; state: CatalogState; queued?: number; compact?: boolean }) {
  const name = `${download.model.repo.split("/").pop()} ${download.model.variant}`.trim();
  const position = download.position ?? 0;
  return (
    <span className="inline-flex flex-none items-center">
      {!compact && download.status === "queued" && (
        <>
          <Button variant="ghost" size="icon-xs" aria-label={`Move ${name} up`} title="Move up" disabled={position === 0} onClick={() => state.move(download.id, position - 1)}>
            <ChevronUp size={12} />
          </Button>
          <Button variant="ghost" size="icon-xs" aria-label={`Move ${name} down`} title="Move down" disabled={position >= queued - 1} onClick={() => state.move(download.id, position + 1)}>
            <ChevronDown size={12} />
          </Button>
        </>
      )}
      {download.status === "paused" ? (
        <Button variant="ghost" size="icon-xs" aria-label={`Resume ${name}`} title="Resume" onClick={() => state.resume(download.id)}>
          <Play size={12} />
        </Button>
      ) : (
        <Button variant="ghost" size="icon-xs" aria-label={`Pause ${name}`} title="Pause" onClick={() => state.pause(download.id)}>
          <Pause size={12} />
        </Button>
      )}
      <Button variant="ghost" size="icon-xs" aria-label={`Cancel ${name}`} title="Cancel" onClick={() => state.cancel(download.id)}>
        <X size={12} />
      </Button>
    </span>
  );
}

/** The download queue: the running download, then the queue in order (movable), then paused ones. */
function DownloadTray({ downloads, state }: { downloads: Download[]; state: CatalogState }) {
  const count = (status: Download["status"]) => downloads.filter((d) => d.status === status).length;
  const queued = count("queued");
  const summary = [
    count("downloading") && `${count("downloading")} downloading`,
    queued && `${queued} queued`,
    count("paused") && `${count("paused")} paused`,
  ]
    .filter(Boolean)
    .join(", ");
  return (
    <aside className="fixed bottom-4 left-4 z-20 flex w-[360px] flex-col rounded-card bg-surface-over py-1 shadow-overlay" aria-label="Downloads">
      <div className="flex items-baseline justify-between gap-2 px-3 pt-1.5 pb-1">
        <span className="font-semibold">Downloads</span>
        <span className="text-xs tracking-[0.005em] tabular-nums text-fg-3">{summary}</span>
      </div>
      <ScrollArea className="max-h-[50vh]">
        <ol aria-label="Download queue">
          {downloads.map((d) => (
            <li key={d.id} className="flex flex-col gap-1 px-3 py-2 shadow-[inset_0_1px_0_var(--border-soft)]">
              <div className="flex items-baseline justify-between gap-2">
                <span className={cn("truncate font-medium", d.status === "paused" && "text-fg-2")}>
                  {d.model.repo.split("/").pop()} {d.model.variant}
                </span>
                <span className="flex-none text-xs tracking-[0.005em] tabular-nums text-fg-3">
                  {formatBytes(d.done)} of {formatBytes(d.total)}
                </span>
              </div>
              <div className="flex items-center gap-2 [&>span:first-child]:flex-1 [&>span:first-child>span:first-child]:flex-1">
                <DownloadProgress download={d} />
                <QueueControls download={d} state={state} queued={queued} />
              </div>
            </li>
          ))}
        </ol>
      </ScrollArea>
    </aside>
  );
}

// ---- installed ----------------------------------------------------------------------------------

function Installed({ state, onDiscover }: { state: CatalogState; onDiscover: () => void }) {
  const ws = useWorkspace();
  const [confirm, setConfirm] = useState<string>();
  const config = ws.config;
  const spinner = useDelayed(!state.loaded);
  const reload = () => {
    void state.reloadInstalled();
    void ws.refreshModels();
  };
  const { storage, setStorage, refresh: refreshStorage } = useModelsStorage(reload);
  const storagePanel = storage && (
    <section className="flex items-center gap-6 rounded-card bg-surface px-4 py-3 shadow-raised max-[900px]:flex-col max-[900px]:items-stretch max-[900px]:gap-3" aria-label="Storage">
      <div className="flex w-[320px] max-w-full flex-none max-[900px]:w-full">
        <StorageBar storage={storage} />
      </div>
      <div className="min-w-0 flex-1">
        <ModelsFolder
          storage={storage}
          onChange={(next) => {
            setStorage(next);
            if (next.move?.status !== "moving") reload();
          }}
        />
      </div>
    </section>
  );

  const inUse = (model: InstalledModel, usage: Usage): boolean => {
    if (!config) return false;
    if (usage.stage === "translation") return config.translation?.provider === usage.provider && config.translation?.model === model.id;
    const stage = usage.stage as "detection" | "ocr" | "inpainting";
    if (config[stage] !== usage.provider) return false;
    try {
      return JSON.parse(config.processorJson[usage.provider] || "{}").model === model.id;
    } catch {
      return false;
    }
  };

  const use = async (model: InstalledModel, usage: Usage) => {
    if (!config) return;
    if (usage.stage === "translation") {
      await ws.saveConfig({
        ...config,
        translation: { ...config.translation!, provider: usage.provider, model: model.id, quantization: model.variant || "default", vision: model.files.some((f) => f.role === "projector"), reasoning: false },
      });
    } else {
      let settings: Record<string, unknown> = {};
      try {
        settings = JSON.parse(config.processorJson[usage.provider] || "{}");
      } catch {
        settings = {};
      }
      await ws.saveConfig({
        ...config,
        [usage.stage]: usage.provider,
        processorJson: { ...config.processorJson, [usage.provider]: JSON.stringify({ ...settings, model: model.id }) },
      });
    }
  };

  const remove = async (model: InstalledModel) => {
    try {
      await catalog.remove(model.id);
      setConfirm(undefined);
      await state.reloadInstalled();
      await ws.refreshModels();
      void refreshStorage();
    } catch (error) {
      ws.reportError(describeError(error));
    }
  };

  if (!state.loaded) {
    return (
      <div className="flex flex-1 items-center justify-center">
        {spinner && (
          <div className={cn(PROGRESS_TRACK, "w-40")} aria-label="Loading installed models">
            <span className={cn(PROGRESS_FILL, PROGRESS_SLIDE)} />
          </div>
        )}
      </div>
    );
  }
  if (state.installed.length === 0) {
    return (
      <div className="flex flex-1 flex-col bg-canvas">
        {storagePanel && <div className="mx-auto w-full max-w-[1120px] px-8 pt-8 max-[900px]:px-4 max-[900px]:pt-5">{storagePanel}</div>}
        <div className="flex flex-1 flex-col items-center justify-center gap-1 p-8 text-center">
          <p className="text-base text-fg">No models installed yet.</p>
          <p className="max-w-[42ch] text-fg-2">Models you download from Discover appear here. The built-in defaults are always available.</p>
          <Button className="mt-2" onClick={onDiscover}>
            <Search size={14} /> Discover models
          </Button>
        </div>
      </div>
    );
  }

  return (
    <ScrollArea className="flex-1">
      <div className="mx-auto flex w-full max-w-[1120px] flex-col gap-4 px-8 pt-8 pb-16 max-[900px]:px-4 max-[900px]:pt-5 max-[900px]:pb-12">
        {storagePanel}
        <Table className="overflow-hidden rounded-card shadow-raised">
          <TableHeader>
            <TableRow>
              <TableHead className="px-3 whitespace-normal">Model</TableHead>
              <TableHead className="px-3 whitespace-normal">Runtime</TableHead>
              <TableHead className="px-3 text-right whitespace-nowrap">Size</TableHead>
              <TableHead className="px-3 whitespace-normal">Status</TableHead>
              <TableHead className="px-3 whitespace-normal">Use for</TableHead>
              <TableHead aria-label="Delete" className="px-3" />
            </TableRow>
          </TableHeader>
          <TableBody className="[&>tr:last-child>td]:shadow-none">
            {state.installed.map((model) => {
              const status = model.compatibility?.status ?? "supported";
              return (
                <TableRow key={model.id} className="[tbody_&]:hover:bg-transparent">
                  <TableCell className="h-auto px-3 py-2">
                    <span className="font-medium">{model.repo.split("/").pop()}</span> {model.variant && <Badge>{model.variant}</Badge>}
                    <p className="text-xs tracking-[0.005em] text-fg-3">
                      {model.repo.split("/")[0]}, installed {formatTimestamp(model.installed_at)}
                    </p>
                  </TableCell>
                  <TableCell className="h-auto px-3 py-2 text-fg-2">{RUNTIME_LABEL[model.runtime]}</TableCell>
                  <TableCell className="h-auto px-3 py-2 text-right whitespace-nowrap tabular-nums">{formatBytes(model.size)}</TableCell>
                  <TableCell className="h-auto px-3 py-2">
                    <StatusCell model={model} status={status} />
                  </TableCell>
                  <TableCell className="h-auto px-3 py-2">
                    <div className="flex flex-wrap gap-1">
                      {model.usable_by.map((usage) =>
                        inUse(model, usage) ? (
                          <Badge key={usage.provider} tone="success">
                            <CircleCheck size={12} /> {STAGE_LABEL[usage.stage] ?? usage.stage}
                          </Badge>
                        ) : (
                          <Button
                            key={usage.provider}
                            variant="outline"
                            size="sm"
                            disabled={status === "unsupported" || model.complete === false}
                            onClick={() => use(model, usage)}
                          >
                            {STAGE_LABEL[usage.stage] ?? usage.stage}
                          </Button>
                        ),
                      )}
                    </div>
                  </TableCell>
                  <TableCell className="h-auto w-[1%] px-3 py-2 text-right whitespace-nowrap">
                    {confirm === model.id ? (
                      <span className="flex items-center gap-1">
                        <Button variant="destructive" size="sm" onClick={() => remove(model)}>
                          Delete files
                        </Button>
                        <Button variant="ghost" size="sm" onClick={() => setConfirm(undefined)}>
                          Cancel
                        </Button>
                      </span>
                    ) : (
                      <Button variant="ghost" size="icon-sm" aria-label={`Delete ${model.id}`} onClick={() => setConfirm(model.id)}>
                        <Trash2 size={14} />
                      </Button>
                    )}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
    </ScrollArea>
  );
}

function StatusCell({ model, status }: { model: InstalledModel; status: string }): ReactNode {
  if (model.complete === false) return <Badge tone="danger">Files missing</Badge>;
  if (status === "needs_update" || status === "unsupported") {
    return (
      <Tooltip content={model.compatibility?.message ?? ""} side="top">
        <span className="inline-flex outline-none focus-visible:rounded-control focus-visible:shadow-focus" tabIndex={0}>
          <Badge tone={status === "needs_update" ? "warning" : "danger"}>
            <CircleAlert size={12} /> {status === "needs_update" ? "Needs newer runtime" : "Unsupported"}
          </Badge>
        </span>
      </Tooltip>
    );
  }
  return <Badge tone="success">Ready</Badge>;
}
