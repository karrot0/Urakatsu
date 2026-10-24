// Settings: a left nav of sections and rows of label + control. Changes save
// automatically a moment after you stop editing.

import { ArrowDown, ArrowLeft, ArrowUp, Boxes, Cpu, Eye, EyeOff, KeyRound, Keyboard, Languages, Library, Monitor, Moon, Palette, Plus, Server, Sun, Trash2, Type, X } from "lucide-react";
import { createContext, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { catalog, RUNTIME_LABEL, type InstalledModel, type Release, type Runtimes as RuntimeList } from "../api/catalog";
import { describeError } from "../api/client";
import { formatDate, formatTimestamp } from "../format";

import type { Config, ProviderInfo, TranslationModel } from "../gen/urakatsu/v1/urakatsu_pb";
import { format as formatShortcut, fromEvent as shortcutFromEvent, MOD_NAME } from "../shortcuts";
import { DEFAULT_SHORTCUTS, NUDGES, TOOLS, useWorkspace, type ShortcutAction, type StageName, type Theme } from "../state/workspace";
import { ModelsFolder, StorageBar, StorageBreakdown, useModelsStorage } from "./ModelsStorage";
import { NumberField, Select, Switch, type Option } from "./primitives";
import { FontPicker } from "./RightPanel";
import { SourcesSettings } from "./SourcesSettings";
import { TOOL_INFO } from "./ToolBelt";
import { ACCENTS, NEUTRALS } from "../theme";
import { Badge, Segmented } from "./ui";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "cn";
import { ScrollArea } from "@/components/ui/scroll-area";

const TABS = [
  ["appearance", "Appearance", Palette],
  ["pipeline", "Pipeline", Cpu],
  ["providers", "Providers", KeyRound],
  ["translation", "Translation", Languages],
  ["typesetting", "Typesetting", Type],
  ["runtimes", "Runtimes", Server],
  ["sources", "Sources", Library],
  ["shortcuts", "Shortcuts", Keyboard],
] as const;
type Tab = (typeof TABS)[number][0];

const SAVE_DELAY = 260;

function parseJson(raw: string | undefined): Record<string, unknown> {
  try {
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

export function SettingsPage() {
  const ws = useWorkspace();
  const [tab, setTab] = useState<Tab>(TABS.some(([id]) => id === ws.settingsTab) ? (ws.settingsTab as Tab) : "appearance");
  const [installed, setInstalled] = useState<InstalledModel[]>([]);
  useEffect(() => {
    catalog.installed().then(setInstalled, () => undefined);
  }, []);
  const [draft, setDraft] = useState<Config | undefined>(ws.config);
  const [status, setStatus] = useState<"saved" | "saving" | "idle">("idle");
  const saved = useRef(JSON.stringify(ws.config ?? null));

  // Opened straight from a URL, Settings mounts before the config has loaded: start the draft then.
  useEffect(() => {
    if (draft || !ws.config) return;
    saved.current = JSON.stringify(ws.config);
    setDraft(ws.config);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ws.config]);

  useEffect(() => {
    if (!draft || !ws.config) return;
    const serialized = JSON.stringify(draft);
    if (serialized === saved.current) return;
    const timer = window.setTimeout(() => {
      saved.current = serialized;
      setStatus("saving");
      void ws.saveConfig(draft).then(() => setStatus("saved"));
    }, SAVE_DELAY);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft]);

  const update = (patch: Partial<Config>) => setDraft((current) => (current ? ({ ...current, ...patch } as Config) : current));

  return (
    <InstalledContext.Provider value={installed}>
    <div className="settings grid h-screen grid-cols-[256px_minmax(0,1fr)] bg-surface-sunk">
      <nav className="flex flex-col gap-0.5 px-2 py-3" aria-label="Settings sections">
        <Button variant="ghost" className="mb-4 justify-start font-normal" onClick={() => ws.setSettingsOpen(false)}>
          <ArrowLeft size={16} /> {ws.settingsFromModels ? "Back to models" : ws.settingsFromSources ? "Back to sources" : ws.project ? "Back to editor" : "Back to projects"}
        </Button>
        <p className="px-2 pb-1 text-xs leading-5 text-fg-3">Settings</p>
        {TABS.map(([id, label, Icon]) => (
          <button
            key={id}
            className={cn(
              "flex h-9 cursor-pointer items-center gap-2 rounded-control px-2 text-left transition-colors duration-[20ms] ease-in focus-visible:shadow-focus focus-visible:outline-none",
              tab === id ? "bg-selected font-medium text-fg [&_svg]:text-brand" : "text-fg-2 hover:bg-hover hover:text-fg [&_svg]:text-icon-2",
            )}
            aria-current={tab === id ? "page" : undefined}
            onClick={() => setTab(id)}
          >
            <Icon size={16} /> {label}
          </button>
        ))}
      </nav>
      <main className="flex min-h-0 min-w-0 flex-col overflow-hidden rounded-tl-modal bg-surface shadow-raised">
        <header className="flex flex-none items-center justify-between h-14 px-8 shadow-[inset_0_-1px_0_var(--border-soft)]">
          <h1 className="text-sm font-semibold">Settings</h1>
          <span className="text-xs leading-4 tracking-[0.005em] text-fg-3" role="status">
            {status === "saving" ? "Saving" : status === "saved" ? "All changes saved" : ""}
          </span>
        </header>
        <ScrollArea className="flex-1">
          <div className="mx-auto w-full max-w-[820px] px-8 pt-10 pb-16">
            {tab === "appearance" && <Appearance />}
            {tab !== "appearance" && tab !== "shortcuts" && tab !== "runtimes" && tab !== "sources" && !draft && <p className="text-fg-2">Loading settings.</p>}
            {tab === "pipeline" && draft && <Pipeline draft={draft} update={update} />}
            {tab === "providers" && draft && <Providers draft={draft} update={update} />}
            {tab === "translation" && draft && <Translation draft={draft} update={update} />}
            {tab === "typesetting" && draft && <Typesetting draft={draft} update={update} />}
            {tab === "shortcuts" && <Shortcuts />}
            {tab === "runtimes" && <Runtimes />}
            {tab === "sources" && <SourcesSettings />}
          </div>
        </ScrollArea>
      </main>
    </div>
    </InstalledContext.Provider>
  );
}

/** Models installed from the Models page, for the per-provider model choice. */
const InstalledContext = createContext<InstalledModel[]>([]);

// ---- layout ------------------------------------------------------------------------------------

/** One label-and-control row: the grid shared by PreferenceRow and the rows built by hand. */
const PREF_ROW = "grid grid-cols-[minmax(160px,0.9fr)_minmax(240px,1.1fr)] gap-5 py-3 shadow-[inset_0_-1px_0_var(--border-soft)] last:shadow-none";
const CONTROL_PAIR = "flex min-w-0 items-center gap-1 [&>:first-child]:min-w-0 [&>:first-child]:flex-1";

export function PreferencePage({ title, description, children }: { title: string; description: string; children: ReactNode }) {
  return (
    <div>
      <header className="mb-8">
        <h2 className="text-xl font-semibold tracking-[-0.01em]">{title}</h2>
        <p className="mt-0.5 max-w-[60ch] text-fg-2">{description}</p>
      </header>
      {children}
    </div>
  );
}

export function PreferenceSection({ title, description, action, children }: { title: string; description?: string; action?: ReactNode; children: ReactNode }) {
  return (
    <section className="[&+&]:mt-8">
      <div className="mb-2 flex items-end justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold">{title}</h3>
          {description && <p className="mt-0.5 text-xs leading-4 text-fg-2">{description}</p>}
        </div>
        {action}
      </div>
      <div className="rounded-card bg-surface px-4 shadow-raised">{children}</div>
    </section>
  );
}

export function PreferenceRow({ title, description, children, align = "center" }: { title: string; description?: string; children: ReactNode; align?: "center" | "start" }) {
  return (
    <div className={cn(PREF_ROW, align === "start" ? "items-start" : "items-center")}>
      <div>
        <p className="font-medium">{title}</p>
        {description && <p className="mt-0.5 text-xs leading-4 text-fg-2">{description}</p>}
      </div>
      <div className="min-w-0">{children}</div>
    </div>
  );
}

// ---- appearance --------------------------------------------------------------------------------

function Appearance() {
  const ws = useWorkspace();
  const themes: [Theme, string, typeof Sun][] = [
    ["light", "Light", Sun],
    ["dark", "Dark", Moon],
    ["system", "System", Monitor],
  ];
  return (
    <PreferencePage title="Appearance" description="How Urakatsu looks on this device.">
      <PreferenceSection title="Interface">
        <PreferenceRow title="Theme" description="System follows your operating system setting.">
          <div className="grid grid-cols-3 rounded-control bg-surface-fill p-0.5" role="radiogroup" aria-label="Theme">
            {themes.map(([value, label, Icon]) => (
              <button
                key={value}
                role="radio"
                aria-checked={ws.theme === value}
                className={cn(
                  "inline-flex h-[30px] cursor-pointer items-center justify-center gap-1 rounded-[calc(var(--r-control)_-_2px)] text-fg-2 focus-visible:shadow-focus focus-visible:outline-none",
                  ws.theme === value && "bg-surface text-fg shadow-raised",
                )}
                onClick={() => ws.setTheme(value)}
              >
                <Icon size={14} /> {label}
              </button>
            ))}
          </div>
        </PreferenceRow>
        <PreferenceRow title="Accent" description="Buttons, selection and focus use it.">
          <div className="flex flex-wrap gap-2" role="radiogroup" aria-label="Accent">
            {ACCENTS.map((accent) => {
              const active = ws.palette.accent === accent.id;
              return (
                <button
                  key={accent.id}
                  type="button"
                  role="radio"
                  aria-checked={active}
                  aria-label={accent.label}
                  title={accent.label}
                  onClick={() => ws.setPalette({ ...ws.palette, accent: accent.id })}
                  className="grid size-7 cursor-pointer place-items-center rounded-full outline-none focus-visible:shadow-focus"
                >
                  <span
                    className={`block size-5 rounded-full ${active ? "shadow-[0_0_0_2px_var(--surface),0_0_0_4px_var(--text)]" : "shadow-[inset_0_0_0_1px_color-mix(in_oklab,var(--n-950)_18%,transparent)]"}`}
                    style={{ background: accent.swatch }}
                  />
                </button>
              );
            })}
          </div>
        </PreferenceRow>
        <PreferenceRow title="Neutrals" description="The temperature of the greys behind everything.">
          <Segmented
            label="Neutrals"
            value={ws.palette.neutral}
            onChange={(neutral) => ws.setPalette({ ...ws.palette, neutral })}
            options={NEUTRALS.map((n) => ({ value: n.id, label: n.label, title: n.hint }))}
          />
        </PreferenceRow>
      </PreferenceSection>
    </PreferencePage>
  );
}

// ---- pipeline ----------------------------------------------------------------------------------

const VISION: { stage: "detection" | "ocr" | "inpainting"; title: string; description: string }[] = [
  { stage: "detection", title: "Detection", description: "Finds text, speech balloons and panels." },
  { stage: "ocr", title: "Text recognition", description: "Reads the source text in each block." },
  { stage: "inpainting", title: "Inpainting", description: "Removes the original lettering." },
];

function providerOptions(providers: ProviderInfo[]): Option[] {
  return providers.map((p) => ({ value: p.id, label: p.name, disabled: !p.available, detail: p.available ? undefined : "Unavailable" }));
}

function RuntimeSelect({ provider, draft, update }: { provider?: ProviderInfo; draft: Config; update: (patch: Partial<Config>) => void }) {
  if (!provider || provider.runtimes.length < 2) return null;
  return (
    <Select
      size="md"
      label={`${provider.name} runtime`}
      value={draft.runtimes[provider.id] || provider.defaultRuntime}
      options={provider.runtimes.map((r) => ({ value: r.name, label: r.name, disabled: !r.available, detail: r.available ? undefined : "Unavailable" }))}
      onChange={(runtime) => update({ runtimes: { ...draft.runtimes, [provider.id]: runtime } })}
      className="w-[120px] flex-none"
    />
  );
}

/** The model the canvas's generative fill runs: an inpainting model, loaded only while it fills. */
function FillModelRow({ providers, draft, update }: { providers: ProviderInfo[]; draft: Config; update: (patch: Partial<Config>) => void }) {
  const inpainting = providers.find((p) => p.id === draft.inpainting);
  const options: Option[] = [
    { value: "", label: "Same as inpainting", detail: inpainting ? (inpainting.prompted ? inpainting.name : `${inpainting.name}, no prompts`) : undefined },
    ...providers.map((p) => ({ value: p.id, label: p.name, disabled: !p.available, detail: !p.available ? "Unavailable" : p.prompted ? "Follows prompts" : "Removes only" })),
  ];
  const provider = draft.fill ? providers.find((p) => p.id === draft.fill) : undefined;
  return (
    <PreferenceRow title="Generative fill" description="Paints a canvas selection from a prompt. A model other than the inpainting one loads only when you generate.">
      <div className={CONTROL_PAIR}>
        <Select label="Generative fill model" value={draft.fill} options={options} onChange={(fill) => update({ fill })} />
        {provider && provider.id !== draft.inpainting && <RuntimeSelect provider={provider} draft={draft} update={update} />}
      </div>
    </PreferenceRow>
  );
}

function Pipeline({ draft, update }: { draft: Config; update: (patch: Partial<Config>) => void }) {
  const ws = useWorkspace();
  const caps = ws.capabilities;
  const byStage = (stage: StageName) => caps?.providers.filter((p) => p.stage === stage) ?? [];
  const devices: Option[] = [
    { value: "auto", label: `Automatic (${caps?.device?.description ?? "best available"})` },
    ...(caps?.devices ?? []).map((d) => ({
      value: d.backend === "cpu" ? "cpu" : `${d.backend}:${d.index}`,
      label: d.backend === "cpu" ? "CPU" : `${d.backend.toUpperCase()}:${d.index}`,
      detail: d.description,
    })),
  ];
  return (
    <PreferencePage title="Pipeline" description="The device, the models each stage uses, and how pages are stored.">
      <PreferenceSection title="Hardware">
        <PreferenceRow title="Device" description="Where models run. Automatic picks the fastest available.">
          <Select label="Device" value={draft.device} options={devices} onChange={(device) => update({ device })} />
        </PreferenceRow>
      </PreferenceSection>
      <PreferenceSection title="Vision models">
        {VISION.map(({ stage, title, description }) => {
          const providers = byStage(stage);
          const provider = providers.find((p) => p.id === draft[stage]);
          return (
            <PreferenceRow key={stage} title={title} description={description}>
              <div className={CONTROL_PAIR}>
                <Select label={`${title} model`} value={draft[stage]} options={providerOptions(providers)} onChange={(id) => update({ [stage]: id } as Partial<Config>)} />
                <RuntimeSelect provider={provider} draft={draft} update={update} />
              </div>
            </PreferenceRow>
          );
        })}
        <FillModelRow providers={byStage("inpainting")} draft={draft} update={update} />
      </PreferenceSection>
      {VISION.map(({ stage }) => {
        const provider = byStage(stage).find((p) => p.id === draft[stage]);
        if (!provider) return null;
        return <SchemaSection key={stage} provider={provider} map="processorJson" draft={draft} update={update} title={`${provider.name} options`} />;
      })}
      {(() => {
        // A separate fill model has its own options; the same model shares the inpainting ones.
        const provider = draft.fill && draft.fill !== draft.inpainting ? byStage("inpainting").find((p) => p.id === draft.fill) : undefined;
        return provider && <SchemaSection provider={provider} map="processorJson" draft={draft} update={update} title={`${provider.name} options`} />;
      })()}
      <PreferenceSection title="Storage" description="Where models are kept, and how pages are stored. JPEG and WebP pages are never re-encoded.">
        <ModelsFolderRow />
        <PreferenceRow
          title="Compress PNG pages losslessly"
          description="Stores PNG pages you add or download as lossless WebP: about 15% smaller, identical pixels. Importing a chapter takes about a second longer. Applies to new pages."
        >
          <Switch label="Compress PNG pages losslessly" checked={draft.losslessPages} onChange={(losslessPages) => update({ losslessPages })} />
        </PreferenceRow>
      </PreferenceSection>
    </PreferencePage>
  );
}

/** The models folder and its drive's space; changing it can move the installed models along. */
function ModelsFolderRow() {
  const ws = useWorkspace();
  const { storage, setStorage } = useModelsStorage(() => void ws.refreshModels());
  return (
    <PreferenceRow title="Models folder" description="Downloads go here and every model loads from here." align="start">
      {storage ? (
        <div className="flex min-w-0 flex-col gap-2">
          <StorageBar storage={storage} />
          <ModelsFolder storage={storage} onChange={setStorage} />
          <StorageBreakdown storage={storage} />
        </div>
      ) : (
        <span className="text-xs leading-4 text-fg-3">Loading.</span>
      )}
    </PreferenceRow>
  );
}

/** A form for a provider's settings, generated from its JSON schema. */
function SchemaSection({
  provider,
  map,
  draft,
  update,
  title,
  description,
  action,
}: {
  provider: ProviderInfo;
  map: "processorJson" | "providersJson";
  draft: Config;
  update: (patch: Partial<Config>) => void;
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  const schema = parseJson(provider.settingsSchemaJson) as { properties?: Record<string, SchemaProperty> };
  const fields = Object.entries(schema.properties ?? {}).filter(([key]) => key !== "translation");
  if (fields.length === 0) return null;
  const values = { ...parseJson(provider.defaultsJson), ...parseJson(draft[map][provider.id]) };
  delete values.translation;
  const set = (key: string, value: unknown) => {
    const next = { ...parseJson(draft[map][provider.id]), [key]: value };
    if (value === null || value === "") delete next[key];
    update({ [map]: { ...draft[map], [provider.id]: JSON.stringify(next) } } as Partial<Config>);
  };
  return (
    <PreferenceSection title={title} description={description} action={action}>
      {fields.map(([key, property]) => (
        <PreferenceRow key={key} title={FIELD_TITLES[key] ?? property.title ?? key} description={FIELD_DESCRIPTIONS[key] ?? property.description} align={schemaType(property) === "string" && key.includes("prompt") ? "start" : "center"}>
          {key === "model" ? (
            <InstalledModelSelect providerId={provider.id} value={typeof values.model === "string" ? values.model : ""} onChange={(value) => set("model", value || null)} />
          ) : (
            <SchemaField name={key} property={property} value={values[key]} onChange={(value) => set(key, value)} />
          )}
        </PreferenceRow>
      ))}
    </PreferenceSection>
  );
}

function InstalledModelSelect({ providerId, value, onChange }: { providerId: string; value: string; onChange: (value: string) => void }) {
  const ws = useWorkspace();
  const installed = useContext(InstalledContext).filter((m) => m.usable_by.some((u) => u.provider === providerId));
  const options: Option[] = [
    { value: "", label: "Built-in default" },
    ...installed.map((m) => ({ value: m.id, label: `${m.repo.split("/").pop()}${m.variant ? ` ${m.variant}` : ""}`, detail: m.compatibility?.status === "needs_update" ? "Needs newer runtime" : RUNTIME_LABEL[m.runtime] })),
  ];
  if (value && !installed.some((m) => m.id === value)) options.push({ value, label: value, detail: "Not installed" });
  return (
    <div className={CONTROL_PAIR}>
      <Select label="Model" value={value} options={options} onChange={onChange} />
      <Button variant="outline" onClick={() => ws.setModelsOpen(true)}>
        <Boxes size={14} /> Find
      </Button>
    </div>
  );
}

// Pydantic titles come from field names ("Api Key"); a few read better spelled out.
const FIELD_TITLES: Record<string, string> = { api_key: "API key", base_url: "Base URL", model: "Model" };
const FIELD_DESCRIPTIONS: Record<string, string> = { api_key: "Separate several keys with commas to rotate between them." };

// Mirrors ``credentials`` in remote/base.py: keys split on commas, semicolons or whitespace.
function apiKeyCount(value: unknown): number {
  return typeof value === "string" ? new Set(value.split(/[\s,;]+/).filter(Boolean)).size : 0;
}

interface SchemaProperty {
  type?: string;
  anyOf?: { type?: string }[];
  title?: string;
  description?: string;
  enum?: string[];
}

function schemaType(property: SchemaProperty): string {
  return property.type ?? property.anyOf?.find((t) => t.type && t.type !== "null")?.type ?? "string";
}

function SchemaField({ name, property, value, onChange }: { name: string; property: SchemaProperty; value: unknown; onChange: (value: unknown) => void }) {
  const type = schemaType(property);
  const label = property.title ?? name;
  if (type === "number" || type === "integer") {
    return <NumberField label={label} stepper={false} value={typeof value === "number" ? value : null} placeholder="Default" step={type === "integer" ? 1 : 0.01} onCommit={(next) => onChange(next)} />;
  }
  if (type === "boolean") return <Switch label={label} checked={!!value} onChange={onChange} />;
  if (property.enum) {
    return <Select label={label} value={String(value ?? "")} options={property.enum.map((v) => ({ value: v, label: v }))} onChange={onChange} />;
  }
  const secret = /key|secret|token|password/i.test(name);
  if (secret) return <SecretInput label={label} value={String(value ?? "")} onChange={(next) => onChange(next || null)} />;
  if (name.includes("prompt")) {
    return <CommitTextarea label={label} value={String(value ?? "")} onCommit={(next) => onChange(next || null)} />;
  }
  return <CommitInput label={label} value={String(value ?? "")} placeholder={name.includes("url") ? "https://" : "Default"} onCommit={(next) => onChange(next || null)} />;
}

function CommitInput({ label, value, placeholder, onCommit, type = "text" }: { label: string; value: string; placeholder?: string; onCommit: (value: string) => void; type?: string }) {
  const [text, setText] = useState(value);
  useEffect(() => setText(value), [value]);
  return <Input type={type} aria-label={label} value={text} placeholder={placeholder} onChange={(e) => setText(e.target.value)} onBlur={() => text !== value && onCommit(text.trim())} />;
}

function CommitTextarea({ label, value, onCommit, placeholder }: { label: string; value: string; onCommit: (value: string) => void; placeholder?: string }) {
  const [text, setText] = useState(value);
  useEffect(() => setText(value), [value]);
  return <Textarea aria-label={label} value={text} placeholder={placeholder} onChange={(e) => setText(e.target.value)} onBlur={() => text !== value && onCommit(text)} />;
}

function SecretInput({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  const [shown, setShown] = useState(false);
  return (
    <div className="flex gap-0.5">
      <CommitInput label={label} type={shown ? "text" : "password"} value={value} placeholder="Not set" onCommit={onChange} />
      <Button type="button" variant="ghost" size="icon-sm" className="shrink" aria-label={shown ? `Hide ${label}` : `Show ${label}`} onClick={() => setShown((s) => !s)}>
        {shown ? <EyeOff size={14} /> : <Eye size={14} />}
      </Button>
    </div>
  );
}

// ---- providers ---------------------------------------------------------------------------------

function Providers({ draft, update }: { draft: Config; update: (patch: Partial<Config>) => void }) {
  const ws = useWorkspace();
  const remote = (ws.capabilities?.providers ?? []).filter((p) => p.stage === "translation" && p.id !== "local");
  return (
    <PreferencePage title="Providers" description="Keys and endpoints for hosted translation services. Keys are stored in your local settings file.">
      {remote.map((provider) => {
        const settings = parseJson(draft.providersJson[provider.id]);
        const keys = apiKeyCount(settings.api_key);
        return (
          <SchemaSection
            key={provider.id}
            provider={provider}
            map="providersJson"
            draft={draft}
            update={update}
            title={provider.name}
            description={provider.description}
            action={<Badge tone={keys > 0 ? "success" : "neutral"}>{keys > 1 ? `${keys} keys` : keys === 1 ? "Configured" : "Not configured"}</Badge>}
          />
        );
      })}
    </PreferencePage>
  );
}

// ---- translation -------------------------------------------------------------------------------

const GENERATION: { key: string; title: string; integer?: boolean; min?: number; max?: number; step?: number }[] = [
  { key: "temperature", title: "Temperature", min: 0, max: 2, step: 0.05 },
  { key: "top_p", title: "Top P", min: 0, max: 1, step: 0.01 },
  { key: "top_k", title: "Top K", integer: true, min: 0 },
  { key: "min_p", title: "Min P", min: 0, max: 1, step: 0.01 },
  { key: "max_tokens", title: "Max tokens", integer: true, min: 1 },
  { key: "repeat_penalty", title: "Repeat penalty", min: 0, step: 0.05 },
  { key: "frequency_penalty", title: "Frequency penalty", min: -2, max: 2, step: 0.05 },
  { key: "presence_penalty", title: "Presence penalty", min: -2, max: 2, step: 0.05 },
];

function Translation({ draft, update }: { draft: Config; update: (patch: Partial<Config>) => void }) {
  const ws = useWorkspace();
  const translation = draft.translation!;
  const providers = (ws.capabilities?.providers ?? []).filter((p) => p.stage === "translation");
  const provider = providers.find((p) => p.id === translation.provider);
  const models = ws.translationModels.filter((m) => m.provider === translation.provider);
  const model = models.find((m) => m.model === translation.model);
  const generation = parseJson(draft.generationJson);
  const setGeneration = (key: string, value: unknown) => {
    const next = { ...generation, [key]: value };
    if (value === null) delete next[key];
    update({ generationJson: JSON.stringify(next) });
  };
  const chooseModel = (choice?: TranslationModel, providerId = translation.provider) =>
    update({
      translation: {
        ...translation,
        provider: providerId,
        model: choice?.model ?? "",
        quantization: choice?.quantizations[0]?.id ?? "",
        vision: !!choice?.vision,
        reasoning: !!choice?.reasoning,
      },
    });
  const languages: Option[] = useMemo(
    () => (ws.capabilities?.languages.length ? ws.capabilities.languages.map((l) => ({ value: l.tag, label: l.name, detail: l.tag })) : [{ value: draft.targetLanguage, label: draft.targetLanguage }]),
    [ws.capabilities, draft.targetLanguage],
  );

  return (
    <PreferencePage title="Translation" description="The model that translates, the language it writes, and how it samples.">
      <PreferenceSection title="Model">
        <PreferenceRow title="Provider" description="A local model, or a hosted service set up under Providers.">
          <div className={CONTROL_PAIR}>
            <Select
              label="Translation provider"
              value={translation.provider}
              options={providerOptions(providers)}
              onChange={(id) => chooseModel(ws.translationModels.find((m) => m.provider === id), id)}
            />
            <RuntimeSelect provider={provider} draft={draft} update={update} />
          </div>
        </PreferenceRow>
        <PreferenceRow title="Translation model" description={translation.provider === "local" ? "Install more GGUF chat models on the Models page." : undefined}>
          {models.length > 0 ? (
            <Select
              label="Translation model"
              value={translation.model}
              options={models.map((m) => ({ value: m.model, label: m.name || m.model, detail: [m.vision && "Vision", m.reasoning && "Reasoning"].filter(Boolean).join(" · ") || undefined }))}
              onChange={(id) => chooseModel(models.find((m) => m.model === id))}
            />
          ) : (
            <CommitInput label="Model ID" value={translation.model} placeholder="Model ID" onCommit={(id) => update({ translation: { ...translation, model: id } })} />
          )}
        </PreferenceRow>
        {model && model.quantizations.length > 0 && (
          <PreferenceRow title="Quantization" description="Smaller files load faster; larger ones translate better.">
            <Select
              label="Quantization"
              value={translation.quantization}
              options={model.quantizations.map((q) => ({ value: q.id, label: q.name, detail: q.downloaded ? "Downloaded" : undefined }))}
              onChange={(quantization) => update({ translation: { ...translation, quantization } })}
            />
          </PreferenceRow>
        )}
      </PreferenceSection>
      <PreferenceSection title="Output">
        <PreferenceRow title="Target language">
          <Select label="Target language" value={draft.targetLanguage} options={languages} onChange={(targetLanguage) => update({ targetLanguage })} />
        </PreferenceRow>
        <PreferenceRow title="Instructions" description="Added to every translation prompt." align="start">
          <CommitTextarea label="Instructions" value={draft.instructions} placeholder="Keep honorifics. Translate sound effects." onCommit={(instructions) => update({ instructions })} />
        </PreferenceRow>
      </PreferenceSection>
      <PreferenceSection title="Sampling" description="Empty fields use the model's defaults.">
        {GENERATION.map(({ key, title, integer, min, max, step }) => (
          <PreferenceRow key={key} title={title}>
            <NumberField
              label={title}
              stepper={false}
              value={typeof generation[key] === "number" ? (generation[key] as number) : null}
              placeholder="Default"
              min={min}
              max={max}
              step={integer ? 1 : step}
              onCommit={(value) => setGeneration(key, value)}
            />
          </PreferenceRow>
        ))}
      </PreferenceSection>
      <PreferenceSection title="Capabilities">
        <PreferenceRow title="Reasoning" description="Let models that support it think before answering. Slower.">
          <Switch label="Reasoning" checked={!!generation.reasoning} onChange={(on) => setGeneration("reasoning", on)} />
        </PreferenceRow>
        <PreferenceRow title="Vision" description="Send the page image with the text, for models that accept images.">
          <Switch label="Vision" checked={generation.vision !== false} onChange={(on) => setGeneration("vision", on)} />
        </PreferenceRow>
      </PreferenceSection>
    </PreferencePage>
  );
}

// ---- typesetting -------------------------------------------------------------------------------

function Typesetting({ draft, update }: { draft: Config; update: (patch: Partial<Config>) => void }) {
  const [adding, setAdding] = useState("");
  const families = draft.fontFamilies;
  const set = (next: string[]) => update({ fontFamilies: next });
  const move = (index: number, delta: number) => {
    const next = [...families];
    const [item] = next.splice(index, 1);
    next.splice(index + delta, 0, item);
    set(next);
  };
  return (
    <PreferencePage title="Typesetting" description="Fonts used for translated text when a block does not choose one.">
      <PreferenceSection title="Default fonts" description="Tried in order; the first font with the needed characters is used.">
        {families.length === 0 && <p className="py-3 text-fg-2 shadow-[inset_0_-1px_0_var(--border-soft)]">No default fonts. The renderer falls back to system fonts.</p>}
        {families.map((family, index) => (
          <div key={family} className={cn(PREF_ROW, "items-center")}>
            <p className="font-medium" style={{ fontFamily: `"${family}"` }}>
              <span className="mr-2 font-sans text-fg-3 tabular-nums">{index + 1}</span>
              {family}
            </p>
            <div className="flex min-w-0 items-center justify-end gap-0.5">
              <Button variant="ghost" size="icon-sm" aria-label={`Move ${family} up`} disabled={index === 0} onClick={() => move(index, -1)}>
                <ArrowUp size={14} />
              </Button>
              <Button variant="ghost" size="icon-sm" aria-label={`Move ${family} down`} disabled={index === families.length - 1} onClick={() => move(index, 1)}>
                <ArrowDown size={14} />
              </Button>
              <Button variant="ghost" size="icon-sm" aria-label={`Remove ${family}`} onClick={() => set(families.filter((f) => f !== family))}>
                <X size={14} />
              </Button>
            </div>
          </div>
        ))}
        <div className={cn(PREF_ROW, "items-center")}>
          <p className="font-medium">Add font</p>
          <div className={CONTROL_PAIR}>
            <FontPicker value={adding} placeholder="Choose a font" onChange={setAdding} />
            <Button
              variant="outline"
              disabled={!adding || families.includes(adding)}
              onClick={() => {
                set([...families, adding]);
                setAdding("");
              }}
            >
              <Plus size={14} /> Add
            </Button>
          </div>
        </div>
      </PreferenceSection>
    </PreferencePage>
  );
}

// ---- shortcuts ---------------------------------------------------------------------------------

const ACTION_LABELS: Partial<Record<ShortcutAction, string>> = {
  fit: "Fit page to view",
  undo: "Undo",
  redo: "Redo",
  nudge_left: "Move left",
  nudge_right: "Move right",
  nudge_up: "Move up",
  nudge_down: "Move down",
};

const shortcutLabel = (action: ShortcutAction) => ACTION_LABELS[action] ?? TOOL_INFO[action as keyof typeof TOOL_INFO].label;

// Keys the canvas uses that are not configurable; a custom shortcut on one of them is flagged.
const FIXED_KEYS: Record<string, string> = {
  space: "panning (hold Space)",
  delete: "deleting the selected block",
  backspace: "deleting the selected block",
  escape: "clearing the selection",
  "1": "the Source view",
  "2": "the Layers view",
  "3": "the Rendered view",
};

const SHORTCUT_GROUPS: { title: string; description?: string; actions: readonly ShortcutAction[] }[] = [
  { title: "Editing", actions: ["undo", "redo"] },
  { title: "Tools", actions: [...TOOLS, "fit"] },
  { title: "Move selected block", description: "Hold Shift to move 10 pixels. With no block selected, these keys scroll the canvas.", actions: NUDGES },
];

const REFERENCE: [string, string][] = [
  ["Scroll wheel", "Scroll the canvas"],
  ["Shift + scroll wheel", "Scroll the canvas sideways"],
  [`${MOD_NAME} + scroll wheel, or pinch`, "Zoom at the pointer"],
  ["Alt + scroll wheel", "Brush size, with a brush tool"],
  ["Space + drag", "Pan the canvas"],
  ["Delete or Backspace", "Delete the selected block"],
  ["Esc", "Clear the selection"],
  ["1, 2, 3", "Source, Layers and Rendered views"],
];

function Shortcuts() {
  const ws = useWorkspace();
  const actions = Object.keys(DEFAULT_SHORTCUTS) as ShortcutAction[];
  const changed = actions.some((action) => ws.shortcuts[action] !== DEFAULT_SHORTCUTS[action]);
  const clashWith = (action: ShortcutAction): string | undefined => {
    const binding = ws.shortcuts[action];
    const other = actions.find((a) => a !== action && ws.shortcuts[a] === binding);
    if (other) return `Also used for ${shortcutLabel(other).toLowerCase()}.`;
    if (FIXED_KEYS[binding]) return `Also used for ${FIXED_KEYS[binding]}.`;
    return undefined;
  };
  return (
    <PreferencePage title="Shortcuts" description="Keys for the canvas. Click a shortcut, then press the keys you want; Esc cancels. They are stored in this browser.">
      {SHORTCUT_GROUPS.map((group, index) => (
        <PreferenceSection
          key={group.title}
          title={group.title}
          description={group.description}
          action={
            index === 0 ? (
              <Button variant="outline" size="sm" disabled={!changed} onClick={() => actions.forEach((action) => ws.setShortcut(action, DEFAULT_SHORTCUTS[action]))}>
                <Trash2 size={14} /> Reset to defaults
              </Button>
            ) : undefined
          }
        >
          {group.actions.map((action) => {
            const clash = clashWith(action);
            const note = action === "redo" && ws.shortcuts.redo === DEFAULT_SHORTCUTS.redo ? `${MOD_NAME}+Y also redoes.` : undefined;
            return (
              <PreferenceRow key={action} title={shortcutLabel(action)} description={clash ?? note}>
                <ShortcutRecorder label={shortcutLabel(action)} value={ws.shortcuts[action]} invalid={!!clash} onChange={(binding) => ws.setShortcut(action, binding)} />
              </PreferenceRow>
            );
          })}
        </PreferenceSection>
      ))}
      <PreferenceSection title="Mouse and fixed keys" description="These work everywhere on the canvas and cannot be changed.">
        {REFERENCE.map(([keys, what]) => (
          <PreferenceRow key={keys} title={what}>
            <span className="block text-right font-mono text-xs leading-4 text-fg-2">{keys}</span>
          </PreferenceRow>
        ))}
      </PreferenceSection>
    </PreferencePage>
  );
}

/** A shortcut that records the next key press when clicked. */
function ShortcutRecorder({ label, value, invalid, onChange }: { label: string; value: string; invalid: boolean; onChange: (binding: string) => void }) {
  const [recording, setRecording] = useState(false);
  return (
    <button
      type="button"
      className={cn(
        "ml-auto flex h-7 min-w-[96px] cursor-pointer items-center justify-center rounded-control bg-surface px-2 text-xs leading-4 font-medium text-fg transition-colors duration-[20ms] ease-in hover:bg-hover focus-visible:shadow-focus focus-visible:outline-none aria-[invalid=true]:shadow-[inset_0_0_0_1px_var(--danger)] aria-[invalid=true]:focus-visible:shadow-[inset_0_0_0_1px_var(--danger)] [&_kbd]:[color:inherit] [&_kbd]:[font:inherit]",
        recording ? "font-sans text-fg-2 shadow-focus" : "font-mono shadow-[inset_0_0_0_1px_var(--border)]",
      )}
      aria-label={recording ? `Press the new shortcut for ${label}` : `${label} shortcut: ${formatShortcut(value)}. Click to change.`}
      aria-invalid={invalid || undefined}
      onClick={() => setRecording(true)}
      onBlur={() => setRecording(false)}
      onKeyDown={(event) => {
        if (!recording) return;
        event.preventDefault();
        event.stopPropagation();
        if (event.key === "Escape") return setRecording(false);
        const binding = shortcutFromEvent(event);
        if (!binding) return; // a modifier on its own: wait for the key
        onChange(binding);
        setRecording(false);
      }}
    >
      {recording ? "Press keys" : <kbd>{formatShortcut(value)}</kbd>}
    </button>
  );
}

// ---- runtimes ----------------------------------------------------------------------------------

const LATEST = "latest";

function Runtimes() {
  const ws = useWorkspace();
  const [data, setData] = useState<RuntimeList>();
  const [releases, setReleases] = useState<Record<string, Release[]>>({});
  const [busy, setBusy] = useState<string>();

  useEffect(() => {
    catalog.runtimes().then(
      (list) => {
        setData(list);
        list.native.forEach((runtime) =>
          catalog.releases(runtime.id).then(
            (found) => setReleases((current) => ({ ...current, [runtime.id]: found })),
            () => undefined,
          ),
        );
      },
      (error) => ws.reportError(describeError(error)),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const act = async (id: string, task: () => Promise<RuntimeList>) => {
    setBusy(id);
    try {
      setData(await task());
      await ws.refreshModels();
    } catch (error) {
      ws.reportError(describeError(error));
    } finally {
      setBusy(undefined);
    }
  };

  return (
    <PreferencePage
      title="Runtimes"
      description="Urakatsu keeps every runtime on its newest release. Pin a version to stay on it; models that need a newer release show a warning instead of failing."
    >
      {!data && <p className="text-fg-2">Loading runtimes.</p>}
      {data?.native.map((runtime) => {
        const kept = new Set(runtime.kept.map((k) => k.tag));
        const available = (releases[runtime.id] ?? []).filter((r) => r.variant);
        const options: Option[] = [
          { value: LATEST, label: "Latest", detail: runtime.latest ? `Updates automatically, now ${runtime.latest}` : "Updates automatically" },
          ...available.map((r) => ({ value: r.tag, label: r.tag, detail: [formatDate(r.published), kept.has(r.tag) && "On disk"].filter(Boolean).join(" · ") })),
        ];
        if (runtime.pinned && !available.some((r) => r.tag === runtime.pinned)) options.push({ value: runtime.pinned, label: runtime.pinned, detail: "Pinned" });
        const outdated = !runtime.pinned && runtime.latest && runtime.installed && runtime.latest !== runtime.installed;
        const working = busy === runtime.id;
        return (
          <PreferenceSection
            key={runtime.id}
            title={runtime.name}
            description={runtime.repository}
            action={
              <Badge tone={runtime.pinned ? "warning" : outdated ? "info" : runtime.installed ? "success" : "neutral"}>
                {runtime.pinned ? "Pinned" : outdated ? "Update available" : runtime.installed ? "Up to date" : "Not installed"}
              </Badge>
            }
          >
            <PreferenceRow title="Installed" description={runtime.checked_at ? `Checked ${formatTimestamp(runtime.checked_at)}` : "Installs the first time a model needs it."}>
              <span className="tabular-nums">{runtime.installed ? `${runtime.installed} (${runtime.variant})` : "None"}</span>
            </PreferenceRow>
            <PreferenceRow title="Version" description="Older versions stay usable; models that need newer support are flagged on the Models page.">
              <Select
                label={`${runtime.name} version`}
                value={runtime.pinned ?? LATEST}
                options={options}
                disabled={working || data.offline}
                onChange={(tag) => act(runtime.id, () => catalog.pin(runtime.id, tag === LATEST ? null : tag))}
              />
            </PreferenceRow>
            <PreferenceRow title="Update" description={working ? "Installing. This can take a minute." : undefined}>
              <Button variant="outline" disabled={!outdated || working || data.offline} onClick={() => act(runtime.id, () => catalog.update(runtime.id))}>
                {outdated ? `Update to ${runtime.latest}` : runtime.pinned ? "Unpin to update" : "Up to date"}
              </Button>
            </PreferenceRow>
          </PreferenceSection>
        );
      })}
      {data && data.python.length > 0 && (
        <PreferenceSection title="Python packages" description="Installed with the app. Update them with uv sync --upgrade.">
          {data.python.map((p) => (
            <PreferenceRow key={p.id} title={p.name}>
              <span className="tabular-nums">{p.installed}</span>
            </PreferenceRow>
          ))}
        </PreferenceSection>
      )}
    </PreferencePage>
  );
}
