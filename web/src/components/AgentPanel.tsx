// The Agent tab: a chat with an assistant that works in the open project through tools.
// Conversations live on the server; a running turn keeps going when the tab is closed and
// the panel re-attaches to it when it opens again.

import { Check, ChevronRight, CircleAlert, KeyRound, LoaderCircle, LogIn, LogOut, MoreHorizontal, Plus, RefreshCw, Trash2, Wrench } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { cn } from "cn";

import { agentApi, followTurn, type AgentCommand, type AgentModel, type ChatGptStatus, type ChatMessage, type ConversationSummary, type TurnEvent } from "../api/agent";
import { assetHref, describeError } from "../api/client";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useWorkspace } from "../state/workspace";
import { AgentComposer as Composer } from "./AgentComposer";
import { ActionMenu, Select, Tooltip, type Option } from "./primitives";

const MODEL_KEY = "urakatsu.agent.model";
const chatKey = (projectId: string) => `urakatsu.agent.chat.${projectId}`;

function stored(key: string): string {
  try {
    return localStorage.getItem(key) ?? "";
  } catch {
    return "";
  }
}

function store(key: string, value: string | undefined): void {
  try {
    if (value) localStorage.setItem(key, value);
    else localStorage.removeItem(key);
  } catch {
    /* storage unavailable */
  }
}

// ---- transcript model --------------------------------------------------------------------------

type Item =
  | { kind: "user"; text: string }
  | { kind: "text"; text: string }
  | { kind: "reasoning"; text: string }
  | { kind: "tool"; id: string; name: string; args: Record<string, unknown>; status: "running" | "ok" | "error"; result?: string; progress?: string }
  | { kind: "error"; text: string }
  | { kind: "status"; text: string };

function failed(result: string): boolean {
  try {
    const value = JSON.parse(result);
    return !!value && typeof value === "object" && !Array.isArray(value) && "error" in value;
  } catch {
    return false;
  }
}

function fromMessages(messages: ChatMessage[]): Item[] {
  const items: Item[] = [];
  for (const message of messages) {
    if (message.role === "user") items.push({ kind: "user", text: message.content });
    else if (message.role === "assistant") {
      if (message.content) items.push({ kind: "text", text: message.content });
      for (const call of message.tool_calls) items.push({ kind: "tool", id: call.id, name: call.name, args: call.arguments, status: "running" });
    } else {
      const index = items.findIndex((item) => item.kind === "tool" && item.id === message.tool_call_id);
      if (index >= 0) items[index] = { ...(items[index] as Extract<Item, { kind: "tool" }>), status: failed(message.content) ? "error" : "ok", result: message.content };
    }
  }
  return items;
}

function progressText(event: Extract<TurnEvent, { type: "progress" }>): string {
  const parts = [event.stage ? event.stage[0].toUpperCase() + event.stage.slice(1) : event.kind, event.total ? `${event.completed}/${event.total}` : ""];
  return parts.filter(Boolean).join(" · ") + (event.message ? `: ${event.message}` : "");
}

function reduce(items: Item[], event: TurnEvent): Item[] {
  const last = items.at(-1);
  switch (event.type) {
    case "user":
      return [...items, { kind: "user", text: event.content }];
    case "text":
      return last?.kind === "text" ? [...items.slice(0, -1), { kind: "text", text: last.text + event.delta }] : [...items, { kind: "text", text: event.delta }];
    case "reasoning":
      return last?.kind === "reasoning" ? [...items.slice(0, -1), { kind: "reasoning", text: last.text + event.delta }] : [...items, { kind: "reasoning", text: event.delta }];
    case "tool_call":
      return [...items, { kind: "tool", id: event.id, name: event.name, args: event.arguments, status: "running" }];
    case "tool_result":
      return items.map((item) => (item.kind === "tool" && item.id === event.id ? { ...item, status: event.ok ? "ok" : "error", result: event.result, progress: undefined } : item));
    case "progress": {
      const index = items.map((item) => item.kind === "tool" && item.status === "running").lastIndexOf(true);
      if (index < 0) return items;
      const next = [...items];
      next[index] = { ...(next[index] as Extract<Item, { kind: "tool" }>), progress: progressText(event) };
      return next;
    }
    case "error":
      return [...items, { kind: "error", text: event.message }];
    case "retry": {
      // Tool calls are published only after a step finishes, so the failed attempt is the
      // trailing text and thinking (and any earlier retry note).
      let end = items.length;
      while (end > 0 && ["text", "reasoning", "status"].includes(items[end - 1].kind)) end--;
      const why = event.reason === "overflow" ? "Conversation too long for the model; retrying shorter" : `Retrying in ${event.delay}s`;
      return [...items.slice(0, end), { kind: "status", text: `${why} (${event.message})` }];
    }
    default:
      return items;
  }
}

// ---- panel -------------------------------------------------------------------------------------

export function AgentPanel() {
  const ws = useWorkspace();
  const projectId = ws.project?.id;
  const [models, setModels] = useState<AgentModel[] | null>(null);
  const [account, setAccount] = useState<ChatGptStatus | null>(null);
  const [model, setModel] = useState(() => stored(MODEL_KEY));
  const [chats, setChats] = useState<ConversationSummary[]>([]);
  const [chatId, setChatId] = useState<string | undefined>();
  const [items, setItems] = useState<Item[]>([]);
  const [running, setRunning] = useState(false);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string>();
  const [commands, setCommands] = useState<AgentCommand[]>([]);
  const unfollow = useRef<(() => void) | null>(null);
  const opened = useRef(0);
  const live = useRef<Item[]>([]);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    agentApi.commands().then(setCommands, () => setCommands([]));
  }, []);

  // The saved list shows at once; a stale one is replaced when the fresh listing arrives.
  const loadModels = useCallback(async (refresh = false) => {
    try {
      const [status, list] = await Promise.all([agentApi.chatgpt(), agentApi.models(refresh)]);
      setAccount(status);
      setModels(list.models);
      if (list.stale) setModels((await agentApi.models(true)).models);
    } catch (e) {
      setModels((current) => current ?? []);
      setError(describeError(e));
    }
  }, []);

  useEffect(() => {
    void loadModels();
  }, [loadModels]);

  // Keep the chosen model valid: the stored one when it still exists, else the first offered.
  useEffect(() => {
    if (models && models.length && !models.some((m) => m.id === model)) setModel(models[0].id);
  }, [models, model]);
  useEffect(() => store(MODEL_KEY, model), [model]);

  // ChatGPT sign-in: the server opens the sign-in page in the system browser and waits for its
  // callback; the panel shows the link too and polls until the flow ends.
  const [signIn, setSignIn] = useState<{ url: string } | null>(null);
  const stopPolling = useRef<(() => void) | null>(null);
  useEffect(() => () => stopPolling.current?.(), []);

  const startSignIn = useCallback(async () => {
    setError(undefined);
    stopPolling.current?.();
    try {
      const { url } = await agentApi.signIn();
      setSignIn({ url });
      stopPolling.current = waitForSignIn((status) => {
        setSignIn(null);
        if (status?.error) setError(status.error);
        void loadModels(true);
      });
    } catch (e) {
      setSignIn(null);
      setError(`Could not start ChatGPT sign-in: ${describeError(e)}`);
    }
  }, [loadModels]);

  const cancelSignIn = useCallback(() => {
    stopPolling.current?.();
    setSignIn(null);
    void agentApi.cancelSignIn().catch(() => undefined);
  }, []);

  const signOut = useCallback(async () => {
    try {
      await agentApi.signOut();
      await loadModels(true);
    } catch (e) {
      setError(describeError(e));
    }
  }, [loadModels]);

  const loadChats = useCallback(async () => {
    if (!projectId) return setChats([]);
    try {
      setChats(await agentApi.conversations(projectId));
    } catch {
      setChats([]);
    }
  }, [projectId]);

  // A different project: its chat list, and the chat that was open there last.
  useEffect(() => {
    void loadChats();
    setChatId(projectId ? stored(chatKey(projectId)) || undefined : undefined);
  }, [projectId, loadChats]);

  const handle = useCallback(
    (event: TurnEvent) => {
      if (event.type === "tool_result") {
        if (event.changed.includes("project") && projectId) void ws.reloadProject(projectId);
        if (event.changed.includes("config")) void ws.refresh();
      }
      if (event.type === "ui" && event.action === "open_page") ws.selectPage(event.page_id);
    },
    [projectId, ws],
  );

  const open = useCallback(
    async (id: string | undefined) => {
      const ticket = ++opened.current;
      unfollow.current?.();
      unfollow.current = null;
      setError(undefined);
      if (!id) {
        setItems([]);
        setRunning(false);
        return;
      }
      try {
        const chat = await agentApi.conversation(id);
        if (ticket !== opened.current) return;
        if (chat.model && !model) setModel(chat.model);
        if (!chat.running) {
          setItems(fromMessages(chat.messages));
          setRunning(false);
          return;
        }
        // The running turn replays from its user message; show what came before it.
        const lastUser = chat.messages.map((m) => m.role).lastIndexOf("user");
        live.current = fromMessages(chat.messages.slice(0, Math.max(0, lastUser)));
        setItems(live.current);
        setRunning(true);
        unfollow.current = followTurn(
          id,
          (event) => {
            if (ticket !== opened.current) return;
            handle(event);
            live.current = reduce(live.current, event);
            setItems(live.current);
            if (event.type === "done") {
              setRunning(false);
              void loadChats();
            }
          },
          () => {
            if (ticket === opened.current) void open(id);
          },
        );
      } catch (e) {
        if (ticket !== opened.current) return;
        // A chat deleted elsewhere: forget it.
        setItems([]);
        setRunning(false);
        if (projectId) store(chatKey(projectId), undefined);
        setChatId(undefined);
        setError(describeError(e));
      }
    },
    [handle, loadChats, model, projectId],
  );

  useEffect(() => {
    if (projectId) store(chatKey(projectId), chatId);
    void open(chatId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatId]);

  useEffect(() => () => unfollow.current?.(), []);

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [items]);

  const send = async (text: string) => {
    const message = text.trim();
    if (!message || !model || running) return;
    setDraft("");
    setError(undefined);
    try {
      const { conversation_id } = await agentApi.send({
        text: message,
        model,
        conversation_id: chatId,
        project_id: projectId,
        page_id: ws.pageId,
        selection: ws.blockId ? [ws.blockId] : [],
      });
      if (conversation_id === chatId) void open(conversation_id);
      else setChatId(conversation_id);
      void loadChats();
    } catch (e) {
      setDraft(message);
      setError(describeError(e));
    }
  };

  const stop = () => chatId && void agentApi.stop(chatId).catch((e) => setError(describeError(e)));

  const remove = async () => {
    if (!chatId) return;
    await agentApi.deleteConversation(chatId).catch((e) => setError(describeError(e)));
    setChatId(undefined);
    void loadChats();
  };

  const chatOptions: Option[] = useMemo(() => chats.map((c) => ({ value: c.id, label: c.title })), [chats]);
  const noModels = models !== null && models.length === 0;
  const lastSent = useMemo(() => [...items].reverse().find((item) => item.kind === "user")?.text, [items]);
  const pageLabel = useMemo(() => {
    const page = ws.project?.pages.find((p) => p.id === ws.pageId);
    if (!page) return undefined;
    const chapter = ws.project?.chapters.find((c) => c.pageIds.includes(page.id));
    return chapter ? `${chapter.name} · ${page.name}` : page.name;
  }, [ws.project, ws.pageId]);

  return (
    <div className="@container/agent flex min-h-0 flex-1 flex-col">
      <header className="flex h-[38px] flex-none items-center gap-1 px-2 shadow-[inset_0_-1px_0_var(--border-soft)]">
        <div className="min-w-0 flex-1">
          {chatOptions.length > 0 ? (
            <Select size="sm" label="Conversation" value={chatId ?? null} placeholder="New chat" options={chatOptions} onChange={setChatId} className="w-full" />
          ) : (
            <span className="px-1 text-xs font-semibold">New chat</span>
          )}
        </div>
        <Tooltip content="New chat" side="bottom">
          <Button variant="ghost" size="icon-sm" aria-label="New chat" disabled={!chatId} onClick={() => setChatId(undefined)}>
            <Plus size={14} />
          </Button>
        </Tooltip>
        <AccountMenu
          account={account}
          canDelete={!!chatId && !running}
          onDelete={() => void remove()}
          onSignIn={() => void startSignIn()}
          onSignOut={() => void signOut()}
          onRefresh={() => void loadModels(true)}
        />
      </header>

      <ScrollArea className="flex-1" contentClassName="flex flex-col gap-3 px-3 py-3">
        {!projectId && <p className="text-xs text-fg-2">Open a project so the assistant can work in it.</p>}
        {signIn && !noModels && <SignInPending url={signIn.url} onCancel={cancelSignIn} />}
        {noModels ? (
          <SignInCard signIn={signIn} onSignIn={() => void startSignIn()} onCancel={cancelSignIn} openProviders={() => ws.setSettingsOpen(true, "providers")} />
        ) : items.length === 0 ? (
          <Suggestions disabled={!model || !projectId} onPick={(text) => void send(text)} />
        ) : (
          items.map((item, index) => <Entry key={index} item={item} />)
        )}
        {running && items.at(-1)?.kind !== "text" && items.at(-1)?.kind !== "tool" && (
          <span className="flex items-center gap-1.5 text-xs text-fg-3">
            <LoaderCircle size={12} className="animate-spin" /> Working
          </span>
        )}
        {error && <Notice>{error}</Notice>}
        <div ref={bottom} />
      </ScrollArea>

      <Composer
        value={draft}
        onChange={setDraft}
        onSend={() => void send(draft)}
        onStop={stop}
        running={running}
        disabled={noModels || !projectId}
        placeholder={!projectId ? "Open a project first" : running ? "The assistant is working" : "Ask anything, or type / for commands"}
        lastSent={lastSent}
        page={pageLabel}
        models={models}
        model={model}
        onModel={setModel}
        commands={commands}
      />
    </div>
  );
}

// ---- pieces ------------------------------------------------------------------------------------

function AccountMenu({
  account,
  canDelete,
  onDelete,
  onSignIn,
  onSignOut,
  onRefresh,
}: {
  account: ChatGptStatus | null;
  canDelete: boolean;
  onDelete: () => void;
  onSignIn: () => void;
  onSignOut: () => void;
  onRefresh: () => void;
}) {
  return (
    <ActionMenu
      trigger={
        <Button variant="ghost" size="icon-sm" aria-label="Assistant options">
          <MoreHorizontal size={14} />
        </Button>
      }
      actions={[
        account?.signed_in
          ? {
              label: (
                <span className="flex items-center gap-2">
                  <LogOut size={14} /> Sign out of ChatGPT{account.email ? ` (${account.email})` : ""}
                </span>
              ),
              onSelect: onSignOut,
            }
          : {
              label: (
                <span className="flex items-center gap-2">
                  <LogIn size={14} /> Sign in with ChatGPT
                </span>
              ),
              onSelect: onSignIn,
            },
        {
          label: (
            <span className="flex items-center gap-2">
              <RefreshCw size={14} /> Refresh models
            </span>
          ),
          onSelect: onRefresh,
        },
        {
          label: (
            <span className="flex items-center gap-2">
              <Trash2 size={14} /> Delete chat
            </span>
          ),
          onSelect: onDelete,
          danger: true,
          disabled: !canDelete,
        },
      ]}
    />
  );
}

/** Polls the sign-in until it ends (signed in, failed, cancelled, or ten minutes passed). */
function waitForSignIn(onEnd: (status: ChatGptStatus | null) => void): () => void {
  let stopped = false;
  const deadline = Date.now() + 10 * 60_000;
  const tick = async () => {
    if (stopped) return;
    let status: ChatGptStatus | null = null;
    try {
      status = await agentApi.chatgpt();
    } catch {
      /* the server may be restarting; keep polling */
    }
    if (stopped) return;
    if ((status && (status.signed_in || !status.pending)) || Date.now() > deadline) {
      onEnd(status);
      return;
    }
    window.setTimeout(() => void tick(), 1500);
  };
  window.setTimeout(() => void tick(), 1500);
  return () => {
    stopped = true;
  };
}

function SignInPending({ url, onCancel }: { url: string; onCancel: () => void }) {
  return (
    <div className="flex flex-col gap-2 rounded-control bg-surface px-3 py-2 shadow-[inset_0_0_0_1px_var(--border-soft)]">
      <span className="flex items-center gap-1.5 text-xs text-fg-2">
        <LoaderCircle size={12} className="animate-spin" /> Finish signing in to ChatGPT in your browser.
      </span>
      <div className="flex gap-1">
        <a
          href={url}
          target="_blank"
          rel="noreferrer"
          className="inline-flex h-7 items-center rounded-control bg-surface px-2 text-sm text-fg shadow-[inset_0_0_0_1px_var(--border)] transition-colors duration-[20ms] ease-in hover:bg-hover focus-visible:shadow-focus focus-visible:outline-none"
        >
          Open sign-in page
        </a>
        <Button variant="ghost" size="sm" onClick={onCancel}>
          Cancel
        </Button>
      </div>
    </div>
  );
}

function SignInCard({ signIn, onSignIn, onCancel, openProviders }: { signIn: { url: string } | null; onSignIn: () => void; onCancel: () => void; openProviders: () => void }) {
  return (
    <div className="flex flex-col gap-3 rounded-card bg-surface p-4 shadow-raised">
      <div className="flex flex-col gap-1">
        <h3 className="text-sm font-semibold">Choose a model for the assistant</h3>
        <p className="text-xs text-fg-2">Sign in with your ChatGPT plan, or add an API key for a chat model such as OpenAI, Claude, Gemini or DeepSeek.</p>
      </div>
      {signIn ? (
        <SignInPending url={signIn.url} onCancel={onCancel} />
      ) : (
        <div className="flex flex-wrap gap-1">
          <Button size="sm" onClick={onSignIn}>
            <LogIn size={14} /> Sign in with ChatGPT
          </Button>
          <Button variant="outline" size="sm" onClick={openProviders}>
            <KeyRound size={14} /> Add API key
          </Button>
        </div>
      )}
    </div>
  );
}

const SUGGESTIONS = [
  "Review the translations on this page",
  "Make character names consistent across the project",
  "Run the full pipeline on this chapter",
];

function Suggestions({ disabled, onPick }: { disabled: boolean; onPick: (text: string) => void }) {
  return (
    <div className="flex flex-col gap-2 pt-2">
      <p className="text-xs text-fg-2">The assistant reads and edits this project: translations, typesetting and pipeline runs. Every edit can be undone.</p>
      <div className="flex flex-col gap-1">
        {SUGGESTIONS.map((text) => (
          <button
            key={text}
            type="button"
            disabled={disabled}
            className="flex items-center justify-between gap-2 rounded-control px-2 py-1.5 text-left text-sm text-fg-2 shadow-[inset_0_0_0_1px_var(--border-soft)] transition-colors duration-[20ms] ease-in hover:bg-hover hover:text-fg focus-visible:shadow-focus focus-visible:outline-none disabled:pointer-events-none disabled:opacity-50"
            onClick={() => onPick(text)}
          >
            {text}
            <ChevronRight size={14} className="flex-none text-icon-2" />
          </button>
        ))}
      </div>
    </div>
  );
}

function Notice({ children }: { children: ReactNode }) {
  return (
    <div className="flex gap-1.5 rounded-control bg-danger-weak px-2 py-1.5 text-xs text-danger shadow-[inset_0_0_0_1px_var(--danger-line)] [word-break:break-word]">
      <CircleAlert size={14} className="mt-px flex-none" />
      <span>{children}</span>
    </div>
  );
}

const TOOL_LABELS: Record<string, string> = {
  list_projects: "Listed projects",
  get_project: "Read the project",
  get_page: "Read a page",
  search_text: "Searched text",
  get_translation_settings: "Read translation settings",
  list_translation_models: "Listed translation models",
  list_fonts: "Listed fonts",
  edit_text_blocks: "Edited text",
  set_typography: "Changed typesetting",
  set_layer_visibility: "Changed layer visibility",
  create_text_block: "Added a text block",
  delete_text_blocks: "Deleted text blocks",
  edit_image: "Edited the art",
  undo: "Undid a change",
  redo: "Redid a change",
  run_pipeline: "Ran the pipeline",
  configure_translation: "Changed translation settings",
  open_page: "Opened a page",
};

interface EditResult {
  page_id: string;
  page: { width: number; height: number };
  region: { x: number; y: number; width: number; height: number };
  revision: number;
}

/** The edited region of the page as it renders now; a click opens the page. */
function EditPreview({ result }: { result: string }) {
  const ws = useWorkspace();
  const projectId = ws.project?.id;
  let edit: EditResult;
  try {
    edit = JSON.parse(result) as EditResult;
  } catch {
    return null;
  }
  if (!projectId || !edit.page || !edit.region) return null;
  const { page, region } = edit;
  return (
    <button
      type="button"
      onClick={() => ws.selectPage(edit.page_id)}
      className="relative w-full overflow-hidden rounded-control bg-surface-sunk shadow-[inset_0_0_0_1px_var(--border-soft)]"
      style={{ aspectRatio: `${region.width} / ${region.height}`, width: `min(100%, ${Math.round((240 * region.width) / region.height)}px)` }}
    >
      <img
        alt="Edited region"
        src={assetHref(projectId, edit.page_id, "render", String(edit.revision))}
        className="absolute max-w-none"
        style={{
          width: `${(page.width / region.width) * 100}%`,
          left: `${(-region.x / region.width) * 100}%`,
          top: `${(-region.y / region.height) * 100}%`,
        }}
      />
    </button>
  );
}

function pretty(text: string): string {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
}

function Entry({ item }: { item: Item }) {
  switch (item.kind) {
    case "user":
      return <div className="self-end max-w-[88%] rounded-card bg-surface-fill px-3 py-2 text-sm whitespace-pre-wrap [word-break:break-word]">{item.text}</div>;
    case "text":
      return <div className="text-sm leading-5 whitespace-pre-wrap [word-break:break-word]">{inline(item.text)}</div>;
    case "reasoning":
      return (
        <details className="group text-xs text-fg-3">
          <summary className="cursor-pointer list-none select-none hover:text-fg-2">Thinking</summary>
          <p className="mt-1 whitespace-pre-wrap">{item.text}</p>
        </details>
      );
    case "error":
      return <Notice>{item.text}</Notice>;
    case "status":
      return <p className="text-xs text-fg-3 [word-break:break-word]">{item.text}</p>;
    case "tool": {
      const label = TOOL_LABELS[item.name] ?? item.name;
      const args = Object.keys(item.args).length ? JSON.stringify(item.args, null, 2) : "";
      const details = (
        <details className="group rounded-control text-xs shadow-[inset_0_0_0_1px_var(--border-soft)]">
          <summary className="flex cursor-pointer list-none items-center gap-1.5 rounded-control px-2 py-1.5 select-none hover:bg-hover-faint">
            {item.status === "running" ? (
              <LoaderCircle size={12} className="flex-none animate-spin text-brand" />
            ) : item.status === "ok" ? (
              <Check size={12} className="flex-none text-success" />
            ) : (
              <CircleAlert size={12} className="flex-none text-danger" />
            )}
            <Wrench size={12} className="flex-none text-icon-2" />
            <span className={cn("min-w-0 flex-1 truncate", item.status === "error" ? "text-danger" : "text-fg-2")}>{item.progress ?? label}</span>
            <ChevronRight size={12} className="flex-none text-icon-2 transition-transform duration-100 group-open:rotate-90" />
          </summary>
          <div className="flex flex-col gap-1 px-2 pb-2 font-mono text-[11px] leading-4 text-fg-2">
            <span className="text-fg-3">{item.name}</span>
            {args && <pre className="max-h-40 overflow-auto whitespace-pre-wrap [word-break:break-word]">{args}</pre>}
            {item.result && <pre className="max-h-60 overflow-auto rounded-control bg-surface-sunk p-1.5 whitespace-pre-wrap [word-break:break-word]">{pretty(item.result)}</pre>}
          </div>
        </details>
      );
      if (item.name !== "edit_image" || item.status !== "ok" || !item.result) return details;
      return (
        <div className="flex flex-col gap-1.5">
          {details}
          <EditPreview result={item.result} />
        </div>
      );
    }
  }
}

/** `code` and **bold** in assistant replies; everything else stays plain text. */
function inline(text: string): ReactNode[] {
  return text.split(/(`[^`\n]+`|\*\*[^*\n]+\*\*)/g).map((part, index) => {
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return (
        <code key={index} className="rounded-[3px] bg-surface-fill px-1 font-mono text-[12px]">
          {part.slice(1, -1)}
        </code>
      );
    }
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return (
        <strong key={index} className="font-semibold">
          {part.slice(2, -2)}
        </strong>
      );
    }
    return part;
  });
}
