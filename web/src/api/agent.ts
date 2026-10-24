// JSON client for agents (/api/v1/agent): models, ChatGPT sign-in, conversations, and the
// server-sent events of a running turn.

import { call } from "./catalog";

export interface AgentModel {
  /** provider:model */
  id: string;
  provider: string;
  provider_name: string;
  model: string;
  name: string;
  source: "chatgpt" | "provider" | "local";
}

export interface ChatGptStatus {
  signed_in: boolean;
  email: string | null;
  plan: string | null;
  pending: boolean;
  login_url: string | null;
  error: string | null;
}

/** A slash command the message box offers: ``/name argument`` runs one tool without the model. */
export interface AgentCommand {
  name: string;
  description: string;
  /** What follows the name, as a hint; empty when nothing does. */
  argument: string;
}

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface ChatMessage {
  role: "user" | "assistant" | "tool";
  content: string;
  tool_calls: ToolCall[];
  tool_call_id: string | null;
  name: string | null;
  model: string | null;
  created_at: number;
}

export interface ConversationSummary {
  id: string;
  agent: string;
  model: string;
  project_id: string | null;
  title: string;
  created_at: number;
  updated_at: number;
}

export interface Conversation extends ConversationSummary {
  messages: ChatMessage[];
  running: boolean;
}

export type TurnEvent =
  | { type: "user"; content: string }
  | { type: "step"; model: string }
  | { type: "text"; delta: string }
  | { type: "reasoning"; delta: string }
  | { type: "tool_call"; id: string; name: string; arguments: Record<string, unknown> }
  | { type: "tool_result"; id: string; name: string; ok: boolean; result: string; changed: string[] }
  | { type: "progress"; kind: string; page_id: string | null; stage: string | null; model: string | null; completed: number; total: number; message: string }
  | { type: "ui"; action: "open_page"; page_id: string }
  | { type: "error"; message: string }
  /** A model step failed in a way that may pass and is being sent again; drop its streamed text. */
  | { type: "retry"; reason: "transient" | "overflow"; delay: number; message: string }
  | { type: "done"; status: "done" | "stopped" | "failed" };

export const agentApi = {
  /** The saved model list (instant); ``stale`` means a refresh (``refresh=true``) is worth doing. */
  models: (refresh = false) => call<{ models: AgentModel[]; stale: boolean }>(`/agent/models${refresh ? "?refresh=true" : ""}`),
  chatgpt: () => call<ChatGptStatus>("/agent/auth/chatgpt"),
  signIn: () => call<{ url: string }>("/agent/auth/chatgpt/login", { method: "POST" }),
  cancelSignIn: () => call<void>("/agent/auth/chatgpt/cancel", { method: "POST" }),
  signOut: () => call<void>("/agent/auth/chatgpt/logout", { method: "POST" }),
  conversations: (projectId?: string) => call<ConversationSummary[]>(`/agent/conversations${projectId ? `?project_id=${encodeURIComponent(projectId)}` : ""}`),
  conversation: (id: string) => call<Conversation>(`/agent/conversations/${id}`),
  deleteConversation: (id: string) => call<void>(`/agent/conversations/${id}`, { method: "DELETE" }),
  stop: (id: string) => call<void>(`/agent/conversations/${id}/stop`, { method: "POST" }),
  commands: () => call<AgentCommand[]>("/agent/commands"),
  /** ``selection``: the text blocks selected in the editor, which slash commands act on. */
  send: (body: { text: string; model: string; conversation_id?: string; project_id?: string; page_id?: string; selection?: string[] }) =>
    call<{ conversation_id: string }>("/agent/messages", { method: "POST", body: JSON.stringify(body) }),
};

/** Follows a turn's events, replaying it from the start. ``onLost`` runs when the stream ends
 * without a "done" event (no turn on the server, or the connection dropped). Returns a function
 * that stops listening. */
export function followTurn(conversationId: string, onEvent: (event: TurnEvent) => void, onLost: () => void): () => void {
  const source = new EventSource(`/api/v1/agent/conversations/${conversationId}/events`);
  let finished = false;
  source.onmessage = (message) => {
    const event = JSON.parse(message.data) as TurnEvent;
    onEvent(event);
    if (event.type === "done") {
      finished = true;
      source.close();
    }
  };
  // EventSource reconnects and would replay the turn, so a dropped stream is closed for good.
  source.onerror = () => {
    source.close();
    if (!finished) {
      finished = true;
      onLost();
    }
  };
  return () => {
    finished = true;
    source.close();
  };
}
