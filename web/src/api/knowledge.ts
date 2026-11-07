// JSON client for a project's knowledge (/api/v1/projects/{id}/knowledge): the glossary and notes the
// translator draws on, and what retrieval picks for a page (urakatsu/pipeline/knowledge.py).

import { call } from "./catalog";

export type KnowledgeKind = "character" | "place" | "term" | "note";

export interface KnowledgeEntry {
  id: string;
  kind: KnowledgeKind;
  /** The name as the source text writes it. */
  source: string;
  aliases: string[];
  translation: string;
  /** The language of `translation`. */
  language: string | null;
  text: string;
  /** Sent with every page. */
  pinned: boolean;
  origin: { kind: "user" | "generated" };
}

export interface Knowledge {
  enabled: boolean;
  /** Story so far: the translator summarizes each page and later pages read the summaries. */
  story: boolean;
  entries: KnowledgeEntry[];
}

export type KnowledgeInput = Pick<KnowledgeEntry, "kind" | "source" | "aliases" | "translation" | "text" | "pinned">;

export interface RetrievedKnowledge {
  glossary: { source: string; translation: string; note: string }[];
  /** Pinned notes, sent with every page. */
  background: string[];
  notes: string[];
  /** Summaries of earlier pages, one string per chapter. */
  story: string[];
  context: { source: string; translation: string }[];
  /** This page's own summary, written when it was translated with Story so far on. */
  summary: string;
}

const json = (method: string, body: unknown): RequestInit => ({ method, body: JSON.stringify(body) });
const root = (projectId: string) => `/projects/${encodeURIComponent(projectId)}/knowledge`;

export const knowledge = {
  get: (projectId: string) => call<Knowledge>(root(projectId)),
  add: (projectId: string, entry: KnowledgeInput) => call<Knowledge>(`${root(projectId)}/entries`, json("POST", { entries: [entry] })),
  update: (projectId: string, entryId: string, entry: KnowledgeInput) => call<Knowledge>(`${root(projectId)}/entries/${encodeURIComponent(entryId)}`, json("PATCH", entry)),
  remove: (projectId: string, entryId: string) => call<void>(`${root(projectId)}/entries/${encodeURIComponent(entryId)}`, { method: "DELETE" }),
  setEnabled: (projectId: string, enabled: boolean) => call<Knowledge>(`${root(projectId)}/enabled`, json("PUT", { enabled })),
  setStory: (projectId: string, enabled: boolean) => call<Knowledge>(`${root(projectId)}/story`, json("PUT", { enabled })),
  page: (projectId: string, pageId: string) => call<RetrievedKnowledge>(`${root(projectId)}/pages/${encodeURIComponent(pageId)}`),
};
