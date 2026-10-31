// Knowledge: what the translator knows about the series (urakatsu/pipeline/knowledge.py). Opened from
// the project header: a glossary of names and terms, free notes, and what retrieval picks for the
// open page. Notes on a linked series start from the source's own summary.

import { BookMarked, Pin, Plus } from "lucide-react";
import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Textarea } from "@/components/ui/textarea";
import { describeError } from "../api/client";
import { knowledge as api, type Knowledge, type KnowledgeEntry, type KnowledgeInput, type KnowledgeKind, type RetrievedKnowledge } from "../api/knowledge";
import { plural } from "../format";
import { useWorkspace } from "../state/workspace";
import { CheckRow, Select, Switch, Tooltip, type Option } from "./primitives";
import { Badge, Segmented } from "./ui";

const KINDS: Option<KnowledgeKind>[] = [
  { value: "character", label: "Character" },
  { value: "place", label: "Place" },
  { value: "term", label: "Term" },
  { value: "note", label: "Note" },
];
const KIND_LABEL = Object.fromEntries(KINDS.map((kind) => [kind.value, kind.label])) as Record<KnowledgeKind, string>;
const EMPTY: KnowledgeInput = { kind: "character", source: "", aliases: [], translation: "", text: "", pinned: false };

// ---- header button ---------------------------------------------------------------------------

/** The project header's Knowledge button. */
export function KnowledgeButton() {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Tooltip content="Knowledge" side="bottom">
        <Button variant="ghost" size="icon-sm" aria-label="Knowledge" onClick={() => setOpen(true)}>
          <BookMarked size={16} />
        </Button>
      </Tooltip>
      {open && <KnowledgeDialog onClose={() => setOpen(false)} />}
    </>
  );
}

// ---- dialog ------------------------------------------------------------------------------------

type Tab = "entries" | "page";
/** The entry being written: an entry's id, or "new". */
type Editing = string | null;

function KnowledgeDialog({ onClose }: { onClose: () => void }) {
  const ws = useWorkspace();
  const projectId = ws.project!.id;
  const pageId = ws.pageId;
  const [data, setData] = useState<Knowledge>();
  const [error, setError] = useState<string>();
  const [tab, setTab] = useState<Tab>("entries");
  const [query, setQuery] = useState("");
  const [editing, setEditing] = useState<Editing>(null);

  const act = useCallback(async (work: () => Promise<Knowledge | void>) => {
    setError(undefined);
    try {
      const next = await work();
      if (next) setData(next);
      return true;
    } catch (failure) {
      setError(describeError(failure));
      return false;
    }
  }, []);

  useEffect(() => {
    void act(() => api.get(projectId));
  }, [act, projectId]);

  const entries = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const all = data?.entries ?? [];
    return needle ? all.filter((e) => [e.source, ...e.aliases, e.translation, e.text].join(" ").toLowerCase().includes(needle)) : all;
  }, [data, query]);

  const save = async (id: string, entry: KnowledgeInput) => {
    if (await act(() => (id === "new" ? api.add(projectId, entry) : api.update(projectId, id, entry)))) setEditing(null);
  };
  const remove = async (id: string) => {
    if (await act(async () => (await api.remove(projectId, id), api.get(projectId)))) setEditing(null);
  };

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      {/* A fixed height: the list and the editor come and go, and a centered dialog would jump. */}
      <DialogContent className="h-[min(620px,calc(100vh-48px))] gap-0 p-0 sm:max-w-2xl">
        <DialogHeader className="px-5 pt-5 pb-3">
          <DialogTitle>Knowledge</DialogTitle>
          <DialogDescription>
            What the translator knows about this series. Each page gets the names it mentions, the notes that fit it, and lines from the pages around it.
          </DialogDescription>
        </DialogHeader>
        {error && <div className="mx-5 mb-3 rounded-control bg-danger-weak px-3 py-2 text-xs text-danger">{error}</div>}
        <div className="flex items-center gap-3 px-5 pb-3">
          <Segmented
            label="Knowledge view"
            value={tab}
            onChange={setTab}
            options={[
              { value: "entries", label: data ? `Entries ${data.entries.length}` : "Entries", title: "The glossary and notes" },
              { value: "page", label: "This page", title: "What the translator gets for the open page" },
            ]}
          />
          <label className="ml-auto flex items-center gap-2 text-xs text-fg-2">
            Use while translating
            <Switch label="Use while translating" checked={data?.enabled ?? true} disabled={!data} onChange={(enabled) => void act(() => api.setEnabled(projectId, enabled))} />
          </label>
        </div>
        {tab === "entries" ? (
          <>
            <div className="flex items-center gap-2 px-5 pb-3">
              <Input aria-label="Search entries" placeholder="Search" value={query} onChange={(event) => setQuery(event.target.value)} />
              <Button variant="secondary" size="sm" onClick={() => setEditing("new")}>
                <Plus size={14} /> Add
              </Button>
            </div>
            <ScrollArea className="min-h-0 flex-1" contentClassName="px-5 pb-5">
              <div className="flex flex-col gap-1">
                {editing === "new" && <EntryEditor initial={EMPTY} onSave={(entry) => void save("new", entry)} onCancel={() => setEditing(null)} />}
                {entries.map((entry) =>
                  editing === entry.id ? (
                    <EntryEditor key={entry.id} initial={entry} onSave={(next) => void save(entry.id, next)} onCancel={() => setEditing(null)} onRemove={() => void remove(entry.id)} />
                  ) : (
                    <EntryRow key={entry.id} entry={entry} onEdit={() => setEditing(entry.id)} />
                  ),
                )}
                {data && !entries.length && editing !== "new" && (
                  <p className="py-6 text-center text-xs leading-4 text-fg-3">
                    {query ? "No entries match." : "Nothing yet. Add character names, places and terms, or ask the assistant to build a glossary from a chapter."}
                  </p>
                )}
              </div>
            </ScrollArea>
          </>
        ) : (
          <PageView projectId={projectId} pageId={pageId} enabled={data?.enabled ?? true} revision={data} />
        )}
      </DialogContent>
    </Dialog>
  );
}

// ---- entries -----------------------------------------------------------------------------------

function EntryRow({ entry, onEdit }: { entry: KnowledgeEntry; onEdit: () => void }) {
  const name = entry.source || entry.text;
  return (
    <button className="flex w-full cursor-pointer flex-col gap-0.5 rounded-control px-2 py-1.5 text-left hover:bg-hover" onClick={onEdit}>
      <span className="flex min-w-0 items-center gap-2">
        <span className="truncate font-medium">{name}</span>
        {entry.translation && <span className="truncate text-fg-2">{entry.translation}</span>}
        <span className="ml-auto flex shrink-0 items-center gap-1.5">
          {entry.pinned && <Pin size={12} className="text-fg-3" aria-label="Sent with every page" />}
          {entry.origin.kind === "generated" && <Badge title="Written from the linked source">Source</Badge>}
          <Badge>{KIND_LABEL[entry.kind] ?? entry.kind}</Badge>
        </span>
      </span>
      {entry.source && entry.text && <span className="truncate text-xs text-fg-3">{entry.text}</span>}
    </button>
  );
}

function EntryEditor({ initial, onSave, onCancel, onRemove }: { initial: KnowledgeInput; onSave: (entry: KnowledgeInput) => void; onCancel: () => void; onRemove?: () => void }) {
  const [entry, setEntry] = useState<KnowledgeInput>({ ...initial });
  const [aliases, setAliases] = useState(initial.aliases.join(", "));
  const set = (patch: Partial<KnowledgeInput>) => setEntry((current) => ({ ...current, ...patch }));
  const blank = !entry.source.trim() && !entry.text.trim();
  const note = entry.kind === "note";

  const submit = () =>
    onSave({
      kind: entry.kind,
      source: entry.source,
      aliases: aliases.split(/[,、]/).map((alias) => alias.trim()).filter(Boolean),
      translation: entry.translation,
      text: entry.text,
      pinned: entry.pinned,
    });

  return (
    <div className="my-1 flex flex-col gap-2 rounded-control p-3 shadow-[inset_0_0_0_1px_var(--border)]">
      <div className="grid grid-cols-[8rem_1fr] gap-2">
        <Labelled label="Kind">
          <Select label="Kind" value={entry.kind} options={KINDS} onChange={(kind) => set({ kind })} />
        </Labelled>
        <Labelled label={note ? "Title" : "Name in the source text"}>
          <Input aria-label="Name" autoFocus value={entry.source} onChange={(event) => set({ source: event.target.value })} />
        </Labelled>
      </div>
      {!note && (
        <div className="grid grid-cols-2 gap-2">
          <Labelled label="Translation">
            <Input aria-label="Translation" value={entry.translation} onChange={(event) => set({ translation: event.target.value })} />
          </Labelled>
          <Labelled label="Other spellings">
            <Input aria-label="Other spellings" placeholder="Separated by commas" value={aliases} onChange={(event) => setAliases(event.target.value)} />
          </Labelled>
        </div>
      )}
      <Labelled label={note ? "Text" : "Notes"}>
        <Textarea
          aria-label="Notes"
          rows={note ? 6 : 3}
          placeholder={note ? "A synopsis, a character sheet, anything worth knowing" : "Gender, how they speak, who they are to others"}
          value={entry.text}
          onChange={(event) => set({ text: event.target.value })}
        />
      </Labelled>
      <div className="flex items-center gap-2">
        <div className="-ml-2 min-w-0">
          <CheckRow checked={entry.pinned} label="Send with every page" onChange={(pinned) => set({ pinned })} />
        </div>
        <span className="ml-auto flex items-center gap-2">
          {onRemove && (
            <Button variant="ghost" size="sm" onClick={onRemove}>
              Delete
            </Button>
          )}
          <Button variant="ghost" size="sm" onClick={onCancel}>
            Cancel
          </Button>
          <Button size="sm" disabled={blank} onClick={submit}>
            Save
          </Button>
        </span>
      </div>
    </div>
  );
}

function Labelled({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <span className="text-xs font-medium text-fg-2">{label}</span>
      {children}
    </div>
  );
}

// ---- this page ---------------------------------------------------------------------------------

/** What retrieval picks for the open page; ``revision`` reads it again when the entries change. */
function PageView({ projectId, pageId, enabled, revision }: { projectId: string; pageId?: string; enabled: boolean; revision: unknown }) {
  const [picked, setPicked] = useState<RetrievedKnowledge>();
  const [error, setError] = useState<string>();

  useEffect(() => {
    if (!pageId) return;
    let stopped = false;
    api
      .page(projectId, pageId)
      .then((next) => !stopped && setPicked(next))
      .catch((failure) => !stopped && setError(describeError(failure)));
    return () => {
      stopped = true;
    };
  }, [projectId, pageId, revision]);

  const nothing = picked && !picked.glossary.length && !picked.notes.length && !picked.context.length;
  return (
    <ScrollArea className="min-h-0 flex-1" contentClassName="flex flex-col gap-4 px-5 pb-5">
      {error && <p className="text-xs text-danger">{error}</p>}
      {!pageId && <p className="py-6 text-center text-xs text-fg-3">Open a page to see what its translation gets.</p>}
      {!enabled && <p className="py-6 text-center text-xs text-fg-3">Knowledge is off, so pages are translated on their own.</p>}
      {enabled && pageId && !picked && !error && <p className="py-6 text-center text-xs text-fg-3">Reading this page…</p>}
      {enabled && nothing && <p className="py-6 text-center text-xs leading-4 text-fg-3">Nothing fits this page yet. It needs source text from OCR, and entries or translated pages to draw on.</p>}
      {enabled && picked && picked.glossary.length > 0 && (
        <Group title={plural(picked.glossary.length, "term")}>
          {picked.glossary.map((term) => (
            <div key={term.source} className="flex min-w-0 flex-col gap-0.5 rounded-control px-2 py-1">
              <span className="flex min-w-0 gap-2">
                <span className="truncate font-medium">{term.source}</span>
                <span className="truncate text-fg-2">{term.translation}</span>
              </span>
              {term.note && <span className="truncate text-xs text-fg-3">{term.note}</span>}
            </div>
          ))}
        </Group>
      )}
      {enabled && picked && picked.notes.length > 0 && (
        <Group title={plural(picked.notes.length, "note")}>
          {picked.notes.map((note, index) => (
            <p key={index} className="rounded-control px-2 py-1 text-xs leading-4 whitespace-pre-wrap text-fg-2">
              {note}
            </p>
          ))}
        </Group>
      )}
      {enabled && picked && picked.context.length > 0 && (
        <Group title={plural(picked.context.length, "line") + " of context"}>
          {picked.context.map((line, index) => (
            <div key={index} className="grid grid-cols-2 gap-3 rounded-control px-2 py-0.5 text-xs leading-4">
              <span className="min-w-0 text-fg-2">{line.source}</span>
              <span className="min-w-0 text-fg-3">{line.translation}</span>
            </div>
          ))}
        </Group>
      )}
    </ScrollArea>
  );
}

function Group({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="flex flex-col gap-1">
      <h3 className="px-2 text-xs font-medium text-fg-2">{title}</h3>
      {children}
    </section>
  );
}
