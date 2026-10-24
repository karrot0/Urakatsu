// Project sidebar: Project -> Chapters -> Pages, reorderable with dnd-kit.
//
// Chapters and pages share one DndContext. Sortable ids carry their kind ("c:" or "p:")
// so the collision check can keep a dragged chapter among chapters, and let a dragged
// page land on another page or on a chapter (appending to it). Pages move between
// chapters live in onDragOver; the final order is committed once, on drop.

import {
  DndContext,
  DragOverlay,
  KeyboardSensor,
  MeasuringStrategy,
  PointerSensor,
  closestCenter,
  pointerWithin,
  useSensor,
  useSensors,
  type CollisionDetection,
  type DragEndEvent,
  type DragOverEvent,
  type DragStartEvent,
  type UniqueIdentifier,
} from "@dnd-kit/core";
import { SortableContext, arrayMove, sortableKeyboardCoordinates, useSortable, verticalListSortingStrategy } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import {
  ArrowLeft,
  Boxes,
  ChevronRight,
  CircleCheck,
  Download,
  Eye,
  EyeOff,
  FileDown,
  GripVertical,
  ImagePlus,
  Link2,
  LoaderCircle,
  MoreHorizontal,
  Pencil,
  Plus,
  RotateCw,
  Settings,
  ShieldCheck,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type HTMLAttributes, type KeyboardEvent, type MouseEvent as ReactMouseEvent, type PointerEventHandler, type ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { cn } from "cn";
import { assetHref, assetVersion, describeError, exportChapter, exportPage, saveBlob } from "../api/client";
import { chapterNumber, sources, type LinkView, type LinkedChapter } from "../api/sources";
import type { Page, Project } from "../gen/urakatsu/v1/urakatsu_pb";
import { STAGES, chapterOf, pageComplete, selectedPages, stageDone, useWorkspace, type ChapterLayout } from "../state/workspace";
import { AutomationsButton } from "./Automations";
import { MemoryMeter } from "./MemoryMeter";
import { ActionMenu, Tooltip } from "./primitives";
import { ScrollArea } from "@/components/ui/scroll-area";
import { TitleBarInset } from "./WindowControls";

const CHAPTER = "c:";
const PAGE = "p:";
const kindOf = (id: UniqueIdentifier) => (String(id).startsWith(CHAPTER) ? "chapter" : "page");
const rawId = (id: UniqueIdentifier) => String(id).slice(2);

/** dnd-kit types its listener map loosely; the row starts a pointer drag, the grip also handles keys. */
const pointerStart = (listeners: Record<string, Function> | undefined) => listeners?.onPointerDown as PointerEventHandler<HTMLDivElement> | undefined;

const ROW = "relative flex min-h-[30px] items-center gap-0.5 rounded-control pr-0.5 transition-[background-color] duration-[20ms] ease-in";
const GRIP =
  "inline-flex h-[22px] w-4 cursor-grab touch-none items-center justify-center rounded-pill border-0 bg-transparent text-icon-2 transition-opacity duration-100 ease-out active:cursor-grabbing focus-visible:shadow-focus";
const GRIP_HIDDEN = "opacity-0 group-hover:opacity-100 focus-visible:opacity-100";
const DISCLOSURE =
  "inline-flex h-[22px] w-[22px] cursor-pointer items-center justify-center rounded-pill border-0 bg-transparent text-icon-2 transition-[background-color] duration-[20ms] ease-in hover:bg-hover hover:text-icon focus-visible:shadow-focus";
const ACTIONS =
  "absolute top-1/2 flex -translate-y-1/2 items-center gap-0.5 pl-1 opacity-0 shadow-[-8px_0_8px_-4px_var(--surface-sunk)] transition-opacity duration-100 ease-out group-focus-within:opacity-100 group-hover:opacity-100";
const ACTIONS_BG = "[background:linear-gradient(var(--hover),var(--hover)),var(--surface-sunk)]";
const ACTIONS_BG_ACTIVE = "[background:linear-gradient(var(--selected),var(--selected)),var(--surface-sunk)]";
const THUMB = "h-8 w-6 flex-none rounded-pill bg-surface-fill object-cover shadow-[0_0_0_1px_var(--border-soft)]";
const COVER = "h-[46px] w-8 flex-none rounded-pill bg-surface-fill object-cover";

function layoutOf(project: Project): ChapterLayout[] {
  return project.chapters.map((chapter) => ({ id: chapter.id, pageIds: [...chapter.pageIds] }));
}

function sameLayout(a: ChapterLayout[], b: ChapterLayout[]): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

type Entry = { kind: "local"; chapter: ChapterLayout; index: number } | { kind: "remote"; chapter: LinkedChapter };

/**
 * Local chapters in their own order, with the linked source's chapters that are not downloaded
 * yet slotted in by source order: each goes before the first downloaded chapter that follows it.
 */
function mergeChapters(layout: ChapterLayout[], link: LinkView | null | undefined, withRemote: boolean): Entry[] {
  const local: Entry[] = layout.map((chapter, index) => ({ kind: "local", chapter, index }));
  if (!link || !withRemote) return local;
  const position = new Map(link.chapters.map((chapter, index) => [chapter.id, index]));
  const downloaded = new Map(link.chapters.filter((c) => c.chapter_id).map((c) => [c.chapter_id!, position.get(c.id)!]));
  const pending = link.chapters.filter((c) => !c.chapter_id);
  const merged: Entry[] = [];
  for (const entry of local) {
    const at = downloaded.get(entry.chapter.id);
    if (at !== undefined) while (pending.length && position.get(pending[0].id)! < at) merged.push({ kind: "remote", chapter: pending.shift()! });
    merged.push(entry);
  }
  for (const chapter of pending) merged.push({ kind: "remote", chapter });
  return merged;
}

/** Whether the source's undownloaded chapters show in the tree, remembered per project. */
function useShowRemote(projectId: string): [boolean, () => void] {
  const key = `urakatsu.remote.${projectId}`;
  const [show, setShow] = useState(() => {
    try {
      return localStorage.getItem(key) !== "hidden";
    } catch {
      return true;
    }
  });
  const toggle = useCallback(() => {
    setShow((current) => {
      try {
        localStorage.setItem(key, current ? "hidden" : "shown");
      } catch {
        /* storage unavailable */
      }
      return !current;
    });
  }, [key]);
  return [show, toggle];
}

/** Which chapters are open, remembered per project in this browser. */
function useExpanded(projectId: string): [(id: string) => boolean, (id: string, open?: boolean) => void] {
  const key = `urakatsu.expanded.${projectId}`;
  const [state, setState] = useState<Record<string, boolean>>(() => {
    try {
      return JSON.parse(localStorage.getItem(key) ?? "{}");
    } catch {
      return {};
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(state));
    } catch {
      /* storage unavailable */
    }
  }, [key, state]);
  const isOpen = useCallback((id: string) => state[id] ?? true, [state]);
  const toggle = useCallback((id: string, open?: boolean) => setState((s) => ({ ...s, [id]: open ?? !(s[id] ?? true) })), []);
  return [isOpen, toggle];
}

export function Sidebar() {
  const ws = useWorkspace();
  const project = ws.project!;
  const base = useMemo(() => layoutOf(project), [project]);
  const [draft, setDraft] = useState<ChapterLayout[] | null>(null);
  const [active, setActive] = useState<UniqueIdentifier | null>(null);
  const [overChapter, setOverChapter] = useState<string | null>(null);
  const [isOpen, toggle] = useExpanded(project.id);
  const [showRemote, toggleRemote] = useShowRemote(project.id);
  const upload = useRef<HTMLInputElement>(null);
  const uploadTarget = useRef<string | undefined>(undefined);
  const [fileOver, setFileOver] = useState(false);

  const layout = draft ?? base;
  const pages = useMemo(() => new Map(project.pages.map((page) => [page.id, page])), [project.pages]);
  const names = useMemo(() => new Map(project.chapters.map((chapter) => [chapter.id, chapter.name])), [project.chapters]);
  const draggingChapter = active !== null && kindOf(active) === "chapter";
  const busy = ws.job?.status === "running";
  const currentChapter = chapterOf(ws, ws.pageId) ?? project.chapters[project.chapters.length - 1];
  // Source rows are not sortable; they step aside while a chapter is dragged.
  const entries = useMemo(() => mergeChapters(layout, ws.link, showRemote && !draggingChapter), [layout, ws.link, showRemote, draggingChapter]);
  const waiting = ws.link?.chapters.filter((c) => !c.chapter_id).length ?? 0;
  const shared = useMemo(() => {
    const seen = new Set<string>();
    const twice = new Set<string>();
    for (const c of ws.link?.chapters ?? []) (seen.has(c.name) ? twice : seen).add(c.name);
    return twice;
  }, [ws.link]);

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  );

  const collision: CollisionDetection = useCallback((args) => {
    if (kindOf(args.active.id) === "chapter") {
      return closestCenter({ ...args, droppableContainers: args.droppableContainers.filter((c) => kindOf(c.id) === "chapter") });
    }
    const hits = pointerWithin(args);
    const page = hits.find((hit) => kindOf(hit.id) === "page");
    if (page) return [page];
    const chapter = hits.find((hit) => kindOf(hit.id) === "chapter");
    if (chapter) return [chapter];
    return closestCenter({ ...args, droppableContainers: args.droppableContainers.filter((c) => kindOf(c.id) === "page") });
  }, []);

  const onDragStart = ({ active }: DragStartEvent) => {
    setActive(active.id);
    setDraft(layoutOf(project));
  };

  const onDragOver = ({ active, over }: DragOverEvent) => {
    if (!over || kindOf(active.id) !== "page") return;
    const pageId = rawId(active.id);
    const overIsChapter = kindOf(over.id) === "chapter";
    setDraft((prev) => {
      if (!prev) return prev;
      const from = prev.findIndex((c) => c.pageIds.includes(pageId));
      const to = overIsChapter ? prev.findIndex((c) => c.id === rawId(over.id)) : prev.findIndex((c) => c.pageIds.includes(rawId(over.id)));
      if (to >= 0) setOverChapter(prev[to].id);
      if (from < 0 || to < 0 || from === to) return prev;
      const next = prev.map((c) => ({ ...c, pageIds: c.pageIds.filter((id) => id !== pageId) }));
      const target = next[to].pageIds;
      let index = target.length;
      if (!overIsChapter) {
        const translated = active.rect.current.translated;
        const below = translated ? translated.top > over.rect.top + over.rect.height / 2 : false;
        index = target.indexOf(rawId(over.id)) + (below ? 1 : 0);
      }
      target.splice(index, 0, pageId);
      return next;
    });
  };

  const onDragEnd = ({ active, over }: DragEndEvent) => {
    let next = draft;
    if (next && over) {
      if (kindOf(active.id) === "chapter" && kindOf(over.id) === "chapter") {
        const from = next.findIndex((c) => c.id === rawId(active.id));
        const to = next.findIndex((c) => c.id === rawId(over.id));
        if (from !== to) next = arrayMove(next, from, to);
      } else if (kindOf(active.id) === "page" && kindOf(over.id) === "page") {
        const chapter = next.find((c) => c.pageIds.includes(rawId(active.id)));
        if (chapter?.pageIds.includes(rawId(over.id))) {
          const from = chapter.pageIds.indexOf(rawId(active.id));
          const to = chapter.pageIds.indexOf(rawId(over.id));
          if (from !== to) next = next.map((c) => (c === chapter ? { ...c, pageIds: arrayMove(c.pageIds, from, to) } : c));
        }
      }
    }
    setActive(null);
    setOverChapter(null);
    setDraft(null);
    if (next && over && !sameLayout(next, base)) void ws.arrange(next);
  };

  const onDragCancel = () => {
    setActive(null);
    setOverChapter(null);
    setDraft(null);
  };

  const addPagesTo = (chapterId?: string) => {
    uploadTarget.current = chapterId;
    upload.current?.click();
  };

  const createChapter = async () => {
    await ws.createChapter(`Chapter ${project.chapters.length + 1}`);
  };

  return (
    <aside className="flex min-h-0 flex-1 flex-col bg-surface-sunk shadow-[inset_-1px_0_0_var(--border-soft)]" aria-label="Chapters and pages">
      <ProjectHead />
      <SourceBar />

      <div className="flex h-7 items-center justify-between pr-2 pl-3 text-xs text-fg-3">
        <span>Chapters</span>
        <span className="inline-flex gap-0.5">
          {ws.link && waiting > 0 && (
            <Button
              variant="ghost"
              size="icon-xs"
              className="aria-pressed:bg-transparent aria-pressed:text-fg-2 aria-pressed:[&_svg]:text-icon-2"
              title={showRemote ? `Hide chapters not downloaded (${waiting})` : `Show chapters not downloaded (${waiting})`}
              aria-label={showRemote ? "Hide chapters not downloaded" : "Show chapters not downloaded"}
              aria-pressed={showRemote}
              onClick={toggleRemote}
            >
              {showRemote ? <Eye size={14} /> : <EyeOff size={14} />}
            </Button>
          )}
          <Button variant="ghost" size="icon-xs" title="New chapter" aria-label="New chapter" disabled={busy} onClick={() => void createChapter()}>
            <Plus size={14} />
          </Button>
        </span>
      </div>
      <SelectionBar />

      <ScrollArea className="flex-1" contentClassName="px-1 pb-3">
        <DndContext
          sensors={sensors}
          collisionDetection={collision}
          measuring={{ droppable: { strategy: MeasuringStrategy.Always } }}
          onDragStart={onDragStart}
          onDragOver={onDragOver}
          onDragEnd={onDragEnd}
          onDragCancel={onDragCancel}
        >
          <SortableContext items={layout.map((c) => CHAPTER + c.id)} strategy={verticalListSortingStrategy}>
            {entries.map((entry) =>
              entry.kind === "remote" ? (
                <RemoteChapterRow key={`remote:${entry.chapter.id}`} chapter={entry.chapter} showGroup={shared.has(entry.chapter.name)} />
              ) : (
              <ChapterItem
                key={entry.chapter.id}
                chapter={entry.chapter}
                name={names.get(entry.chapter.id) ?? `Chapter ${entry.index + 1}`}
                pages={pages}
                open={isOpen(entry.chapter.id) && !draggingChapter}
                dropTarget={overChapter === entry.chapter.id && !isOpen(entry.chapter.id)}
                disabled={busy}
                onToggle={() => toggle(entry.chapter.id)}
                onOpen={() => toggle(entry.chapter.id, true)}
                onAddPages={() => addPagesTo(entry.chapter.id)}
              />
              ),
            )}
          </SortableContext>
          {entries.length === 0 && (
            <div className="px-2 py-1 text-xs text-fg-3">
              No chapters. <button className="cursor-pointer p-0 text-brand" onClick={() => void createChapter()}>Add chapter</button>
            </div>
          )}
          <DragOverlay dropAnimation={{ duration: 150, easing: "cubic-bezier(.2, 1, .2, 1)" }}>
            {active !== null && kindOf(active) === "chapter" && (
              <div className="cursor-grabbing rounded-control bg-surface-over shadow-overlay">
                <ChapterRowView name={names.get(rawId(active)) ?? ""} count={layout.find((c) => c.id === rawId(active))?.pageIds.length ?? 0} open={false} overlay />
              </div>
            )}
            {active !== null && kindOf(active) === "page" && pages.get(rawId(active)) && (
              <div className="cursor-grabbing rounded-control bg-surface-over shadow-overlay">
                <PageRowView page={pages.get(rawId(active))!} number={0} projectId={project.id} overlay />
              </div>
            )}
          </DragOverlay>
        </DndContext>
      </ScrollArea>

      <div className="flex flex-col gap-1 p-2 shadow-[inset_0_1px_0_var(--border-soft)]">
        <div
          className={cn(
            "flex h-10 cursor-pointer items-center justify-center gap-2 rounded-control border border-dashed text-xs text-fg-3 [transition:background-color_var(--t-hover),color_var(--t-ui)]",
            fileOver ? "border-brand bg-hover text-fg-2" : "border-line-strong hover:bg-hover hover:text-fg-2",
          )}
          role="button"
          tabIndex={0}
          onClick={() => addPagesTo(currentChapter?.id)}
          onKeyDown={(event) => (event.key === "Enter" || event.key === " ") && addPagesTo(currentChapter?.id)}
          onDragOver={(event) => {
            if (!event.dataTransfer.types.includes("Files")) return;
            event.preventDefault();
            setFileOver(true);
          }}
          onDragLeave={() => setFileOver(false)}
          onDrop={(event) => {
            event.preventDefault();
            setFileOver(false);
            void ws.uploadPages(event.dataTransfer.files, currentChapter?.id);
          }}
        >
          <Upload size={14} />
          <span className="truncate">Add pages{currentChapter ? ` to ${currentChapter.name}` : ""}</span>
        </div>
        <input
          ref={upload}
          type="file"
          accept="image/*"
          multiple
          hidden
          onChange={(event) => {
            if (event.target.files) {
              const target = uploadTarget.current;
              if (target) toggle(target, true);
              void ws.uploadPages(event.target.files, target);
            }
            event.target.value = "";
          }}
        />
        <div className="flex items-center gap-1">
          <Tooltip content="Settings" side="right">
            <Button variant="ghost" size="icon-sm" aria-label="Settings" onClick={() => ws.setSettingsOpen(true)}>
              <Settings size={16} />
            </Button>
          </Tooltip>
          <Tooltip content="Models" side="top">
            <Button variant="ghost" size="icon-sm" aria-label="Models" onClick={() => ws.setModelsOpen(true)}>
              <Boxes size={16} />
            </Button>
          </Tooltip>
          <MemoryMeter />
        </div>
      </div>
    </aside>
  );
}

function ProjectHead() {
  const ws = useWorkspace();
  const project = ws.project!;
  const [editing, setEditing] = useState(false);
  return (
    <div className="flex h-12 items-center gap-1 px-2" data-titlebar>
      <TitleBarInset side="left" />
      <Button variant="ghost" size="icon-sm" title="All projects" aria-label="All projects" onClick={ws.closeProject}>
        <ArrowLeft size={16} />
      </Button>
      {editing ? (
        <InlineName
          value={project.name}
          label="Project name"
          onDone={(name) => {
            setEditing(false);
            if (name && name !== project.name) void ws.renameProject(name);
          }}
        />
      ) : (
        <span
          className="min-w-0 flex-1 cursor-text truncate rounded-control px-1 py-0.5 font-semibold hover:bg-hover"
          title="Rename project"
          onDoubleClick={() => setEditing(true)}
        >
          {project.name}
        </span>
      )}
      <AutomationsButton />
    </div>
  );
}

/** The linked series: its cover, title and source, or a way to link one. */
function SourceBar() {
  const ws = useWorkspace();
  const project = ws.project!;
  const link = ws.link;
  const [broken, setBroken] = useState(false);
  if (link === undefined) return null;
  if (link === null) {
    return (
      <div className="mx-1 mb-1 flex items-center gap-2 rounded-card">
        <Button variant="ghost" size="sm" className="w-full justify-start" onClick={() => ws.setSourcesOpen(true)}>
          <Link2 size={14} /> Link a series
        </Button>
      </div>
    );
  }
  const downloaded = link.chapters.filter((c) => c.chapter_id).length;
  const pending = link.chapters.filter((c) => !c.chapter_id && !c.download);
  return (
    <div className="mx-1 mb-1 flex items-center gap-2 rounded-card bg-surface p-1 shadow-raised">
      {broken ? (
        <span className={COVER} />
      ) : (
        <img className={COVER} src={sources.projectCoverUrl(project.id, link.refreshed_at)} alt="" onError={() => setBroken(true)} />
      )}
      <div className="flex min-w-0 flex-1 flex-col">
        <span className="truncate font-medium" title={link.series.title}>
          {link.series.title}
        </span>
        <span className="truncate text-xs tracking-[0.005em] text-fg-3">
          {link.source_name} · <span className="tabular-nums">{downloaded}</span> of <span className="tabular-nums">{link.chapters.length}</span> downloaded
        </span>
      </div>
      <ActionMenu
        trigger={
          <Button variant="ghost" size="icon-xs" aria-label="Series options" title="Series options">
            <MoreHorizontal size={14} />
          </Button>
        }
        actions={[
          { label: "Check for new chapters", onSelect: () => void ws.refreshSource() },
          {
            label: `Download all ${pending.length} remaining`,
            disabled: pending.length === 0,
            onSelect: () => {
              if (window.confirm(`Download ${pending.length} chapters of ${link.series.title}?`)) void ws.downloadChapters(pending.map((c) => c.id));
            },
          },
          { label: "Change series", onSelect: () => ws.setSourcesOpen(true) },
          ...(link.series.url ? [{ label: "Open on the site", onSelect: () => void window.open(link.series.url!, "_blank", "noreferrer") }] : []),
          {
            label: "Unlink series",
            danger: true,
            onSelect: () => {
              if (window.confirm(`Unlink ${link.series.title}? Downloaded chapters stay in the project.`)) void ws.unlinkSource();
            },
          },
        ]}
      />
    </div>
  );
}

/** A chapter that is still on the source: grayed out until downloaded, and not openable. */
function RemoteChapterRow({ chapter, showGroup }: { chapter: LinkedChapter; showGroup: boolean }) {
  const ws = useWorkspace();
  const state = chapter.download;
  const number = chapterNumber(chapter);
  const detail = [number && !chapter.name.includes(String(chapter.number)) ? number : null, chapter.scanlator].filter(Boolean).join(" · ");
  const emphasized = state?.state === "downloading" || state?.state === "importing" || state?.state === "queued";
  return (
    <div className="rounded-control">
      <div className={cn(ROW, "hover:bg-hover", emphasized ? "text-fg-2" : "text-fg-3")} title={detail ? `${chapter.name}\n${detail}` : chapter.name}>
        <span className="w-10 flex-none" aria-hidden />
        <span className="min-w-0 flex-1 truncate px-0.5 font-normal">
          {chapter.name}
          {showGroup && chapter.scanlator && <span className="text-fg-disabled"> · {chapter.scanlator}</span>}
        </span>
        {!state && (
          <Button variant="ghost" size="icon-xs" className="hover:[&_svg]:text-brand" aria-label={`Download ${chapter.name}`} title="Download chapter" onClick={() => void ws.downloadChapters([chapter.id])}>
            <Download size={14} />
          </Button>
        )}
        {state && state.state !== "failed" && (
          <>
            <span
              className="relative h-1 w-10 flex-none overflow-hidden rounded-full bg-surface-fill"
              role="progressbar"
              aria-label={`Downloading ${chapter.name}`}
              aria-valuemin={0}
              aria-valuemax={state.total || 1}
              aria-valuenow={state.done}
            >
              <span
                className="absolute inset-y-0 left-0 rounded-[inherit] bg-brand transition-[width] duration-200 ease-settle"
                style={{ width: state.total ? `${(100 * state.done) / state.total}%` : "0%" }}
              />
            </span>
            <span className="px-1 text-xs tabular-nums text-fg-3">{state.state === "queued" ? "Queued" : state.state === "importing" ? "Adding" : `${state.done}/${state.total || "?"}`}</span>
            {state.state !== "importing" && (
              <Button variant="ghost" size="icon-xs" aria-label={`Cancel ${chapter.name}`} title="Cancel download" onClick={() => void ws.cancelDownload(chapter.id)}>
                <X size={14} />
              </Button>
            )}
          </>
        )}
        {state?.state === "failed" && state.challenge && (
          <Button
            variant="ghost"
            size="xs"
            className="gap-0.5 px-1 text-sm text-warning hover:text-warning"
            title={`${new URL(state.challenge).host} wants a quick check. Open the browser to continue.`}
            onClick={() => ws.openBrowser({ system: ws.link!.system, url: state.challenge! }, () => void ws.downloadChapters([chapter.id]))}
          >
            <ShieldCheck size={14} /> Check
          </Button>
        )}
        {state?.state === "failed" && !state.challenge && (
          <>
            <span className="px-1 text-xs tabular-nums text-danger" title={state.error ?? undefined}>
              Failed
            </span>
            <Button
              variant="ghost"
              size="icon-xs"
              aria-label={`Retry ${chapter.name}`}
              title={state.error ? `Retry. ${state.error}` : "Retry"}
              onClick={() => void ws.downloadChapters([chapter.id])}
            >
              <RotateCw size={14} />
            </Button>
          </>
        )}
      </div>
    </div>
  );
}

function InlineName({ value, label, onDone }: { value: string; label: string; onDone: (value: string | null) => void }) {
  const [text, setText] = useState(value);
  return (
    <input
      className="h-6 w-full rounded-pill border-0 bg-surface px-1 shadow-focus outline-none"
      aria-label={label}
      autoFocus
      value={text}
      onChange={(event) => setText(event.target.value)}
      onFocus={(event) => event.target.select()}
      onBlur={() => onDone(text.trim() || null)}
      onKeyDown={(event) => {
        if (event.key === "Enter") onDone(text.trim() || null);
        if (event.key === "Escape") onDone(null);
        event.stopPropagation();
      }}
      onPointerDown={(event) => event.stopPropagation()}
    />
  );
}

function ChapterItem({
  chapter,
  name,
  pages,
  open,
  dropTarget,
  disabled,
  onToggle,
  onOpen,
  onAddPages,
}: {
  chapter: ChapterLayout;
  name: string;
  pages: Map<string, Page>;
  open: boolean;
  dropTarget: boolean;
  disabled: boolean;
  onToggle: () => void;
  onOpen: () => void;
  onAddPages: () => void;
}) {
  const ws = useWorkspace();
  const [editing, setEditing] = useState(false);
  const [fileOver, setFileOver] = useState(false);
  const sortable = useSortable({ id: CHAPTER + chapter.id, data: { type: "chapter" }, disabled: editing || disabled });
  const style = { transform: CSS.Translate.toString(sortable.transform), transition: sortable.transition };
  const items = chapter.pageIds.filter((id) => pages.has(id));
  const complete = items.length > 0 && items.every((id) => pageComplete(pages.get(id)!));

  const [exporting, setExporting] = useState<string>();
  /** Renders every page with its translation and saves the archive; large chapters take a few seconds. */
  const exportAs = async (format: "cbz" | "zip") => {
    if (!ws.project) return;
    setExporting(format);
    try {
      const { blob, filename } = await exportChapter(ws.project.id, chapter.id, format);
      saveBlob(blob, filename);
    } catch (error) {
      ws.reportError(describeError(error));
    } finally {
      setExporting(undefined);
    }
  };

  const remove = () => {
    const count = items.length;
    const detail = count ? ` and its ${count} page${count === 1 ? "" : "s"}` : "";
    if (window.confirm(`Delete ${name}${detail}? You can undo this.`)) void ws.deleteChapter(chapter.id);
  };

  return (
    <div
      ref={sortable.setNodeRef}
      style={style}
      className={cn("rounded-control", sortable.isDragging && "opacity-40")}
      onDragOver={(event) => {
        if (!event.dataTransfer.types.includes("Files")) return;
        event.preventDefault();
        setFileOver(true);
      }}
      onDragLeave={() => setFileOver(false)}
      onDrop={(event) => {
        if (!event.dataTransfer.files.length) return;
        event.preventDefault();
        setFileOver(false);
        onOpen();
        void ws.uploadPages(event.dataTransfer.files, chapter.id);
      }}
    >
      <ChapterRowView
        name={name}
        count={items.length}
        open={open}
        complete={complete}
        dropTarget={dropTarget || fileOver}
        onToggle={onToggle}
        rowProps={{ onPointerDown: pointerStart(sortable.listeners), onDoubleClick: () => setEditing(true) }}
        gripProps={{ ...sortable.attributes, ...sortable.listeners, ref: sortable.setActivatorNodeRef }}
        editor={
          editing ? (
            <InlineName
              value={name}
              label="Chapter name"
              onDone={(next) => {
                setEditing(false);
                if (next && next !== name) void ws.renameChapter(chapter.id, next);
              }}
            />
          ) : undefined
        }
        actions={
          <>
            <Button variant="ghost" size="icon-xs" title="Add pages" aria-label={`Add pages to ${name}`} onClick={onAddPages}>
              <ImagePlus size={14} />
            </Button>
            {items.length > 0 && (
              <ActionMenu
                trigger={
                  <Button variant="ghost" size="icon-xs" title={exporting ? "Exporting" : "Export chapter"} aria-label={`Export ${name}`} disabled={!!exporting}>
                    {exporting ? <LoaderCircle size={14} className="animate-spin motion-reduce:animate-none" /> : <FileDown size={14} />}
                  </Button>
                }
                actions={[
                  { label: "Comic book (.cbz)", onSelect: () => void exportAs("cbz") },
                  { label: "Images (.zip)", onSelect: () => void exportAs("zip") },
                ]}
              />
            )}
            <Button variant="ghost" size="icon-xs" title="Rename" aria-label={`Rename ${name}`} onClick={() => setEditing(true)}>
              <Pencil size={14} />
            </Button>
            <Button variant="ghost" size="icon-xs" title="Delete chapter" aria-label={`Delete ${name}`} disabled={disabled} onClick={remove}>
              <Trash2 size={14} />
            </Button>
          </>
        }
      />
      <div
        className={cn(
          "grid",
          open
            ? "grid-rows-[1fr] opacity-100 [transition:grid-template-rows_200ms_var(--ease-settle),opacity_var(--t-in)]"
            : "grid-rows-[0fr] opacity-0 [transition:grid-template-rows_200ms_var(--ease-settle),opacity_var(--t-out)]",
        )}
        aria-hidden={!open}
      >
        <div className="min-h-0 overflow-hidden">
          <SortableContext items={items.map((id) => PAGE + id)} strategy={verticalListSortingStrategy}>
            <div className="flex flex-col gap-0.5 pt-0.5 pb-1 pl-4" role="list">
              {items.map((id, index) => (
                <PageItem key={id} page={pages.get(id)!} number={index + 1} disabled={disabled} />
              ))}
              {items.length === 0 && (
                <div className="px-2 py-1 text-xs text-fg-3">
                  No pages in this chapter. <button className="cursor-pointer p-0 text-brand" onClick={onAddPages}>Add pages</button>
                </div>
              )}
            </div>
          </SortableContext>
        </div>
      </div>
    </div>
  );
}

type GripProps = HTMLAttributes<HTMLButtonElement> & { ref?: (element: HTMLElement | null) => void };

function ChapterRowView({
  name,
  count,
  open,
  complete,
  dropTarget,
  overlay,
  onToggle,
  rowProps,
  gripProps,
  editor,
  actions,
}: {
  name: string;
  count: number;
  open: boolean;
  complete?: boolean;
  dropTarget?: boolean;
  overlay?: boolean;
  onToggle?: () => void;
  rowProps?: HTMLAttributes<HTMLDivElement>;
  gripProps?: GripProps;
  editor?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className={cn(ROW, "group", dropTarget ? "bg-selected" : "hover:bg-hover")} {...rowProps}>
      <button className={cn(GRIP, overlay ? "opacity-100" : GRIP_HIDDEN)} aria-label={`Reorder ${name}`} {...gripProps}>
        <GripVertical size={12} />
      </button>
      <button className={DISCLOSURE} aria-expanded={open} aria-label={open ? `Collapse ${name}` : `Expand ${name}`} onClick={onToggle}>
        <ChevronRight size={14} className={cn("transition-transform duration-150 ease-settle", open && "rotate-90")} />
      </button>
      {editor ?? (
        <span className="min-w-0 flex-1 truncate px-0.5 font-medium" title={name}>
          {name}
        </span>
      )}
      {!editor && actions && <span className={cn(ACTIONS, "right-7", ACTIONS_BG)}>{actions}</span>}
      {complete && <CircleCheck size={14} className="text-success" aria-label="All pages translated" />}
      <span className="px-1 text-xs tabular-nums text-fg-3" aria-label={`${count} pages`}>
        {count}
      </span>
    </div>
  );
}

/** "4 pages selected" with a way out, while a Shift/Ctrl selection is active. */
function SelectionBar() {
  const ws = useWorkspace();
  const pages = selectedPages(ws);
  const active = pages.length > 1;
  const clear = () => ws.pageId && ws.selectPage(ws.pageId);
  // Esc clears it, unless the key is for a field being typed in.
  useEffect(() => {
    if (!active) return;
    const onKey = (event: globalThis.KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (event.key === "Escape" && !target?.closest("input, textarea, [contenteditable]")) clear();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, ws.pageId]);
  if (!active) return null;
  return (
    <div className="mx-2 mb-1 flex h-7 items-center gap-2 rounded-control bg-brand-weak pr-0.5 pl-2 text-xs text-fg" role="status">
      <span className="min-w-0 flex-1 truncate">
        <span className="font-medium tabular-nums">{pages.length}</span> pages selected
      </span>
      <Button variant="ghost" size="icon-xs" title="Clear selection (Esc)" aria-label="Clear page selection" onClick={clear}>
        <X size={14} />
      </Button>
    </div>
  );
}

function PageItem({ page, number, disabled }: { page: Page; number: number; disabled: boolean }) {
  const ws = useWorkspace();
  const sortable = useSortable({ id: PAGE + page.id, data: { type: "page" }, disabled });
  const style = { transform: CSS.Translate.toString(sortable.transform), transition: sortable.transition };
  // Shift selects a run of pages, Ctrl (Cmd on a Mac) adds or removes one, both add a run.
  const select = (event?: { shiftKey: boolean; ctrlKey: boolean; metaKey: boolean }) => {
    const add = !!event && (event.ctrlKey || event.metaKey);
    ws.selectPage(page.id, event?.shiftKey ? (add ? "add-range" : "range") : add ? "toggle" : "replace");
  };
  const inSelection = selectedPages(ws).includes(page.id);
  const [exporting, setExporting] = useState(false);
  return (
    <div ref={sortable.setNodeRef} style={style} role="listitem">
      <PageRowView
        page={page}
        number={number}
        projectId={ws.project!.id}
        active={ws.pageId === page.id}
        selected={inSelection}
        dragging={sortable.isDragging}
        activeStage={ws.job?.status === "running" && ws.job.current?.page === page.id ? ws.job.current.stage : undefined}
        rowProps={{
          onPointerDown: pointerStart(sortable.listeners),
          // Shift-click would otherwise select the text between the rows.
          onMouseDown: (event: ReactMouseEvent<HTMLDivElement>) => event.shiftKey && event.preventDefault(),
          onClick: (event: ReactMouseEvent<HTMLDivElement>) => select(event),
          onKeyDown: (event: KeyboardEvent<HTMLDivElement>) => {
            if (event.target === event.currentTarget && (event.key === "Enter" || event.key === " ")) {
              event.preventDefault();
              select();
            }
          },
          tabIndex: 0,
          role: "button",
          "aria-current": ws.pageId === page.id ? "page" : undefined,
        }}
        gripProps={{ ...sortable.attributes, ...sortable.listeners, ref: sortable.setActivatorNodeRef }}
        actions={
          <>
          <Button
            variant="ghost"
            size="icon-xs"
            title={exporting ? "Exporting" : "Export page as PNG"}
            aria-label={`Export ${page.name}`}
            disabled={exporting}
            onClick={async (event) => {
              event.stopPropagation();
              setExporting(true);
              try {
                await exportPage(ws.project!.id, page);
              } catch (error) {
                ws.reportError(describeError(error));
              } finally {
                setExporting(false);
              }
            }}
          >
            {exporting ? <LoaderCircle size={14} className="animate-spin motion-reduce:animate-none" /> : <FileDown size={14} />}
          </Button>
          <Button
            variant="ghost"
            size="icon-xs"
            title="Remove page"
            aria-label={`Remove ${page.name}`}
            disabled={disabled}
            onClick={(event) => {
              event.stopPropagation();
              if (window.confirm(`Remove ${page.name}? You can undo this.`)) void ws.removePage(page.id);
            }}
          >
            <X size={14} />
          </Button>
          </>
        }
      />
    </div>
  );
}

function PageRowView({
  page,
  number,
  projectId,
  active,
  selected,
  dragging,
  overlay,
  activeStage,
  rowProps,
  gripProps,
  actions,
}: {
  page: Page;
  number: number;
  projectId: string;
  active?: boolean;
  /** In a sidebar multi-selection (the current page is `active` as well). */
  selected?: boolean;
  dragging?: boolean;
  overlay?: boolean;
  activeStage?: string;
  rowProps?: HTMLAttributes<HTMLDivElement>;
  gripProps?: GripProps;
  actions?: ReactNode;
}) {
  // A lazy <img> by URL: only on-screen rows load, after the canvas image, and the browser caches them.
  const thumb = assetHref(projectId, page.id, "thumbnail", assetVersion(page, "thumbnail"));

  return (
    <div
      className={cn(ROW, "group cursor-pointer pl-0", active ? "bg-selected" : selected ? "bg-[color-mix(in_oklab,var(--selected)_55%,transparent)] hover:bg-selected" : "hover:bg-hover", dragging && "opacity-40")}
      data-selected={selected || undefined}
      {...rowProps}
    >
      <button className={cn(GRIP, overlay ? "opacity-100" : GRIP_HIDDEN)} aria-label={`Reorder ${page.name}`} onClick={(event) => event.stopPropagation()} {...gripProps}>
        <GripVertical size={12} />
      </button>
      <img className={THUMB} src={thumb} alt="" loading="lazy" decoding="async" draggable={false} />
      <span className="flex min-w-0 flex-1 flex-col py-0.5">
        <span className="truncate" title={page.name}>
          {page.name}
        </span>
        {number > 0 && <span className="text-xs tabular-nums text-fg-3">Page {number}</span>}
      </span>
      {actions && <span className={cn(ACTIONS, "right-11", active ? ACTIONS_BG_ACTIVE : ACTIONS_BG)}>{actions}</span>}
      <span className="inline-flex gap-[3px] px-1" title={STAGES.map((s) => `${s}: ${stageDone(page, s) ? "done" : "pending"}`).join("\n")}>
        {STAGES.map((stage) => (
          <i
            key={stage}
            className={cn(
              "h-1.5 w-1.5 rounded-full",
              activeStage === stage ? "animate-[soft-pulse_1s_ease-in-out_infinite_alternate] bg-brand" : stageDone(page, stage) ? "bg-success" : "bg-line-strong",
            )}
          />
        ))}
      </span>
    </div>
  );
}
