// The phone app: a tab bar over four sections, and in the library a navigation stack of
// project, chapter and page editor. Desktop keeps App's own layout; this replaces it only on
// phone widths (useMobile).

import { BookOpen, Boxes, Compass, Download, FileDown, FolderPlus, ImagePlus, Link2, Pencil, Play, Plus, RotateCw, Settings, ShieldCheck, Sparkles, Square, Trash2, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { cn } from "cn";

import { assetHref, assetVersion, describeError, exportPage } from "../../api/client";
import { sources, type LinkedChapter } from "../../api/sources";
import { plural } from "../../format";
import type { Page } from "../../gen/urakatsu/v1/urakatsu_pb";
import { STAGES, pageComplete, stageDone, useWorkspace } from "../../state/workspace";
import { ModelsPage } from "../ModelsPage";
import { SourcesPage } from "../SourcesPage";
import { ProjectCover } from "../Start";
import { ActionList, GlassButton, ListRow, ListSection, Screen, SearchField, Sheet, TabBar, useLongPress, type Enter, type TabItem } from "./glass";
import { MobileEditor } from "./MobileEditor";
import { MobileSettings } from "./MobileSettings";

type Tab = "library" | "browse" | "models" | "settings";

const TABS: TabItem<Tab>[] = [
  { id: "library", label: "Library", icon: BookOpen },
  { id: "browse", label: "Browse", icon: Compass },
  { id: "models", label: "Models", icon: Boxes },
  { id: "settings", label: "Settings", icon: Settings },
];

export function MobileApp() {
  const ws = useWorkspace();
  const tab: Tab = ws.settingsOpen ? "settings" : ws.modelsOpen ? "models" : ws.sourcesOpen ? "browse" : "library";
  // The library's stack below the project: a chapter, then the editor on one of its pages.
  const [chapterId, setChapterId] = useState<string>();
  const [editing, setEditing] = useState(false);
  const [enter, setEnter] = useState<Enter>("none");
  const projectId = ws.project?.id;
  useEffect(() => {
    setChapterId(undefined);
    setEditing(false);
  }, [projectId]);

  const select = (next: Tab) => {
    setEnter("none");
    if (next === tab) {
      // Tapping the current tab again goes to its root, as on iOS.
      if (next === "library") {
        if (editing) setEditing(false);
        else if (chapterId) setChapterId(undefined);
        else if (ws.project) ws.closeProject();
      }
      window.scrollTo({ top: 0, behavior: "smooth" });
      return;
    }
    ws.setSourcesOpen(false);
    if (next === "browse") ws.setSourcesOpen(true);
    if (next === "models") ws.setModelsOpen(true);
    if (next === "settings") ws.setSettingsOpen(true);
  };
  const push = (step: () => void) => {
    setEnter("push");
    step();
  };
  const pop = (step: () => void) => (swiped?: boolean) => {
    setEnter(swiped ? "none" : "pop");
    step();
  };

  const chapter = chapterId ? ws.project?.chapters.find((c) => c.id === chapterId) : undefined;
  const inEditor = tab === "library" && !!ws.project && editing && !!ws.pageId;

  return (
    <div className="min-h-dvh bg-grouped text-fg">
      {tab === "settings" ? (
        <MobileSettings />
      ) : tab === "models" ? (
        <ModelsPage />
      ) : tab === "browse" ? (
        <SourcesPage />
      ) : inEditor ? (
        <MobileEditor
          onBack={() => {
            setEnter("pop");
            setChapterId((current) => current ?? ws.project?.chapters.find((c) => c.pageIds.includes(ws.pageId!))?.id);
            setEditing(false);
          }}
        />
      ) : ws.project && chapter ? (
        <ChapterScreen
          key={chapter.id}
          chapterId={chapter.id}
          enter={enter}
          onBack={pop(() => setChapterId(undefined))}
          onOpenPage={(pageId) =>
            push(() => {
              ws.selectPage(pageId);
              setEditing(true);
            })
          }
        />
      ) : ws.project ? (
        <ProjectScreen key={ws.project.id} enter={enter} onBack={pop(ws.closeProject)} onOpenChapter={(id) => push(() => setChapterId(id))} />
      ) : (
        <Library enter={enter} onOpen={(id) => push(() => void ws.openProject(id))} />
      )}
      {!inEditor && <TabBar items={TABS} active={tab} onSelect={select} />}
    </div>
  );
}

// ---- library ------------------------------------------------------------------------------------

function Library({ enter, onOpen }: { enter: Enter; onOpen: (id: string) => void }) {
  const ws = useWorkspace();
  const [query, setQuery] = useState("");
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [menu, setMenu] = useState<string>();
  const [confirming, setConfirming] = useState(false);
  const shown = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    const matching = needle ? ws.projects.filter((p) => p.name.toLocaleLowerCase().includes(needle)) : ws.projects;
    return [...matching].sort((a, b) => Number(b.updatedAt) - Number(a.updatedAt));
  }, [ws.projects, query]);
  const target = ws.projects.find((p) => p.id === menu);

  const create = () => {
    const value = name.trim() || "Untitled";
    setName("");
    setCreating(false);
    void ws.createProject(value);
  };

  return (
    <Screen
      title="Library"
      enter={enter}
      actions={
        <GlassButton label="New project" onClick={() => setCreating(true)}>
          <Plus size={22} strokeWidth={2.25} />
        </GlassButton>
      }
      accessory={ws.projects.length > 0 && <SearchField value={query} onChange={setQuery} placeholder="Search projects" />}
    >
      {ws.ready && ws.projects.length === 0 && (
        <Empty title="No projects yet." detail="Create one from your own pages, or follow a series from a source.">
          <GlassButton label="New project" shape="capsule" tone="prominent" onClick={() => setCreating(true)}>
            <Plus size={18} /> New project
          </GlassButton>
        </Empty>
      )}
      {ws.projects.length > 0 && shown.length === 0 && <Empty title={`No projects match “${query.trim()}”.`} />}
      <ul className="grid grid-cols-2 gap-x-4 gap-y-5 pt-1 min-[420px]:grid-cols-3" aria-label="Projects">
        {shown.map((project) => (
          <ProjectTile key={project.id} id={project.id} name={project.name} version={project.updatedAt} detail={plural(project.pages, "page")} onOpen={() => onOpen(project.id)} onMenu={() => setMenu(project.id)} />
        ))}
      </ul>

      <Sheet open={creating} onOpenChange={setCreating} title="New project" detents="fit">
        <form
          className="flex flex-col gap-3 pt-1"
          onSubmit={(event) => {
            event.preventDefault();
            create();
          }}
        >
          <input
            className="h-[50px] w-full rounded-group bg-grouped-cell px-4 text-body-lg text-fg outline-none placeholder:text-fg-3 focus-visible:shadow-focus"
            aria-label="Project name"
            placeholder="Project name"
            enterKeyHint="done"
            value={name}
            onChange={(event) => setName(event.target.value)}
          />
          <GlassButton label="Create project" shape="capsule" tone="prominent" type="submit" className="h-[50px] w-full">
            Create project
          </GlassButton>
          <ActionList
            actions={[
              {
                label: "Follow a series from a source",
                icon: Link2,
                onSelect: () => {
                  setCreating(false);
                  ws.setSourcesOpen(true);
                },
              },
            ]}
          />
        </form>
      </Sheet>

      <Sheet
        open={!!target}
        onOpenChange={(open) => {
          if (!open) {
            setMenu(undefined);
            setConfirming(false);
          }
        }}
        title={target?.name ?? ""}
        detents="fit"
      >
        {target && !confirming && (
          <div className="flex flex-col gap-3 pt-1">
            <ActionList actions={[{ label: "Open", icon: BookOpen, onSelect: () => (setMenu(undefined), onOpen(target.id)) }]} />
            <ActionList actions={[{ label: "Delete project", icon: Trash2, destructive: true, onSelect: () => setConfirming(true) }]} />
          </div>
        )}
        {target && confirming && (
          <div className="flex flex-col gap-3 pt-1">
            <p className="px-1 text-center text-[13px] leading-[18px] text-fg-2">This removes its chapters, pages and renders. It cannot be undone.</p>
            <ActionList
              actions={[
                {
                  label: `Delete ${target.name}`,
                  destructive: true,
                  onSelect: () => {
                    void ws.deleteProject(target.id);
                    setMenu(undefined);
                    setConfirming(false);
                  },
                },
              ]}
            />
          </div>
        )}
      </Sheet>
    </Screen>
  );
}

function ProjectTile({ id, name, version, detail, onOpen, onMenu }: { id: string; name: string; version: number | bigint; detail: string; onOpen: () => void; onMenu: () => void }) {
  const press = useLongPress(onMenu);
  return (
    <li className="min-w-0">
      <button
        type="button"
        className="flex w-full cursor-pointer flex-col gap-1 text-left outline-none select-none focus-visible:rounded-card focus-visible:shadow-focus active:opacity-70 [-webkit-tap-highlight-color:transparent] [-webkit-touch-callout:none]"
        onClick={onOpen}
        {...press}
      >
        <ProjectCover id={id} version={version} name={name} />
        <span className="line-clamp-2 text-[15px] leading-5 font-medium [overflow-wrap:anywhere]">{name}</span>
        <span className="truncate text-[13px] leading-[18px] text-fg-2 tabular-nums">{detail}</span>
      </button>
    </li>
  );
}

function Empty({ title, detail, children }: { title: string; detail?: string; children?: React.ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-1 px-6 py-16 text-center">
      <p className="text-body-lg font-semibold">{title}</p>
      {detail && <p className="max-w-[34ch] text-[15px] leading-5 text-fg-2">{detail}</p>}
      {children && <div className="mt-4">{children}</div>}
    </div>
  );
}

// ---- project ------------------------------------------------------------------------------------

const REMOTE_PAGE = 40;

function ProjectScreen({ enter, onBack, onOpenChapter }: { enter: Enter; onBack: (swiped?: boolean) => void; onOpenChapter: (id: string) => void }) {
  const ws = useWorkspace();
  const project = ws.project!;
  const [adding, setAdding] = useState(false);
  const [renaming, setRenaming] = useState<{ id?: string; value: string }>();
  const [menu, setMenu] = useState<string>();
  const [remoteShown, setRemoteShown] = useState(REMOTE_PAGE);
  const upload = useRef<HTMLInputElement>(null);
  const pages = useMemo(() => new Map(project.pages.map((page) => [page.id, page])), [project.pages]);
  const remote = ws.link?.chapters.filter((c) => !c.chapter_id) ?? [];
  const busy = ws.job?.status === "running";
  const target = project.chapters.find((c) => c.id === menu);
  const last = project.chapters[project.chapters.length - 1];

  return (
    <Screen
      title={project.name}
      enter={enter}
      back={{ label: "Library", onBack }}
      actions={
        <GlassButton label="Add" onClick={() => setAdding(true)}>
          <Plus size={22} strokeWidth={2.25} />
        </GlassButton>
      }
    >
      {ws.link && (
        <div className="mb-2 flex items-center gap-3 rounded-group bg-grouped-cell p-3">
          {ws.link.series.cover && (
            <img className="h-[72px] w-12 flex-none rounded-control bg-surface-fill object-cover" src={sources.coverUrl(ws.link.system, ws.link.source, ws.link.series.cover)} alt="" />
          )}
          <div className="min-w-0 flex-1">
            <p className="line-clamp-2 text-[15px] leading-5 font-semibold">{ws.link.series.title}</p>
            <p className="truncate text-[13px] leading-[18px] text-fg-2">
              {ws.link.source_name} · {ws.link.system_name}
            </p>
          </div>
          <GlassButton label="Check for new chapters" onClick={() => void ws.refreshSource()}>
            <RotateCw size={18} />
          </GlassButton>
        </div>
      )}

      {project.chapters.length > 0 ? (
        <ListSection header="Chapters">
          {project.chapters.map((chapter) => (
            <ChapterRow
              key={chapter.id}
              name={chapter.name}
              pages={chapter.pageIds.map((id) => pages.get(id)).filter((p): p is Page => !!p)}
              projectId={project.id}
              onOpen={() => onOpenChapter(chapter.id)}
              onMenu={() => setMenu(chapter.id)}
            />
          ))}
        </ListSection>
      ) : (
        <Empty title="No chapters." detail="Add pages from your photos or files, or link a series to download chapters.">
          <GlassButton label="Add pages" shape="capsule" tone="prominent" disabled={busy} onClick={() => upload.current?.click()}>
            <ImagePlus size={18} /> Add pages
          </GlassButton>
        </Empty>
      )}

      {remote.length > 0 && (
        <ListSection header="Not downloaded" footer={remote.length > remoteShown ? undefined : `${plural(remote.length, "chapter")} on ${ws.link?.source_name}.`}>
          {remote.slice(0, remoteShown).map((chapter) => (
            <RemoteRow key={chapter.id} chapter={chapter} />
          ))}
          {remote.length > remoteShown && <ListRow title={`Show ${Math.min(REMOTE_PAGE, remote.length - remoteShown)} more`} className="[&_span]:text-brand" onClick={() => setRemoteShown((n) => n + REMOTE_PAGE)} />}
        </ListSection>
      )}

      <input
        ref={upload}
        type="file"
        accept="image/*"
        multiple
        hidden
        onChange={(event) => {
          if (event.target.files) void ws.uploadPages(event.target.files, last?.id);
          event.target.value = "";
        }}
      />

      <Sheet open={adding} onOpenChange={setAdding} title="Add" detents="fit">
        <div className="flex flex-col gap-3 pt-1">
          <ActionList
            actions={[
              { label: last ? `Add pages to ${last.name}` : "Add pages", icon: ImagePlus, disabled: busy, onSelect: () => (setAdding(false), upload.current?.click()) },
              { label: "New chapter", icon: FolderPlus, disabled: busy, onSelect: () => (setAdding(false), void ws.createChapter(`Chapter ${project.chapters.length + 1}`)) },
            ]}
          />
          <ActionList
            actions={[
              { label: ws.link ? "Link another series" : "Link a series", icon: Link2, onSelect: () => (setAdding(false), ws.setSourcesOpen(true)) },
              { label: "Rename project", icon: Pencil, onSelect: () => (setAdding(false), setRenaming({ value: project.name })) },
            ]}
          />
        </div>
      </Sheet>

      <Sheet open={!!target} onOpenChange={(open) => !open && setMenu(undefined)} title={target?.name ?? ""} detents="fit">
        {target && (
          <div className="flex flex-col gap-3 pt-1">
            <ActionList
              actions={[
                { label: "Open", icon: BookOpen, onSelect: () => (setMenu(undefined), onOpenChapter(target.id)) },
                { label: "Rename chapter", icon: Pencil, onSelect: () => (setMenu(undefined), setRenaming({ id: target.id, value: target.name })) },
              ]}
            />
            <ActionList
              actions={[
                {
                  label: "Delete chapter",
                  icon: Trash2,
                  destructive: true,
                  disabled: busy,
                  onSelect: () => {
                    setMenu(undefined);
                    if (window.confirm(`Delete ${target.name} and its ${plural(target.pageIds.length, "page")}?`)) void ws.deleteChapter(target.id);
                  },
                },
              ]}
            />
          </div>
        )}
      </Sheet>

      <Sheet open={!!renaming} onOpenChange={(open) => !open && setRenaming(undefined)} title={renaming?.id ? "Rename chapter" : "Rename project"} detents="fit">
        {renaming && (
          <form
            className="flex flex-col gap-3 pt-1"
            onSubmit={(event) => {
              event.preventDefault();
              const value = renaming.value.trim();
              if (value) void (renaming.id ? ws.renameChapter(renaming.id, value) : ws.renameProject(value));
              setRenaming(undefined);
            }}
          >
            <input
              className="h-[50px] w-full rounded-group bg-grouped-cell px-4 text-body-lg text-fg outline-none focus-visible:shadow-focus"
              aria-label="Name"
              enterKeyHint="done"
              value={renaming.value}
              onChange={(event) => setRenaming({ ...renaming, value: event.target.value })}
            />
            <GlassButton label="Save" shape="capsule" tone="prominent" type="submit" className="h-[50px] w-full">
              Save
            </GlassButton>
          </form>
        )}
      </Sheet>
    </Screen>
  );
}

function ChapterRow({ name, pages, projectId, onOpen, onMenu }: { name: string; pages: Page[]; projectId: string; onOpen: () => void; onMenu: () => void }) {
  const press = useLongPress(onMenu);
  const first = pages[0];
  const done = pages.filter(pageComplete).length;
  return (
    <ListRow
      leading={
        first ? (
          <img className="h-12 w-9 flex-none rounded-[5px] bg-surface-fill object-cover shadow-[0_0_0_0.5px_var(--border)]" src={assetHref(projectId, first.id, "thumbnail", assetVersion(first, "thumbnail"))} alt="" loading="lazy" draggable={false} />
        ) : (
          <span className="h-12 w-9 flex-none rounded-[5px] bg-surface-fill" aria-hidden />
        )
      }
      title={name}
      subtitle={pages.length === 0 ? "No pages" : done === pages.length ? `${plural(pages.length, "page")}, translated` : `${plural(pages.length, "page")}, ${done} translated`}
      chevron
      onClick={onOpen}
      {...press}
    />
  );
}

function RemoteRow({ chapter }: { chapter: LinkedChapter }) {
  const ws = useWorkspace();
  const state = chapter.download;
  const running = state && state.state !== "failed";
  return (
    <ListRow
      title={chapter.name}
      subtitle={
        running
          ? state.state === "queued"
            ? "Queued"
            : state.state === "importing"
              ? "Adding"
              : state.state === "stitching"
                ? "Stitching"
                : `Downloading ${state.done} of ${state.total || "?"}`
          : state?.state === "failed"
            ? state.challenge
              ? `${new URL(state.challenge).host} wants a quick check`
              : "Download failed"
            : chapter.scanlator || undefined
      }
      trailing={
        !state ? (
          <RowIcon label={`Download ${chapter.name}`} onClick={() => void ws.downloadChapters([chapter.id])}>
            <Download size={20} />
          </RowIcon>
        ) : running ? (
          state.state !== "importing" && state.state !== "stitching" ? (
            <RowIcon label={`Cancel ${chapter.name}`} onClick={() => void ws.cancelDownload(chapter.id)}>
              <ProgressRing value={state.total ? state.done / state.total : 0} />
            </RowIcon>
          ) : (
            <ProgressRing value={state.total ? state.done / state.total : 0} />
          )
        ) : state.challenge ? (
          <RowIcon label="Open the site check" onClick={() => ws.openBrowser({ system: ws.link!.system, url: state.challenge! }, () => void ws.downloadChapters([chapter.id]))}>
            <ShieldCheck size={20} className="text-warning" />
          </RowIcon>
        ) : (
          <RowIcon label={`Retry ${chapter.name}`} onClick={() => void ws.downloadChapters([chapter.id])}>
            <RotateCw size={20} className="text-danger" />
          </RowIcon>
        )
      }
    />
  );
}

function RowIcon({ label, onClick, children }: { label: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className="-my-2 -mr-2 grid size-touch flex-none cursor-pointer place-items-center rounded-full text-brand outline-none focus-visible:shadow-focus active:bg-hover [-webkit-tap-highlight-color:transparent]"
      onClick={onClick}
    >
      {children}
    </button>
  );
}

/** A circular progress mark with a stop square, as App Store downloads show. */
function ProgressRing({ value }: { value: number }) {
  const r = 10;
  const length = 2 * Math.PI * r;
  return (
    <svg width="26" height="26" viewBox="0 0 26 26" className="text-brand" role="progressbar" aria-valuenow={Math.round(value * 100)} aria-valuemin={0} aria-valuemax={100}>
      <circle cx="13" cy="13" r={r} fill="none" stroke="var(--border-strong)" strokeWidth="2.5" />
      <circle cx="13" cy="13" r={r} fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeDasharray={length} strokeDashoffset={length * (1 - value)} transform="rotate(-90 13 13)" className="transition-[stroke-dashoffset] duration-200" />
      <rect x="10" y="10" width="6" height="6" rx="1" fill="currentColor" />
    </svg>
  );
}

// ---- chapter ------------------------------------------------------------------------------------

function ChapterScreen({ chapterId, enter, onBack, onOpenPage }: { chapterId: string; enter: Enter; onBack: (swiped?: boolean) => void; onOpenPage: (pageId: string) => void }) {
  const ws = useWorkspace();
  const project = ws.project!;
  const chapter = project.chapters.find((c) => c.id === chapterId)!;
  const pages = useMemo(() => {
    const byId = new Map(project.pages.map((page) => [page.id, page]));
    return chapter.pageIds.map((id) => byId.get(id)).filter((p): p is Page => !!p);
  }, [project.pages, chapter.pageIds]);
  const upload = useRef<HTMLInputElement>(null);
  const [menu, setMenu] = useState<Page>();
  const [exporting, setExporting] = useState(false);
  const running = ws.job?.status === "running";
  const progress = ws.job && ws.job.total ? ws.job.completed / ws.job.total : 0;

  return (
    <Screen
      title={chapter.name}
      enter={enter}
      back={{ label: `Back to ${project.name}`, onBack }}
      actions={
        <>
          <GlassButton label="Add pages" disabled={running} onClick={() => upload.current?.click()}>
            <ImagePlus size={20} />
          </GlassButton>
          {running ? (
            <GlassButton label="Stop" shape="capsule" onClick={() => void ws.stop()}>
              <Square size={14} fill="currentColor" /> Stop
            </GlassButton>
          ) : (
            <GlassButton
              label="Translate chapter"
              shape="capsule"
              tone="prominent"
              disabled={pages.length === 0}
              onClick={() => void ws.run({ kind: "full", scope: "chapter", pageIds: chapter.pageIds })}
            >
              <Sparkles size={16} /> Translate
            </GlassButton>
          )}
        </>
      }
    >
      {running && (
        <div className="mb-3 flex items-center gap-3 rounded-group bg-grouped-cell px-4 py-3" role="status">
          <span className="min-w-0 flex-1 truncate text-[15px]">
            Translating{ws.job?.total ? ` ${ws.job.completed} of ${ws.job.total}` : ""}
          </span>
          <span className="relative h-1.5 w-24 overflow-hidden rounded-full bg-surface-fill">
            <span className="absolute inset-y-0 left-0 rounded-[inherit] bg-brand transition-[width] duration-200 ease-settle" style={{ width: `${Math.round(progress * 100)}%` }} />
          </span>
        </div>
      )}
      {pages.length === 0 ? (
        <Empty title="No pages." detail="Add pages from your photos or files.">
          <GlassButton label="Add pages" shape="capsule" tone="prominent" onClick={() => upload.current?.click()}>
            <ImagePlus size={18} /> Add pages
          </GlassButton>
        </Empty>
      ) : (
        <ul className="grid grid-cols-3 gap-x-3 gap-y-4 pt-1" aria-label="Pages">
          {pages.map((page, index) => (
            <PageTile
              key={page.id}
              page={page}
              number={index + 1}
              projectId={project.id}
              stage={running && ws.job?.current?.page === page.id ? ws.job.current.stage : undefined}
              onOpen={() => onOpenPage(page.id)}
              onMenu={() => setMenu(page)}
            />
          ))}
        </ul>
      )}
      <input
        ref={upload}
        type="file"
        accept="image/*"
        multiple
        hidden
        onChange={(event) => {
          if (event.target.files) void ws.uploadPages(event.target.files, chapter.id);
          event.target.value = "";
        }}
      />
      <Sheet open={!!menu} onOpenChange={(open) => !open && setMenu(undefined)} title={menu?.name ?? ""} detents="fit">
        {menu && (
          <div className="flex flex-col gap-3 pt-1">
            <ActionList
              actions={[
                { label: "Open", icon: BookOpen, onSelect: () => (setMenu(undefined), onOpenPage(menu.id)) },
                { label: "Translate page", icon: Play, disabled: running, onSelect: () => (setMenu(undefined), void ws.run({ kind: "full", scope: "page", pageIds: [menu.id] })) },
                {
                  label: exporting ? "Exporting" : "Export as PNG",
                  icon: FileDown,
                  disabled: exporting,
                  onSelect: async () => {
                    setExporting(true);
                    try {
                      await exportPage(project.id, menu);
                    } catch (error) {
                      ws.reportError(describeError(error));
                    } finally {
                      setExporting(false);
                      setMenu(undefined);
                    }
                  },
                },
              ]}
            />
            <ActionList
              actions={[
                {
                  label: "Remove page",
                  icon: X,
                  destructive: true,
                  disabled: running,
                  onSelect: () => {
                    void ws.removePage(menu.id);
                    setMenu(undefined);
                  },
                },
              ]}
            />
          </div>
        )}
      </Sheet>
    </Screen>
  );
}

function PageTile({ page, number, projectId, stage, onOpen, onMenu }: { page: Page; number: number; projectId: string; stage?: string; onOpen: () => void; onMenu: () => void }) {
  const press = useLongPress(onMenu);
  const done = STAGES.filter((s) => stageDone(page, s)).length;
  return (
    <li className="min-w-0">
      <button
        type="button"
        className="flex w-full cursor-pointer flex-col gap-1 text-left outline-none select-none focus-visible:rounded-control focus-visible:shadow-focus active:opacity-70 [-webkit-tap-highlight-color:transparent] [-webkit-touch-callout:none]"
        onClick={onOpen}
        {...press}
      >
        <img
          className="block aspect-[5/7] w-full rounded-control bg-surface-fill object-cover object-top shadow-[0_0_0_0.5px_var(--border)]"
          src={assetHref(projectId, page.id, "thumbnail", assetVersion(page, "thumbnail"))}
          alt=""
          loading="lazy"
          decoding="async"
          draggable={false}
        />
        <span className="flex items-center gap-1.5">
          <span className="min-w-0 flex-1 truncate text-[13px] leading-[18px] text-fg-2 tabular-nums">Page {number}</span>
          <span className="inline-flex gap-[3px]" aria-label={`${done} of ${STAGES.length} stages done`}>
            {STAGES.map((s) => (
              <i
                key={s}
                className={cn(
                  "size-1.5 rounded-full",
                  stage === s ? "animate-[soft-pulse_1s_ease-in-out_infinite_alternate] bg-brand" : stageDone(page, s) ? "bg-success" : "bg-line-strong",
                )}
              />
            ))}
          </span>
        </span>
      </button>
    </li>
  );
}
