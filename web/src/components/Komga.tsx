// Komga: the project as a series on a Komga server, one book per chapter (urakatsu/sources/komga). Opened
// from the project header: add the project as a new series or link one Komga already has, then
// export chapters. Automations can export the chapters they finish (Automations.tsx).

import { BookUp, ExternalLink } from "lucide-react";
import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";

import { Button, buttonVariants } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "cn";
import { describeError } from "../api/client";
import { komga, type KomgaLink, type KomgaSeries, type KomgaStatus } from "../api/komga";
import { formatTimestamp, plural } from "../format";
import { useWorkspace } from "../state/workspace";
import { CheckRow, Select, Tooltip, type Option } from "./primitives";
import { Badge, Segmented } from "./ui";

/** How often an export on its way is read. */
const EXPORT_POLL_MS = 1000;
const SEARCH_DELAY_MS = 250;

// ---- header button ---------------------------------------------------------------------------

/** The project header's Komga button. */
export function KomgaButton() {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Tooltip content="Komga" side="bottom">
        <Button variant="ghost" size="icon-sm" aria-label="Komga" onClick={() => setOpen(true)}>
          <BookUp size={16} />
        </Button>
      </Tooltip>
      <KomgaDialog open={open} onOpenChange={setOpen} />
    </>
  );
}

/** The project's Komga series: null when it is not on Komga, undefined while loading. Read again
 * every second while an export is on its way. */
export function useKomgaLink(projectId: string, active = true): [KomgaLink | null | undefined, (link: KomgaLink | null) => void, string | undefined] {
  const [link, setLink] = useState<KomgaLink | null>();
  const [error, setError] = useState<string>();
  const exporting = !!link?.export;
  useEffect(() => {
    if (!active) return;
    let stopped = false;
    let timer: number | undefined;
    const read = async () => {
      try {
        const next = await komga.link(projectId);
        if (stopped) return;
        setLink(next);
        setError(undefined);
        if (next?.export) timer = window.setTimeout(read, EXPORT_POLL_MS);
      } catch (failure) {
        if (stopped) return;
        setError(describeError(failure));
        // A read that fails mid-export (a network blip, Komga restarting) tries again rather than leaving the dialog stuck.
        if (exporting) timer = window.setTimeout(read, EXPORT_POLL_MS);
      }
    };
    void read();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [projectId, active, exporting]);
  return [link, setLink, error];
}

// ---- dialog --------------------------------------------------------------------------------------

function KomgaDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const ws = useWorkspace();
  const projectId = ws.project!.id;
  const [status, setStatus] = useState<KomgaStatus>();
  const [link, setLink, linkError] = useKomgaLink(projectId, open);
  const [error, setError] = useState<string>();

  useEffect(() => {
    if (!open) return;
    setError(undefined);
    komga.status().then(setStatus, (failure) => setError(describeError(failure)));
  }, [open]);

  const act = async (work: () => Promise<KomgaLink | null | void>) => {
    try {
      const next = await work();
      setLink(next ?? null);
      setError(undefined);
      return true;
    } catch (failure) {
      setError(describeError(failure));
      return false;
    }
  };

  const shown = error ?? linkError;
  const loading = !status || link === undefined;
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[min(720px,calc(100vh-48px))] gap-0 p-0 sm:max-w-xl">
        <DialogHeader className="px-5 pt-5 pb-3">
          <DialogTitle>Komga</DialogTitle>
          <DialogDescription>
            {link
              ? "Each chapter is a book of this series, its pages as rendered. Exporting a chapter again replaces its book."
              : "Read this project in Komga: add it as a new series, or link a series Komga already has."}
          </DialogDescription>
        </DialogHeader>
        {shown && <div className="mx-5 mb-3 rounded-control bg-danger-weak px-3 py-2 text-xs text-danger">{shown}</div>}
        {loading && !shown && <p className="px-5 pb-5 text-sm text-fg-3">Loading.</p>}
        {!loading && !status.connected && (
          <div className="flex flex-col items-start gap-3 px-5 pb-5 text-sm text-fg-2">
            <p>{status.configured ? `Komga cannot be reached: ${status.error ?? "unknown error"}` : "Connect Urakatsu to your Komga server first."}</p>
            <Button variant="outline" size="sm" onClick={() => ws.setSettingsOpen(true, "sources")}>
              Open Komga settings
            </Button>
          </div>
        )}
        {!loading && status.connected && link === null && <AddToKomga status={status} act={act} />}
        {!loading && status.connected && link && <SeriesPanel link={link} act={act} onClose={() => onOpenChange(false)} />}
      </DialogContent>
    </Dialog>
  );
}

function Row({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="flex items-start gap-4 py-3 [&+&]:shadow-[inset_0_1px_0_var(--border-soft)]">
      <span className="w-32 shrink-0 pt-1.5 text-sm font-medium">{title}</span>
      <div className="flex min-w-0 flex-1 flex-col gap-2">{children}</div>
    </div>
  );
}

// ---- not on Komga yet ----------------------------------------------------------------------------

function AddToKomga({ status, act }: { status: KomgaStatus; act: (work: () => Promise<KomgaLink>) => Promise<boolean> }) {
  const ws = useWorkspace();
  const project = ws.project!;
  const usable = status.libraries.filter((library) => !library.problem);
  const [mode, setMode] = useState<"new" | "existing">("new");
  const [libraryId, setLibraryId] = useState(usable[0]?.id ?? "");
  const [title, setTitle] = useState(ws.link?.series.title ?? project.name);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<KomgaSeries[]>();
  const [busy, setBusy] = useState(false);

  const libraryOptions: Option[] = status.libraries.map((library) => ({
    value: library.id,
    label: library.name,
    disabled: !!library.problem,
    detail: library.problem ? "Choose its folder in Settings" : undefined,
  }));

  // Series in the library, by title as you type.
  useEffect(() => {
    if (mode !== "existing" || !libraryId) return;
    let stopped = false;
    const timer = window.setTimeout(() => {
      komga.search(libraryId, query).then(
        (found) => !stopped && setResults(found.series),
        () => !stopped && setResults([]),
      );
    }, SEARCH_DELAY_MS);
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [mode, libraryId, query]);

  const run = async (work: () => Promise<KomgaLink>) => {
    setBusy(true);
    await act(work);
    setBusy(false);
  };

  if (status.libraries.length === 0 || usable.length === 0) {
    return (
      <div className="flex flex-col items-start gap-3 px-5 pb-5 text-sm text-fg-2">
        <p>{status.libraries.length === 0 ? "This Komga has no libraries yet. Add one in Komga first." : "Choose where your Komga libraries' folders are on this computer first."}</p>
        <Button variant="outline" size="sm" onClick={() => ws.setSettingsOpen(true, "sources")}>
          Open Komga settings
        </Button>
      </div>
    );
  }

  return (
    <>
      <ScrollArea className="min-h-0 flex-1" contentClassName="px-5">
        <Row title="Series">
          <Segmented
            label="New or existing series"
            value={mode}
            options={[
              { value: "new", label: "Add as a new series" },
              { value: "existing", label: "Link an existing series" },
            ]}
            onChange={setMode}
          />
        </Row>
        <Row title="Library">
          <Select label="Komga library" className="min-w-52" value={libraryId} options={libraryOptions} onChange={setLibraryId} />
        </Row>
        {mode === "new" ? (
          <Row title="Title">
            <Input aria-label="Series title" value={title} onChange={(event) => setTitle(event.target.value)} />
            <span className="text-xs text-fg-3">
              A folder with this name is made in the library. Komga lists the series after the first export; Urakatsu then gives it the source's summary, genres
              and cover.
            </span>
          </Row>
        ) : (
          <Row title="Find">
            <Input aria-label="Search the library's series" placeholder="Series title" value={query} onChange={(event) => setQuery(event.target.value)} />
            <ul className="flex max-h-64 flex-col overflow-y-auto rounded-control py-1 shadow-[inset_0_0_0_1px_var(--border-soft)]">
              {results === undefined && <li className="px-2 py-1 text-xs text-fg-3">Searching.</li>}
              {results?.length === 0 && <li className="px-2 py-1 text-xs text-fg-3">No series match.</li>}
              {results?.map((series) => (
                <li key={series.id} className="flex items-center gap-2 px-2 py-1">
                  <span className="min-w-0 flex-1 truncate" title={series.url}>
                    {series.title}
                  </span>
                  <span className="shrink-0 text-xs text-fg-3">{plural(series.books, "book")}</span>
                  <Button variant="outline" size="xs" disabled={busy} onClick={() => void run(() => komga.linkSeries(project.id, series.id))}>
                    Link
                  </Button>
                </li>
              ))}
            </ul>
            <span className="text-xs text-fg-3">Books are added beside the series' own; files Urakatsu did not write are never replaced.</span>
          </Row>
        )}
      </ScrollArea>
      {mode === "new" && (
        <DialogFooter className="px-5 py-4 shadow-[inset_0_1px_0_var(--border-soft)]">
          <Button disabled={busy || !libraryId || !title.trim()} onClick={() => void run(() => komga.add(project.id, libraryId, title.trim()))}>
            Add to Komga
          </Button>
        </DialogFooter>
      )}
    </>
  );
}

// ---- on Komga ------------------------------------------------------------------------------------

function SeriesPanel({ link, act, onClose }: { link: KomgaLink; act: (work: () => Promise<KomgaLink | void>) => Promise<boolean>; onClose: () => void }) {
  const ws = useWorkspace();
  const project = ws.project!;
  const chapters = useMemo(() => project.chapters.filter((chapter) => chapter.pageIds.length > 0), [project.chapters]);
  // Chapters not exported yet are chosen to begin with, and again after each export (an automation's too).
  const unexported = useCallback(() => chapters.filter((chapter) => !(chapter.id in link.exported)).map((chapter) => chapter.id), [chapters, link.exported]);
  const [chosen, setChosen] = useState<string[]>(unexported);
  const exporting = link.export;
  const last = link.last_export;
  const finished = last?.finished_at;
  useEffect(() => {
    setChosen(unexported());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [finished, chapters.length]);
  const exported = chapters.filter((chapter) => chapter.id in link.exported).length;

  const start = useCallback(() => void act(() => komga.export(project.id, chosen.length === chapters.length ? null : chosen)), [act, chosen, chapters.length, project.id]);
  const exportLabel =
    exporting?.state === "scanning"
      ? "Komga is scanning"
      : exporting
        ? `Exporting ${Math.min(exporting.done + 1, exporting.total)} of ${exporting.total}${exporting.chapter ? `: ${exporting.chapter}` : ""}`
        : "";

  return (
    <>
      <ScrollArea className="min-h-0 flex-1" contentClassName="px-5">
        <Row title="Series">
          <div className="flex items-center gap-2">
            <span className="min-w-0 flex-1 truncate font-medium">{link.title}</span>
            {link.series_url ? (
              <a className={buttonVariants({ variant: "ghost", size: "sm" })} href={link.series_url} target="_blank" rel="noreferrer">
                Open in Komga <ExternalLink size={12} />
              </a>
            ) : (
              <Badge tone="neutral" title="Komga lists a new series once its folder holds a book">
                Appears after the first export
              </Badge>
            )}
          </div>
          <span className="text-xs text-fg-3">
            {link.library_name} · {link.folder}
          </span>
        </Row>
        <Row title="Chapters">
          <span className="text-sm text-fg-2">
            {exported} of {plural(chapters.length, "chapter")} exported
          </span>
          <div className="flex max-h-64 flex-col overflow-y-auto rounded-control py-1 shadow-[inset_0_0_0_1px_var(--border-soft)]">
            {chapters.length === 0 && <span className="px-2 py-1 text-xs text-fg-3">This project has no pages yet.</span>}
            {chapters.length > 1 && (
              <CheckRow checked={chosen.length === chapters.length} label="All chapters" onChange={(all) => setChosen(all ? chapters.map((c) => c.id) : [])} />
            )}
            {chapters.map((chapter) => (
              <CheckRow
                key={chapter.id}
                checked={chosen.includes(chapter.id)}
                label={chapter.name}
                detail={chapter.id in link.exported ? `Exported ${formatTimestamp(link.exported[chapter.id])}` : "Not exported"}
                onChange={(on) => setChosen(on ? [...chosen, chapter.id] : chosen.filter((id) => id !== chapter.id))}
              />
            ))}
          </div>
        </Row>
        {(exporting || last) && (
          <Row title="Export">
            {exporting ? (
              <>
                <span className="truncate text-sm text-fg-2">{exportLabel}</span>
                <Progress value={exporting.total ? (exporting.done / exporting.total) * 100 : 0} aria-label="Export progress" />
              </>
            ) : (
              last && (
                <span className={cn("text-sm", last.error ? "text-danger" : "text-fg-2")}>
                  {last.error
                    ? `${last.books ? `${plural(last.books, "chapter")} exported, then: ` : ""}${last.error}`
                    : `${plural(last.books, "chapter")} exported ${formatTimestamp(last.finished_at ?? last.started_at)}${last.stopped ? ", then the run was stopped" : ""}${last.pending ? ". Komga has not listed the series yet; it appears after Komga's next scan." : ""}`}
                </span>
              )
            )}
          </Row>
        )}
      </ScrollArea>
      <DialogFooter className="px-5 py-4 shadow-[inset_0_1px_0_var(--border-soft)]">
        <Button
          variant="ghost"
          className="mr-auto"
          disabled={!!exporting}
          title="The books stay on Komga"
          onClick={async () => {
            if (await act(() => komga.unlink(project.id))) onClose();
          }}
        >
          Unlink series
        </Button>
        <Button disabled={!!exporting || chosen.length === 0} onClick={start}>
          {exporting ? "Exporting" : `Export ${plural(chosen.length, "chapter")}`}
        </Button>
      </DialogFooter>
    </>
  );
}
