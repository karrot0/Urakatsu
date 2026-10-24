// The models folder: how much its models take on their drive, where it is, and changing it
// (with the installed models moved along). Shown on Models > Installed and in Settings.

import { ChevronDown, ChevronUp, FolderOpen } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { catalog, formatBytes, type Storage, type StorageItem } from "../api/catalog";
import { describeError } from "../api/client";
import { useWorkspace } from "../state/workspace";
import { Switch, Tooltip } from "./primitives";
import { OTHER_FILL, UsageBar, UsageLegend } from "./UsageBar";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";

const MOVE_POLL_MS = 800;

/** Storage facts, refreshed every moment while a move runs; ``onMoved`` fires when one finishes. */
export function useModelsStorage(onMoved?: () => void) {
  const ws = useWorkspace();
  const [storage, setStorage] = useState<Storage>();

  const refresh = useCallback(async () => {
    try {
      setStorage(await catalog.storage());
    } catch (error) {
      ws.reportError(describeError(error));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => void refresh(), [refresh]);

  const moving = storage?.move?.status === "moving";
  useEffect(() => {
    if (!moving) return;
    const timer = window.setInterval(async () => {
      const next = await catalog.storage().catch(() => undefined);
      if (!next) return;
      setStorage(next);
      if (next.move?.status !== "moving") {
        if (next.move?.status === "failed") ws.reportError(`Moving the models failed: ${next.move.error}. They stay in ${next.move.source}.`);
        onMoved?.();
      }
    }, MOVE_POLL_MS);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [moving]);

  return { storage, setStorage, refresh };
}

/** "Models 34.2 GB / 931 GB" on the models folder's drive: models in brand, other files neutral. */
export function StorageBar({ storage }: { storage: Storage }) {
  const pool = { name: `Drive of ${storage.models_dir}`, total: storage.disk.total, used: storage.disk.used, app: storage.size };
  return (
    <Tooltip side="bottom" content={<UsageLegend pool={pool} owner="Models" others="Other files" />}>
      <div className="min-w-0 flex-1 cursor-default">
        <UsageBar label="Models" owner="Models" pool={pool} />
      </div>
    </Tooltip>
  );
}

const ITEM_KIND: Record<StorageItem["kind"], string> = { installed: "Installed", builtin: "Built-in", partial: "Unfinished", other: "Other" };
const SHOWN_ITEMS = 6;

/** What takes the space, largest first, each with a bar sized against the largest. */
export function StorageBreakdown({ storage }: { storage: Storage }) {
  const [all, setAll] = useState(false);
  const items = storage.items.filter((item) => item.size > 0);
  if (items.length === 0) return <span className="text-xs leading-4 text-fg-3">No model files yet.</span>;
  const largest = items[0].size;
  const shown = all ? items : items.slice(0, SHOWN_ITEMS);
  return (
    <div className="flex min-w-0 flex-col">
      <ul className="flex flex-col" aria-label="Storage by model">
        {shown.map((item) => {
          const [org, ...rest] = item.name.split("/");
          const title = rest.length ? rest.join("/") : item.name;
          const detail = item.kind === "builtin" ? (item.detail ?? "No provider uses it") : item.detail;
          return (
            <li key={`${item.kind}:${item.name}:${item.id ?? ""}`} className="flex min-w-0 flex-col gap-1 py-1.5 shadow-[inset_0_-1px_0_var(--border-soft)] last:shadow-none">
              <div className="flex min-w-0 items-baseline gap-2">
                <span className="min-w-0 flex-1 truncate" title={item.name}>
                  {title}
                  {rest.length > 0 && <span className="text-xs leading-4 text-fg-3"> {org}</span>}
                </span>
                <span className="flex-none text-xs leading-4 text-fg-2 tabular-nums">{formatBytes(item.size)}</span>
              </div>
              <div className="flex min-w-0 items-center gap-2">
                <span className="relative h-1 min-w-0 flex-1 overflow-hidden rounded-full bg-surface-fill" aria-hidden>
                  <span className={`absolute inset-y-0 left-0 rounded-[inherit] ${item.kind === "installed" ? "bg-brand" : OTHER_FILL}`} style={{ width: `${Math.max(1, (item.size / largest) * 100)}%` }} />
                </span>
                <span className="w-[45%] flex-none truncate text-xs leading-4 text-fg-3" title={detail ?? undefined}>
                  {ITEM_KIND[item.kind]}
                  {detail && `, ${detail}`}
                </span>
              </div>
            </li>
          );
        })}
      </ul>
      {items.length > SHOWN_ITEMS && (
        <Button variant="ghost" size="sm" className="justify-start text-fg-2" aria-expanded={all} onClick={() => setAll((open) => !open)}>
          {all ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          {all ? "Show fewer" : `Show all ${items.length}`}
        </Button>
      )}
    </div>
  );
}

/** The folder path with a button to change it; a move in progress shows here with its progress. */
export function ModelsFolder({ storage, onChange }: { storage: Storage; onChange: (storage: Storage) => void }) {
  const [open, setOpen] = useState(false);
  const move = storage.move;
  const moving = move?.status === "moving";
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <div className="flex min-w-0 items-center gap-2">
        <span className="min-w-0 flex-1 truncate font-mono text-xs leading-4 text-fg-2" title={storage.models_dir}>
          {storage.models_dir}
        </span>
        <Button variant="outline" size="sm" disabled={moving} onClick={() => setOpen(true)}>
          <FolderOpen size={14} /> Change
        </Button>
      </div>
      {moving && (
        <div className="flex flex-col gap-0.5" role="status">
          <div className="relative h-1 overflow-hidden rounded-full bg-surface-fill">
            <span className="absolute inset-y-0 left-0 rounded-[inherit] bg-brand transition-[width] duration-200 ease-settle" style={{ width: `${move.total ? (move.done / move.total) * 100 : 0}%` }} />
          </div>
          <span className="truncate text-xs leading-4 text-fg-3 tabular-nums">
            Moving to {move.target}: {formatBytes(move.done)} of {formatBytes(move.total)}
          </span>
        </div>
      )}
      {move?.status === "completed" && move.left_behind > 0 && (
        <span className="text-xs leading-4 text-warning">
          {move.left_behind} {move.left_behind === 1 ? "file was" : "files were"} in use and stayed in {move.source}. Delete them once the model is unloaded.
        </span>
      )}
      <ChangeFolderDialog open={open} storage={storage} onOpenChange={setOpen} onChange={onChange} />
    </div>
  );
}

function ChangeFolderDialog({ open, storage, onOpenChange, onChange }: { open: boolean; storage: Storage; onOpenChange: (open: boolean) => void; onChange: (storage: Storage) => void }) {
  const ws = useWorkspace();
  const [path, setPath] = useState(storage.models_dir);
  const [move, setMove] = useState(true);
  const [busy, setBusy] = useState<"browse" | "save">();

  useEffect(() => {
    if (open) {
      setPath(storage.models_dir);
      setMove(true);
    }
  }, [open, storage.models_dir]);

  const browse = async () => {
    setBusy("browse");
    try {
      const picked = await catalog.browseFolder();
      if (picked.path) setPath(picked.path);
    } catch (error) {
      ws.reportError(describeError(error));
    } finally {
      setBusy(undefined);
    }
  };

  const save = async (target: string | null) => {
    setBusy("save");
    try {
      onChange(await catalog.setModelsFolder(target, move));
      onOpenChange(false);
    } catch (error) {
      ws.reportError(describeError(error));
    } finally {
      setBusy(undefined);
    }
  };

  const unchanged = path.trim() === storage.models_dir;
  const isDefault = storage.models_dir === storage.default_dir;
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Models folder</DialogTitle>
          <DialogDescription>New downloads go here, and every model loads from here. Pick a drive with room: the models take {formatBytes(storage.size)} now.</DialogDescription>
        </DialogHeader>
        <form
          className="flex flex-col gap-3"
          onSubmit={(event) => {
            event.preventDefault();
            if (!unchanged && path.trim()) void save(path.trim());
          }}
        >
          <div className="flex gap-1">
            <Input className="min-w-0 flex-1 font-mono text-xs" aria-label="Models folder" value={path} onChange={(e) => setPath(e.target.value)} spellCheck={false} />
            <Button variant="outline" size="sm" className="h-8" disabled={!!busy} onClick={browse}>
              {busy === "browse" ? "Choosing" : "Browse"}
            </Button>
          </div>
          <label className="flex items-center justify-between gap-3">
            <span className="flex flex-col">
              <span>Move the installed models</span>
              <span className="text-xs leading-4 text-fg-3">{move ? `Moves ${formatBytes(storage.size)}. Large models across drives take a few minutes.` : "The models stay where they are and look missing until you switch back."}</span>
            </span>
            <Switch label="Move the installed models" checked={move} onChange={setMove} />
          </label>
          <DialogFooter className="items-center">
            {!isDefault && (
              <Button variant="ghost" size="sm" className="mr-auto" disabled={!!busy} onClick={() => save(null)}>
                Use the default folder
              </Button>
            )}
            <Button variant="ghost" size="sm" disabled={!!busy} onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button size="sm" type="submit" disabled={!!busy || unchanged || !path.trim()}>
              {busy === "save" ? "Changing" : move ? "Move models" : "Change folder"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
