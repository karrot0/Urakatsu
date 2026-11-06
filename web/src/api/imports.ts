// Pages from what the user picks, drops or pastes: images, ZIP/CBZ archives and whole folders.
// Loose images go through AddPage one by one; anything with an archive or a folder goes to
// POST /api/v1/projects/{id}/import, which makes chapters of them in one undo step
// (urakatsu/pipeline/imports.py says how files become chapters).

/** A file with the path it had: `Series/Ch 2/003.jpg` from a folder, else its name. */
export interface Incoming {
  file: File;
  path: string;
}

/** What pages can come from: a file picker (with or without `webkitdirectory`), a paste, or a drop. */
export type PageSource = FileList | File[] | DataTransfer;

const IMAGE = /\.(png|jpe?g|jfif|webp|bmp|gif|tiff?|avif)$/i;
const ARCHIVE = /\.(zip|cbz)$/i;

/** What the page pickers accept: images and comic archives. */
export const PAGE_ACCEPT = "image/*,.zip,.cbz,application/zip,application/vnd.comicbook+zip";

export const isArchive = (path: string) => ARCHIVE.test(path);
const isImage = (item: Incoming) => item.file.type.startsWith("image/") || IMAGE.test(item.path);

/** Whether these files need the server's import (an archive, or files from a folder) rather than AddPage. */
export const needsImport = (items: Incoming[]) => items.some((item) => isArchive(item.path) || item.path.includes("/"));

function fromList(files: FileList | File[]): Incoming[] {
  return Array.from(files).map((file) => ({ file, path: file.webkitRelativePath || file.name }));
}

/**
 * The files of a pick, paste or drop, folders walked, keeping only images and archives.
 * Call it synchronously in the event handler: a drop's DataTransfer is emptied once the handler
 * returns, so its entries are taken here and only the folder walk runs later.
 */
export function collect(source: PageSource): Promise<Incoming[]> {
  const keep = (items: Incoming[]) => items.filter((item) => isArchive(item.path) || isImage(item));
  if (!(source instanceof DataTransfer)) return Promise.resolve(keep(fromList(source)));
  const items = Array.from(source.items).filter((item) => item.kind === "file");
  const entries = items.map((item) => item.webkitGetAsEntry?.() ?? null);
  if (!entries.some((entry) => entry?.isDirectory)) return Promise.resolve(keep(fromList(source.files)));
  const files = items.map((item) => item.getAsFile());
  return (async () => {
    const found: Incoming[] = [];
    for (const [index, entry] of entries.entries()) {
      const file = files[index];
      if (entry?.isDirectory) await walk(entry as FileSystemDirectoryEntry, found);
      else if (file) found.push({ file, path: file.name });
    }
    return keep(found);
  })();
}

async function walk(folder: FileSystemDirectoryEntry, found: Incoming[]): Promise<void> {
  const reader = folder.createReader();
  // readEntries hands a folder over in batches (100 in Chrome) until it returns none.
  for (;;) {
    const batch = await new Promise<FileSystemEntry[]>((resolve, reject) => reader.readEntries(resolve, reject));
    if (!batch.length) return;
    for (const entry of batch) {
      if (entry.isDirectory) await walk(entry as FileSystemDirectoryEntry, found);
      else {
        const file = await new Promise<File>((resolve, reject) => (entry as FileSystemFileEntry).file(resolve, reject));
        found.push({ file, path: entry.fullPath.replace(/^\/+/, "") });
      }
    }
  }
}

/** Sends files to the server's import; resolves to the new pages' ids in project order. */
export async function importFiles(projectId: string, items: Incoming[], chapterId?: string): Promise<{ pages: string[]; chapters: number }> {
  const form = new FormData();
  for (const item of items) {
    form.append("files", item.file, item.file.name);
    form.append("paths", item.path);
  }
  if (chapterId) form.append("chapter_id", chapterId);
  const response = await fetch(`/api/v1/projects/${encodeURIComponent(projectId)}/import`, { method: "POST", body: form });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(typeof body?.detail === "string" ? body.detail : `Import failed: ${response.status} ${response.statusText}`);
  }
  return response.json();
}
