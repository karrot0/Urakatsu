// gRPC-web client. The Python service hosts the bridge on the same origin as
// the UI, so the browser talks real gRPC framing without a proxy.

import { createClient } from "@connectrpc/connect";
import { createGrpcWebTransport } from "@connectrpc/connect-web";

import { Urakatsu, type Page } from "../gen/urakatsu/v1/urakatsu_pb";

export const transport = createGrpcWebTransport({
  baseUrl: window.location.origin,
  useBinaryFormat: true,
});

export const client = createClient(Urakatsu, transport);

export type AssetKind = "source" | "cleanup" | "render" | "thumbnail" | string;

/** Scene revision the page view was read at; every committed edit bumps it. */
export function pageVersion(page: Page): string {
  return String(page.revision);
}

/**
 * The cache key of one page asset: its content address, so an edit refetches only the assets it
 * changed. Servers without per-asset versions fall back to the revision.
 */
export function assetVersion(page: Page, kind: AssetKind): string {
  return page.assetVersions[kind] ?? pageVersion(page);
}

/**
 * A page asset as a plain HTTP URL, for `<img src>`: the browser loads it lazily, decodes it off the
 * main thread and caches it (the version makes the URL immutable; the server says so).
 */
export function assetHref(projectId: string, pageId: string, kind: AssetKind, version: string): string {
  return `/api/v1/projects/${encodeURIComponent(projectId)}/pages/${encodeURIComponent(pageId)}/assets/${encodeURIComponent(kind)}?v=${encodeURIComponent(version)}`;
}

const urls = new Map<string, string>();
const pending = new Map<string, Promise<string | null>>();

export function assetUrl(projectId: string, pageId: string, kind: AssetKind, version: string): Promise<string | null> {
  const prefix = `${projectId}/${pageId}/${kind}@`;
  const key = prefix + version;
  const cached = urls.get(key);
  if (cached) return Promise.resolve(cached);
  const inflight = pending.get(key);
  if (inflight) return inflight;
  const task = (async () => {
    try {
      const asset = await client.getAsset({ projectId, pageId, kind });
      if (!asset.found) return null;
      const url = URL.createObjectURL(new Blob([asset.data as BlobPart], { type: asset.contentType }));
      for (const [other, value] of urls) {
        if (other.startsWith(prefix) && other !== key) {
          URL.revokeObjectURL(value);
          urls.delete(other);
        }
      }
      urls.set(key, url);
      return url;
    } finally {
      pending.delete(key);
    }
  })();
  pending.set(key, task);
  return task;
}

export function forgetAssets(projectId: string, pageId?: string): void {
  const prefix = pageId ? `${projectId}/${pageId}/` : `${projectId}/`;
  for (const [key, value] of urls) {
    if (key.startsWith(prefix)) {
      URL.revokeObjectURL(value);
      urls.delete(key);
    }
  }
}

/** Hands ``blob`` to the browser as a download named ``filename``. */
export function saveBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  Object.assign(document.createElement("a"), { href: url, download: filename }).click();
  window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

/** One page as rendered (translation typeset), named after the page: "012.jpg" saves as "012.png". */
export async function exportPage(projectId: string, page: Page): Promise<void> {
  const response = await fetch(assetHref(projectId, page.id, "render", pageVersion(page)));
  if (!response.ok) throw new Error(`Export failed: ${response.status} ${response.statusText}`);
  const stem = page.name.trim().replace(/\.[A-Za-z0-9]{1,5}$/, "").replace(/[<>:"/\\|?*\u0000-\u001f]/g, "_") || "page";
  saveBlob(await response.blob(), `${stem}.png`);
}

/** A chapter's rendered pages as one archive, with the file name the server chose. */
export async function exportChapter(projectId: string, chapterId: string, format: "cbz" | "zip"): Promise<{ blob: Blob; filename: string }> {
  const response = await fetch(`/api/v1/projects/${projectId}/chapters/${chapterId}/export?format=${format}`);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(typeof body?.detail === "string" ? body.detail : `Export failed: ${response.status} ${response.statusText}`);
  }
  const disposition = response.headers.get("content-disposition") ?? "";
  const encoded = /filename\*=UTF-8''([^;]+)/i.exec(disposition)?.[1];
  const plain = /filename="([^"]+)"/i.exec(disposition)?.[1];
  return { blob: await response.blob(), filename: encoded ? decodeURIComponent(encoded) : plain ?? `chapter.${format}` };
}

export function describeError(error: unknown): string {
  if (error instanceof Error) return error.message.replace(/^\[\w+\]\s*/, "");
  return String(error);
}
