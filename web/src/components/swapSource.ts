// Swaps pages' source images from the sidebar's right-click menu or the phone's page sheet.

import type { Page } from "../gen/urakatsu/v1/urakatsu_pb";

type Swap = (swaps: [pageId: string, file: File][]) => unknown;

/**
 * Picks one image per page in `pages` and hands the pairs to `swap`. Call it inside a click, or
 * the browser refuses the file dialog. Nothing asks first: each page keeps its work (scaled to its
 * new image) and Undo reverts the whole swap.
 *
 * Several pages take as many images, matched by name when every image is named like a page
 * ("012.png" for page "012.jpg"), else in name order to the pages in project order.
 */
export function swapSources(pages: Page[], swap: Swap, report: (message: string) => void) {
  pickImages(pages.length > 1, (files) => {
    if (files.length !== pages.length) {
      report(`Pick ${pages.length} images, one for each selected page (${files.length} picked).`);
      return;
    }
    swap(pair(pages, files));
  });
}

function pair(pages: Page[], files: File[]): [string, File][] {
  const byName = new Map(pages.map((page) => [stem(page.name), page.id]));
  const named = files.map((file) => byName.get(stem(file.name)));
  if (byName.size === pages.length && named.every((id) => id !== undefined) && new Set(named).size === files.length) {
    return files.map((file, index) => [named[index]!, file]);
  }
  const sorted = [...files].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  return pages.map((page, index) => [page.id, sorted[index]]);
}

const stem = (name: string) => name.replace(/\.[^.]+$/, "").toLowerCase();

function pickImages(multiple: boolean, onPick: (files: File[]) => void) {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/*";
  input.multiple = multiple;
  input.hidden = true;
  // In the document: iOS Safari does not report a choice on a detached input.
  document.body.append(input);
  input.onchange = () => {
    if (input.files?.length) onPick([...input.files]);
    input.remove();
  };
  input.oncancel = () => input.remove();
  input.click();
}
