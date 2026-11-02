// Swaps a page's source image from the sidebar's right-click menu or the phone's page sheet.

import type { Page } from "../gen/urakatsu/v1/urakatsu_pb";

/**
 * Picks an image and hands it to `swap`. Call it inside a click, or the browser refuses the file
 * dialog. Nothing asks first: the page keeps its work (scaled to the new image) and Undo reverts it.
 */
export function swapSource(page: Page, swap: (pageId: string, file: File) => unknown) {
  pickImage((file) => swap(page.id, file));
}

function pickImage(onPick: (file: File) => void) {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/*";
  input.hidden = true;
  // In the document: iOS Safari does not report a choice on a detached input.
  document.body.append(input);
  input.onchange = () => {
    if (input.files?.[0]) onPick(input.files[0]);
    input.remove();
  };
  input.oncancel = () => input.remove();
  input.click();
}
