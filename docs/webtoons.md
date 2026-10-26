# Webtoons, manhwa and manhua

Webtoons, manhwa and manhua are read as one long vertical strip. Sites serve that strip cut into
images at whatever height suits them, often straight through a speech bubble, a line of
lettering or a face. Translated page by page, a bubble cut in two is read as two half bubbles.

A webtoon project fixes this. It joins a chapter's images back into one strip and cuts the strip
again, this time where a cut harms nothing: a gutter between panels or a flat background. It
never cuts through text or bubbles. The pipeline then runs on those pages as on any other.

## Make a project a webtoon

The format button beside the project name (a book for manga, a strip for webtoon) chooses:

- **Manga:** pages are kept as they come.
- **Webtoon:** for webtoons, manhwa and manhua.

The format is stored in the project's `project.json`. Changing it changes nothing already in the
project. Chapters downloaded from a linked source afterwards are restitched as they arrive, before
they appear. Images added to an empty chapter are restitched once they are all added.

## Restitch a chapter

In a webtoon project each chapter row has a **Restitch** button (scissors). It joins the
chapter's pages and cuts them again with the current settings, replacing the chapter's pages in
one step that Undo reverts. Work done on the old pages (text, cleanup) goes with them, so
restitch a chapter before translating it. The chapter keeps its name and its link to the source.

## Read and edit as one strip

In a webtoon project the canvas shows the chapter as it is read: its pages joined edge to edge in
one strip, fitted to the width, scrolled from top to bottom. The page in the middle of the view is
the current one, with its text blocks, layers and tools, edited where it sits in the strip.
Scrolling on makes the next page current without moving anything; clicking a page makes it
current; choosing a page in the sidebar scrolls to it. While a stroke, a move, text editing or a
fill selection is in progress, the current page stays put.

In the Rendered view, pages that have been through the pipeline show their render and the others
their source. **Strip** in the status bar switches to one page at a time and back; the choice is
remembered.

## How the cuts are chosen

Slicing follows [SmartStitch](https://github.com/MechTechnology/SmartStitch)'s pixel comparison.
A row may be cut when no two neighbouring pixels in it differ by more than the sensitivity
allows, so rows through line art, lettering and bubble outlines are refused.

1. Images narrower or wider than most of the chapter are scaled to its most common width, then
   joined top to bottom.
2. Near each target height, the cut goes in a quiet run of rows. Wide runs, such as panel
   gutters, are preferred over narrow ones near the target: the gap between two lines of
   borderless lettering is quiet too. The cut sits inside its run, away from the art on either
   side.
3. Pages stay between half and one and a half times the target height. When nothing in that
   range is quiet, the cut goes through the least busy row.
4. With **Avoid detected text** on, the detection model (the one the Detection stage uses,
   wherever it runs) looks at a square around each cut. Text, bubbles and sound effects it
   finds are never cut, and a cut that moves to avoid them is looked at again. Text already
   detected on the pages being restitched is avoided too.

Restitched pages from lossless sources (PNG) stay lossless. From JPEG or WebP sources they are
stored as WebP at quality 95.

## Settings

| Setting | Default | What it does |
| --- | --- | --- |
| Page height | Auto | The target page height in pixels. Auto is 1.5 times the strip width: detection models squeeze a page into a square, so much taller pages shrink their lettering. |
| Sensitivity | 90 | 0 cuts anywhere, 100 only perfectly flat rows. Lower it for strips with noisy or textured gutters. |
| Scan step | 5 | Cuts land on rows this far apart. |
| Side margin | 0 | Columns ignored at each side, for borders or watermarks at the edge that would make every row busy. |
| Avoid detected text | On | Ask the detection model where text is before cutting. Off, cuts rely on pixel comparison alone. |

## Export

Besides the CBZ and ZIP exports, a webtoon chapter's export menu has **Long strip (.png)**: the
chapter's rendered pages joined into one image.

On [Komga](komga.md), a webtoon project's books say `<Format>Webtoon</Format>` in their
`ComicInfo.xml`, and a series Urakatsu creates is set to Komga's webtoon (vertical) reading
direction.
