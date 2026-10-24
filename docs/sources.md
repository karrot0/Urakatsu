# Sources

A project can follow one series on a manga site and download its chapters directly.

## Systems, sources and series

From the outside in:

- a **source system** is a scraper runtime: Suwayomi Server (Tachiyomi and Mihon extensions)
  or Mana (built `.mana` bundles, run under Bun);
- a **source** is one site inside a system (an extension may add several, one per language:
  the Sources page shows those as one site with a language picker);
- a **series** is one title on a source.

## Linking a series

1. Turn a system on in Settings, then Sources. Urakatsu installs and starts its runtime.
2. Add an extension repository and install the sites you read.
3. Use **From a source** on the project list, or **Link a series** in a project, to pick a
   source and search it.

The series cover becomes the project cover. The sidebar lists the series' chapters among the
project's own: chapters not downloaded yet are grayed out with a download button, and a
downloaded chapter becomes an ordinary chapter (one undoable commit) in source order. A
downloaded chapter is never fetched again, so your edits stay.

Covers are kept at tile size in `URAKATSU_HOME/cache/covers`, and a search fetches its covers
in parallel, so results fill in as fast as the site allows.

## The built-in browser (Mana)

Mana has a built-in browser, as the Mana app has a WebView:

- When a site answers with a check (Cloudflare and the like), the host opens it in a headless
  Chromium driven through `Bun.WebView`, waits for the check to clear, and from then on sends
  that site's requests from inside the browser tab. A cleared check is bound to the browser's
  own network stack, so its cookies alone do not get a plain request through.
- A check that needs a person (a "verify you are human" box) shows up as a prompt. Urakatsu
  opens the page live in its browser panel; you clear it there, and the search or download
  continues.
- Sources that use `WebViewPage` get the real browser too, and Settings, then Sources, can open
  any page in it (to sign in, for example).

## For developers

Systems plug in behind `urakatsu.sources.base.SourceSystem`; third-party ones register through
the `urakatsu.source_systems` entry-point group. Links and downloads live in
`urakatsu/sources/links.py`. The JSON routes are under `/api/v1/sources` and
`/api/v1/projects/{id}/source`.
