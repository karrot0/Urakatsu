# Komga

Urakatsu can put a project on a [Komga](https://komga.org) server as a series, one book per
chapter, so translated chapters can be read in Komga's web reader and the apps that sync with
it. Export chapters by hand, or let an automation export the chapters it finishes.

## Connect Komga

Open Settings, then Sources; Komga is below the source systems:

1. Enter Komga's address, such as `http://localhost:25600`.
2. Enter an API key from an admin account (in Komga: Account Settings, API Keys; Komga 1.13
   or newer). Exporting starts library scans, which only admins may do. The key is kept in
   `URAKATSU_HOME/komga.json` and is never sent back to the UI. It is only sent to the address
   it was saved with: changing the address needs the key again.
3. Check each library's folder.

Komga has no upload: it reads books from its libraries' folders. Urakatsu writes the books
into those folders itself, so it needs each folder as this computer sees it:

- **Komga on this computer:** the library's own path works as it is. Nothing to set.
- **Komga in Docker, or on another computer:** Komga's path (`/data/manga` in the container)
  is not this computer's. Enter the folder it is mounted from (the `-v ~/komga/manga:/data/manga`
  side of the bind mount), or a share of it mounted here. Urakatsu checks that the folder exists,
  can be written to, and holds the library's series.

## Put a project on Komga

The Komga button beside the project name opens the project's series:

- **Add as a new series:** pick a library and a title (the linked source's title, else the
  project name). Urakatsu makes a folder with that name. Komga lists a series once its folder
  holds a book, so the series appears with the first export. Urakatsu then gives it its title,
  language, and the linked source's summary, status, genres, link and cover. These fields are
  locked in Komga so later scans keep them. They are set once, so later edits made in Komga stay.
- **Link an existing series:** search a library and pick a series Komga already has, such as
  the raws. Books are added beside the series' own, and its metadata is left alone.

**Unlink series** forgets the link. The books stay on Komga.

## Export

Each chapter becomes `<chapter name>.cbz` in the series' folder: its pages as rendered (the
same images as the chapter's CBZ export) and a `ComicInfo.xml` with:

| Field | From |
| --- | --- |
| Series, Title | The Komga series' title, the chapter's name |
| Number | The source's chapter number for a downloaded chapter, else its place in the project |
| Writer, Genre, Web | The linked source's authors, tags and series page |
| LanguageISO | The translation's target language |
| Manga | Right to left when the source is Japanese |

Exporting a chapter again replaces its book in place. Files Urakatsu did not write are never
replaced: a name that is taken gets a number, `Chapter 1 (2).cbz`. After writing, Urakatsu
asks Komga to scan the library and, for a new series, waits up to a minute for the series to
appear. Deleting a chapter in Urakatsu leaves its book on Komga.

In the dialog, choose chapters (the ones not exported yet are chosen to begin with) and
select **Export**. Its progress and the last export's outcome show there.

## Automations

Turn on **Export finished chapters to Komga** in an automation (the project must be on Komga
first). When a run ends, the chapters it finished without errors are exported before the run
is reported. Partly done or failed chapters wait for a later run. The run history and the
Discord message say how many chapters were exported, or why the export stopped. A failed
export does not fail the run: the chapters are translated, and the next run exports the
chapters the automation finished whose books were never written, even when it has nothing new
to translate (so do chapters finished before exporting was turned on). They can also be
exported from the Komga dialog. Stopping a run while it exports ends the export after the
chapter it is writing; the run is reported as stopped.

## Storage and API

The connection lives in `URAKATSU_HOME/komga.json`, and a project's series in
`projects/<id>/komga.json`.

| Route | Purpose |
| --- | --- |
| `GET`, `PUT /api/v1/komga` | The connection and its libraries; set the address and key (`api_key: null` keeps the saved key) |
| `PUT /api/v1/komga/libraries/{library}/folder` | Where the library's folder is on this computer (`null`: Komga's own path) |
| `GET /api/v1/komga/libraries/{library}/series?q=` | Search the library's series |
| `GET`, `DELETE /api/v1/projects/{id}/komga` | The project's series, with the export on its way; unlink it |
| `PUT /api/v1/projects/{id}/komga` | Link a series Komga has (`series_id`) |
| `POST /api/v1/projects/{id}/komga` | Add the project as a new series (`library_id`, `title`) |
| `POST /api/v1/projects/{id}/komga/export` | Export chapters in the background (`chapter_ids`, or `null` for all) |

The code is `urakatsu/sources/komga/`: `client.py` (Komga's REST API), `connection.py` (the server,
key and library folders), `archive.py` (the CBZ and `ComicInfo.xml`) and `links.py` (projects
as series, the export). The routes are `urakatsu/service/komga_api.py`, and the automation step
is in `urakatsu/service/automations.py`. Koharu has no equivalent.

## Try it with a local Komga

```bash
mkdir -p ~/komga/config ~/komga/manga
docker run -d --name komga -p 25600:25600 -v ~/komga/config:/config -v ~/komga/manga:/data/manga --user "$(id -u):$(id -g)" gotson/komga
```

Open http://localhost:25600, create the admin account and a library at `/data/manga`, then make
an API key. In Urakatsu, connect `http://localhost:25600` with the key and set the library's
folder to `~/komga/manga`.
