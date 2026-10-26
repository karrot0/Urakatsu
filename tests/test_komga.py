"""Komga export: the client, library folders, project links, the export and the automation hook,
against a fake Komga that scans the library folder on disk the way Komga does."""

import asyncio
import io
import json
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import httpx
import pytest

from urakatsu.core.registry import registry
from urakatsu.pipeline.batch import BatchResult
from urakatsu.service import automations as automations_module
from urakatsu.service.app import build_app
from urakatsu.service.automations import AutomationBody, Schedule
from urakatsu.service.servicer import UrakatsuServicer
from urakatsu.service.workspace import Workspace
from urakatsu.sources import Image, RemoteChapter, Series
from urakatsu.sources.komga import BookInfo, KomgaConnection, comic_info, pack_cbz
from urakatsu.sources.komga import links as links_module
from urakatsu.sources.komga.connection import relative_folder
from urakatsu.sources.komga.links import series_folder
from urakatsu.sources.links import SourceLink, remote_key

from fakes import page_png

ROOT = "/data/manga"  # the library's folder as Komga (in Docker) sees it
KEY = "komga-test-key"


class FakeKomga:
    """Komga's API over one library whose folder is ``folder`` here and ``ROOT`` for Komga.
    A scan turns every subfolder holding a CBZ into a series, as Komga does."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.series: dict[str, dict] = {}  # id -> SeriesDto
        self.patches: list[tuple[str, dict]] = []
        self.covers: list[tuple[str, bytes]] = []
        self.scans = 0
        self.transport = httpx.MockTransport(self.handle)

    def add_series(self, name: str) -> str:
        (self.folder / name).mkdir(parents=True, exist_ok=True)
        (self.folder / name / "Their own.cbz").write_bytes(b"not ours")
        self.scan()
        return next(s["id"] for s in self.series.values() if s["name"] == name)

    def scan(self) -> None:
        self.scans += 1
        known = {s["name"] for s in self.series.values()}
        for path in sorted(self.folder.iterdir()):
            if path.is_dir() and path.name not in known and any(path.glob("*.cbz")):
                series_id = f"S{len(self.series) + 1}"
                self.series[series_id] = {"id": series_id, "libraryId": "L1", "name": path.name, "url": f"{ROOT}/{path.name}", "metadata": {"title": path.name}}
        for series in self.series.values():
            series["booksCount"] = len(list((self.folder / series["name"]).glob("*.cbz")))

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.headers.get("X-API-Key") != KEY:
            return httpx.Response(401)
        path, method = request.url.path, request.method
        library = {"id": "L1", "name": "Manga", "root": ROOT}
        if path == "/api/v2/users/me":
            return httpx.Response(200, json={"email": "admin@example.com"})
        if path == "/actuator/info":
            return httpx.Response(200, json={"build": {"version": "1.29.1"}})
        if path == "/api/v1/libraries":
            return httpx.Response(200, json=[library])
        if path == "/api/v1/libraries/L1":
            return httpx.Response(200, json=library)
        if path == "/api/v1/libraries/L1/scan" and method == "POST":
            self.scan()
            return httpx.Response(202)
        if path == "/api/v1/series":
            query = request.url.params.get("search", "").lower()
            found = [s for s in self.series.values() if query in s["metadata"]["title"].lower()]
            return httpx.Response(200, json={"content": found, "last": True})
        if path.startswith("/api/v1/series/"):
            series_id = path.split("/")[4]
            if series_id not in self.series:
                return httpx.Response(404, json={"message": "series not found"})
            if path.endswith("/metadata") and method == "PATCH":
                patch = json.loads(request.content)
                self.patches.append((series_id, patch))
                self.series[series_id]["metadata"]["title"] = patch.get("title", self.series[series_id]["metadata"]["title"])
                return httpx.Response(204)
            if path.endswith("/thumbnails") and method == "POST":
                assert request.url.params["selected"] == "true"
                self.covers.append((series_id, request.content))
                return httpx.Response(200, json={})
            return httpx.Response(200, json=self.series[series_id])
        return httpx.Response(404, json={"message": f"no route {method} {path}"})


@pytest.fixture
def komga(tmp_path):
    folder = tmp_path / "komga-library"
    folder.mkdir()
    return FakeKomga(folder)


@pytest.fixture
def workspace(isolated_home, komga, monkeypatch):
    registry.discover()
    ws = Workspace(device="cpu")
    monkeypatch.setattr(ws.automations.wake, "schedule", lambda when: None)
    ws.komga.transport = komga.transport
    ws.komga.update("http://komga.local:25600/", KEY)
    ws.komga.set_folder("L1", str(komga.folder))
    monkeypatch.setattr(links_module, "SCAN_POLL_SECONDS", 0.01)
    yield ws


def project_with_chapters(ws: Workspace, names=("Chapter 1", "Chapter 2")) -> tuple[str, list[str]]:
    project_id = ws.create_project("Blue Box").id
    for name in names:
        chapter = ws.store.create_chapter(project_id, name)
        for page in range(2):
            ws.add_page(project_id, f"{name}-{page}.png", page_png(), None, chapter)
    return project_id, [chapter.id for chapter in ws.project_view(project_id).chapters if chapter.page_ids]


def read_cbz(path: Path) -> tuple[list[str], ElementTree.Element]:
    with zipfile.ZipFile(path) as archive:
        return archive.namelist(), ElementTree.fromstring(archive.read("ComicInfo.xml"))


# ---- books and paths ---------------------------------------------------------------------------------


def test_a_book_carries_its_metadata_in_comic_info():
    book = BookInfo(series="Blue Box", title="Chapter 12.5", number=12.5, writers=["Kouji Miura"], genres=["Romance", "Sports"], language="en-US", right_to_left=True)
    info = ElementTree.fromstring(comic_info(book, pages=2))
    assert [element.tag for element in info] == ["Title", "Series", "Number", "Writer", "Genre", "PageCount", "LanguageISO", "Manga"]  # the schema's order
    assert info.findtext("Number") == "12.5" and info.findtext("LanguageISO") == "en" and info.findtext("Manga") == "YesAndRightToLeft"
    assert ElementTree.fromstring(comic_info(BookInfo(series="S", title="T", number=3.0), 1)).findtext("Number") == "3"

    with zipfile.ZipFile(io.BytesIO(pack_cbz([("0001_a.png", b"1"), ("0002_b.png", b"2")], book))) as archive:
        assert archive.namelist() == ["0001_a.png", "0002_b.png", "ComicInfo.xml"]


def test_series_folders_are_found_inside_their_library():
    assert relative_folder("/data/manga", "/data/manga/Blue Box") == "Blue Box"
    assert relative_folder("/data/manga/", "/data/manga/Shelf/Blue Box") == "Shelf/Blue Box"
    assert relative_folder("D:\\Manga", "D:\\Manga\\Blue Box") == "Blue Box"
    with pytest.raises(ValueError):
        relative_folder("/data/manga", "/elsewhere/Blue Box")
    with pytest.raises(ValueError):
        relative_folder("/data/manga", "/data/manga")
    assert series_folder(Path("/x"), "Shelf/Blue Box") == Path("/x/Shelf/Blue Box")
    for escape in ("../outside", "/abs", ""):
        with pytest.raises(ValueError):
            series_folder(Path("/x"), escape)


# ---- the connection -----------------------------------------------------------------------------------


def test_the_connection_finds_each_librarys_folder(isolated_home, komga):
    connection = KomgaConnection(transport=komga.transport)
    assert connection.status().configured is False

    status = connection.update("http://komga.local:25600/", KEY)
    assert status.connected and status.url == "http://komga.local:25600" and status.user == "admin@example.com" and status.version == "1.29.1"
    (library,) = status.libraries
    assert library.root == ROOT and library.folder is None and "choose where" in library.problem  # Komga's path is not here

    assert "rejected the API key" in connection.update("http://komga.local:25600", "wrong").error
    connection.update("http://komga.local:25600", KEY)
    with pytest.raises(ValueError, match="http://"):
        connection.update("komga.local:25600")

    komga.add_series("Existing")
    empty = isolated_home / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="holds none of Manga's series"):
        connection.set_folder("L1", str(empty))
    with pytest.raises(ValueError, match="does not exist"):
        connection.set_folder("L1", str(isolated_home / "missing"))
    (library,) = connection.set_folder("L1", str(komga.folder)).libraries
    assert (library.folder, library.folder_source, library.problem) == (str(komga.folder), "set", None)

    connection.update("http://komga.local:25600")  # the same server: the key and folders stay
    assert connection.settings().api_key == KEY and connection.settings().folders == {"L1": str(komga.folder)}
    connection.update("http://other:25600", KEY)  # another server: its libraries are others
    assert connection.settings().folders == {}


# ---- projects on Komga ------------------------------------------------------------------------------------


def test_a_project_added_as_a_new_series_is_exported_and_described(workspace, komga):
    ws = workspace
    project_id, chapters = project_with_chapters(ws, names=("Chapter 1",))
    remote = RemoteChapter(id="c2", name="Chapter 2", number=2.0)
    source = SourceLink(
        system="mana",
        source="site",
        source_name="Site",
        source_language="ja",
        series=Series(id="s1", title="Blue Box", summary="Taiki trains at dawn.", authors=["Kouji Miura"], status="Ongoing", tags=["Romance", "Sports"], url="https://site/blue-box"),
        chapters=[RemoteChapter(id="c1", name="Chapter 1", number=1.0), remote],
        linked_at=0.0,
        refreshed_at=0.0,
    )
    ws.links._save(project_id, source, Image(b"cover-bytes", "image/png"))
    ws.store.import_chapter(project_id, "Chapter 2", [("p1.png", page_png())], remote=remote_key(source, remote.id))
    chapters = [chapter.id for chapter in ws.project_view(project_id).chapters if chapter.page_ids]

    view = ws.komga_links.add(project_id, "L1")
    assert (view.title, view.folder, view.series_id, view.created) == ("Blue Box", "Blue Box", None, True)
    assert (komga.folder / "Blue Box").is_dir()
    with pytest.raises(ValueError, match="already has a folder"):
        ws.komga_links.add(project_with_chapters(ws)[0], "L1")  # another project, the same title

    report = ws.komga_links.export(project_id)
    assert report.error is None and report.books == 2 and not report.pending
    names, info = read_cbz(komga.folder / "Blue Box" / "Chapter 2.cbz")
    assert names[-1] == "ComicInfo.xml" and len(names) == 2
    assert (info.findtext("Series"), info.findtext("Number"), info.findtext("Writer"), info.findtext("LanguageISO"), info.findtext("Manga")) == (
        "Blue Box",
        "2",  # the source's number for a downloaded chapter
        "Kouji Miura",
        "en",
        "YesAndRightToLeft",
    )

    view = ws.komga_links.view(project_id)
    assert view.series_id == "S1" and view.series_url == "http://komga.local:25600/series/S1" and set(view.exported) == set(chapters)
    ((series_id, patch),) = komga.patches
    assert series_id == "S1" and patch["title"] == "Blue Box" and patch["titleLock"] is True
    assert (patch["summary"], patch["status"], patch["genres"], patch["readingDirection"]) == ("Taiki trains at dawn.", "ONGOING", ["Romance", "Sports"], "RIGHT_TO_LEFT")
    assert patch["links"] == [{"label": "Site", "url": "https://site/blue-box"}]
    assert komga.covers == [("S1", komga.covers[0][1])] and b"cover-bytes" in komga.covers[0][1]

    # Exporting again replaces the books in place, and leaves the series' metadata to Komga.
    again = ws.komga_links.export(project_id, chapters[:1])
    assert again.books == 1 and sorted(p.name for p in (komga.folder / "Blue Box").glob("*.cbz")) == ["Chapter 1.cbz", "Chapter 2.cbz"]
    assert len(komga.patches) == 1 and len(komga.covers) == 1


def test_a_linked_series_keeps_the_books_it_had(workspace, komga):
    ws = workspace
    project_id, chapters = project_with_chapters(ws, names=("Their own",))
    series_id = komga.add_series("Blue Box Raw")
    view = ws.komga_links.link(project_id, series_id)
    assert (view.folder, view.series_id, view.created) == ("Blue Box Raw", series_id, False)

    report = ws.komga_links.export(project_id)
    assert report.books == 1
    folder = komga.folder / "Blue Box Raw"
    assert (folder / "Their own.cbz").read_bytes() == b"not ours"  # never replaced
    assert read_cbz(folder / "Their own (2).cbz")[1].findtext("Series") == "Blue Box Raw"  # Komga's title, so a scan does not rename it
    assert komga.patches == [] and komga.covers == []  # a series Urakatsu did not make is not described

    ws.komga_links.unlink(project_id)
    assert ws.komga_links.view(project_id) is None and (folder / "Their own (2).cbz").exists()
    assert ws.komga_links.export(project_id).error == "the project is not on Komga any more"


def test_export_failures_are_reported_not_raised(workspace, komga):
    ws = workspace
    project_id, _ = project_with_chapters(ws)
    ws.komga_links.add(project_id, "L1")
    ws.komga.update("http://komga.local:25600", "revoked")
    report = ws.komga_links.export(project_id)
    assert report.books == 0 and "rejected the API key" in report.error
    assert ws.komga_links.view(project_id).last_export.error == report.error


# ---- automations and routes --------------------------------------------------------------------------------


async def test_an_automation_exports_the_chapters_it_finished(workspace, komga, monkeypatch):
    ws = workspace
    project_id, chapters = project_with_chapters(ws)
    with pytest.raises(ValueError, match="not on Komga"):
        ws.automations.create(project_id, AutomationBody(komga=True))
    ws.komga_links.add(project_id, "L1")

    async def batch(pipeline, session, picked, stages, stop, **kwargs):
        return BatchResult(stopped=False, failed={picked[1].pages[0]: ["translation: failed"]})  # chapter 2 is partly done

    monkeypatch.setattr(automations_module, "run_batch", batch)
    sent = []

    async def post(url, payload):
        sent.append(payload)

    monkeypatch.setattr(automations_module, "post_discord", post)
    notify = automations_module.Notify(enabled=True, webhook_url="https://discord.com/api/webhooks/1/abc")
    automation = ws.automations.create(project_id, AutomationBody(name="Nightly", schedule=Schedule(kind="manual"), komga=True, notify=notify))
    ws.automations.run_now(project_id, automation.id)
    while ws.automations._worker is not None and not ws.automations._worker.done():
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)

    record = ws.automations.runs(project_id)[0]
    assert record.status == "completed" and record.komga.books == 1 and record.komga.error is None
    assert set(ws.komga_links.view(project_id).exported) == {chapters[0]}  # only the chapter that finished
    assert [p.name for p in (komga.folder / "Blue Box").glob("*.cbz")] == ["Chapter 1.cbz"]
    fields = {field["name"]: field["value"] for field in sent[0]["embeds"][0]["fields"]}
    assert fields["Komga"] == "1 chapter exported"


async def test_komga_routes(workspace, komga):
    ws = workspace
    project_id, chapters = project_with_chapters(ws)
    app = build_app(ws, UrakatsuServicer(ws), serve_ui=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test/api/v1") as client:
        status = (await client.get("/komga")).json()
        assert status["connected"] and status["libraries"][0]["folder"] == str(komga.folder) and "api_key" not in json.dumps(status)
        assert (await client.put("/komga", json={"url": "nope"})).status_code == 400
        assert (await client.get(f"/projects/{project_id}/komga")).json() is None
        assert (await client.post(f"/projects/{project_id}/komga/export", json={})).status_code == 404

        added = await client.post(f"/projects/{project_id}/komga", json={"library_id": "L1", "title": "Blue Box (EN)"})
        assert added.status_code == 200 and added.json()["folder"] == "Blue Box (EN)"
        started = await client.post(f"/projects/{project_id}/komga/export", json={"chapter_ids": chapters[:1]})
        assert started.status_code == 200
        deadline = time.monotonic() + 30
        while (view := (await client.get(f"/projects/{project_id}/komga")).json())["export"] is not None and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert view["last_export"]["books"] == 1 and view["series_url"].endswith("/series/S1")
        found = (await client.get("/komga/libraries/L1/series", params={"q": "blue"})).json()["series"]
        assert [s["title"] for s in found] == ["Blue Box (EN)"]
        assert (await client.delete(f"/projects/{project_id}/komga")).status_code == 204
