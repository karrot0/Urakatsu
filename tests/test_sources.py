"""Source systems: the layer, project links, chapter downloads and the JSON routes, with a fake system."""

import io
import time

import pytest
from PIL import Image as PILImage

from urakatsu.pipeline.projects import ProjectStore, chapters_of
from urakatsu.runtimes.packages import PackageManager
from urakatsu.sources import Image, RemoteChapter, Repository, SearchPage, Series, SourceChallenge, SourceInfo, SourceSystem, SourceSystemError, SystemStatus
from urakatsu.sources.links import SourceLinks, placement
from urakatsu.sources.mana.system import ManaSystem, repository_root, usable
from urakatsu.sources.registry import SourceSystems, discover


def png(color: str = "white", size=(60, 90)) -> bytes:
    buffer = io.BytesIO()
    PILImage.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


class FakeSystem(SourceSystem):
    id = "fake"
    name = "Fake"
    description = "In-memory source system for tests."
    homepage = "https://example.invalid"

    CHAPTERS = [
        RemoteChapter(id="c1", name="Chapter 1", number=1),
        RemoteChapter(id="c2a", name="Chapter 2", number=2, scanlator="Group A"),
        RemoteChapter(id="c2b", name="Chapter 2", number=2, scanlator="Group B"),
        RemoteChapter(id="c3", name="Chapter 3", number=3),
    ]

    def __init__(self, root, packages):
        super().__init__(root, packages)
        self.running = False
        self.broken: set[str] = set()
        self.checked: set[str] = set()

    def status(self):
        return SystemStatus(id=self.id, name=self.name, description=self.description, homepage=self.homepage, state="running" if self.running else "stopped")

    def start(self):
        self.running = True
        return self.status()

    def stop(self):
        self.running = False

    def repositories(self):
        return [Repository(url="https://example.invalid/repo", name="Repo", extensions=1)]

    def add_repository(self, url):
        return self.repositories()

    def remove_repository(self, url):
        return []

    def extensions(self, refresh=False):
        return []

    def install_extension(self, extension_id):
        return []

    def uninstall_extension(self, extension_id):
        return []

    def sources(self):
        return [SourceInfo(id="site", name="Site", language="ja")]

    def search(self, source_id, query, page=1):
        if query == "checked":
            raise SourceChallenge("example.invalid asks for a browser check", "https://example.invalid/check", self.id)
        return SearchPage(results=[Series(id="s1", title="Series One", cover="cover://s1")], has_more=False)

    def series(self, source_id, series_id):
        return Series(id=series_id, title="Series One", cover="cover://s1", summary="A test series.")

    def chapters(self, source_id, series_id):
        return list(self.CHAPTERS)

    def pages(self, source_id, series_id, chapter_id):
        return [f"{chapter_id}/{n}" for n in range(3)]

    def page(self, source_id, series_id, chapter_id, handle):
        if chapter_id in self.broken:
            raise SourceSystemError("gone")
        if chapter_id in self.checked:
            raise SourceChallenge("example.invalid asks for a browser check", "https://example.invalid/check", self.id)
        return Image(png("gray"), "image/jpeg")

    def cover(self, source_id, url):
        return Image(png("red"), "image/jpeg")


@pytest.fixture
def setup(isolated_home):
    store = ProjectStore()
    systems = SourceSystems(classes=[FakeSystem], packages=PackageManager(offline=True))
    links = SourceLinks(store, systems)
    yield store, systems, links
    links.close()


def wait_for(links, project, chapter_ids, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        view = links.view(project)
        states = {c.id: c for c in view.chapters if c.id in chapter_ids}
        if all(c.chapter_id or (c.download and c.download.state == "failed") for c in states.values()):
            return states
        time.sleep(0.05)
    raise AssertionError("downloads did not finish")


def test_builtin_systems_are_discovered():
    ids = [cls.id for cls in discover()]
    assert ids[:2] == ["suwayomi", "mana"]


def test_enabled_systems_persist(setup):
    _, systems, _ = setup
    assert systems.status("fake").enabled is False
    assert systems.set_enabled("fake", True).enabled is True
    assert SourceSystems(classes=[FakeSystem]).enabled() == {"fake"}
    with pytest.raises(KeyError):
        systems.get("missing")


def test_link_names_project_drops_placeholder_and_keeps_cover(setup):
    store, _, links = setup
    project = store.create("Untitled")
    view = links.link(project, "fake", "site", "s1")
    assert view.series.title == "Series One"
    assert store.name(project) == "Series One"
    assert [c.id for c in view.chapters] == ["c1", "c2a", "c2b", "c3"]
    assert all(c.chapter_id is None for c in view.chapters)
    assert chapters_of(store.session(project).snapshot()) == []
    assert links.cover(project).media_type == "image/jpeg"


def test_downloads_land_in_source_order(setup):
    store, _, links = setup
    project = store.create("Mine")
    links.link(project, "fake", "site", "s1")
    assert store.name(project) == "Mine"
    links.download(project, ["c3"])
    wait_for(links, project, {"c3"})
    links.download(project, ["c1", "c2b"])
    done = wait_for(links, project, {"c1", "c2b"})
    chapters = chapters_of(store.session(project).snapshot())
    assert [c.name for c in chapters] == ["Chapter 1", "Chapter 2 (Group B)", "Chapter 3"]
    assert [len(c.pages) for c in chapters] == [3, 3, 3]
    assert done["c1"].chapter_id == chapters[0].id
    # A downloaded chapter is never fetched again.
    view = links.download(project, ["c1"])
    assert next(c for c in view.chapters if c.id == "c1").download is None


def test_failed_download_reports_the_page(setup):
    store, systems, links = setup
    project = store.create("Mine")
    links.link(project, "fake", "site", "s1")
    systems.get("fake").broken.add("c1")
    links.download(project, ["c1"])
    state = wait_for(links, project, {"c1"})
    assert state["c1"].download.state == "failed"
    assert "page 1 of 3" in state["c1"].download.error
    systems.get("fake").broken.clear()
    links.download(project, ["c1"])
    assert wait_for(links, project, {"c1"})["c1"].chapter_id


def test_relinking_another_series_does_not_claim_old_chapters(setup):
    store, _, links = setup
    project = store.create("Mine")
    links.link(project, "fake", "site", "s1")
    links.download(project, ["c1"])
    wait_for(links, project, {"c1"})
    view = links.link(project, "fake", "site", "s2")
    assert all(c.chapter_id is None for c in view.chapters)
    links.unlink(project)
    assert links.view(project) is None
    assert len(chapters_of(store.session(project).snapshot())) == 1


def test_remote_key_survives_arrange(setup):
    store, _, links = setup
    project = store.create("Mine")
    links.link(project, "fake", "site", "s1")
    links.download(project, ["c1", "c3"])
    wait_for(links, project, {"c1", "c3"})
    chapters = chapters_of(store.session(project).snapshot())
    store.arrange(project, [(c.id, c.pages) for c in reversed(chapters)])
    assert {c.remote for c in chapters_of(store.session(project).snapshot())} == {c.remote for c in chapters}


def test_placement_goes_before_the_next_downloaded_chapter():
    from urakatsu.scene import Chapter

    order = {"a": 0, "b": 1, "c": 2, "d": 3}
    chapters = [Chapter(id="1", name="A", remote="a"), Chapter(id="x", name="Mine"), Chapter(id="3", name="D", remote="d")]
    assert placement(order, "b")(chapters) == 2
    assert placement(order, "d")([]) == 0
    assert placement(order, "c")(chapters[:2]) == 1


def test_import_chapter_is_one_undoable_commit(isolated_home):
    store = ProjectStore()
    project = store.create("p")
    store.import_page(project, "loose", png())
    session = store.session(project)
    revision = session.snapshot().revision
    store.import_chapter(project, "New", [("1", png()), ("2", png())], place=lambda chapters: 0, remote="x")
    snapshot = session.snapshot()
    assert snapshot.revision == revision + 1
    chapters = chapters_of(snapshot)
    assert chapters[0].name == "New" and chapters[0].remote == "x" and len(chapters[0].pages) == 2
    session.undo()
    assert [c.name for c in chapters_of(session.snapshot())] == ["Chapter 1"]


def test_mana_repository_urls_and_intents():
    assert repository_root("https://x.github.io/repo/main/") == "https://x.github.io/repo/main"
    assert repository_root("https://x.github.io/repo/main/sources.json") == "https://x.github.io/repo/main"
    with pytest.raises(ValueError):
        repository_root("x.github.io/repo")
    assert usable({"intents": 102763528})  # Batcave: search + chapters
    assert not usable({"intents": 1 << 10})  # search only
    assert not usable({"intents": (1 << 10) | (1 << 21) | (1 << 1)})  # needs in-app setup


def test_mana_chapters_sort_by_number(isolated_home, monkeypatch):
    system = ManaSystem(isolated_home / "mana", PackageManager(offline=True))
    raw = [
        {"chapterId": "3", "title": None, "number": 3, "index": 0},
        {"chapterId": "1", "title": "First", "number": 1, "index": 2},
        {"chapterId": "2", "title": None, "number": 2, "index": 1},
    ]
    monkeypatch.setattr(system, "_call", lambda *args, **kwargs: raw)
    chapters = system.chapters("src", "series")
    assert [c.id for c in chapters] == ["1", "2", "3"]
    assert [c.name for c in chapters] == ["First", "Chapter 2", "Chapter 3"]


def test_routes(isolated_home):
    from fastapi import FastAPI

    from urakatsu.service.sources_api import sources_router
    from urakatsu.service.workspace import Workspace

    ws = Workspace(device="cpu")
    ws.systems = SourceSystems(classes=[FakeSystem], packages=PackageManager(offline=True))
    ws.links = SourceLinks(ws.store, ws.systems)
    app = FastAPI()
    app.include_router(sources_router(ws), prefix="/api/v1")
    from fastapi.testclient import TestClient

    client = TestClient(app)
    assert client.get("/api/v1/sources/systems").json()[0]["id"] == "fake"
    listed = client.get("/api/v1/sources").json()
    assert listed[0]["sources"] == []  # not enabled yet
    assert client.put("/api/v1/sources/systems/fake/enabled", json={"enabled": True}).json()["state"] == "running"
    assert client.get("/api/v1/sources").json()[0]["sources"][0]["id"] == "site"
    assert client.get("/api/v1/sources/systems/fake/sources/site/search", params={"q": "one"}).json()["results"][0]["id"] == "s1"
    project = ws.create_project("Untitled").id
    assert client.get(f"/api/v1/projects/{project}/source").json() is None
    linked = client.put(f"/api/v1/projects/{project}/source", json={"system": "fake", "source": "site", "series": "s1"}).json()
    assert linked["series"]["title"] == "Series One"
    cover = client.get(f"/api/v1/projects/{project}/cover")
    assert cover.status_code == 200 and cover.headers["content-type"] == "image/jpeg"
    assert client.post(f"/api/v1/projects/{project}/source/downloads", json={"chapter_ids": ["nope"]}).status_code == 404
    assert client.get("/api/v1/sources/systems/missing/extensions").status_code == 404
    assert client.delete(f"/api/v1/projects/{project}/source").status_code == 204
    ws.links.close()


def test_download_reports_a_site_check(setup):
    store, systems, links = setup
    project = store.create("Mine")
    links.link(project, "fake", "site", "s1")
    systems.get("fake").checked.add("c1")
    links.download(project, ["c1"])
    state = wait_for(links, project, {"c1"})["c1"].download
    assert state.state == "failed" and state.challenge == "https://example.invalid/check"


def test_systems_start_in_the_background(setup):
    _, systems, _ = setup
    systems.start_background("fake").join(5)
    assert systems.status("fake").state == "running"


def test_challenge_and_browser_routes(isolated_home):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from urakatsu.service.sources_api import sources_router
    from urakatsu.service.workspace import Workspace

    ws = Workspace(device="cpu")
    ws.systems = SourceSystems(classes=[FakeSystem], packages=PackageManager(offline=True))
    ws.links = SourceLinks(ws.store, ws.systems)
    app = FastAPI()
    app.include_router(sources_router(ws), prefix="/api/v1")
    client = TestClient(app)
    answer = client.get("/api/v1/sources/systems/fake/sources/site/search", params={"q": "checked"})
    assert answer.status_code == 409
    assert answer.json()["detail"]["challenge"] == {"system": "fake", "url": "https://example.invalid/check"}
    # A system without a built-in browser says so.
    assert client.post("/api/v1/sources/systems/fake/browser", json={"url": "https://example.invalid"}).status_code == 501
    ws.links.close()


def test_browser_runtime_lookup(isolated_home, monkeypatch, tmp_path):
    from urakatsu.runtimes.browser import BrowserRuntime

    runtime = BrowserRuntime(tmp_path / "chrome", offline=True)
    executable = tmp_path / "chrome.exe"
    executable.write_bytes(b"")
    monkeypatch.setenv("URAKATSU_BROWSER", str(executable))
    assert runtime.find().kind == "configured"
    monkeypatch.setenv("URAKATSU_BROWSER", "managed")
    assert runtime.find() is None
    with pytest.raises(RuntimeError, match="offline"):
        runtime.ensure()


# -- covers and the cached source list --------------------------------------------------------------


def test_covers_are_kept_at_tile_size_and_fetched_once(tmp_path):
    import io
    import threading

    from PIL import Image as Picture

    from urakatsu.sources.covers import TILE, CoverCache

    buffer = io.BytesIO()
    Picture.new("RGB", (1200, 1800), (200, 30, 30)).save(buffer, format="PNG")
    big = Image(buffer.getvalue(), "image/png")
    fetched = []
    gate = threading.Event()

    def fetch():
        fetched.append(1)
        gate.wait(5)
        return big

    covers = CoverCache(tmp_path)
    key = CoverCache.key("fake", "site", "cover://s1")
    threads = [threading.Thread(target=lambda: results.append(covers.get(key, fetch))) for _ in range(4)]
    results: list = []
    for thread in threads:
        thread.start()
    gate.set()
    for thread in threads:
        thread.join(5)
    assert len(fetched) == 1 and len(results) == 4  # concurrent requests share one download
    stored = results[0]
    assert stored.media_type == "image/webp" and len(stored.data) < len(big.data) / 10
    assert Picture.open(io.BytesIO(stored.data)).size[1] <= TILE[1]
    assert CoverCache(tmp_path).get(key, lambda: pytest.fail("cached covers are not fetched again")).data == stored.data
    with pytest.raises(SourceSystemError):
        covers.get(CoverCache.key("fake", "site", "broken"), lambda: (_ for _ in ()).throw(SourceSystemError("gone")))
    assert covers.cached(CoverCache.key("fake", "site", "broken")) is None  # a failure is not cached


def test_the_source_list_is_reused_until_extensions_change(isolated_home):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from urakatsu.service.sources_api import sources_router
    from urakatsu.service.workspace import Workspace

    calls = []

    class Counting(FakeSystem):
        def sources(self):
            calls.append(1)
            return super().sources()

    ws = Workspace(device="cpu")
    ws.systems = SourceSystems(classes=[Counting], packages=PackageManager(offline=True))
    ws.links = SourceLinks(ws.store, ws.systems)
    app = FastAPI()
    app.include_router(sources_router(ws), prefix="/api/v1")
    client = TestClient(app)
    client.put("/api/v1/sources/systems/fake/enabled", json={"enabled": True})
    for _ in range(3):
        assert client.get("/api/v1/sources").json()[0]["sources"][0]["id"] == "site"
    assert len(calls) == 1
    client.post("/api/v1/sources/systems/fake/extensions/x/install")
    client.get("/api/v1/sources")
    assert len(calls) == 2  # installing an extension drops the cached list
    cover = client.get("/api/v1/sources/systems/fake/sources/site/cover", params={"url": "cover://s1"})
    assert cover.status_code == 200 and "max-age=604800" in cover.headers["cache-control"]
    ws.links.close()
