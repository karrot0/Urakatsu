"""Project -> Chapters -> Pages: storage, arrangement, undo, and legacy projects."""

import io

import pytest
from PIL import Image

from urakatsu.pipeline.projects import DEFAULT_CHAPTER_ID, ProjectStore, chapters_of


def png(color: int = 0) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (color, color, color)).save(buffer, format="PNG")
    return buffer.getvalue()


def layout(store: ProjectStore, project: str) -> list[tuple[str, list[str]]]:
    return [(ch.id, ch.pages) for ch in chapters_of(store.session(project).snapshot())]


def test_new_project_has_one_implicit_chapter_that_collects_pages(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create("p")
    assert layout(store, project) == [(DEFAULT_CHAPTER_ID, [])]
    a = store.import_page(project, "a.png", png())
    b = store.import_page(project, "b.png", png())
    assert layout(store, project) == [(DEFAULT_CHAPTER_ID, [a, b])]
    assert store.list()[0].chapters == 1


def test_swapping_a_source_keeps_the_place_and_label_and_can_be_undone(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create("p")
    a = store.import_page(project, "a.png", png())
    b = store.import_page(project, "b.png", png())
    second = store.create_chapter(project, "Chapter 2")
    c = store.import_page(project, "c.png", png(), chapter_id=second)
    bigger = io.BytesIO()
    Image.new("RGB", (16, 24), (9, 9, 9)).save(bigger, format="PNG")

    swapped = store.swap_page_source(project, a, bigger.getvalue())

    snapshot = store.session(project).snapshot()
    assert layout(store, project) == [(DEFAULT_CHAPTER_ID, [swapped, b]), (second, [c])]
    assert snapshot.pages() == [swapped, b, c]
    page = snapshot.page(swapped)
    assert (page.label, page.width, page.height) == ("a.png", 16, 24)
    assert snapshot.blobs.read(snapshot.asset(swapped, "source").blob) == bigger.getvalue()

    store.session(project).undo()
    assert layout(store, project) == [(DEFAULT_CHAPTER_ID, [a, b]), (second, [c])]


def test_a_swap_that_cannot_read_the_image_leaves_the_page_alone(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create("p")
    a = store.import_page(project, "a.png", png())
    with pytest.raises(OSError):
        store.swap_page_source(project, a, b"not an image")
    assert layout(store, project) == [(DEFAULT_CHAPTER_ID, [a])]


def test_pages_import_into_the_requested_chapter_and_keep_page_order_flat(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create("p")
    one = store.import_page(project, "1.png", png())
    second = store.create_chapter(project, "Chapter 2")
    two = store.import_page(project, "2.png", png(), chapter_id=second)
    first_b = store.import_page(project, "1b.png", png(), chapter_id=DEFAULT_CHAPTER_ID)
    first_a = store.import_page(project, "1a.png", png(), index=0, chapter_id=DEFAULT_CHAPTER_ID)
    assert layout(store, project) == [(DEFAULT_CHAPTER_ID, [first_a, one, first_b]), (second, [two])]
    assert store.session(project).snapshot().pages() == [first_a, one, first_b, two]


def test_arrange_moves_pages_between_chapters_and_reorders_chapters(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create("p")
    a = store.import_page(project, "a.png", png())
    b = store.import_page(project, "b.png", png())
    second = store.create_chapter(project, "Two")
    store.arrange(project, [(second, [b]), (DEFAULT_CHAPTER_ID, [a])])
    assert layout(store, project) == [(second, [b]), (DEFAULT_CHAPTER_ID, [a])]
    assert store.session(project).snapshot().pages() == [b, a]
    with pytest.raises(ValueError):
        store.arrange(project, [(second, [b, a])])  # a chapter is missing
    with pytest.raises(ValueError):
        store.arrange(project, [(second, [b]), (DEFAULT_CHAPTER_ID, [])])  # a page is missing


def test_chapter_edits_are_undoable(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create("p")
    a = store.import_page(project, "a.png", png())
    second = store.create_chapter(project, "Two")
    store.arrange(project, [(DEFAULT_CHAPTER_ID, []), (second, [a])])
    store.session(project).undo()
    assert layout(store, project) == [(DEFAULT_CHAPTER_ID, [a]), (second, [])]
    store.rename_chapter(project, second, "Renamed")
    assert chapters_of(store.session(project).snapshot())[1].name == "Renamed"


def test_deleting_a_chapter_removes_its_pages(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create("p")
    keep = store.import_page(project, "keep.png", png())
    doomed = store.create_chapter(project, "Doomed")
    store.import_page(project, "x.png", png(), chapter_id=doomed)
    store.delete_chapter(project, doomed)
    assert layout(store, project) == [(DEFAULT_CHAPTER_ID, [keep])]
    assert store.session(project).snapshot().pages() == [keep]


def test_removing_a_page_updates_its_chapter(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create("p")
    a = store.import_page(project, "a.png", png())
    b = store.import_page(project, "b.png", png())
    store.remove_page(project, a)
    assert layout(store, project) == [(DEFAULT_CHAPTER_ID, [b])]


def test_chapters_survive_reopening_the_project(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create("p")
    second = store.create_chapter(project, "Two")
    a = store.import_page(project, "a.png", png(), chapter_id=second)
    reopened = ProjectStore(tmp_path)
    assert layout(reopened, project) == [(DEFAULT_CHAPTER_ID, []), (second, [a])]


# -- how page images are stored ----------------------------------------------------------------


def encoded(image: Image.Image, format: str, **options) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=format, **options)
    return buffer.getvalue()


def page_art() -> Image.Image:
    """A page-like image: flat paper, ink lines, a gradient, so lossless WebP has work to do."""
    image = Image.new("L", (96, 128), 245)
    for y in range(0, 128, 9):
        image.paste(30, (8, y, 88, y + 2))
    image.paste(Image.linear_gradient("L").resize((40, 40)), (28, 44))
    return image.convert("RGB")


def source_of(store: ProjectStore, project: str, page: str):
    snapshot = store.session(project).snapshot()
    asset = snapshot.asset(page, "source")
    return asset.media_type, snapshot.blobs.read(asset.blob)


@pytest.mark.parametrize("format, media_type", [("JPEG", "image/jpeg"), ("PNG", "image/png"), ("WEBP", "image/webp")])
def test_pages_keep_the_bytes_they_arrive_as(tmp_path, format, media_type):
    store = ProjectStore(tmp_path)
    project = store.create("p")
    data = encoded(page_art(), format)
    page = store.import_page(project, "page", data)
    assert source_of(store, project, page) == (media_type, data)


def test_pages_the_stages_or_browsers_would_read_differently_become_png(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create("p")
    cmyk = encoded(page_art().convert("CMYK"), "JPEG")
    exif = Image.Exif()
    exif[0x0112] = 6  # rotated: browsers turn it, Pillow and the stages would not
    rotated = encoded(page_art(), "JPEG", exif=exif.tobytes())
    for data in (cmyk, rotated):
        media_type, stored = source_of(store, project, store.import_page(project, "page", data))
        assert media_type == "image/png" and Image.open(io.BytesIO(stored)).mode in ("RGB", "L")


def test_lossless_pages_store_png_as_smaller_webp_with_the_same_pixels(tmp_path):
    store = ProjectStore(tmp_path)
    store.lossless_pages = True
    project = store.create("p")
    png_data = encoded(page_art(), "PNG")
    jpeg_data = encoded(page_art(), "JPEG")
    chapter = store.import_chapter(project, "c", [("1", png_data), ("2", jpeg_data)])
    first, second = next(c.pages for c in chapters_of(store.session(project).snapshot()) if c.id == chapter)
    media_type, stored = source_of(store, project, first)
    assert media_type == "image/webp" and len(stored) < len(png_data)
    assert Image.open(io.BytesIO(stored)).convert("RGB").tobytes() == page_art().tobytes()
    assert source_of(store, project, second) == ("image/jpeg", jpeg_data)  # never re-encoded


def test_a_chapter_import_writes_every_page_blob(tmp_path):
    """Blobs of one commit are written concurrently; each page must still land intact."""
    store = ProjectStore(tmp_path)
    project = store.create("p")
    pages = [(f"{n:02d}", png(n * 9)) for n in range(24)]
    chapter = store.import_chapter(project, "c", pages)
    ids = next(c.pages for c in chapters_of(store.session(project).snapshot()) if c.id == chapter)
    assert [source_of(store, project, page)[1] for page in ids] == [data for _, data in pages]
    reopened = ProjectStore(tmp_path)
    assert [source_of(reopened, project, page)[1] for page in ids] == [data for _, data in pages]


def test_the_source_route_swaps_a_page_and_waits_for_running_jobs(isolated_home):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from urakatsu.service.http_api import api_router
    from urakatsu.service.workspace import Workspace

    workspace = Workspace(device="cpu")
    app = FastAPI()
    app.include_router(api_router(workspace), prefix="/api/v1")
    client = TestClient(app)
    project = workspace.create_project("p").id
    page = workspace.add_page(project, "a.png", png()).id
    url = f"/api/v1/projects/{project}/pages/{page}/source"
    image = {"file": ("new.png", png(255), "image/png")}

    swapped = client.put(url, files=image)
    assert swapped.status_code == 200
    assert swapped.json()["name"] == "a.png" and swapped.json()["id"] != page
    assert [p.id for p in workspace.project_view(project).pages] == [swapped.json()["id"]]

    assert "not found" in client.put(url, files=image).json()["detail"]  # the old page is gone
    workspace.jobs["running"] = type("Job", (), {"project_id": project, "status": "running"})()
    assert client.put(f"/api/v1/projects/{project}/pages/{swapped.json()['id']}/source", files=image).status_code == 412
