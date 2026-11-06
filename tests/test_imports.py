"""Importing archives (ZIP/CBZ) and folders: how files become chapters, and the one-commit import."""

import io
import zipfile

import pytest
from PIL import Image

from urakatsu.pipeline.imports import gather, natural_key
from urakatsu.pipeline.projects import DEFAULT_CHAPTER_ID, ProjectStore, chapters_of
from urakatsu.pipeline.stitch import ProjectFormat


def png(color: int = 0, size: tuple[int, int] = (8, 8)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (color, color, color)).save(buffer, format="PNG")
    return buffer.getvalue()


def archive(entries: dict[str, bytes], utf8: bool = True) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, data in entries.items():
            zf.writestr(zipfile.ZipInfo(name) if utf8 else _ShiftJis(name), data)
    return buffer.getvalue()


class _ShiftJis(zipfile.ZipInfo):
    """An entry named the way a Japanese Windows zip tool names it: Shift-JIS, no UTF-8 flag."""

    def _encodeFilenameFlags(self):
        return self.filename.encode("cp932"), self.flag_bits & ~0x800


def shape(batch) -> tuple[list[tuple[str, list[str]]], list[str]]:
    return [(name, [label for label, _ in pages]) for name, pages in batch.chapters], [label for label, _ in batch.loose]


def test_natural_order_puts_2_before_10():
    assert sorted(["10.jpg", "2.jpg", "1.jpg", "Ch 10", "ch 9"], key=natural_key) == ["1.jpg", "2.jpg", "10.jpg", "ch 9", "Ch 10"]


def test_an_archive_is_one_chapter_named_after_it_in_page_order():
    data = archive({"10.png": png(), "2.png": png(), "ComicInfo.xml": b"<x/>", "__MACOSX/._2.png": b"junk", ".hidden.png": b"junk"})
    assert shape(gather([("Vol 1 Ch 3.cbz", data)])) == ([("Vol 1 Ch 3", ["2.png", "10.png"])], [])


def test_an_archive_with_its_pages_in_one_inner_folder_keeps_the_archive_name():
    data = archive({"inner/b.png": png(), "inner/a.png": png()})
    assert shape(gather([("c12.zip", data)])) == ([("c12", ["a.png", "b.png"])], [])


def test_an_archive_of_folders_is_a_chapter_per_folder():
    data = archive({"Ch 10/1.png": png(), "Ch 2/1.png": png(), "Ch 2/2.png": png(), "cover.png": png()})
    assert shape(gather([("vol.zip", data)])) == ([("vol", ["cover.png"]), ("Ch 2", ["1.png", "2.png"]), ("Ch 10", ["1.png"])], [])


def test_a_zip_of_cbz_files_is_a_chapter_per_cbz():
    data = archive({"x/Ch 2.cbz": archive({"1.png": png()}), "Ch 1.cbz": archive({"1.png": png(), "2.png": png()})})
    assert shape(gather([("Volume 1.zip", data)])) == ([("Ch 1", ["1.png", "2.png"]), ("Ch 2", ["1.png"])], [])


def test_folders_become_chapters_and_loose_images_stay_loose():
    files = [
        ("Series/Ch 2/b.png", png()),
        ("Series/Ch 2/a.png", png()),
        ("Series/Ch 1/a.png", png()),
        ("Series/Ch 3.cbz", archive({"x.png": png()})),
        ("Other/1.png", png()),
        ("loose 2.png", png()),
        ("loose 10.png", png()),
        ("notes.txt", b"text"),
    ]
    chapters, loose = shape(gather(files))
    assert chapters == [("Other", ["1.png"]), ("Ch 1", ["a.png"]), ("Ch 2", ["a.png", "b.png"]), ("Ch 3", ["x.png"])]
    assert loose == ["loose 2.png", "loose 10.png"]


def test_shift_jis_entry_names_are_read_as_japanese():
    data = archive({"第1話/001.png": png(), "第1話/002.png": png(), "第2話/001.png": png()}, utf8=False)
    assert [name for name, _ in gather([("巻.zip", data)]).chapters] == ["第1話", "第2話"]


def test_not_a_zip_and_password_protected_archives_are_named_in_the_error():
    with pytest.raises(ValueError, match="book.cbz is not a ZIP archive"):
        gather([("book.cbz", b"Rar!\x1a\x07\x00")])
    data = bytearray(archive({"1.png": png()}))
    # Set the "encrypted" flag on the entry's local and central headers.
    for signature, offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        at = data.index(signature) + offset
        data[at] |= 1
    with pytest.raises(ValueError, match="locked.cbz is password protected"):
        gather([("locked.cbz", bytes(data))])


def test_a_new_project_takes_an_archive_as_its_first_chapter_in_one_undo_step(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create("p")
    batch = gather([("Ch 1.cbz", archive({"1.png": png(), "2.png": png()})), ("Ch 2.cbz", archive({"1.png": png()}))])
    pages = store.import_chapters(project, batch.chapters, batch.loose)
    chapters = chapters_of(store.session(project).snapshot())
    assert [(ch.name, len(ch.pages)) for ch in chapters] == [("Ch 1", 2), ("Ch 2", 1)]
    assert chapters[0].id == DEFAULT_CHAPTER_ID  # the empty implicit chapter was filled, not left behind
    assert pages == [page for ch in chapters for page in ch.pages]
    store.session(project).undo()
    assert [(ch.name, ch.pages) for ch in chapters_of(store.session(project).snapshot())] == [("Chapter 1", [])]


def test_chapters_follow_the_target_and_loose_pages_join_it(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create("p")
    first = store.import_page(project, "a.png", png())
    middle = store.create_chapter(project, "Middle")
    store.import_page(project, "m.png", png(), chapter_id=middle)
    store.create_chapter(project, "Last")
    batch = gather([("z.png", png()), ("New.cbz", archive({"1.png": png()}))])
    pages = store.import_chapters(project, batch.chapters, batch.loose, chapter_id=middle)
    chapters = chapters_of(store.session(project).snapshot())
    assert [(ch.name, len(ch.pages)) for ch in chapters] == [("Chapter 1", 1), ("Middle", 2), ("New", 1), ("Last", 0)]
    assert first not in pages and len(pages) == 2


def test_an_empty_chapter_the_user_named_keeps_its_name(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create("p")
    store.import_page(project, "a.png", png())
    named = store.create_chapter(project, "The Return")
    batch = gather([("c012.cbz", archive({"1.png": png()}))])
    store.import_chapters(project, batch.chapters, batch.loose, chapter_id=named)
    assert [(ch.name, len(ch.pages)) for ch in chapters_of(store.session(project).snapshot())] == [("Chapter 1", 1), ("The Return", 1)]


def test_an_unreadable_image_is_named_and_nothing_is_added(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create("p")
    batch = gather([("Ch.cbz", archive({"1.png": png(), "2.png": b"not an image"}))])
    with pytest.raises(ValueError, match="2.png is not an image"):
        store.import_chapters(project, batch.chapters, batch.loose)
    assert store.session(project).snapshot().pages() == []


def test_webtoon_chapters_are_restitched_on_import(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create("p")
    store.set_format(project, ProjectFormat(kind="webtoon"))
    strips = {f"{n}.png": png(255, (200, 300)) for n in range(1, 5)}
    batch = gather([("Ep 1.cbz", archive(strips))])
    store.import_chapters(project, batch.chapters, batch.loose)
    (chapter,) = chapters_of(store.session(project).snapshot())
    assert chapter.name == "Ep 1" and chapter.pages
    snapshot = store.session(project).snapshot()
    assert sum(snapshot.page(page).height for page in chapter.pages) == 1200
