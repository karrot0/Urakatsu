"""Webtoon stitching and slicing: where cuts land, and restitching chapters in a project."""

import io

import pytest
from PIL import Image, ImageDraw

from urakatsu.core.results import LayoutDetection, LayoutDetections, LayoutMask
from urakatsu.pipeline.projects import ProjectStore, chapters_of
from urakatsu.pipeline.stitch import (
    ProjectFormat,
    StitchSettings,
    common_width,
    find_cuts,
    restitch,
    stitch,
)

WIDTH = 400


def art(draw: ImageDraw.ImageDraw, top: int, bottom: int) -> None:
    """Busy line art: no row in it is quiet."""
    for y in range(top, bottom, 3):
        draw.line((0, y, WIDTH, y + 40), fill=(40, 40, 40), width=1)
        draw.line((0, y + 1, WIDTH, y - 30), fill=(200, 60, 60), width=1)


def lettering(draw: ImageDraw.ImageDraw, top: int, lines: int, gap: int = 12, height: int = 20) -> int:
    """Borderless text on white: dark rows separated by narrow quiet gaps. Returns the bottom."""
    y = top
    for _ in range(lines):
        for x in range(60, WIDTH - 60, 14):
            draw.rectangle((x, y, x + 8, y + height), fill="black")
        y += height + gap
    return y


def strip_with_gutters(height: int, gutters: list[tuple[int, int]]) -> Image.Image:
    """Art everywhere except the white ``gutters`` (top, bottom)."""
    image = Image.new("RGB", (WIDTH, height), "white")
    draw = ImageDraw.Draw(image)
    edges = [0] + [edge for gutter in gutters for edge in gutter] + [height]
    for top, bottom in zip(edges[0::2], edges[1::2]):
        art(draw, top, bottom)
    return image


def png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_cuts_land_inside_gutters_and_pages_stay_near_the_target():
    gutters = [(560, 640), (1230, 1300), (1850, 1950), (2500, 2560)]
    strip = strip_with_gutters(3200, gutters)
    cuts = find_cuts(strip, StitchSettings(height=600))
    assert cuts[0] == 0 and cuts[-1] == 3200
    for cut in cuts[1:-1]:
        assert any(top <= cut < bottom for top, bottom in gutters), cut
    for top, bottom in zip(cuts, cuts[1:]):
        assert 300 <= bottom - top <= 900


def test_a_wide_gutter_wins_over_the_gap_between_lines_of_lettering():
    image = Image.new("RGB", (WIDTH, 2000), "white")
    draw = ImageDraw.Draw(image)
    art(draw, 0, 500)
    bottom = lettering(draw, 520, lines=4)  # gaps of 12 rows around the 600-row target
    art(draw, bottom + 160, 2000)  # a 160-row gutter below the text, further from the target
    cuts = find_cuts(image, StitchSettings(height=600))
    assert bottom <= cuts[1] < bottom + 160


def test_no_quiet_row_cuts_the_least_busy_row_within_bounds():
    strip = strip_with_gutters(2400, [])
    cuts = find_cuts(strip, StitchSettings(height=600))
    for top, bottom in zip(cuts, cuts[1:]):
        assert 300 <= bottom - top <= 900


def test_avoided_bands_are_never_cut():
    strip = Image.new("RGB", (WIDTH, 2000), "white")  # everything is quiet
    band = (450, 800)
    cuts = find_cuts(strip, StitchSettings(height=600), avoid=[band])
    assert all(not band[0] <= cut <= band[1] for cut in cuts)


def test_the_last_page_is_not_a_sliver():
    strip = Image.new("RGB", (WIDTH, 1300), "white")
    cuts = find_cuts(strip, StitchSettings(height=600))
    assert min(b - a for a, b in zip(cuts, cuts[1:])) >= 300


def test_sensitivity_zero_cuts_anywhere_and_margin_ignores_edges():
    busy = strip_with_gutters(2000, [])
    assert find_cuts(busy, StitchSettings(height=500, sensitivity=0))[1] == 500
    # A watermark column at the edge makes every row busy unless the margin skips it.
    image = Image.new("RGB", (WIDTH, 2000), "white")
    ImageDraw.Draw(image).rectangle((WIDTH - 6, 0, WIDTH, 2000), fill="black")
    art(ImageDraw.Draw(image), 0, 450)
    art(ImageDraw.Draw(image), 700, 2000)
    cuts = find_cuts(image, StitchSettings(height=600, margin=8))
    assert 450 <= cuts[1] < 700


def test_stitch_scales_to_the_most_common_width():
    images = [Image.new("RGB", (400, 300)), Image.new("RGB", (400, 500)), Image.new("RGB", (800, 200))]
    assert common_width(image.size for image in images) == 400
    strip = stitch(images)
    assert strip.size == (400, 300 + 500 + 100)


BUBBLE = (0, 0, 255)


def find_bubbles(tiles):
    """A stand-in detector: a bubble wherever the left column is BUBBLE blue (a flat colour, so
    pixel comparison alone would happily cut through it)."""
    found = []
    for tile in tiles:
        rows = [y for y in range(tile.height) if tile.getpixel((0, y)) == BUBBLE]
        box = (0.0, float(rows[0]), 100.0, float(rows[-1] + 1)) if rows else None
        found.append(LayoutDetections(tile.width, tile.height, [LayoutDetection(0, "bubble", 0.9, box, 1, LayoutMask(0, 0, 0, 0, b""))] if box else []))
    return found


def test_restitch_keeps_every_row_and_avoids_detected_text():
    strip = Image.new("RGB", (WIDTH, 1800), "white")
    ImageDraw.Draw(strip).rectangle((0, 500, WIDTH, 760), fill=BUBBLE)
    pieces = [strip.crop((0, top, WIDTH, top + 600)) for top in (0, 600, 1200)]
    seen = []

    def detect(tiles):
        seen.extend(tiles)
        return find_bubbles(tiles)

    pages = restitch(pieces, StitchSettings(height=600), detect=detect)
    assert seen and all(tile.size == (WIDTH, WIDTH) for tile in seen)
    assert len(seen) <= 4  # tiles around the cuts, not the whole strip
    assert sum(page.height for page in pages) == 1800
    edges = [sum(page.height for page in pages[:index]) for index in range(1, len(pages))]
    assert all(not 492 <= edge <= 768 for edge in edges)
    # Without detection the same strip is cut at the target, through the bubble.
    plain = restitch(pieces, StitchSettings(height=600, detect=False), detect=detect)
    assert plain[0].height == 600


def test_a_cut_moved_by_detection_is_checked_again():
    # Two bubbles: avoiding what the tile around the first cut sees moves the cut into the
    # second bubble, past that tile's edge, so it is detected again around the new cut.
    strip = Image.new("RGB", (WIDTH, 2400), "white")
    draw = ImageDraw.Draw(strip)
    draw.rectangle((0, 300, WIDTH, 640), fill=BUBBLE)
    draw.rectangle((0, 712, WIDTH, 900), fill=BUBBLE)
    seen = []

    def detect(tiles):
        seen.extend(tiles)
        return find_bubbles(tiles)

    pages = restitch([strip], StitchSettings(height=600), detect=detect)
    edges = [sum(page.height for page in pages[:index]) for index in range(1, len(pages))]
    assert all(not 292 <= edge <= 908 for edge in edges), edges
    assert len(seen) > len(edges)  # some cut needed a second look


def test_restitch_chapter_replaces_pages_in_one_undoable_commit(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create("p")
    gutters = [(560, 640), (1230, 1300)]
    strip = strip_with_gutters(1800, gutters)
    chapter = store.import_chapter(project, "Ch 1", [(str(i), png(strip.crop((0, i * 450, WIDTH, (i + 1) * 450)))) for i in range(4)], remote="src:1")
    store.set_format(project, ProjectFormat(kind="webtoon", stitch=StitchSettings(height=600)))
    before = chapters_of(store.session(project).snapshot())[0].pages
    assert len(before) == 4

    store.restitch_chapter(project, chapter)
    snapshot = store.session(project).snapshot()
    [after] = chapters_of(snapshot)
    assert after.remote == "src:1" and after.name == "Ch 1"
    assert len(after.pages) == 3 and not set(after.pages) & set(before)
    heights = [int(snapshot.page(page).height) for page in after.pages]
    assert sum(heights) == 1800
    assert all(any(top <= edge < bottom for top, bottom in gutters) for edge in (heights[0], heights[0] + heights[1]))
    assert snapshot.asset(after.pages[0], "source").media_type == "image/png"  # lossless sources stay lossless

    store.session(project).undo()
    assert chapters_of(store.session(project).snapshot())[0].pages == before


def test_webtoon_projects_restitch_imported_chapters(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create("p")
    assert store.format(project).kind == "manga"
    store.set_format(project, ProjectFormat(kind="webtoon", stitch=StitchSettings(height=600)))
    strip = strip_with_gutters(1800, [(560, 640), (1230, 1300)])
    pieces = [(str(i), png(strip.crop((0, i * 300, WIDTH, (i + 1) * 300)).convert("RGB"))) for i in range(6)]
    store.import_chapter(project, "Ch 1", pieces)
    snapshot = store.session(project).snapshot()
    [chapter] = chapters_of(snapshot)
    assert len(chapter.pages) == 3
    assert [snapshot.page(page).label for page in chapter.pages] == ["1", "2", "3"]


def test_restitch_refuses_an_empty_chapter(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create("p")
    chapter = store.create_chapter(project, "Empty")
    with pytest.raises(ValueError):
        store.restitch_chapter(project, chapter)


def test_format_reaches_the_proto_and_back():
    from urakatsu.service import convert

    value = ProjectFormat(kind="webtoon", stitch=StitchSettings(height=1200, sensitivity=0, detect=False))
    assert convert.format_from_proto(convert.format_to_proto(value)) == value
