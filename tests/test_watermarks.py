"""Repeated watermark removal: the matte learned across pages, its placements, and the stage."""

from __future__ import annotations

import asyncio
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from urakatsu.pipeline import watermarks as wm
from urakatsu.pipeline.images import ImageCache
from urakatsu.pipeline.stages.base import StageInput
from urakatsu.pipeline.stages.watermarks import LAYER_NAME, WatermarksStage, page_image, watermark_layer
from urakatsu.providers.base import WatermarkBox
from urakatsu.scene import At, AssetInput, AssetMetadata, Page, Session
from urakatsu.scene import components as c

SIZE = (360, 300)  # width, height
MARK_ALPHA = 0.25


def _art(seed: int) -> np.ndarray:
    """A page of smooth colour with a few shapes and lines of art, different per seed (manga pages
    are mostly flat, which is where a seed box shows the mark's own edges)."""
    rng = np.random.default_rng(seed)
    width, height = SIZE
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.stack([128 + 90 * np.sin(xx / (25 + 10 * k) + rng.uniform(0, 6)) * np.cos(yy / (30 + 7 * k)) for k in range(3)], -1)
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for _ in range(4):
        x, y = rng.integers(0, width), rng.integers(0, height)
        r = int(rng.integers(8, 40))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
    for _ in range(3):
        draw.line([tuple(int(v) for v in rng.integers(0, max(SIZE), 2)) for _ in range(2)], fill=(20, 20, 20), width=2)
    return np.asarray(image.filter(ImageFilter.GaussianBlur(0.7)))


def _mark() -> tuple[np.ndarray, np.ndarray]:
    """A logo: light lettering in a dark frame, as alpha and colour (premultiplied)."""
    shape = Image.new("L", (120, 40), 0)
    draw = ImageDraw.Draw(shape)
    draw.rectangle((2, 2, 117, 37), outline=200, width=2)
    draw.text((8, 5), "MaRk", fill=255, font=ImageFont.load_default(size=26))
    alpha = np.asarray(shape, np.float32) / 255.0 * MARK_ALPHA
    color = np.full((40, 120, 3), 235.0, np.float32)
    color[np.asarray(shape) == 200] = 60.0  # some strokes dark, some light
    return alpha, alpha[..., None] * color


def _blend(art: np.ndarray, alpha: np.ndarray, color: np.ndarray, x: int, y: int) -> np.ndarray:
    out = art.astype(np.float32).copy()
    h, w = alpha.shape
    region = out[y:y + h, x:x + w]
    out[y:y + h, x:x + w] = color + (1 - alpha[..., None]) * region
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _chapter(positions):
    alpha, color = _mark()
    arts = {f"p{i}": _art(i) for i in range(len(positions))}
    pages = {name: _blend(art, alpha, color, *positions[i]) for i, (name, art) in enumerate(arts.items())}
    return arts, pages


POSITIONS = [(200, 240), (200, 240), (30, 40), (200, 240), (110, 130)]


def test_a_repeated_mark_is_unblended_and_the_art_under_it_kept():
    arts, pages = _chapter(POSITIONS)
    candidates = [wm.Candidate("p0", (196, 236, 324, 284), 0.6), wm.Candidate("p2", (26, 36, 154, 84), 0.4)]
    result = wm.remove_watermarks(pages, candidates)
    assert len(result.watermarks) == 1
    assert {p.page for p in result.watermarks[0].placements} == set(pages)
    on_mark = _mark()[0] > 0
    for name, (x, y) in zip(pages, POSITIONS):
        page = result.pages[name]
        before = np.abs(pages[name][y:y + 40, x:x + 120].astype(int) - arts[name][y:y + 40, x:x + 120]).mean(2)[on_mark].mean()
        after = np.abs(page.image[y:y + 40, x:x + 120].astype(int) - arts[name][y:y + 40, x:x + 120]).mean(2)[on_mark].mean()
        assert after < 0.3 * before, (name, before, after)
        assert not page.footprint[: y - 4].any()  # nothing changed away from the mark


def test_a_box_on_art_that_does_not_repeat_is_left_alone():
    arts, pages = _chapter(POSITIONS)
    result = wm.remove_watermarks(pages, [wm.Candidate("p1", (10, 10, 90, 70), 0.9)])
    assert result.watermarks == [] and result.pages == {}


def test_a_matte_whose_edges_all_run_one_way_is_not_a_mark():
    line = np.zeros((40, 120, 3), np.float32)
    for k in range(120):
        line[max(0, 39 - k // 3) : max(0, 39 - k // 3) + 3, k] = 50.0
    alpha, color = _mark()
    assert wm.coherence(line) > wm.MAX_COHERENCE > wm.coherence(color)


# ---- the stage -------------------------------------------------------------------------------------


class _Detector:
    id = "fake-watermark"

    def __init__(self, boxes):
        self.boxes = boxes
        self.runtimes = None
        self.settings = type("S", (), {"model_dump_json": lambda self: "{}"})()

    async def detect_watermarks(self, image):
        return self.boxes.get(image.info.get("page"), [])


def _session(tmp_path, pages: dict[str, np.ndarray]):
    session = Session(tmp_path)
    edit = session.snapshot().edit()
    ids = {}
    for name, pixels in pages.items():
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        page = edit.add_page(Page(label=name, width=float(SIZE[0]), height=float(SIZE[1])), At.END)
        edit.set_asset(page, "source", AssetInput(data=buffer.getvalue(), media_type="image/png", metadata=AssetMetadata(width=SIZE[0], height=SIZE[1])))
        ids[name] = page
    return session, session.commit(edit.finish()), ids


def test_the_stage_commits_a_watermarks_layer_that_later_stages_read_through(tmp_path, monkeypatch):
    arts, pages = _chapter(POSITIONS)
    session, snapshot, ids = _session(tmp_path, pages)
    # The fake detector proposes the mark on the first page only; the stage finds the rest.
    first = ids["p0"]
    images = ImageCache()
    original_get = images.get

    def tagged(scene, entity, role):
        image = original_get(scene, entity, role)
        if image is not None and role == "source" and entity in ids.values():
            image.info["page"] = entity
        return image

    monkeypatch.setattr(images, "get", tagged)
    detector = _Detector({first: [WatermarkBox(box=(196, 236, 324, 284), score=0.6)]})
    stage = WatermarksStage()
    for name, page in ids.items():
        stage_input = StageInput(snapshot, page, None, None, images)
        assert not stage.skip(stage_input, detector)
        snapshot = session.commit(asyncio.run(stage.process(stage_input, detector)))
        layer = watermark_layer(snapshot, page)
        assert layer is not None
        raster = snapshot.component(layer, c.RasterLayer)
        assert raster.name == LAYER_NAME and raster.kind is c.RasterLayerKind.WATERMARK
        assert snapshot.children(page)[0] == layer  # right over the source
        x, y = POSITIONS[list(ids.values()).index(page)]
        cleaned = np.asarray(page_image(StageInput(snapshot, page, None, None, ImageCache())))
        error = np.abs(cleaned[y:y + 40, x:x + 120].astype(int) - arts[name][y:y + 40, x:x + 120]).mean()
        assert error < 6, (name, error)
        assert stage.skip(StageInput(snapshot, page, None, None, ImageCache()), detector)  # done
