"""OCR providers and stage: weight-free tests, a stage run through a Session, and weight tests.

The weight tests skip unless the pinned checkpoint is already in the model store, so the suite
stays offline. They compare each provider on the balloon fixture when the weights are present.
"""

from __future__ import annotations

from pathlib import Path

import json
import os

import numpy as np
import pytest
from PIL import Image

from urakatsu.pipeline.stages.base import StageInput
from urakatsu.providers.base import OcrProvider
from urakatsu.pipeline.stages.ocr import OcrStage, crop, normalize_ocr_text
from urakatsu.providers.ocr._common import pillow_grayscale, read_lines
from urakatsu.catalog.facts import ModelFacts
from urakatsu.providers.ocr.baberu_ocr import BaberuOcrProvider
from urakatsu.providers.ocr.charspot_ocr import COMBOS, CharspotOcrProvider, load_vocabulary, pad_grid, preprocess
from urakatsu.providers.ocr.hayai_ocr import _image_size_for_max_patches
from urakatsu.providers.ocr.manga_ocr import (
    MangaOcrProvider,
    _BeamHypotheses,
    _banned_ngram_tokens,
    halfwidth_to_fullwidth,
    post_process,
)
from urakatsu.providers.ocr.paddle_ocr_vl import (
    PaddleOCRVLTask,
    PaddleOcrVlProvider,
    smart_resize,
)
from urakatsu.runtimes.store import HuggingFaceFile, ModelStore
from urakatsu.scene import (
    At,
    Authored,
    Geometry,
    Page,
    RegionKinds,
    Relations,
    Session,
    SourceText,
    TextDirection,
)
from urakatsu.scene import components as c

KOHARU_ROOT = Path(os.environ.get("KOHARU_ROOT", r"D:\dev\koharu"))
FIXTURE = KOHARU_ROOT / "crates" / "koharu-ml" / "benches" / "fixtures" / "object_detection" / "1.jpg"

# Balloons (x, y, w, h) and ground truth from the task brief. Ground truth uses full-width
# punctuation; Manga OCR and PaddleOCR-VL print full width, Baberu prints the same glyphs, and
# Hayai prints ASCII punctuation, so comparisons normalise with NFKC.
BALLOONS = [
    (362, 281, 19, 138, "エビ漁船バイト！"),
    (635, 828, 28, 141, "銀行を襲う"),
    (98, 628, 35, 97, "さすがシロコ先輩！"),
    (69, 25, 256, 35, "対策委員会です！"),
    (273, 123, 56, 130, "アビドス高校の廃校を防ぐための会議を始めます！"),
    (595, 561, 63, 128, "み、皆さん…私たちにできる範囲で…"),
    (279, 559, 56, 117, "その通りだよ現実的な方法を提案しないと"),
    (111, 229, 55, 143, "スクール☆アイドルはどうでしょう〜？"),
]


# ---- weight-free: processor and decoder numerics (Rust unit tests) ---------------------


def test_pillow_grayscale_matches_koharu_fixture():
    # manga_ocr/processor.rs: pillow_image_processing_matches_reference
    image = Image.new("RGB", (3, 2))
    image.putdata([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0), (128, 64, 32)])
    gray = pillow_grayscale(image)
    values = [gray.getpixel((x, y))[0] for y in range(2) for x in range(3)]
    assert values == [76, 150, 29, 255, 0, 79]
    assert gray.getpixel((0, 0)) == (76, 76, 76)


def test_manga_bilinear_resize_matches_koharu_fixture():
    # manga_ocr/processor.rs expects [104, 74, 159, 49] from the 3x2 grayscale image resized to 2x2.
    image = Image.new("RGB", (3, 2))
    image.putdata([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0), (128, 64, 32)])
    resized = pillow_grayscale(image).resize((2, 2), Image.BILINEAR)
    assert [resized.getpixel((x, y))[0] for y in range(2) for x in range(2)] == [104, 74, 159, 49]


def test_manga_post_processing_matches_koharu():
    # manga_ocr/processor.rs: post_processing_matches_manga_ocr
    assert post_process(" A 12…・・. ｶﾞﾊﾟ ") == "Ａ１２．．．．．．ガパ"
    assert post_process("・ ﾞ") == "・ﾞ"


def test_halfwidth_to_fullwidth_composes_voiced_marks():
    assert halfwidth_to_fullwidth("ｶﾞ") == "ガ"
    assert halfwidth_to_fullwidth("ﾊﾟ") == "パ"
    assert halfwidth_to_fullwidth("a b") == "ａ\u3000ｂ"


def test_manga_beam_hypotheses_keep_the_best_scored_sequence():
    hyp = _BeamHypotheses(num_beams=2, length_penalty=1.0, early_stopping=True)
    hyp.add([2, 5], -1.0)
    hyp.add([2, 6], -4.0)
    hyp.add([2, 7], -0.5)
    assert hyp.best() == [2, 7]


def test_manga_no_repeat_ngram_bans_the_completing_token():
    assert _banned_ngram_tokens([2, 5, 6, 2, 5], 3) == [6]
    assert _banned_ngram_tokens([2, 5, 6, 5], 3) == []
    assert _banned_ngram_tokens([2, 5], 3) == []


def test_read_lines_follows_rust_lines_semantics(tmp_path):
    path = tmp_path / "vocab.txt"
    path.write_bytes("a\r\nb\n\nc\n".encode("utf-8"))
    assert read_lines(path) == ["a", "b", "", "c"]


def test_hayai_patch_budget_matches_upstream_search():
    # hayai_ocr/processor.rs: the NaFlex size search never exceeds 256 patches of 16x16.
    height, width = _image_size_for_max_patches(1024, 1024, 16, 256)
    assert (height // 16) * (width // 16) <= 256
    assert height == width == 256
    height, width = _image_size_for_max_patches(120, 900, 16, 256)
    assert (height // 16) * (width // 16) <= 256 and height % 16 == 0 and width % 16 == 0


def test_paddle_smart_resize_matches_transformers_fixture_shapes():
    # paddle_ocr_vl/processor.rs: smart_resize_matches_transformers_fixture_shapes
    cases = [
        ((178, 99), (476, 252)),
        ((251, 188), (392, 308)),
        ((103, 62), (448, 280)),
        ((115, 406), (196, 644)),
        ((313, 69), (728, 168)),
        ((54, 204), (196, 672)),
    ]
    for (height, width), expected in cases:
        assert smart_resize(height, width, 28, 112_896, 1_003_520) == expected


def test_paddle_smart_resize_rejects_extreme_aspect_ratios():
    with pytest.raises(ValueError):
        smart_resize(28, 28 * 201, 28, 112_896, 1_003_520)


def test_paddle_task_prompts_match_koharu():
    # paddle_ocr_vl/processor.rs: PaddleOCRVLTask::prompt
    expected = {
        PaddleOCRVLTask.OCR: "OCR:",
        PaddleOCRVLTask.TABLE: "Table Recognition:",
        PaddleOCRVLTask.FORMULA: "Formula Recognition:",
        PaddleOCRVLTask.CHART: "Chart Recognition:",
        PaddleOCRVLTask.SPOTTING: "Spotting:",
        PaddleOCRVLTask.SEAL: "Seal Recognition:",
    }
    assert {task: task.prompt for task in PaddleOCRVLTask} == expected


def test_placeholder_glyphs_become_an_ellipsis():
    # stages/ocr.rs: repeated_placeholder_glyphs_are_an_ellipsis
    assert normalize_ocr_text("☐ ☐ ☐") == "…"
    assert normalize_ocr_text("□\n□") == "…"


def test_ordinary_text_and_single_boxes_are_unchanged():
    # stages/ocr.rs: ordinary_text_and_single_boxes_are_unchanged
    assert normalize_ocr_text("待って…") == "待って…"
    assert normalize_ocr_text("☐") == "☐"


def test_crop_floors_and_ceils_and_rejects_empty_overlap():
    source = Image.new("RGB", (100, 80))
    box = crop(source, Geometry.rectangle(10.4, 5.2, 20.1, 10.3))
    assert box.size == (21, 11)  # x 10..31, y 5..16
    with pytest.raises(ValueError):
        crop(source, Geometry.rectangle(200, 200, 5, 5))


def test_direction_rule_uses_1_15_aspect(tmp_path):
    tall = Geometry.rectangle(0, 0, 10, 20)
    wide = Geometry.rectangle(0, 0, 20, 10)
    from urakatsu.pipeline.stages.ocr import _text_direction

    assert _text_direction(tall) is c.TextDirection.VERTICAL
    assert _text_direction(wide) is c.TextDirection.HORIZONTAL


# ---- stage through a Session (no weights): a stub OCR provider ----------------------------------


class _StubOcr(OcrProvider):
    """OcrProvider with no runtime: answers a fixed string per crop and records the crop sizes."""

    id = "stub-ocr"

    def __init__(self, answer: str = "テスト…"):
        super().__init__()
        self.answer = answer
        self.sizes: list[tuple[int, int]] = []

    async def recognize(self, crops):
        self.sizes = [crop.size for crop in crops]
        return [self.answer for _ in crops]


def _page_png(width: int = 200, height: int = 160) -> bytes:
    import io

    image = Image.new("RGB", (width, height), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _build_scene(session: Session, user_text: str | None = None):
    edit = session.snapshot().edit()
    page = edit.add_page(Page(label="p1", width=200, height=160))
    edit.set_asset(page, "source", c.AssetInput(data=_page_png(), media_type="image/png"))
    region = edit.add_analysis_region(page, At.END, RegionKinds.TEXT, Geometry.rectangle(20, 30, 60, 100))
    content = edit.add_text_content(page)
    edit.relate(Relations.RECOGNIZED_FROM, content, region)
    if user_text is not None:
        edit.set(content, SourceText(text=Authored.user(user_text), language="ja-JP"))
    session.commit(edit.finish())
    return page, region, content


def _run_stage(session: Session, page: str, provider) -> None:
    import asyncio

    from urakatsu.pipeline.images import ImageCache

    snapshot = session.snapshot()
    stage = OcrStage()
    input = StageInput(scene=snapshot, page=page, entities=None, region=None, images=ImageCache())
    assert not stage.skip(input, provider)
    patch = asyncio.run(stage.process(input, provider))
    session.commit(patch)


def test_ocr_stage_commits_generated_source_text_and_analysis(tmp_path):
    session = Session(tmp_path)
    page, region, content = _build_scene(session)
    provider = _StubOcr("テスト…")
    _run_stage(session, page, provider)

    snap = session.snapshot()
    source = snap.component(content, c.SourceText)
    assert source.text.value == "テスト…"  # only all-placeholder output is rewritten
    assert source.text.origin.kind == "generated"
    assert source.text.origin.generation.producer == "dev.koharu.pipeline.ocr"
    assert source.text.origin.generation.model == "stub-ocr"
    assert source.language == "ja-JP"
    analysis = snap.component(region, c.OcrAnalysis)
    assert analysis.direction is TextDirection.VERTICAL  # 60 wide, 100 tall
    assert provider.sizes == [(60, 100)]


def test_ocr_stage_never_overwrites_user_text(tmp_path):
    session = Session(tmp_path)
    page, region, content = _build_scene(session, user_text="ユーザー")
    provider = _StubOcr("別のテキスト")
    import asyncio

    from urakatsu.pipeline.images import ImageCache

    stage = OcrStage()
    input = StageInput(scene=session.snapshot(), page=page, entities=None, region=None, images=ImageCache())
    assert stage.skip(input, provider)
    assert asyncio.run(stage.process(input, provider)).is_empty()
    assert provider.sizes == []
    assert session.snapshot().component(content, c.SourceText).text.value == "ユーザー"


def test_ocr_stage_rejects_a_region_outside_the_source(tmp_path):
    session = Session(tmp_path)
    edit = session.snapshot().edit()
    page = edit.add_page(Page(label="p1", width=200, height=160))
    edit.set_asset(page, "source", c.AssetInput(data=_page_png(), media_type="image/png"))
    region = edit.add_analysis_region(page, At.END, RegionKinds.TEXT, Geometry.rectangle(500, 500, 10, 10))
    content = edit.add_text_content(page)
    edit.relate(Relations.RECOGNIZED_FROM, content, region)
    session.commit(edit.finish())

    import asyncio

    from urakatsu.pipeline.images import ImageCache

    stage = OcrStage()
    input = StageInput(scene=session.snapshot(), page=page, entities=None, region=None, images=ImageCache())
    with pytest.raises(ValueError):
        asyncio.run(stage.process(input, _StubOcr()))


# ---- Charspot: preprocessing, vocabulary and catalog matching -----------------------------------


@pytest.mark.parametrize("height, width", [(40, 300), (300, 40), (128, 128), (1, 900), (3000, 2000)])
@pytest.mark.parametrize("budget", [196, 256])
def test_charspot_grid_fits_the_budget(height, width, budget):
    scale, rows, columns = pad_grid(height, width, budget, 16)
    assert 1 <= rows * columns <= budget
    assert scale > 0


def test_charspot_preprocess_pads_with_the_border_median():
    image = Image.new("RGB", (40, 300), (200, 100, 50))
    pixels, factors = preprocess(image, 256, 16)
    assert factors[0] > 0 and factors[1] > 0
    assert pixels.dtype == np.uint8
    assert pixels.shape[0] == 3
    assert pixels.shape[1] % 16 == 0 and pixels.shape[2] % 16 == 0
    assert (pixels[0] == 200).all() and (pixels[1] == 100).all() and (pixels[2] == 50).all()


def test_charspot_vocabulary_ends_with_the_combined_glyphs(tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"characters": ["あ", "い"]}), encoding="utf-8")
    vocabulary = load_vocabulary(path, 4)
    assert vocabulary[:4] == ["あ", "い", "", ""]
    assert vocabulary[4:] == COMBOS
    assert len(COMBOS) == 12


def test_charspot_reads_characters_in_model_order():
    class Session:
        def run(self, outputs, inputs):
            assert outputs == ["ids", "order"]
            assert inputs["img"].dtype == np.float32 and inputs["img"].shape[:2] == (1, 3)
            return np.array([0, 1, 2]), np.array([2, 0, 1])

    provider = CharspotOcrProvider()
    provider._session = Session()
    provider._cell, provider._budget = 16, 256
    provider._vocabulary = ["字", "の", "!?"]
    assert provider._recognize([Image.new("RGB", (32, 64), "white")]) == ["!?字の"]


def test_charspot_reads_an_empty_crop_as_no_text():
    class Session:
        def run(self, outputs, inputs):
            return np.zeros(0, np.int64), np.zeros(0, np.int64)

    provider = CharspotOcrProvider()
    provider._session = Session()
    provider._cell, provider._budget = 16, 256
    provider._vocabulary = []
    assert provider._recognize([Image.new("RGB", (32, 64), "white")]) == [""]


@pytest.mark.parametrize(
    "runtime, repo, files, expected",
    [
        ("onnx", "muscgab/manga-ocr-charspot-small", ["config.json", "vocab.json", "model.onnx"], "onnx"),
        ("onnx", "muscgab/manga-ocr-charspot-nano", ["config.json", "vocab.json", "model.onnx"], "onnx"),
        ("onnx", "muscgab/manga-ocr-charspot-small", ["config.json", "model.onnx"], None),
        ("onnx", "someone/other-ocr", ["config.json", "vocab.json", "model.onnx"], None),
        ("torch", "muscgab/manga-ocr-charspot-small", ["config.json", "vocab.json", "model.onnx"], None),
    ],
)
def test_charspot_accepts_its_family(runtime, repo, files, expected):
    assert CharspotOcrProvider.accepts(ModelFacts(runtime=runtime, repo=repo, files=files)) == expected


# ---- weight tests (skip when the checkpoint is not cached) --------------------------------------


def _cached(store: ModelStore, *sources) -> bool:
    return all(store.cached(source) for source in sources)


def _fixture_crops():
    if not FIXTURE.exists():
        pytest.skip("fixture image is not available (set KOHARU_ROOT)")
    source = Image.open(FIXTURE).convert("RGB")
    return [crop(source, Geometry.rectangle(x, y, w, h)) for x, y, w, h, _ in BALLOONS]


def _check_balloons(results: list[str]) -> int:
    matched = 0
    import unicodedata

    for (_, _, _, _, truth), got in zip(BALLOONS, results):
        if unicodedata.normalize("NFKC", normalize_ocr_text(got)) == unicodedata.normalize("NFKC", truth):
            matched += 1
    return matched


@pytest.fixture
def real_home(monkeypatch) -> Path:
    """The conftest isolates URAKATSU_HOME; weight tests opt back in to a real model store."""
    home = os.environ.get("URAKATSU_TEST_HOME")
    if not home:
        pytest.skip("set URAKATSU_TEST_HOME to a model store that holds the OCR weights")
    monkeypatch.setenv("URAKATSU_HOME", home)
    return Path(home)


def _require_weights(home: Path, provider_cls, runtime: str) -> None:
    store = ModelStore(root=home / "models", offline=True)
    if not _cached(store, *provider_cls.weight_sources(runtime)):
        pytest.skip(f"{provider_cls.id} ({runtime}) weights are not cached under {home}")


@pytest.mark.parametrize("provider_cls", [MangaOcrProvider, BaberuOcrProvider])
def test_torch_ocr_matches_balloons(provider_cls, real_home):
    _require_weights(real_home, provider_cls, "torch")
    from urakatsu.runtimes import RuntimeManager

    provider = provider_cls({}, RuntimeManager(), "torch")
    import asyncio

    asyncio.run(provider.load())
    try:
        results = provider._recognize(_fixture_crops())
    finally:
        provider.unload()
    assert _check_balloons(results) >= 7, results


def test_hayai_ocr_matches_balloons(real_home):
    from urakatsu.providers.ocr.hayai_ocr import HayaiOcrProvider

    _require_weights(real_home, HayaiOcrProvider, "torch")
    from urakatsu.runtimes import RuntimeManager

    provider = HayaiOcrProvider({}, RuntimeManager(), "torch")
    import asyncio

    asyncio.run(provider.load())
    try:
        results = provider._recognize(_fixture_crops())
    finally:
        provider.unload()
    assert _check_balloons(results) >= 7, results


def test_charspot_ocr_matches_balloons(real_home):
    _require_weights(real_home, CharspotOcrProvider, "onnx")
    from urakatsu.runtimes import RuntimeManager

    provider = CharspotOcrProvider({}, RuntimeManager(), "onnx")
    import asyncio

    asyncio.run(provider.load())
    try:
        results = provider._recognize(_fixture_crops())
    finally:
        provider.unload()
    assert _check_balloons(results) >= 7, results


@pytest.mark.parametrize("runtime", ["llama", "torch"])
def test_paddle_ocr_vl_matches_balloons(runtime, real_home):
    _require_weights(real_home, PaddleOcrVlProvider, runtime)
    from urakatsu.runtimes import RuntimeManager

    provider = PaddleOcrVlProvider({}, RuntimeManager(), runtime)
    import asyncio

    asyncio.run(provider.load())
    try:
        results = provider._recognize(_fixture_crops())
    finally:
        provider.unload()
    assert _check_balloons(results) >= 7, results
