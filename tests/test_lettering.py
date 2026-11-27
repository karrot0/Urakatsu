"""Lettering styles: the style table, font weights from names, the placement rules, the stage
through a Session with a stub font reader, its gating in a run, and the renderer's fonts. No weights."""

from __future__ import annotations

import asyncio
import io
import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from urakatsu.core.config import PipelineConfig, TypesettingConfig
from urakatsu.core.stage import PREREQUISITES, Operation, Stage
from urakatsu.core.styles import SCRIPTS, STYLES, StyleFace, default_styles, face, fallback_families, group, script_of
from urakatsu.pipeline.images import ImageCache
from urakatsu.pipeline.lettering import BALLOON, BOXED, FREE, SOUND_EFFECT, Placement, Reading, boxed, measure_stroke, reading
from urakatsu.pipeline.pipeline import _lettering_stages
from urakatsu.pipeline.stages.base import StageInput, generation
from urakatsu.pipeline.stages.lettering import MARGINS, LetteringStage, crops, mean_reading
from urakatsu.providers.base import FontReading, LetteringProvider
from urakatsu.providers.lettering.font_names import weight_of
from urakatsu.providers.lettering.yuzumarker_font_detection import YuzuMarkerFontDetection, font_weights
from urakatsu.scene import At, Authored, Geometry, Page, RegionKinds, Relations, Session, SourceText, TextLayout, TextLayoutKind
from urakatsu.scene import components as c
from urakatsu.service import convert
from urakatsu.service.gen import urakatsu_pb2 as pb

# ---- the style table -------------------------------------------------------------------------------


def test_every_style_has_a_latin_face_and_groups_share_one():
    styles = default_styles()
    assert set(styles) <= set(STYLES)
    assert all(script in SCRIPTS for faces in styles.values() for script in faces)
    for style in STYLES:
        assert face(styles, style, "Latn").family, style
    # A group's first style's face is each of its styles' that names none of its own.
    assert styles["kaisho"]["Latn"] == styles["mincho"]["Latn"]
    assert styles["gyosho"]["Latn"] == styles["brush"]["Latn"]
    assert styles["kointai"]["Latn"].family == "ManEater BB" and styles["brush"]["Latn"].family == "Ninjutsu BB"
    assert group("pop") == "design" and group("gothic") == "gothic"


def test_a_face_takes_what_it_leaves_out_from_antique():
    styles = default_styles()
    thoughts = face(styles, "rounded", "Cyrl")  # italic only: the dialogue font, slanted
    assert thoughts == StyleFace(family="CCWildWords", italic=True)
    assert face(styles, "gothic", "Jpan").family == "小塚ゴシック Pr6N"  # no gothic of its own: antique's
    assert face(styles, None, "Latn").family == "CCWildWords"
    assert face({}, "gothic", "Latn") == StyleFace()
    assert fallback_families(styles, "Hans") == ["Adobe 黑体 Std", "CCWildWords"]
    assert fallback_families(styles, "Latn") == ["CCWildWords"]


@pytest.mark.parametrize(
    "text, language, script",
    [
        ("Hello there", "en-US", "Latn"),
        ("Hello there", None, "Latn"),
        ("你好", "zh-CN", "Hans"),
        ("你好", "zh-TW", "Hant"),
        ("你好", "zh-Hant-HK", "Hant"),
        ("你好", None, "Hans"),
        ("こんにちは世界", None, "Jpan"),
        ("Привет", None, "Cyrl"),
        ("Hello", "ru", "Cyrl"),
        ("Γειά", None, "Grek"),
    ],
)
def test_script_of_reads_the_language_then_the_characters(text, language, script):
    assert script_of(text, language) == script


# ---- font weights from names -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "path, weight",
    [
        ("Adobe/CJK/SourceHanSans-Bold 1.004.otf", 700),
        ("Adobe/CJK/SourceHanSans-ExtraLight.otf", 200),
        ("Adobe/日文/Heisei Mincho Std W9.otf", 900),
        ("DynaFont/DFKaiShoPro5-W14.otf", 900),
        ("DynaFont/DFHannotate-W5 - DFPHannotate-W5 - DFGHannotate-W5.ttc", 500),
        ("Fontworks/FOT-CezanneProN-DB.otf", 600),
        ("Fontworks/FOT-RowdyStd-EB.otf", 800),
        ("Fontworks/FOT-TsukuGoPr5N-M.otf", 500),
        ("Morisawa/A-OTF-ShinGoPr6-Ultra.otf", 900),
        ("Morisawa/A-OTF-ShinMGoPro-Light.otf", 300),
        ("Morisawa/A-OTF-UDShinGoCOeizPr6-ExL.otf", 200),
        ("方正/方正中等线_GBK.OTF", 500),
        ("汉仪/汉仪细行楷简.ttf", 300),
        ("汉仪/汉仪大黑.ttf", 900),
        ("方正/方正粗圆_GBK.ttf", 700),
        ("Monotype/MGentle HK Xbold.ttf", 800),
        ("其他/TA_ichirou.otf", None),
        ("方正/方正彩云_GBK.ttf", None),
    ],
)
def test_weight_of_reads_a_font_files_name(path, weight):
    assert weight_of(path) == weight


def test_font_weights_mark_the_fonts_whose_names_state_none():
    weights, known = font_weights([{"path": "a/Foo-Bold.otf"}, {"path": "a/Plain.ttf"}, {"path": "a/Bar W3.otf"}])
    assert weights.tolist() == [700.0, 0.0, 300.0] and known.tolist() == [1.0, 0.0, 1.0]


def test_the_font_reader_is_registered_for_the_lettering_stage():
    from urakatsu.core.registry import DEFAULTS, registry

    assert registry.get(Stage.LETTERING, DEFAULTS[Stage.LETTERING]) is YuzuMarkerFontDetection
    assert PipelineConfig().provider_for(Stage.LETTERING) == YuzuMarkerFontDetection.id
    assert [source.filename for source in YuzuMarkerFontDetection.weight_sources()] == ["model.safetensors", "kernels.json", "fonts.json"]


# ---- placement -------------------------------------------------------------------------------------

KANJI_TEXT = "世界の終わりが近い"  # 5 Han of 9: kanji-heavy
KANA_TEXT = "うるさいな"


def test_a_reading_counts_the_mincho_styles_as_antique_without_han():
    with_han = reading({"mincho": 0.7, "kaisho": 0.2, "gothic": 0.1}, KANJI_TEXT, FREE)
    assert with_han.kanji and with_han.kernels == {"mincho": 0.7, "kaisho": 0.2, "gothic": 0.1}
    kana = reading({"mincho": 0.7, "kaisho": 0.2, "gothic": 0.1}, KANA_TEXT, FREE)
    assert not kana.kanji and kana.kernels == pytest.approx({"antique": 0.9, "gothic": 0.1})
    assert reading({}, "漢字かな", FREE).kanji is True  # 2 of 4
    assert reading({}, "漢かなかなかな", FREE).kanji is False  # 1 of 7


def test_a_text_takes_its_likeliest_groups_likeliest_style_when_sure():
    placement = Placement(
        {
            "shout": reading({"gothic": 0.9, "rounded": 0.1}, KANJI_TEXT, FREE),
            # The brush hands sum to 0.7, past any single style: the likeliest of them.
            "brush": reading({"gyosho": 0.4, "reisho": 0.3, "gothic": 0.3}, KANJI_TEXT, SOUND_EFFECT),
            "unsure": reading({"gothic": 0.5, "rounded": 0.3, "pop": 0.2}, KANJI_TEXT, BOXED),
            "none": Reading(),
        }
    )
    assert placement.of("shout") == ("gothic", None)
    assert placement.of("brush") == ("gyosho", None)
    assert placement.of("unsure") == ("antique", None)
    assert placement.of("none") == ("antique", None)
    assert placement.of("missing") == ("antique", None)


def test_mincho_needs_to_be_all_but_certain():
    placement = Placement(
        {
            "likely": reading({"mincho": 0.97, "gothic": 0.03}, KANJI_TEXT, BOXED),
            "certain": reading({"mincho": 0.999, "gothic": 0.001}, KANJI_TEXT, FREE),
        }
    )
    assert placement.of("likely")[0] == "antique"
    assert placement.of("certain")[0] == "mincho"


def test_a_books_dialogue_keeps_one_face():
    # Five kanji-heavy balloon texts read as mincho set the dialogue in mincho: a balloon's text
    # read as antique takes it too, while a shout and a box's narration keep their own.
    readings = {f"m{i}": reading({"mincho": 0.9, "gothic": 0.1}, KANJI_TEXT, BALLOON) for i in range(5)}
    readings["kana"] = reading({"mincho": 0.95, "gothic": 0.05}, KANA_TEXT, BALLOON)
    readings["shout"] = reading({"gothic": 0.95, "mincho": 0.05}, KANJI_TEXT, BALLOON)
    placement = Placement(readings)
    assert placement.face == "mincho"
    assert placement.of("m0")[0] == "mincho" and placement.of("kana")[0] == "mincho"
    assert placement.of("shout")[0] == "gothic"

    # Four are too few, and antique's gothic kanji seldom read as mincho: the dialogue is antique,
    # and a balloon's text read as mincho takes it.
    few = {f"m{i}": reading({"mincho": 0.9, "gothic": 0.1}, KANJI_TEXT, BALLOON) for i in range(4)}
    few["certain"] = reading({"mincho": 0.999, "gothic": 0.001}, KANJI_TEXT, BALLOON)
    few["narration"] = reading({"mincho": 0.999, "gothic": 0.001}, KANJI_TEXT, BOXED)
    antique = Placement({**few, **{f"a{i}": reading({"gothic": 0.8, "mincho": 0.2}, KANJI_TEXT, BALLOON) for i in range(8)}})
    assert antique.face == "antique"
    assert antique.of("certain")[0] == "antique"
    assert antique.of("narration")[0] == "mincho"


def test_pooling_leans_a_reading_to_its_kinds_others():
    readings = {f"g{i}": reading({"gothic": 0.9, "rounded": 0.1}, KANJI_TEXT, SOUND_EFFECT) for i in range(6)}
    readings["swung"] = reading({"gothic": 0.45, "rounded": 0.55}, KANJI_TEXT, SOUND_EFFECT)
    readings["free"] = reading({"gothic": 0.45, "rounded": 0.55}, KANJI_TEXT, FREE)  # alone of its kind: as read
    placement = Placement(readings)
    assert placement.readings["swung"].kernels["gothic"] > 0.6
    assert sum(placement.readings["swung"].kernels.values()) == pytest.approx(1.0)
    assert placement.readings["free"].kernels == {"gothic": 0.45, "rounded": 0.55}
    assert readings["swung"].kernels["gothic"] == 0.45  # the caller's readings are not changed


def _dialogue(extra: dict[str, Reading]) -> Placement:
    readings = {f"d{i}": reading({"mincho": 0.8, "gothic": 0.2}, KANA_TEXT, BALLOON, stroke=0.10, weight=400.0) for i in range(5)}
    return Placement({**readings, **extra})


def test_a_source_is_bold_by_its_stroke_unless_its_fonts_read_lighter_than_the_dialogue():
    shout = lambda stroke, weight: reading({"gothic": 0.95}, KANJI_TEXT, BALLOON, stroke=stroke, weight=weight)  # noqa: E731
    cases = {
        "heavy": (shout(0.14, 650.0), 700),
        "heavy, same fonts": (shout(0.14, 400.0), 700),  # weights named in different families do not compare
        "heavy, fonts unread": (shout(0.14, None), 700),
        "thick box, light fonts": (shout(0.30, 240.0), None),  # a balloon's outline in the box, not a heavy face
        "plain": (shout(0.10, 400.0), None),
        "plain, heavy fonts": (shout(0.11, 800.0), None),
        "no stroke, heavy fonts": (shout(None, 650.0), 700),
        "no stroke, same fonts": (shout(None, 500.0), None),
        "neither": (shout(None, None), None),
    }
    placement = _dialogue({key: value for key, (value, _) in cases.items()})
    assert placement.stroke == 0.10 and placement.weight == 400.0
    assert {key: placement.of(key)[1] for key in cases} == {key: bold for key, (_, bold) in cases.items()}


def test_the_dialogue_is_every_text_when_none_is_in_a_balloon():
    readings = {f"f{i}": reading({"mincho": 0.8, "gothic": 0.2}, KANA_TEXT, FREE, stroke=0.1, weight=400.0) for i in range(3)}
    readings["sfx"] = reading({"gothic": 0.9}, KANA_TEXT, SOUND_EFFECT, stroke=0.3, weight=900.0)
    placement = Placement(readings)
    assert placement.stroke == 0.1 and placement.of("sfx")[1] == 700
    assert Placement({"sfx": readings["sfx"]}).of("sfx")[1] is None  # no dialogue to weigh it against


def test_a_box_is_an_outline_that_runs_square():
    assert boxed([(0, 0), (100, 0), (100, 40), (0, 40)])
    assert boxed([(0, 0), (50, 1), (100, 0), (101, 20), (100, 40), (0, 40)])  # a scanned box, slightly off
    ellipse = [(50 + 50 * np.cos(a), 30 + 30 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 60, endpoint=False)]
    assert not boxed(ellipse)
    assert not boxed([(0, 0), (100, 0), (50, 80)])
    spiky = [(0, 20), (30, 0), (50, 25), (70, 0), (100, 20), (75, 45), (100, 70), (70, 90), (50, 65), (30, 90), (0, 70), (25, 45)]
    assert not boxed(spiky)


def _columns(stroke: int, size: int = 30, columns: int = 3) -> np.ndarray:
    """Vertical text as crosses: each glyph a horizontal and a vertical bar ``stroke`` thick."""
    image = Image.new("RGB", (columns * (size + 12) + 12, 5 * (size + 6) + 12), "white")
    draw = ImageDraw.Draw(image)
    for column in range(columns):
        for row in range(5):
            x, y = 12 + column * (size + 12), 12 + row * (size + 6)
            draw.rectangle((x, y + size // 2 - stroke // 2, x + size - 1, y + size // 2 - stroke // 2 + stroke - 1), fill="black")
            draw.rectangle((x + size // 2 - stroke // 2, y, x + size // 2 - stroke // 2 + stroke - 1, y + size - 1), fill="black")
    return np.asarray(image)


def test_measure_stroke_is_the_glyph_stroke_over_the_glyph_size():
    thin, thick = measure_stroke(_columns(3), vertical=True), measure_stroke(_columns(7), vertical=True)
    assert thin == pytest.approx(3 / 30, abs=0.02) and thick == pytest.approx(7 / 30, abs=0.04)
    assert thick / thin == pytest.approx(7 / 3, rel=0.15)
    # The same lettering at twice the size weighs the same.
    doubled = np.asarray(Image.fromarray(_columns(3)).resize((2 * _columns(3).shape[1], 2 * _columns(3).shape[0]), Image.NEAREST))
    assert measure_stroke(doubled, vertical=True) == pytest.approx(thin, abs=0.02)
    # White on black reads the same; a blank box, and one too small, read nothing.
    assert measure_stroke(255 - _columns(3), vertical=True) == pytest.approx(thin, abs=0.01)
    assert measure_stroke(np.full((40, 40, 3), 255, np.uint8), vertical=True) is None
    assert measure_stroke(np.zeros((1, 1, 3), np.uint8), vertical=True) is None


# ---- the stage through a Session -------------------------------------------------------------------


class _Reader(LetteringProvider):
    """Reads every crop as ``answer``; records the crops it was given."""

    id = "stub-font-reader"
    name = "Stub font reader"

    def __init__(self, answer: FontReading):
        super().__init__()
        self.answer = answer
        self.sizes: list[tuple[int, int]] = []

    async def read_fonts(self, crops):
        self.sizes += [crop.size for crop in crops]
        return [self.answer.model_copy(deep=True) for _ in crops]


GOTHIC = FontReading(kernels={"gothic": 0.93, "rounded": 0.065, "pop": 0.005}, weight=812.34)


def _page_png() -> bytes:
    image = Image.fromarray(np.full((240, 200, 3), 255, np.uint8))
    image.paste(Image.fromarray(_columns(4, size=20, columns=2)), (20, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _scene(session: Session, text: str = KANJI_TEXT, typography: bool = True):
    edit = session.snapshot().edit()
    page = edit.add_page(Page(label="p1", width=200, height=240))
    edit.set_asset(page, "source", c.AssetInput(data=_page_png(), media_type="image/png"))
    region = edit.add_analysis_region(page, At.END, RegionKinds.TEXT, Geometry.rectangle(20, 30, 76, 142))
    content = edit.add_text_content(page)
    layer = edit.add_text_layer(page, At.END, content, TextLayout(kind=TextLayoutKind.PARAGRAPH))
    edit.relate(Relations.RECOGNIZED_FROM, content, region)
    edit.set(content, SourceText(text=Authored.user(text), language="ja-JP"))
    session.commit(edit.finish())
    if typography:  # as detection leaves it: generated, with the colours it read
        edit = session.snapshot().edit_as(generation("dev.koharu.pipeline.detection", "detector"))
        edit.set(layer, c.Typography(color=(10, 20, 30, 255), writing_mode=c.WritingMode.VERTICAL))
        session.commit(edit.finish())
    return page, region, content, layer


def _input(session: Session, page: str, force: bool = False) -> StageInput:
    return StageInput(scene=session.snapshot(), page=page, entities=None, region=None, images=ImageCache(), force=force)


def _run(session: Session, page: str, provider, force: bool = False) -> None:
    stage, input = LetteringStage(), _input(session, page, force)
    assert not stage.skip(input, provider)
    session.commit(asyncio.run(stage.process(input, provider)))


def test_the_stage_keeps_the_reading_and_styles_the_layer(tmp_path):
    session = Session(tmp_path)
    page, region, content, layer = _scene(session)
    reader = _Reader(GOTHIC)
    assert not LetteringStage().done(_input(session, page))
    _run(session, page, reader)

    snap = session.snapshot()
    analysis = snap.component(region, c.LetteringAnalysis)
    assert analysis.kernels == {"gothic": 0.93, "rounded": 0.065}  # under a hundredth is not kept
    assert analysis.weight == 812.3
    assert analysis.stroke == pytest.approx(4 / 20, abs=0.04)
    assert analysis.origin.generation.producer == "dev.urakatsu.pipeline.lettering"
    assert analysis.origin.generation.model == "stub-font-reader"
    # Five crops, the box with each margin more on each side, clamped to the page.
    assert len(reader.sizes) == len(MARGINS) and reader.sizes[0] == (76, 142)
    assert reader.sizes == sorted(reader.sizes)

    placed = snap.component(layer, c.LetteringStyle)
    assert placed.style == "gothic" and placed.font_weight is None  # nothing to be heavier than
    assert placed.origin.generation.producer == "dev.urakatsu.pipeline.lettering"
    # Detection's typography is its own: the stage leaves it as it is.
    typography = snap.component(layer, c.Typography)
    assert typography.style is None and typography.color == (10, 20, 30, 255)
    assert typography.origin.generation.producer == "dev.koharu.pipeline.detection"

    # Read once: a later run skips the page, a forced one reads it again.
    stage = LetteringStage()
    assert stage.done(_input(session, page)) and stage.skip(_input(session, page), reader)
    assert not stage.skip(_input(session, page, force=True), reader)


def test_the_stage_places_a_layer_without_typography_and_leaves_the_users(tmp_path):
    session = Session(tmp_path)
    page, _, _, layer = _scene(session, typography=False)
    _run(session, page, _Reader(GOTHIC))
    snap = session.snapshot()
    assert snap.component(layer, c.LetteringStyle).style == "gothic" and snap.component(layer, c.Typography) is None

    edit = snap.edit()
    edit.set(layer, c.Typography(preferred_font="Comic Neue", style="rounded"))
    session.commit(edit.finish())
    _run(session, page, _Reader(FontReading(kernels={"handwriting": 0.9})), force=True)
    snap = session.snapshot()
    typography = snap.component(layer, c.Typography)
    assert typography.style == "rounded" and typography.preferred_font == "Comic Neue" and typography.origin.kind == "user"
    assert snap.component(layer, c.LetteringStyle).style == "handwriting"


def test_a_forced_run_places_the_style_again(tmp_path):
    session = Session(tmp_path)
    page, _, _, layer = _scene(session)
    _run(session, page, _Reader(GOTHIC))
    _run(session, page, _Reader(FontReading(kernels={"handwriting": 0.8, "gothic": 0.2})), force=True)
    assert session.snapshot().component(layer, c.LetteringStyle).style == "handwriting"


def _add_text(session: Session, page: str, text: str, analysis: c.LetteringAnalysis | None = None):
    edit = session.snapshot().edit()
    region = edit.add_analysis_region(page, At.END, RegionKinds.TEXT, Geometry.rectangle(20, 30, 76, 142))
    content = edit.add_text_content(page)
    layer = edit.add_text_layer(page, At.END, content, TextLayout(kind=TextLayoutKind.PARAGRAPH))
    edit.relate(Relations.RECOGNIZED_FROM, content, region)
    edit.set(content, SourceText(text=Authored.user(text), language="ja-JP"))
    session.commit(edit.finish())
    if analysis is not None:
        edit = session.snapshot().edit_as(generation("dev.urakatsu.pipeline.lettering", "reader"))
        edit.set(region, analysis)
        session.commit(edit.finish())
    return region, layer


def test_a_page_is_placed_from_the_projects_readings(tmp_path):
    # A heavy shout on the second page is bold against the first page's dialogue.
    session = Session(tmp_path)
    page, _, _, layer = _scene(session, KANA_TEXT)
    for _ in range(4):
        _add_text(session, page, KANA_TEXT, c.LetteringAnalysis(kernels={"antique": 0.9, "gothic": 0.1}, weight=400.0, stroke=0.05))
    edit = session.snapshot().edit()
    second = edit.add_page(Page(label="p2", width=200, height=240))
    edit.set_asset(second, "source", c.AssetInput(data=_page_png(), media_type="image/png"))
    session.commit(edit.finish())
    _, shout_layer = _add_text(session, second, KANJI_TEXT)

    stage, input = LetteringStage(), _input(session, second)
    assert not stage.done(input)
    _run(session, second, _Reader(GOTHIC))
    snap = session.snapshot()
    placed = snap.component(shout_layer, c.LetteringStyle)
    assert (placed.style, placed.font_weight) == ("gothic", 700)
    assert snap.component(layer, c.LetteringStyle) is None  # the first page was not run


def test_the_stage_skips_a_page_without_text(tmp_path):
    session = Session(tmp_path)
    edit = session.snapshot().edit()
    page = edit.add_page(Page(label="p1", width=200, height=240))
    session.commit(edit.finish())
    stage = LetteringStage()
    assert stage.skip(_input(session, page), _Reader(GOTHIC)) and not stage.done(_input(session, page))


def test_crops_and_mean_reading():
    image = Image.new("RGB", (100, 100))
    sizes = [crop.size for crop in crops(image, (40.0, 40.0, 60.0, 80.0))]
    assert sizes == [(20, 40), (22, 44), (24, 46), (26, 50), (28, 56)]
    assert [crop.size for crop in crops(image, (0.0, 0.0, 100.0, 100.0))] == [(100, 100)] * 5  # clamped to the page
    mean = mean_reading([FontReading(kernels={"gothic": 1.0}, weight=700.0), FontReading(kernels={"gothic": 0.5, "pop": 0.5})])
    assert mean.kernels == {"gothic": 0.75, "pop": 0.25} and mean.weight == 700.0
    assert mean_reading([FontReading(), FontReading()]).weight is None


# ---- gating in a run -------------------------------------------------------------------------------


def test_a_run_that_reads_or_translates_text_letters_it():
    assert Stage.all().index(Stage.LETTERING) == Stage.all().index(Stage.OCR) + 1
    assert PREREQUISITES[Stage.LETTERING] == (Stage.DETECTION, Stage.OCR)
    assert Operation.through(Stage.LETTERING).resolve() == [Stage.DETECTION, Stage.OCR, Stage.LETTERING]
    assert _lettering_stages([Stage.DETECTION, Stage.OCR, Stage.TRANSLATION], "match") == [Stage.DETECTION, Stage.OCR, Stage.LETTERING, Stage.TRANSLATION]
    assert _lettering_stages([Stage.TRANSLATION], "match") == [Stage.LETTERING, Stage.TRANSLATION]
    assert _lettering_stages([Stage.DETECTION, Stage.INPAINTING], "match") == [Stage.DETECTION, Stage.INPAINTING]
    assert _lettering_stages([Stage.LETTERING], "match") == [Stage.LETTERING]
    assert _lettering_stages(Stage.all(), "off") == [stage for stage in Stage.all() if stage is not Stage.LETTERING]
    with pytest.raises(ValueError, match="lettering is off"):
        _lettering_stages([Stage.LETTERING], "off")


def test_a_run_goes_on_without_lettering_when_its_font_reader_cannot_run(tmp_path):
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from fakes import fake_config, fake_processors, fake_registry, page_png

    from urakatsu.pipeline.pipeline import Pipeline, Request
    from urakatsu.pipeline.scope import ProjectScope
    from urakatsu.runtimes.manager import RuntimeManager

    config = fake_config()
    config.lettering.mode = "match"  # the fake registry holds no font reader
    pipeline = Pipeline(config, RuntimeManager("cpu"), fake_registry(), fake_processors())
    assert not pipeline.can_letter()
    session = Session(tmp_path)
    edit = session.snapshot().edit()
    page = edit.add_page(Page(label="p1", width=400, height=300))
    edit.set_asset(page, "source", c.AssetInput(data=page_png(), media_type="image/png"))
    session.commit(edit.finish())
    report = asyncio.run(pipeline.execute(session, Request(scope=ProjectScope(), operation=Operation.full())))
    assert report.status.value == "completed"
    assert session.snapshot().component(session.snapshot().text_layers(page)[0], c.LetteringStyle) is None


# ---- the API ---------------------------------------------------------------------------------------


def test_config_carries_lettering_and_the_styles():
    config = PipelineConfig()
    message = convert.config_to_proto(config)
    assert message.lettering == "match" and message.lettering_model == "yuzumarker-font-detection"
    assert json.loads(message.styles_json)["rounded"]["Latn"] == {"family": "Anime Ace 3 BB", "italic": True}
    assert message.styles_json == message.default_styles_json

    styles = json.loads(message.styles_json)
    styles["gothic"]["Latn"] = {"family": "BadaBoom BB", "weight": 700}
    styles["mincho"]["Latn"] = {}  # a face that sets nothing is dropped
    message.styles_json, message.lettering, message.font_families[:] = json.dumps(styles), "off", ["Arial"]
    updated = convert.config_from_proto(message, config)
    assert updated.lettering.mode == "off" and updated.lettering.model == "yuzumarker-font-detection"
    assert updated.typesetting.styles["gothic"]["Latn"] == StyleFace(family="BadaBoom BB", weight=700)
    assert "Latn" not in updated.typesetting.styles["mincho"]
    assert updated.typesetting.font_families == ["Arial"]

    # Empty keeps what is there; unknown names are refused.
    kept = convert.config_from_proto(pb.Config(), updated)
    assert kept.typesetting.styles == updated.typesetting.styles and kept.lettering.mode == "off"
    for bad in ('{"comic": {}}', '{"gothic": {"Xxxx": {}}}', '{"gothic": {"Latn": {"weight": 5000}}}', "[]", "{"):
        with pytest.raises(ValueError):
            convert.config_from_proto(pb.Config(styles_json=bad), config)
    with pytest.raises(ValueError, match="unknown lettering mode"):
        convert.config_from_proto(pb.Config(lettering="always"), config)


def test_the_styles_survive_the_config_file(tmp_path):
    from urakatsu.core.config import load_config, save_config

    config = PipelineConfig()
    config.typesetting.styles["gothic"]["Latn"] = StyleFace(family="BadaBoom BB", italic=False)
    loaded = load_config(save_config(config, tmp_path / "pipeline.toml"))
    assert loaded.typesetting.styles == config.typesetting.styles and loaded.lettering == config.lettering


def test_typography_carries_its_style_and_an_edit_keeps_it():
    from urakatsu.service.views import typography_view
    from urakatsu.service.workspace import _typography

    current = c.Typography(origin=c.generated(generation("p", "m")), style="gothic", font_weight=700)
    message = convert.typography_to_proto(typography_view(current))
    assert message.style == "gothic" and convert.typography_from_proto(message)["style"] == "gothic"
    assert _typography({"color": "#ff0000ff"}, current).style == "gothic"  # changing the colour keeps the style
    assert _typography({"style": "rounded"}, current).style == "rounded"
    assert _typography({"style": None}, current).style is None
    with pytest.raises(ValueError, match="unknown lettering style"):
        _typography({"style": "comic"}, current)


# ---- the renderer ----------------------------------------------------------------------------------


def _descriptors(monkeypatch, snapshot, typesetting: TypesettingConfig | None = None) -> list:
    """The text descriptors a render hands the text renderer."""
    from urakatsu.renderer import renderer as module

    seen: list = []
    real = module.TextNodeDescriptor

    def capture(**fields):
        seen.append(real(**fields))
        return seen[-1]

    monkeypatch.setattr(module, "TextNodeDescriptor", capture)
    module.Renderer(typesetting or TypesettingConfig())._compile(snapshot, snapshot.pages()[0])
    return seen


def _translated(tmp_path, typography: c.Typography, text: str = "WAIT!", language: str = "en-US"):
    session = Session(tmp_path)
    page, _, content, layer = _scene(session, typography=False)
    edit = session.snapshot().edit()
    edit.set(content, c.Translation(text=Authored.user(text), language=language))
    edit.set(layer, Geometry.rectangle(20, 30, 76, 142))
    edit.set(layer, typography)
    session.commit(edit.finish())
    return session.snapshot()


def test_a_styled_text_is_set_in_its_styles_font(tmp_path, monkeypatch):
    [text] = _descriptors(monkeypatch, _translated(tmp_path, c.Typography(style="rounded")))
    assert text.preferred_font == "Anime Ace 3 BB" and text.font_style.value == "italic" and text.font_weight is None
    assert text.font_families == ["CCWildWords", "Adobe 黑体 Std"]


def test_a_styled_texts_own_font_weight_and_slant_win(tmp_path, monkeypatch):
    typography = c.Typography(style="rounded", preferred_font="Arial", font_weight=700, font_style=c.FontStyle.NORMAL)
    [text] = _descriptors(monkeypatch, _translated(tmp_path, typography))
    assert (text.preferred_font, text.font_weight, text.font_style.value) == ("Arial", 700, "normal")


def test_a_style_follows_the_translations_script_and_settings(tmp_path, monkeypatch):
    snapshot = _translated(tmp_path, c.Typography(style="mincho", font_weight=700), "等一下", "zh-CN")
    [text] = _descriptors(monkeypatch, snapshot)
    assert text.preferred_font == "Adobe 宋体 Std" and text.font_weight == 700
    assert text.font_families[:2] == ["Adobe 黑体 Std", "CCWildWords"]

    typesetting = TypesettingConfig()
    typesetting.styles["mincho"]["Hans"] = StyleFace(family="仿宋_GB2312", weight=300)
    [text] = _descriptors(monkeypatch, snapshot, typesetting)
    assert text.preferred_font == "仿宋_GB2312" and text.font_weight == 700  # the block's own bold stands


def test_a_placed_style_stands_where_the_typography_sets_none(tmp_path, monkeypatch):
    session = Session(tmp_path)
    page, _, content, layer = _scene(session)  # detection's typography: no style, font or weight
    edit = session.snapshot().edit()
    edit.set(content, c.Translation(text=Authored.user("WAIT!"), language="en-US"))
    edit.set(layer, Geometry.rectangle(20, 30, 76, 142))
    session.commit(edit.finish())
    edit = session.snapshot().edit_as(generation("dev.urakatsu.pipeline.lettering", "reader"))
    edit.set(layer, c.LetteringStyle(style="gothic", font_weight=700))
    session.commit(edit.finish())
    [text] = _descriptors(monkeypatch, session.snapshot())
    assert (text.preferred_font, text.font_weight) == ("CreativeBlock BB", 700)

    from urakatsu.service.views import page_view

    [block] = page_view(session.snapshot(), page).blocks
    assert (block.lettering_style, block.lettering_weight, block.typography.style) == ("gothic", 700, None)
    message = convert.block_to_proto(block)
    assert message.lettering_style == "gothic" and message.lettering_weight == 700

    # The typography's own style and weight win over the placed ones.
    edit = session.snapshot().edit()
    edit.set(layer, c.Typography(style="rounded", font_weight=400))
    session.commit(edit.finish())
    [text] = _descriptors(monkeypatch, session.snapshot())
    assert (text.preferred_font, text.font_weight) == ("Anime Ace 3 BB", 400)


def test_a_text_without_a_style_is_set_as_before(tmp_path, monkeypatch):
    [text] = _descriptors(monkeypatch, _translated(tmp_path, c.Typography()))
    assert text.preferred_font is None and text.font_weight is None and text.font_style is None
    assert text.font_families == ["CCWildWords", "Adobe 黑体 Std"]
