"""Renderer port tests: selected unit tests from the Rust renderer plus frame-level checks.

Each test names the Rust test it mirrors.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from urakatsu.renderer import bubble as B
from urakatsu.renderer import layout as L
from urakatsu.renderer import renderer as R
from urakatsu.renderer import script as S
from urakatsu.renderer import segment as G
from urakatsu.renderer.fonts import Attributes, Font, FontSystem, Fonts
from urakatsu.renderer.shape import ShapingOptions, TextShaper, shape_script_runs
from urakatsu.renderer.types import TextAlign, WritingMode
from urakatsu.scene import components as c
from urakatsu.scene import (
    At,
    AssetInput,
    AssetMetadata,
    Authored,
    Geometry,
    Page,
    RegionKinds,
    Relations,
    Session,
    Translation,
    TextLayout as SceneTextLayout,
    TextLayoutKind,
    Typography,
)
from urakatsu.core.config import TypesettingConfig
from urakatsu.renderer import Renderer


@pytest.fixture(scope="module")
def system() -> FontSystem:
    return FontSystem()


@pytest.fixture(scope="module")
def font(system: FontSystem) -> Font:
    for name in ("Yu Gothic", "Arial", "Segoe UI", "Liberation Sans"):
        try:
            return system.query_family(name)
        except LookupError:
            continue
    pytest.skip("no usable Latin font installed")


def texts(breaks, text):
    return [text[b.offset : n.offset] for b, n in zip(breaks, breaks[1:])]


# ---- segment.rs ------------------------------------------------------------------------------------


def test_break_on_whitespace():
    text = "The quick brown fox jumps over the lazy dog."
    breaks = G.LineBreaker().line_break_opportunities(text)
    assert texts(breaks, text) == [
        "The ", "quick ", "brown ", "fox ", "jumps ", "over ", "the ", "lazy ", "dog.",
    ]


def test_break_on_newline():
    text = "Hello, \nWorld!"
    breaks = G.LineBreaker().line_break_opportunities(text)
    assert breaks == [
        G.LineBreakOpportunity(0, False),
        G.LineBreakOpportunity(8, True),
        G.LineBreakOpportunity(14, False),
    ]


def test_line_segments_trim_newline_suffixes():
    text = "Hello, \nWorld!"
    segments = G.LineBreaker().line_segments(text)
    assert len(segments) == 2
    assert segments[0].text(text) == "Hello, "
    assert segments[0].next_offset == 8
    assert segments[0].is_mandatory
    assert segments[0].break_suffix is None
    assert segments[1].text(text) == "World!"
    assert segments[1].next_offset == len(text)
    assert not segments[1].is_mandatory


def test_chinese_word_segmentation_keeps_jieba_words_together():
    text = "南京市长江大桥"
    segments = [s.text(text) for s in G.LineBreaker().line_segments(text)]
    assert segments == ["南京市", "长江大桥"]


def test_chinese_word_segmentation_does_not_resegment_kana_text():
    text = "吾輩は猫"
    segments = [s.text(text) for s in G.LineBreaker().line_segments(text)]
    assert segments == ["吾", "輩", "は", "猫"]


def test_korean_word_segmentation_uses_keep_all():
    text = "B6층 모험가들이 돌아갔으니까 청소 부탁해"
    segments = [s.text(text) for s in G.LineBreaker().line_segments(text)]
    assert segments == ["B6층 ", "모험가들이 ", "돌아갔으니까 ", "청소 ", "부탁해"]


def test_japanese_break_on_characters():
    text = "吾輩は猫である。名前はまだない。"
    breaks = G.LineBreaker().line_break_opportunities(text)
    assert texts(breaks, text) == [
        "吾", "輩", "は", "猫", "で", "あ", "る。", "名", "前", "は", "ま", "だ", "な", "い。",
    ]


def test_hyphenation_adds_discretionary_segments_to_long_latin_words():
    text = "antidisestablishmentarianism"
    breaker = G.LineBreaker().with_hyphenation(G.HyphenationOptions.new("English", 8))
    segments = breaker.line_segments(text)
    assert len(segments) > 1
    for segment in segments[:-1]:
        assert segment.break_suffix is G.LineBreakSuffix.HYPHEN
        assert not segment.is_mandatory
    assert segments[-1].break_suffix is None
    assert "".join(s.text(text) for s in segments) == text


def test_hyphenation_does_not_invent_breaks_missing_from_the_patterns():
    text = "idols"
    segments = G.LineBreaker().with_hyphenation(G.HyphenationOptions.new("English", 5)).line_segments(text)
    assert len(segments) == 1
    assert segments[0].text(text) == text
    assert segments[0].break_suffix is None


def test_configured_fragment_bounds_relax_language_defaults():
    text = "tower within"
    options = G.HyphenationOptions.new("English", 5).with_fragment_bounds(2, 2)
    segments = G.LineBreaker().with_hyphenation(options).line_segments(text)
    assert [s.text(text) for s in segments] == ["tow", "er ", "with", "in"]
    assert segments[0].break_suffix is G.LineBreakSuffix.HYPHEN
    assert segments[2].break_suffix is G.LineBreakSuffix.HYPHEN


def test_hyphenation_language_tags_cover_hypher_languages():
    cases = {
        "af": "Afrikaans", "sq": "Albanian", "as": "Assamese", "be": "Belarusian", "bn": "Bengali",
        "bg": "Bulgarian", "ca": "Catalan", "hr": "Croatian", "cs": "Czech", "da": "Danish",
        "nl": "Dutch", "en-US": "English", "et": "Estonian", "fi": "Finnish", "fr-FR": "French",
        "gl": "Galician", "ka": "Georgian", "de-DE": "German", "el": "Greek", "gu": "Gujarati",
        "hi": "Hindi", "hu": "Hungarian", "is": "Icelandic", "it-IT": "Italian", "kn": "Kannada",
        "ku": "Kurmanji", "la": "Latin", "lt": "Lithuanian", "ml": "Malayalam", "mr": "Marathi",
        "mn": "Mongolian", "no": "Norwegian", "nb": "Norwegian", "nn": "Norwegian", "or": "Oriya",
        "pa": "Panjabi", "pl": "Polish", "pt-BR": "Portuguese", "ru": "Russian", "sa": "Sanskrit",
        "sr": "Serbian", "sk": "Slovak", "sl": "Slovenian", "es-ES": "Spanish", "sv": "Swedish",
        "ta": "Tamil", "te": "Telugu", "tr": "Turkish", "tk": "Turkmen", "uk": "Ukrainian",
    }
    for tag, language in cases.items():
        assert G.hyphenation_lang_from_tag(tag) == language, tag
    assert G.hyphenation_lang_from_tag("German") is None
    assert G.hyphenation_lang_from_tag("ja-JP") is None


def test_hyphenation_supports_unicode_words():
    text = "электрификация"
    segments = G.LineBreaker().with_hyphenation(G.HyphenationOptions.new("Russian", 8)).line_segments(text)
    assert len(segments) > 1
    assert "".join(s.text(text) for s in segments) == text


# ---- script.rs ----------------------------------------------------------------------------------------


def test_fallback_scripts_preserve_cjk_locale_selection():
    assert S.fontique_scripts("日本語") == ["Hani"]
    assert S.fontique_scripts("한국어") == ["Hang"]
    assert S.fontique_scripts("مرحبا") == ["Arab"]


def test_arabic_text_uses_rtl_shaping():
    direction, script = S.shaping_direction_for_text("مرحبا", WritingMode.HORIZONTAL)
    assert direction == "rtl"
    assert script == "Arab"


def test_source_writing_mode_is_limited_to_chinese_and_japanese():
    assert S.is_chinese_or_japanese_text("日本語")
    assert S.is_chinese_or_japanese_text("繁體中文")
    assert not S.is_chinese_or_japanese_text("한국어")
    assert not S.is_chinese_or_japanese_text("English")


# ---- shape.rs / fonts.rs ------------------------------------------------------------------------------


def test_shape_arabic_with_fallback_preserves_context(system: FontSystem):
    fonts = system.resolve_with_fallback_families(["Arial"], ["Segoe UI Symbol"], Attributes(), ["Arab"])
    options = ShapingOptions(direction="rtl", script="Arab", font_size=16.0)
    text = "مرحبا"
    runs = shape_script_runs(TextShaper(), text, fonts, options)
    assert runs
    first = runs[0].glyphs[0].font
    assert all(glyph.font is first for glyph in runs[0].glyphs)


def test_font_resolution_orders_requested_family_first_then_script_fallbacks(system: FontSystem):
    fonts = system.resolve_with_fallback_families(["Arial"], [], Attributes(), ["Hani"])
    assert fonts[0].family_name == "Arial"
    if system.family_exists("Microsoft YaHei"):
        assert "Microsoft YaHei" in [font.family_name for font in fonts[1:]]


def test_font_covers_ignores_whitespace_and_variation_selectors(system: FontSystem):
    latin = system.query_family("Arial")
    assert latin.covers("Hi there‍")
    assert latin.covers("A️")
    assert not latin.covers("一")  # a Latin face has no Han glyph


# ---- layout.rs ----------------------------------------------------------------------------------------


def _measure(advance, trailing=0.0, suffix=0.0, penalty=0.0, mandatory=False):
    return L.LineBreakMeasure(
        advance=advance,
        trailing_advance=trailing,
        break_suffix_advance=suffix,
        break_penalty=penalty,
        is_mandatory=mandatory,
    )


def test_optimal_line_breaks_balance_ragged_lines():
    segments = [_measure(30.0)] * 7
    assert L.optimal_uniform_line_breaks(segments, 100.0, True).breaks == [2, 4, 7]


def test_line_fit_excludes_invisible_trailing_break_space():
    segments = [_measure(55.0, trailing=10.0), _measure(40.0)]
    result = L.line_breaks_with_policy(segments, 50.0, L.HyphenationPolicy.DISABLED)
    assert result.breaks == [1, 2]
    assert not result.overflowed


def test_largest_fitting_font_size_handles_non_monotonic_balloon_fit():
    calls = []

    def layout_at(size):
        calls.append(size)
        return size

    fitted = L.largest_fitting_font_size(9.0, 24.0, layout_at, lambda size: 10.0 <= size <= 12.0)
    assert fitted is not None and 11.99 <= fitted <= 12.0
    assert len(calls) <= 18


def test_capped_auto_size_shrinks_text_to_fit_the_available_height(font: Font):
    fixed = L.TextLayout(font).with_font_size(24.0).with_max_width(1_000.0).run("First\nSecond\nThird")
    max_height = fixed.height * 0.65
    fitted = (
        L.TextLayout(font)
        .with_max_font_size(24.0)
        .with_max_width(1_000.0)
        .with_max_height(max_height)
        .run("First\nSecond\nThird")
    )
    assert fitted.font_size < 24.0
    assert fitted.width <= 1_000.0
    assert fitted.height <= max_height + 0.01


def test_capped_auto_size_does_not_enlarge_text_that_already_fits(font: Font):
    fitted = L.TextLayout(font).with_max_font_size(12.0).with_max_width(1_000.0).with_max_height(1_000.0).run("Hi")
    assert fitted.font_size == pytest.approx(12.0)


def test_mandatory_newlines_are_not_shaped_as_glyphs(font: Font):
    text = "A\nB"
    run = L.TextLayout(font).with_font_size(16.0).with_max_width(500.0).run(text)
    assert [line.range for line in run.lines] == [(0, 1), (2, 3)]
    for line in run.lines:
        for glyph in line.glyphs:
            assert text[glyph.cluster] != "\n"


def test_horizontal_layout_hyphenates_long_words(font: Font):
    run = (
        L.TextLayout(font)
        .with_font_size(20.0)
        .with_max_width(90.0)
        .with_hyphenation_language_tag("en-US")
        .run("antidisestablishmentarianism")
    )
    assert len(run.lines) > 1


def test_horizontal_layout_wraps_chinese_on_jieba_word_boundaries(font: Font):
    text = "南京市长江大桥"
    run = L.TextLayout(font).with_font_size(16.0).with_max_width(16.0 * 3.2).run(text)
    starts = [line.range[0] for line in run.lines]
    assert 3 in starts  # the break falls before 长江大桥, not inside 南京市


def test_vertical_layout_preserves_original_source_offsets(font: Font):
    # jieba keeps "一二三四五六七八九十" as one word, so use text that breaks per character.
    text = "吾輩は猫である。名前はまだない。"
    run = (
        L.TextLayout(font)
        .with_font_size(16.0)
        .with_writing_mode(WritingMode.VERTICAL_RL)
        .with_max_height(60.0)
        .run(text)
    )
    assert len(run.lines) > 1
    covered = []
    for line in run.lines:
        covered.extend(range(line.range[0], line.range[1]))
        for glyph in line.glyphs:
            assert line.range[0] <= glyph.cluster < line.range[1] + 1
    assert covered == list(range(len(text)))


def test_cjk_emphasis_normalization_only_expands_ascii_marks():
    assert L.normalize_cjk_emphasis_punctuation("Hi!?") == ("Hi！？", [0, 1, 2, 3])
    assert L.normalize_cjk_emphasis_punctuation("！？") is None
    assert L.normalize_cjk_emphasis_punctuation("plain") is None


def test_centered_x_offset_uses_absolute_center():
    assert L.centered_x_offset(-10.0, 30.0) == -10.0


def test_reorder_visual_reverses_rtl_runs():
    assert L.reorder_visual([0, 0, 1, 1, 0]) == [0, 1, 3, 2, 4]
    assert L.reorder_visual([1, 1]) == [1, 0]


def test_comic_profiles_are_widest_in_the_middle_of_a_round_balloon():
    balloon = L.ComicBalloon(width=200.0, height=120.0, contour=[], minimum_air=0.0)
    ink = L.InkBand(before=10.0, after=4.0)
    candidates = balloon.line_profile_candidates(WritingMode.HORIZONTAL, 3, 16.0, ink, 0.0, 0.0)
    assert candidates
    widths = [profile.width for profile in candidates[0]]
    assert widths[1] >= widths[0] and widths[1] >= widths[2]


# ---- bubble.rs ----------------------------------------------------------------------------------------


def test_rotated_rectangle_preserves_layout_dimensions_and_angle():
    geometry = Geometry(points=[c.Point(x=0.0, y=0.0), c.Point(x=0.0, y=100.0), c.Point(x=-40.0, y=100.0), c.Point(x=-40.0, y=0.0)])
    frame = B.geometry_frame(geometry, None)
    assert frame is not None
    assert frame.bounds.width == pytest.approx(100.0)
    assert frame.bounds.height == pytest.approx(40.0)
    assert frame.angle_degrees == pytest.approx(90.0)


def test_flow_cells_split_the_frame_between_source_anchors():
    geometry = Geometry.rectangle(0.0, 0.0, 200.0, 100.0)
    frame = B.geometry_frame(geometry, None)
    polygon = B.contour(geometry, frame)
    cells = B.flow_cells(frame, polygon, [(50.0, 50.0), (150.0, 50.0)])
    assert len(cells) == 2
    assert all(len(cell) >= 3 for cell in cells)
    assert B._point_in_polygon(cells[0], (50.0, 50.0), 1e-3)
    assert B._point_in_polygon(cells[1], (150.0, 50.0), 1e-3)


def test_flow_cells_disambiguate_coincident_source_anchors():
    geometry = Geometry.rectangle(0.0, 0.0, 200.0, 100.0)
    frame = B.geometry_frame(geometry, None)
    polygon = B.contour(geometry, frame)
    cells = B.flow_cells(frame, polygon, [(100.0, 50.0), (100.0, 50.0)])
    assert len(cells) == 2
    assert all(cells)


def test_flow_cells_cut_a_dumbbell_neck_into_two_lobes():
    # Two squares joined by a narrow neck, as in a joined balloon.
    points = [
        (0, 0), (60, 0), (60, 30), (140, 30), (140, 0), (200, 0), (200, 80), (140, 80),
        (140, 50), (60, 50), (60, 80), (0, 80),
    ]
    geometry = Geometry(points=[c.Point(x=float(x), y=float(y)) for x, y in points])
    frame = B.geometry_frame(geometry, None)
    polygon = B.contour(geometry, frame)
    cells = B.flow_cells(frame, polygon, [(30.0, 40.0), (170.0, 40.0)])
    assert len(cells) == 2
    assert B._point_in_polygon(cells[0], (30.0, 40.0), 1e-3)
    assert B._point_in_polygon(cells[1], (170.0, 40.0), 1e-3)


# ---- renderer.rs / frame.rs --------------------------------------------------------------------------


def test_vertical_alignment_defaults_to_start_and_uses_the_inline_axis():
    assert R._resolve_alignment(None, WritingMode.VERTICAL_RL, False) is TextAlign.LEFT
    assert R._resolve_alignment(c.TextAlignment.START, WritingMode.VERTICAL_RL, True) is TextAlign.LEFT
    assert R._resolve_alignment(c.TextAlignment.END, WritingMode.VERTICAL_RL, False) is TextAlign.RIGHT
    assert R._resolve_alignment(c.TextAlignment.START, WritingMode.HORIZONTAL, True) is TextAlign.RIGHT


def test_user_writing_mode_overrides_language_default():
    typography = c.Typography(writing_mode=c.WritingMode.VERTICAL, origin=c.UserOrigin())
    mode = R._resolve_writing_mode("English words", LayoutBoxFor(10, 10), typography, None)
    assert mode is WritingMode.VERTICAL_RL


def test_generated_vertical_mode_is_limited_to_chinese_and_japanese():
    typography = c.Typography(writing_mode=c.WritingMode.VERTICAL, origin=c.generated(c.Generation(producer="x")))
    assert R._resolve_writing_mode("English words", LayoutBoxFor(10, 10), typography, None) is WritingMode.HORIZONTAL
    assert R._resolve_writing_mode("日本語", LayoutBoxFor(10, 10), typography, None) is WritingMode.VERTICAL_RL


def LayoutBoxFor(width, height):
    return B.LayoutBox(x=0.0, y=0.0, width=float(width), height=float(height))


def _png(width: int, height: int, color=(255, 255, 255)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture()
def shared_fonts(system: FontSystem, monkeypatch):
    original = Fonts.system.fget

    def shared(self):
        if self._system is None:
            self._system = system
        return self._system

    monkeypatch.setattr(Fonts, "system", property(shared))
    yield
    monkeypatch.setattr(Fonts, "system", property(original))


def _scene(tmp_path, *, source=True, translation="Hello there", vertical=False):
    session = Session(tmp_path)
    edit = session.snapshot().edit()
    page = edit.add_page(Page(label="p", width=200, height=120))
    if source:
        edit.set_asset(page, "source", AssetInput(data=_png(200, 120), media_type="image/png",
                                                  metadata=AssetMetadata(width=200, height=120)))
    balloon = edit.add_analysis_region(page, At.END, RegionKinds.BUBBLE, Geometry.rectangle(20, 20, 120, 80))
    content = edit.add_text_content(page)
    edit.set(content, Translation(text=Authored.user(translation), language="en"))
    layer = edit.add_text_layer(page, At.END, content, SceneTextLayout(kind=TextLayoutKind.PARAGRAPH))
    edit.relate(Relations.FLOWS_IN, layer, balloon)
    if vertical:
        edit.set(layer, Typography(writing_mode=c.WritingMode.VERTICAL, origin=c.user()))
    session.commit(edit.finish())
    return session


def test_render_composites_source_and_visible_text(tmp_path, shared_fonts):
    session = _scene(tmp_path)
    snapshot = session.snapshot()
    page = snapshot.pages()[0]
    renderer = Renderer(TypesettingConfig())
    frame = renderer.render(snapshot, page)
    assert frame.size() == (200, 120)
    assert frame.layers[0].kind.__class__.__name__ == "ImageMetadata"
    text_layers = [layer for layer in frame.layers if layer.is_text()]
    assert len(text_layers) == 1
    assert text_layers[0].kind.text == "Hello there"
    composite = frame.composite()
    assert composite.mode == "RGBA" and composite.size == (200, 120)
    # The balloon interior is white; text pixels must make it darker somewhere.
    pixels = list(composite.crop((20, 20, 140, 100)).getdata())
    assert any(p[0] < 128 for p in pixels)


def test_render_reports_a_missing_source_asset(tmp_path, shared_fonts):
    session = _scene(tmp_path, source=False)
    snapshot = session.snapshot()
    frame = Renderer(TypesettingConfig()).render(snapshot, snapshot.pages()[0])
    assert any(d.kind == "missing_asset" for d in frame.diagnostics)


def test_layer_image_and_cropped_frame_share_the_layer_bounds(tmp_path, shared_fonts):
    session = _scene(tmp_path)
    snapshot = session.snapshot()
    frame = Renderer(TypesettingConfig()).render(snapshot, snapshot.pages()[0])
    layer = next(layer for layer in frame.layers if layer.is_text())
    image = layer.image()
    assert image.mode == "RGBA"
    cropped = frame.cropped(layer.entity)
    assert cropped is not None
    assert cropped.size() == image.size
    assert cropped.layers[0].presentation.opacity == 1.0


def test_available_fonts_lists_system_families(shared_fonts, tmp_path):
    renderer = Renderer(TypesettingConfig())
    families = renderer.available_fonts()
    assert families
    assert all(family.name for family in families)
    assert any(family.name.lower() == "arial" for family in families)


# ---- more layout.rs ports: readable floor, contour spans, natural pauses, last-resort hyphens -------


def test_capped_auto_size_respects_the_readable_floor(font: Font):
    fitted = (
        L.TextLayout(font)
        .with_max_font_size(16.0)
        .with_min_font_size(8.0)
        .with_max_width(12.0)
        .with_max_height(12.0)
        .run("This translation cannot fit")
    )
    assert fitted.font_size == 8.0


def test_comic_contour_selects_the_widest_concave_span():
    balloon = L.ComicBalloon(
        width=100.0,
        height=100.0,
        contour=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (70.0, 100.0), (70.0, 20.0), (40.0, 20.0), (40.0, 100.0), (0.0, 100.0)],
        minimum_air=0.0,
    )
    assert balloon.contour_inline_span(WritingMode.HORIZONTAL, 50.0, 0.0) == (0.0, 40.0)


def test_comic_contour_keeps_air_at_both_edges():
    balloon = L.ComicBalloon(
        width=100.0,
        height=100.0,
        contour=[(50.0, 0.0), (100.0, 0.0), (100.0, 100.0), (50.0, 100.0)],
        minimum_air=0.0,
    )
    assert balloon.contour_inline_span(WritingMode.HORIZONTAL, 50.0, 10.0) == (60.0, 90.0)


def test_comic_breaks_prefer_natural_pauses():
    segments = [_measure(30.0) for _ in range(5)]
    segments[2] = _measure(30.0, penalty=500.0)
    assert L.optimal_uniform_line_breaks(segments, 90.0, True).breaks == [2, 5]
    assert L.comic_break_penalty("Stop! Now", 6) < L.comic_break_penalty("go and", 3)
    assert L.comic_break_penalty("go and", 3) < L.comic_break_penalty("hello world", 6)
    assert L.comic_break_penalty("hello world", 6) < L.comic_break_penalty("the word", 4)


def test_last_resort_hyphenation_is_used_only_to_avoid_overflow():
    fits_without_hyphen = [_measure(30.0, suffix=5.0), _measure(30.0)]
    unbroken = L.line_breaks_with_policy(fits_without_hyphen, 70.0, L.HyphenationPolicy.LAST_RESORT)
    assert unbroken.breaks == [2]
    assert not unbroken.overflowed

    needs_hyphen = [_measure(55.0, suffix=10.0), _measure(55.0)]
    hyphenated = L.line_breaks_with_policy(needs_hyphen, 70.0, L.HyphenationPolicy.LAST_RESORT)
    assert hyphenated.breaks == [1, 2]
    assert not hyphenated.overflowed


def test_comic_auto_size_uses_tall_balloon_capacity(font: Font):
    width, height = 130.0, 250.0
    fitted = (
        L.TextLayout(font)
        .with_max_font_size(width)
        .with_min_font_size(9.0)
        .with_line_height(1.2)
        .with_hyphenation_policy(L.HyphenationPolicy.LAST_RESORT)
        .with_max_width(width)
        .with_max_height(height)
        .with_comic_balloon(width, height, [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)], 4.0)
        .run("The Justice Realization Committee is a volunteer organization dedicated to the cause of justice.")
    )
    assert not fitted.overflowed()
    assert fitted.height >= height * 0.5
