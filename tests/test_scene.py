"""Scene kernel: structure, authorship, stale-patch rebase, persistence, undo."""

import pytest

from urakatsu.scene import (
    At,
    AuthorshipError,
    Generation,
    Geometry,
    Page,
    RasterLayer,
    RasterLayerKind,
    RegionKinds,
    Relations,
    RemovePolicy,
    Session,
    SourceText,
    SceneError,
    StalePatch,
    TextLayout,
    TextLayoutKind,
    Translation,
    Authored,
    AssetInput,
)

PIPELINE = Generation(producer="dev.koharu.pipeline.translation", model="test")
OTHER = Generation(producer="dev.koharu.pipeline.inpainting", model="test")


def make_page(session: Session):
    edit = session.snapshot().edit()
    page = edit.add_page(Page(label="p", width=100, height=100))
    content = edit.add_text_content(page)
    region = edit.add_analysis_region(page, At.END, RegionKinds.TEXT, Geometry.rectangle(1, 2, 3, 4))
    layer = edit.add_text_layer(page, At.END, content, TextLayout(kind=TextLayoutKind.PARAGRAPH))
    edit.relate(Relations.RECOGNIZED_FROM, content, region)
    edit.relate(Relations.FITS_TO, layer, region)
    edit.set(content, SourceText(text=Authored.user("before")))
    session.commit(edit.finish())
    return page, content, region, layer


def test_document_views(tmp_path):
    session = Session(tmp_path)
    page, content, region, layer = make_page(session)
    snap = session.snapshot()
    assert snap.text_layers(page) == [layer]
    assert snap.layer_content(layer) == content
    assert snap.source_region(content) == region
    assert snap.frame(layer).bounds() == (1, 2, 4, 6)
    with pytest.raises(Exception):
        snap.edit().relate(Relations.FITS_TO, layer, region)  # functional relation


def test_generated_edits_cannot_overwrite_user_values(tmp_path):
    session = Session(tmp_path)
    page, content, region, layer = make_page(session)
    edit = session.snapshot().edit_as(PIPELINE)
    edit.set(content, Translation(text=Authored.user("x")))
    session.commit(edit.finish())
    stored = session.snapshot().component(content, Translation)
    assert stored.text.origin.generation.producer == PIPELINE.producer  # stamped as generated
    edit = session.snapshot().edit_as(PIPELINE)
    with pytest.raises(AuthorshipError):
        edit.set(content, SourceText(text=Authored.user("overwrite")))  # user-owned
    edit = session.snapshot().edit_as(OTHER)
    with pytest.raises(AuthorshipError):
        edit.set(content, Translation(text=Authored.user("y")))  # other producer


def test_concurrent_stage_patches_rebase_and_stale_ones_fail(tmp_path):
    session = Session(tmp_path)
    page, content, region, layer = make_page(session)
    base = session.snapshot()

    text_edit = base.edit_as(PIPELINE)
    text_edit.observe(content, SourceText)
    text_edit.set(content, Translation(text=Authored.user("after")))

    image_edit = base.edit_as(OTHER)
    image_edit.observe_assets(page)
    cleanup = image_edit.add_entity(page, At.START)
    image_edit.set(cleanup, RasterLayer(name="Cleanup", kind=RasterLayerKind.CLEANUP))
    image_edit.set_asset(cleanup, "source", AssetInput(data=b"png", media_type="image/png"))

    session.commit(image_edit.finish())
    session.commit(text_edit.finish())  # rebases onto the image commit
    snap = session.snapshot()
    assert snap.component(content, Translation).text.value == "after"
    assert snap.raster_layers(page, RasterLayerKind.CLEANUP) == [cleanup]
    assert snap.asset_bytes(cleanup, "source") == b"png"

    stale = base.edit_as(PIPELINE)
    stale.observe(content, SourceText)
    stale.set(content, Translation(text=Authored.user("late")))
    changer = session.snapshot().edit()
    changer.set(content, SourceText(text=Authored.user("changed")))
    session.commit(changer.finish())
    with pytest.raises(StalePatch):
        session.commit(stale.finish())


def test_persistence_undo_and_cascade(tmp_path):
    session = Session(tmp_path)
    page, content, region, layer = make_page(session)
    reopened = Session(tmp_path)
    assert reopened.snapshot().layer_content(layer) == content

    edit = session.snapshot().edit()
    edit.remove_entity(region, RemovePolicy.CASCADE)
    session.commit(edit.finish())
    snap = session.snapshot()
    assert not snap.contains(region) and snap.fit_target(layer) is None
    session.undo()
    assert session.snapshot().fit_target(layer) == region
    session.redo()
    assert not session.snapshot().contains(region)


def test_move_entity_rejects_a_hierarchy_cycle(tmp_path):
    session = Session(tmp_path)
    page, *_ = make_page(session)
    edit = session.snapshot().edit()
    outer = edit.add_entity(page)
    inner = edit.add_entity(outer)
    with pytest.raises(SceneError, match="cycle"):
        edit.move_entity(outer, inner)
    with pytest.raises(SceneError, match="cycle"):
        edit.move_entity(outer, outer)


def test_unreadable_project_state_is_an_error_not_an_empty_project(tmp_path):
    session = Session(tmp_path)
    make_page(session)
    for slot in tmp_path.glob("state.*.json"):
        slot.write_text("{corrupt", encoding="utf-8")
    with pytest.raises(SceneError, match="unreadable"):
        Session(tmp_path)
    assert Session(tmp_path / "fresh").snapshot().revision == 0
