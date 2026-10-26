"""Service transports (native gRPC, gRPC-web bridge, JSON API) over one workspace, with fakes."""

import base64
import io
import json

import grpc
import httpx
import pytest
from PIL import Image

from urakatsu.core.registry import registry
from urakatsu.core.stage import Stage
from urakatsu.pipeline import Pipeline
from urakatsu.runtimes import RuntimeManager
from urakatsu.service import convert
from urakatsu.service.app import build_app
from urakatsu.service.gen import urakatsu_pb2 as pb
from urakatsu.service.gen import urakatsu_pb2_grpc as rpc
from urakatsu.service.grpc_web import DATA_FRAME, frame, parse_frames
from urakatsu.service.servicer import UrakatsuServicer
from urakatsu.service.workspace import Workspace

from fakes import FakeDetector, FakeInpainter, FakeOcr, FakeTranslator, fake_config, fake_processors, page_png


class FakeFrame:
    def __init__(self, size):
        self.size = size
        self.layers = []

    def composite(self):
        return Image.new("RGBA", self.size, (255, 255, 255, 255))


class FakeRenderer:
    def render(self, snapshot, page):
        meta = snapshot.page(page)
        return FakeFrame((int(meta.width), int(meta.height)))

    def available_fonts(self):
        return [{"name": "Test Sans", "faces": []}]


@pytest.fixture
def workspace(isolated_home):
    fakes = (FakeDetector, FakeOcr, FakeTranslator, FakeInpainter)
    registry.discover()
    for cls in fakes:
        registry.add(cls)
    ws = Workspace(device="cpu")
    config = fake_config()
    ws.pipeline = Pipeline(config, RuntimeManager("cpu"), registry, fake_processors())
    ws._renderer = FakeRenderer()
    ws.set_config(config)
    yield ws
    for cls in fakes:
        registry.remove(cls.stage, cls.id)


@pytest.mark.asyncio
async def test_native_grpc_roundtrip(workspace):
    server = grpc.aio.server()
    rpc.add_UrakatsuServicer_to_server(UrakatsuServicer(workspace), server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = rpc.UrakatsuStub(channel)
            caps = await stub.GetCapabilities(pb.Empty())
            assert caps.device.backend == "cpu" and list(caps.stages) == ["detection", "ocr", "translation", "inpainting", "sound_effects"]
            detector = next(p for p in caps.providers if p.id == "fake-detector")
            assert [r.name for r in detector.runtimes] == ["torch", "onnx", "remote", "cloud"]

            config = await stub.GetConfig(pb.Empty())
            config.runtimes["fake-detector"] = "onnx"
            config = await stub.SetConfig(config)
            assert config.runtimes["fake-detector"] == "onnx"
            config.runtimes["fake-detector"] = "llama"
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.SetConfig(config)
            assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT

            project = await stub.CreateProject(pb.CreateProjectRequest(name="grpc"))
            page = await stub.AddPage(pb.AddPageRequest(project_id=project.id, name="p.png", data=page_png()))
            assert page.width == 400 and "source" in page.assets

            events = [e async for e in stub.Run(pb.RunRequest(project_id=project.id, operation=pb.Operation(kind="full")))]
            assert events[-1].kind == "completed"
            finished = [e for e in events if e.kind == "finished" and e.stage == "translation"]
            assert finished and finished[0].page.blocks[0].translation == "EN:テスト"

            page = await stub.GetPage(pb.PageRef(project_id=project.id, page_id=page.id))
            assert [s.done for s in page.stages] == [True, True, True, True]
            block = page.blocks[0]
            assert block.translation_origin == "generated" and block.region_id

            updated = await stub.UpdateTextBlock(
                pb.UpdateTextBlockRequest(project_id=project.id, page_id=page.id, layer_id=block.layer_id, translation="By hand", fields=["translation"])
            )
            assert updated.blocks[0].translation == "By hand" and updated.blocks[0].translation_origin == "user"
            restored = await stub.Undo(pb.ProjectRef(project_id=project.id))
            assert restored.pages[0].blocks[0].translation == "EN:テスト" and restored.can_redo

            render = await stub.GetAsset(pb.AssetRequest(project_id=project.id, page_id=page.id, kind="render"))
            assert render.found and render.data[:8] == b"\x89PNG\r\n\x1a\n"
            cleanup = await stub.GetAsset(pb.AssetRequest(project_id=project.id, page_id=page.id, kind="cleanup"))
            assert cleanup.found

            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.GetProject(pb.ProjectRef(project_id="nope"))
            assert error.value.code() == grpc.StatusCode.NOT_FOUND
    finally:
        await server.stop(None)


@pytest.mark.asyncio
async def test_grpc_web_bridge_and_json_api(workspace):
    app = build_app(workspace, UrakatsuServicer(workspace), serve_ui=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        body = frame(DATA_FRAME, pb.CreateProjectRequest(name="web").SerializeToString())
        response = await client.post("/urakatsu.v1.Urakatsu/CreateProject", content=body, headers={"content-type": "application/grpc-web+proto"})
        frames = parse_frames(response.content)
        assert len(frames) == 2 and b"grpc-status:0" in frames[1][1]
        project = pb.Project.FromString(frames[0][1])

        response = await client.post(f"/api/v1/projects/{project.id}/pages", files=[("files", ("page.png", page_png(), "image/png"))])
        assert response.status_code == 201, response.text
        page_id = response.json()[0]["id"]

        response = await client.post(f"/api/v1/projects/{project.id}/run", json={"operation": "through", "stage": "ocr"})
        job_id = response.json()["job_id"]
        events = []
        async with client.stream("GET", f"/api/v1/jobs/{job_id}/events") as stream:
            async for line in stream.aiter_lines():
                if line.startswith("data: ") and line != "data: {}":
                    events.append(json.loads(line[6:]))
        assert any(e["kind"] == "finished" and e["stage"] == "ocr" for e in events)

        body = frame(DATA_FRAME, pb.RunRequest(project_id=project.id, operation=pb.Operation(kind="full")).SerializeToString())
        frames = parse_frames((await client.post("/urakatsu.v1.Urakatsu/Run", content=body)).content)
        assert b"grpc-status:0" in frames[-1][1]
        progress = [pb.ProgressEvent.FromString(p) for flag, p in frames if flag == DATA_FRAME]
        assert any(e.kind == "finished" and e.stage == "inpainting" for e in progress)

        page = (await client.get(f"/api/v1/projects/{project.id}/pages/{page_id}")).json()
        layer = page["blocks"][0]["layer_id"]
        response = await client.patch(
            f"/api/v1/projects/{project.id}/pages/{page_id}/blocks/{layer}",
            json={"typography": {"writing_mode": "Vertical", "color": "#ff0000", "auto_fit": True}},
        )
        assert response.status_code == 200, response.text
        assert response.json()["blocks"][0]["typography"]["color"] == "#ff0000ff"

        response = await client.get(f"/api/v1/projects/{project.id}/pages/{page_id}/assets/render")
        assert response.status_code == 200 and response.headers["content-type"] == "image/png"
        assert response.headers["cache-control"] == "no-cache"
        # A URL with the page revision names one image for good, so the browser may keep it.
        response = await client.get(f"/api/v1/projects/{project.id}/pages/{page_id}/assets/thumbnail?v=7")
        assert response.status_code == 200 and response.headers["cache-control"] == "private, max-age=31536000, immutable"

        assert (await client.get("/api/v1/config")).json()["detection"]["model"] == "fake-detector"
        proto = convert.config_to_proto(workspace.get_config())
        assert convert.config_from_proto(proto, workspace.get_config()) == workspace.get_config()
        lossless = workspace.get_config().model_copy(deep=True)
        lossless.storage.lossless_pages = True
        assert convert.config_from_proto(convert.config_to_proto(lossless), workspace.get_config()).storage.lossless_pages

        response = await client.post("/api/v1/process?operation=full", files=[("file", ("page.png", page_png(), "image/png"))])
        assert response.status_code == 200 and response.headers["content-type"] == "image/png"


def test_layer_operations(workspace):
    project = workspace.create_project("layers").id
    page = workspace.add_page(project, "p.png", page_png()).id
    for bounds in ({"x": 10, "y": 10, "width": 40, "height": 20}, {"x": 60, "y": 60, "width": 40, "height": 20}):
        workspace.create_text_block(project, page, bounds)
    first, second = [block.layer_id for block in workspace.page(project, page).blocks]

    view = workspace.set_layer_visibility(project, page, [first], False, 0.25)
    block = next(b for b in view.blocks if b.layer_id == first)
    assert (block.visible, block.opacity) == (False, 0.25)

    view = workspace.move_layer(project, page, first, 1)
    assert [b.layer_id for b in view.blocks] == [second, first]

    paint = Image.new("RGBA", (400, 300), (0, 0, 0, 0))
    paint.putpixel((5, 5), (255, 0, 0, 255))
    buffer = io.BytesIO()
    paint.save(buffer, format="PNG")
    view = workspace.set_raster_layer(project, page, "paint", None, buffer.getvalue())
    [layer] = [r for r in view.raster_layers if r.kind == "paint"]
    data, media_type, *_ = workspace.asset(project, page, f"layer:{layer.id}")
    assert media_type == "image/png" and Image.open(io.BytesIO(data)).getpixel((5, 5)) == (255, 0, 0, 255)
    # A second stroke replaces the same layer rather than adding one.
    view = workspace.set_raster_layer(project, page, "paint", None, buffer.getvalue())
    assert [r.id for r in view.raster_layers if r.kind == "paint"] == [layer.id]
    with pytest.raises(ValueError):
        workspace.set_raster_layer(project, page, "paint", None, page_png(10, 10))

    view = workspace.delete_layer(project, page, layer.id)
    assert not view.raster_layers
    view = workspace.delete_layer(project, page, second)
    assert [b.layer_id for b in view.blocks] == [first]


def test_rich_text_styles_travel_with_the_translation(workspace):
    project = workspace.create_project("rich").id
    page = workspace.add_page(project, "p.png", page_png()).id
    workspace.create_text_block(project, page, {"x": 10, "y": 10, "width": 200, "height": 60})
    [block] = workspace.page(project, page).blocks

    styles = [
        {"start": 4, "end": 8, "bold": True},
        {"start": 9, "end": 99, "underline": True, "color": "#ff0000ff"},  # past the end: clamped
        {"start": 0, "end": 3},  # no formatting: dropped
    ]
    view = workspace.update_text_block(project, page, block.layer_id, {"translation": "One bold word", "translation_styles": styles})
    [block] = view.blocks
    assert [(s.start, s.end, s.bold, s.underline, s.color) for s in block.translation_styles] == [
        (4, 8, True, False, None),
        (9, 13, False, True, "#ff0000ff"),
    ]
    message = convert.page_to_proto(view).blocks[0]
    assert [(s.start, s.end, s.bold, s.color) for s in message.translation_styles] == [(4, 8, True, ""), (9, 13, False, "#ff0000ff")]

    # A family and a relative size of its own; 100% is no size change.
    view = workspace.update_text_block(
        project, page, block.layer_id,
        {"translation": "One bold word", "translation_styles": [{"start": 0, "end": 3, "font": "Times New Roman", "scale": 1.5}, {"start": 4, "end": 8, "scale": 1.0}]},
    )
    assert [(s.start, s.end, s.font, s.scale) for s in view.blocks[0].translation_styles] == [(0, 3, "Times New Roman", 1.5)]
    message = convert.page_to_proto(view).blocks[0].translation_styles[0]
    assert (message.font, message.scale) == ("Times New Roman", 1.5)
    workspace.update_text_block(project, page, block.layer_id, {"translation": "One bold word", "translation_styles": styles})

    # Other edits keep the formatting; new text without styles is plain.
    view = workspace.update_text_block(project, page, block.layer_id, {"typography": {"size": 30}})
    assert len(view.blocks[0].translation_styles) == 2
    view = workspace.update_text_block(project, page, block.layer_id, {"translation": "Different words"})
    assert view.blocks[0].translation_styles == []


def test_new_layers_are_empty_paint_layers_placed_like_photoshop(workspace):
    project = workspace.create_project("new layers").id
    page = workspace.add_page(project, "p.png", page_png()).id

    first = workspace.add_empty_layer(project, page)
    [one] = first.raster_layers
    assert (one.name, one.kind) == ("Layer 1", "paint")
    assert workspace.asset(project, page, f"layer:{one.id}") is None  # no pixels until a stroke
    workspace.asset(project, page, "render")  # an empty layer renders as nothing

    top = workspace.add_empty_layer(project, page).raster_layers
    assert [r.name for r in top] == ["Layer 1", "Layer 2"]  # later draws on top
    between = workspace.add_empty_layer(project, page, "Shading", above=one.id).raster_layers
    assert [r.name for r in between] == ["Layer 1", "Shading", "Layer 2"]
    with pytest.raises(KeyError):
        workspace.add_empty_layer(project, page, above="missing")

    # Strokes go to the layer they name, even when it is not the newest paint layer.
    paint = Image.new("RGBA", (400, 300), (0, 0, 0, 0))
    paint.putpixel((3, 3), (0, 0, 0, 255))
    buffer = io.BytesIO()
    paint.save(buffer, format="PNG")
    view = workspace.set_raster_layer(project, page, "paint", one.id, buffer.getvalue())
    assert len(view.raster_layers) == 3 and f"layer:{one.id}" in view.asset_versions


def test_layer_saves_store_full_pngs_as_sent_and_paste_patches(workspace):
    project = workspace.create_project("patches").id
    page = workspace.add_page(project, "p.png", page_png()).id

    def png(image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    full = Image.new("RGBA", (400, 300), (0, 0, 0, 0))
    full.putpixel((10, 10), (255, 0, 0, 255))
    full.putpixel((200, 100), (0, 0, 255, 255))
    view = workspace.set_raster_layer(project, page, "paint", None, png(full))
    [layer] = [r.id for r in view.raster_layers]
    # An RGBA PNG of the page's size is stored byte for byte, not re-encoded.
    assert workspace.asset(project, page, f"layer:{layer}")[0] == png(full)

    # A patch replaces its rectangle, cleared pixels included (the eraser), and leaves the rest.
    patch = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    patch.putpixel((5, 5), (0, 255, 0, 255))
    workspace.set_raster_layer(project, page, "paint", layer, png(patch), at=(195, 95))
    stored = Image.open(io.BytesIO(workspace.asset(project, page, f"layer:{layer}")[0]))
    assert stored.getpixel((10, 10)) == (255, 0, 0, 255)
    assert stored.getpixel((200, 100)) == (0, 255, 0, 255)
    assert stored.getpixel((201, 101))[3] == 0
    with pytest.raises(ValueError):
        workspace.set_raster_layer(project, page, "paint", layer, png(patch), at=(390, 0))


def test_edits_mark_the_project_updated_without_writing_it_each_time(workspace, monkeypatch):
    from urakatsu.pipeline import projects

    project = workspace.create_project("touch").id
    page = workspace.add_page(project, "p.png", page_png()).id
    written = workspace.store._read_meta(project)["updated_at"]
    writes = []
    write_text = projects.files.write_text
    monkeypatch.setattr(projects.files, "write_text", lambda path, text: (writes.append(path.name), write_text(path, text)))
    for x in range(3):
        workspace.create_text_block(project, page, {"x": 10 + x, "y": 10, "width": 40, "height": 20})
    assert "project.json" not in writes  # the scene state is written per edit, project.json is not
    assert workspace.project_view(project).updated_at > written
    assert next(s for s in workspace.store.list() if s.id == project).updated_at > written
    monkeypatch.undo()
    workspace.store.flush()
    assert workspace.store._read_meta(project)["updated_at"] > written


def test_a_stroke_changes_only_its_layers_asset_version(workspace):
    project = workspace.create_project("versions").id
    page = workspace.add_page(project, "p.png", page_png()).id

    def stroke(x):
        paint = Image.new("RGBA", (400, 300), (0, 0, 0, 0))
        paint.putpixel((x, 5), (255, 0, 0, 255))
        buffer = io.BytesIO()
        paint.save(buffer, format="PNG")
        return workspace.set_raster_layer(project, page, "paint", None, buffer.getvalue())

    first = stroke(5)
    [layer] = [r.id for r in first.raster_layers]
    second = stroke(6)
    before, after = first.asset_versions, second.asset_versions
    # The client keys its asset cache on these: one stroke must refetch its layer and nothing else.
    assert after[f"layer:{layer}"] != before[f"layer:{layer}"]
    assert after["source"] == before["source"] and after["thumbnail"] == before["source"]
    assert after["render"] == str(second.revision) != before["render"]
    # Undo brings back the earlier pixels, and with them the earlier version.
    workspace.undo(project)
    assert workspace.page(project, page).asset_versions[f"layer:{layer}"] == before[f"layer:{layer}"]
    assert convert.page_to_proto(second).asset_versions[f"layer:{layer}"] == after[f"layer:{layer}"]


def test_chapter_export_packs_rendered_pages_in_chapter_order(workspace):
    import zipfile

    from urakatsu.pipeline.projects import DEFAULT_CHAPTER_ID

    project = workspace.create_project("My: Series").id
    first = workspace.add_page(project, "b?.jpg", page_png()).id
    workspace.add_page(project, "a.png", page_png(), index=0)
    data, name = workspace.export_chapter(project, DEFAULT_CHAPTER_ID, "cbz")
    assert name == "My_ Series - Chapter 1.cbz"
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        assert [e.filename for e in entries] == ["0001_a.png", "0002_b_.png"]
        assert all(e.compress_type == zipfile.ZIP_STORED for e in entries)
        assert archive.read(entries[1]) == workspace.render(project, first)
    with pytest.raises(ValueError):
        workspace.export_chapter(project, DEFAULT_CHAPTER_ID, "pdf")
    empty = workspace.create_chapter(project, "Empty").chapters[-1].id
    with pytest.raises(ValueError):
        workspace.export_chapter(project, empty, "zip")


def test_editing_a_generated_style_keeps_only_what_the_user_set():
    from urakatsu.scene import components as c
    from urakatsu.service.workspace import _typography

    generated = c.Typography(
        origin=c.generated(c.Generation(producer="detector")),
        color=(255, 255, 255, 255),
        writing_mode=c.WritingMode.VERTICAL,
    )
    recolored = _typography({"color": "#000000ff"}, generated)
    assert isinstance(recolored.origin, c.UserOrigin)
    assert recolored.color == (0, 0, 0, 255)
    assert recolored.writing_mode is None  # the detector's hint goes back to automatic
    assert _typography({"writing_mode": "Vertical"}, generated).writing_mode == c.WritingMode.VERTICAL
    authored = _typography({"writing_mode": "Vertical"}, None)
    assert _typography({"color": "#ff0000ff"}, authored).writing_mode == c.WritingMode.VERTICAL  # the user's own choice stays


@pytest.mark.asyncio
async def test_grpc_typography_field_mask_merges_only_named_properties(workspace):
    from urakatsu.scene import components as c

    servicer = UrakatsuServicer(workspace)
    project = workspace.create_project("p").id
    page = workspace.add_page(project, "1.png", page_png()).id
    layer = workspace.create_text_block(project, page, {"x": 1, "y": 1, "width": 40, "height": 20}).blocks[0].layer_id
    workspace.update_text_block(project, page, layer, {"typography": {"preferred_font": "Comic", "writing_mode": "Vertical"}})
    request = pb.UpdateTextBlockRequest(
        project_id=project,
        page_id=page,
        layer_id=layer,
        typography=pb.Typography(color="#123456ff", auto_fit=True),  # every other property blank
        fields=["typography", "typography.color"],
    )
    await servicer.UpdateTextBlock(request, None)
    style = workspace.session(project).snapshot().component(layer, c.Typography)
    assert style.color == (0x12, 0x34, 0x56, 255)
    assert style.preferred_font == "Comic" and style.writing_mode == c.WritingMode.VERTICAL


def test_frame_edit_clears_a_stored_angle_so_the_frame_sets_the_rotation(workspace):
    import math

    from urakatsu.renderer.bubble import geometry_frame
    from urakatsu.scene import components as c

    project = workspace.create_project("p").id
    page = workspace.add_page(project, "1.png", page_png()).id
    layer = workspace.create_text_block(project, page, {"x": 10, "y": 10, "width": 40, "height": 20}).blocks[0].layer_id
    session = workspace.session(project)
    edit = session.snapshot().edit()
    edit.set(layer, c.TextLayout(kind=c.TextLayoutKind.PARAGRAPH, angle_degrees=0.0))
    session.commit(edit.finish())
    # The box turned 30 degrees about its center (30, 20).
    turn = math.radians(30)
    corners = [(-20, -10), (20, -10), (20, 10), (-20, 10)]
    frame = [{"x": 30 + x * math.cos(turn) - y * math.sin(turn), "y": 20 + x * math.sin(turn) + y * math.cos(turn)} for x, y in corners]
    workspace.update_text_block(project, page, layer, {"frame": frame})
    snapshot = session.snapshot()
    layout = snapshot.component(layer, c.TextLayout)
    assert layout.angle_degrees is None and isinstance(layout.origin, c.UserOrigin)
    placed = geometry_frame(snapshot.component(layer, c.Geometry), layout.angle_degrees)
    assert round(placed.angle_degrees) == 30 and round(placed.bounds.width) == 40


def test_font_faces_are_served_for_the_canvas_editor(workspace):
    workspace._renderer = None  # the real renderer and its fonts, not the fixture's stand-in
    found = workspace.font_face("Arial")
    if found is None:  # pragma: no cover - depends on installed fonts
        pytest.skip("Arial is not installed")
    data, media = found
    assert media in ("font/ttf", "font/otf") and len(data) > 10_000
    bold = workspace.font_face("Arial", 700)
    assert bold is not None and bold[0] != data  # the bold face, not the regular one
    assert workspace.font_face("No Such Family Anywhere") is None


def test_text_gradients_round_trip_and_clear(workspace):
    project = workspace.create_project("gradients").id
    page = workspace.add_page(project, "p.png", page_png()).id
    workspace.create_text_block(project, page, {"x": 10, "y": 10, "width": 200, "height": 60})
    [block] = workspace.page(project, page).blocks
    sunset = {"kind": "linear", "angle": 45, "stops": [{"offset": 0, "color": "#ff512fff"}, {"offset": 1, "color": "#f09819ff"}]}
    ring = {"kind": "radial", "angle": 0, "stops": [{"offset": 0, "color": "#ffffffff"}, {"offset": 0.6, "color": "#000000ff"}]}
    view = workspace.update_text_block(project, page, block.layer_id, {"typography": {"fill_gradient": sunset, "stroke_gradient": ring, "stroke_width": 3}})
    typography = view.blocks[0].typography
    assert (typography.fill_gradient.kind, typography.fill_gradient.angle) == ("linear", 45)
    assert [s.color for s in typography.fill_gradient.stops] == ["#ff512fff", "#f09819ff"]
    assert typography.stroke_gradient.kind == "radial"
    message = convert.page_to_proto(view).blocks[0].typography
    assert message.fill_gradient.stops[1].color == "#f09819ff" and message.stroke_gradient.kind == "radial"
    assert convert.typography_from_proto(message)["fill_gradient"]["angle"] == 45

    # A gradient can also belong to a run of the translation.
    view = workspace.update_text_block(project, page, block.layer_id, {"translation": "hot words", "translation_styles": [{"start": 0, "end": 3, "gradient": sunset}]})
    [style] = view.blocks[0].translation_styles
    assert (style.start, style.end, style.gradient.angle) == (0, 3, 45)
    assert convert.page_to_proto(view).blocks[0].translation_styles[0].gradient.stops[0].color == "#ff512fff"

    # Another property keeps them; unset or a single stop clears one.
    view = workspace.update_text_block(project, page, block.layer_id, {"typography": {"size": 30, "auto_fit": False}})
    assert view.blocks[0].typography.fill_gradient is not None
    view = workspace.update_text_block(project, page, block.layer_id, {"typography": {"fill_gradient": None, "stroke_gradient": {"stops": [{"offset": 0, "color": "#000000ff"}]}}})
    assert view.blocks[0].typography.fill_gradient is None and view.blocks[0].typography.stroke_gradient is None


def test_text_layout_reports_the_rendered_size_and_line_starts(workspace):
    workspace._renderer = None  # the real renderer, not the fixture's stand-in
    project = workspace.create_project("layout").id
    page = workspace.add_page(project, "p.png", page_png()).id
    workspace.create_text_block(project, page, {"x": 20, "y": 20, "width": 120, "height": 200})
    [block] = workspace.page(project, page).blocks
    text = "A long enough sentence that it has to wrap onto several lines"
    workspace.update_text_block(project, page, block.layer_id, {"translation": text})
    layout = workspace.text_layout(project, page)[block.layer_id]
    starts = layout["line_starts"]
    assert layout["font_size"] > 5 and len(starts) >= 3 and starts[0] == 0
    assert starts == sorted(starts) and starts[-1] < len(text)  # breaks may fall inside a hyphenated word
    assert workspace.text_layout(project, page) is workspace.text_layout(project, page)  # from the render's cache


def test_a_lifted_block_over_the_page_under_it_is_the_render(workspace):
    workspace._renderer = None  # the real renderer, not the fixture's stand-in
    project = workspace.create_project("lift").id
    page = workspace.add_page(project, "p.png", page_png()).id
    workspace.create_text_block(project, page, {"x": 20, "y": 20, "width": 120, "height": 80})
    [block] = workspace.page(project, page).blocks
    workspace.update_text_block(project, page, block.layer_id, {"translation": "Lift me"})
    lift = workspace.lift(project, page, block.layer_id)

    def image(url):
        return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGBA")

    text, under = image(lift["text"]), image(lift["under"])
    assert text.size == under.size == (lift["width"], lift["height"])
    assert text.getchannel("A").getextrema()[1] > 0  # the text is on its own
    box = (lift["x"], lift["y"], lift["x"] + lift["width"], lift["y"] + lift["height"])
    render = Image.open(io.BytesIO(workspace.render(project, page))).convert("RGBA").crop(box)
    assert Image.alpha_composite(under, text).tobytes() == render.tobytes()
    with pytest.raises(KeyError):
        workspace.lift(project, page, "no-such-block")


def test_settings_accept_the_remote_runtime(workspace):
    config = workspace.get_config().model_copy(deep=True)
    config.runtimes = {"fake-detector": "remote", "fake-inpainter": "remote"}
    config.processor = {"fake-detector": {"remote_url": "http://worker.test", "remote_api_key": "t"}}
    saved = workspace.set_config(config)
    assert saved.runtimes["fake-detector"] == "remote"
    assert workspace.pipeline.provider(Stage.DETECTION).runtime_name == "remote"
    config.runtimes = {"fake-detector": "warp-drive"}
    with pytest.raises(ValueError, match="cannot run on 'warp-drive'"):
        workspace.set_config(config)
