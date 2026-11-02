"""The colorization stage, its gating in a run, and the generic diffusion colorizer."""

from __future__ import annotations

import asyncio
import base64
import io

import numpy as np
import pytest
from PIL import Image

from urakatsu.catalog.facts import ModelFacts
from urakatsu.catalog.recipes import Recipe, edit_recipe
from urakatsu.core.config import ColorizationConfig, PipelineConfig
from urakatsu.core.errors import ProviderUnavailable
from urakatsu.core.stage import Operation, Stage
from urakatsu.pipeline.images import ImageCache
from urakatsu.pipeline.pipeline import _colorization_stages
from urakatsu.pipeline.scope import Bounds
from urakatsu.pipeline.stages.base import StageInput
from urakatsu.pipeline.stages.colorization import LAYER_NAME, ColorizationStage, color_layer, is_colored, lettering_masks, paint_over_paper
from urakatsu.providers.colorization.diffusion_colorizer import PROMPT, ColorizeProcessor, DiffusionColorizerProvider, DiffusionColorizerSettings, fit_to_page, keep_ink
from urakatsu.scene import At, AssetInput, AssetMetadata, Page, Session, user
from urakatsu.scene import components as c
from urakatsu.service import convert

SIZE = (64, 48)  # width, height
PAINT = (200, 40, 90)


class _Colorizer:
    id = "fake-colorizer"

    def __init__(self):
        self.seen = []

    async def colorize(self, image):
        self.seen.append(image)
        return Image.new("RGB", image.size, PAINT)


class _Server:
    def __init__(self, reply=None):
        self.reply = reply
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return [self.reply or Image.new("RGB", (request["width"], request["height"]), (255, 0, 0))]


def _png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array, "RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _session(tmp_path, array: np.ndarray | None = None) -> tuple[Session, str]:
    session = Session(tmp_path / "scene")
    edit = session.snapshot().edit()
    page = edit.add_page(Page(label="1", width=float(SIZE[0]), height=float(SIZE[1])), At.END)
    pixels = array if array is not None else np.full((SIZE[1], SIZE[0], 3), 120, dtype=np.uint8)
    edit.set_asset(page, "source", AssetInput(data=_png(pixels), media_type="image/png", metadata=AssetMetadata(width=SIZE[0], height=SIZE[1])))
    session.commit(edit.finish())
    return session, page


def _input(session: Session, page: str, **kwargs) -> StageInput:
    return StageInput(session.snapshot(), page, None, kwargs.pop("region", None), ImageCache(), **kwargs)


def _run(session: Session, page: str, provider=None, **kwargs) -> _Colorizer:
    provider = provider or _Colorizer()
    session.commit(asyncio.run(ColorizationStage().process(_input(session, page, **kwargs), provider)))
    return provider


def _layer_pixels(session: Session, page: str) -> np.ndarray:
    scene = session.snapshot()
    return np.array(ImageCache().get(scene, color_layer(scene, page), "source").convert("RGBA"))


def test_the_stage_runs_on_the_cleaned_page_and_before_sound_effects():
    order = Stage.all()
    assert order.index(Stage.DECENSOR) < order.index(Stage.COLORIZATION) < order.index(Stage.SOUND_EFFECTS)
    assert Operation.through(Stage.COLORIZATION).resolve() == [Stage.DETECTION, Stage.INPAINTING, Stage.COLORIZATION]


def test_a_run_leaves_the_stage_out_while_colorization_is_off():
    assert Stage.COLORIZATION not in _colorization_stages(Stage.all(), "off")
    assert Stage.COLORIZATION in _colorization_stages(Stage.all(), "colorize")
    with pytest.raises(ValueError, match="colorization is off"):
        _colorization_stages([Stage.COLORIZATION], "off")


def test_the_model_is_configured_like_any_other_stage():
    config = PipelineConfig()
    assert config.colorization.mode == "off"
    assert config.provider_for(Stage.COLORIZATION) == DiffusionColorizerProvider.id
    config.set_provider(Stage.COLORIZATION, "other")
    assert config.colorization.model == "other"


def test_the_config_message_keeps_the_mode_and_refuses_an_unknown_one():
    config = PipelineConfig(colorization=ColorizationConfig(mode="colorize", model="other"))
    message = convert.config_to_proto(config)
    assert (message.colorization, message.colorization_model) == ("colorize", "other")
    assert convert.config_from_proto(message, PipelineConfig()).colorization == config.colorization
    message.colorization = "repaint"
    with pytest.raises(ValueError, match="unknown colorization mode"):
        convert.config_from_proto(message, PipelineConfig())


def test_is_colored_tells_a_color_page_from_a_scan_on_tinted_paper():
    assert is_colored(Image.new("RGB", (32, 32), PAINT))
    assert not is_colored(Image.new("RGB", (32, 32), (128, 128, 128)))
    assert not is_colored(Image.new("RGB", (32, 32), (240, 228, 205)))  # yellowed paper


def test_the_result_is_a_layer_of_its_own(tmp_path):
    session, page = _session(tmp_path)
    _run(session, page)
    scene = session.snapshot()
    layer = color_layer(scene, page)
    component = scene.component(layer, c.RasterLayer)
    assert component.name == LAYER_NAME and component.kind is c.RasterLayerKind.COLOR
    assert scene.user_authored(layer) is False
    assert tuple(_layer_pixels(session, page)[0, 0]) == (*PAINT, 255)


def test_the_model_sees_the_cleanup_layer_and_the_layer_goes_over_it(tmp_path):
    session, page = _session(tmp_path)
    edit = session.snapshot().edit()
    cleanup = edit.add_entity(page, At.START)
    edit.set(cleanup, c.RasterLayer(origin=user(), name="Cleanup", kind=c.RasterLayerKind.CLEANUP))
    white = io.BytesIO()
    Image.new("RGBA", SIZE, (255, 255, 255, 255)).save(white, format="PNG")
    edit.set_asset(cleanup, "source", AssetInput(data=white.getvalue(), media_type="image/png", metadata=AssetMetadata(width=SIZE[0], height=SIZE[1])))
    session.commit(edit.finish())

    provider = _run(session, page)

    assert provider.seen[0].getpixel((0, 0)) == (255, 255, 255)
    scene = session.snapshot()
    rasters = scene.raster_layers(page)
    assert rasters.index(cleanup) < rasters.index(color_layer(scene, page))


def test_skips_color_pages_and_done_pages_until_forced(tmp_path):
    stage = ColorizationStage()
    session, page = _session(tmp_path, np.full((SIZE[1], SIZE[0], 3), PAINT, dtype=np.uint8))
    assert stage.skip(_input(session, page), _Colorizer())
    assert not stage.done(_input(session, page))  # nothing to do is not the same as done
    assert not stage.skip(_input(session, page, force=True), _Colorizer())

    session, page = _session(tmp_path / "gray")
    assert not stage.skip(_input(session, page), _Colorizer())
    _run(session, page)
    assert stage.done(_input(session, page))
    assert stage.skip(_input(session, page), _Colorizer())
    assert not stage.skip(_input(session, page, force=True), _Colorizer())


def test_redo_replaces_the_layer_but_not_one_the_user_edited(tmp_path):
    session, page = _session(tmp_path)
    _run(session, page)
    layer = color_layer(session.snapshot(), page)
    _run(session, page, force=True)
    assert color_layer(session.snapshot(), page) == layer

    edit = session.snapshot().edit()
    component = session.snapshot().component(layer, c.RasterLayer)
    component.origin = user()
    edit.set(layer, component)
    session.commit(edit.finish())
    assert ColorizationStage().skip(_input(session, page, force=True), _Colorizer())


def test_a_regional_run_replaces_only_the_region(tmp_path):
    session, page = _session(tmp_path)
    _run(session, page, region=Bounds(x=0, y=0, width=16, height=16))
    pixels = _layer_pixels(session, page)
    assert pixels[0, 0, 3] == 255 and pixels[40, 60, 3] == 0

    class Blue(_Colorizer):
        async def colorize(self, image):
            return Image.new("RGB", image.size, (0, 0, 255))

    _run(session, page, Blue(), region=Bounds(x=32, y=0, width=16, height=16))
    pixels = _layer_pixels(session, page)
    assert tuple(pixels[0, 0]) == (*PAINT, 255) and tuple(pixels[0, 40]) == (0, 0, 255, 255)


def test_the_layer_leaves_detected_bubbles_as_cleaned(tmp_path):
    session, page = _session(tmp_path)
    edit = session.snapshot().edit()
    edit.add_analysis_region(page, At.END, c.RegionKinds.BUBBLE, c.Geometry.rectangle(8, 8, 16, 16))
    session.commit(edit.finish())
    _run(session, page)
    pixels = _layer_pixels(session, page)
    assert pixels[12, 12, 3] == 0  # the model's lettering would be here
    assert tuple(pixels[40, 40]) == (*PAINT, 255)


def test_invented_lettering_on_erased_paper_is_painted_over():
    image = Image.new("RGB", (64, 48), (255, 255, 255))
    color = np.full((48, 64, 3), PAINT, dtype=np.uint8)
    color[20:28, 20:44] = 0  # lettering the model made up
    box = np.zeros((48, 64), dtype=np.uint8)
    box[16:32, 16:48] = 255
    pixels = np.asarray(paint_over_paper(image, Image.fromarray(color, "RGB"), box))
    assert np.abs(pixels[24, 32].astype(int) - PAINT).max() < 12


def test_art_in_a_text_box_is_not_painted_over():
    page = np.full((48, 64, 3), 255, dtype=np.uint8)
    page[20:28, 20:44] = 0  # a line of the page's own art
    color = np.full((48, 64, 3), PAINT, dtype=np.uint8)
    color[20:28, 20:44] = 0
    box = np.full((48, 64), 255, dtype=np.uint8)
    pixels = np.asarray(paint_over_paper(Image.fromarray(page, "RGB"), Image.fromarray(color, "RGB"), box))
    assert tuple(pixels[24, 32]) == (0, 0, 0)


def test_free_text_is_found_but_text_in_a_bubble_is_left_to_the_bubble(tmp_path):
    session, page = _session(tmp_path)
    edit = session.snapshot().edit()
    bubble = edit.add_analysis_region(page, At.END, c.RegionKinds.BUBBLE, c.Geometry.rectangle(0, 0, 20, 20))
    inside = edit.add_analysis_region(page, At.END, c.RegionKinds.TEXT, c.Geometry.rectangle(4, 4, 8, 8))
    edit.relate(c.Relations.INSIDE, inside, bubble)
    edit.add_analysis_region(page, At.END, c.RegionKinds.TEXT, c.Geometry.rectangle(44, 30, 8, 8))
    session.commit(edit.finish())
    bubbles, free_text = lettering_masks(session.snapshot(), page, SIZE)
    assert bubbles[10, 10] and not bubbles[40, 40]
    assert free_text[34, 48] and not free_text[8, 8]


def _lines(shift: int = 0, color: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    random = np.random.default_rng(1)
    pixels = np.full((256, 192, 3), 255, dtype=np.uint8)
    for _ in range(40):
        top, left = random.integers(8, 232), random.integers(8, 170)
        pixels[top + shift : top + shift + 3, left : left + 14] = color
        pixels[top + shift : top + shift + 16, left : left + 3] = color
    return Image.fromarray(pixels, "RGB")


def test_the_drawing_is_fitted_back_onto_the_lines_of_the_page():
    page = _lines()
    drifted = _lines(shift=4)
    gray = np.asarray(page.convert("L"), dtype=np.float32)
    before = np.abs(np.asarray(drifted.convert("L"), dtype=np.float32) - gray).mean()
    after = np.abs(np.asarray(fit_to_page(page, drifted, page.size).convert("L"), dtype=np.float32) - gray).mean()
    assert after < before / 2


def test_the_fit_works_at_the_size_of_the_page():
    page = _lines()
    assert fit_to_page(page, _lines(shift=2), (960, 1280)).size == (960, 1280)


def test_a_size_mismatch_is_refused(tmp_path):
    class Wrong(_Colorizer):
        async def colorize(self, image):
            return Image.new("RGB", (8, 8), PAINT)

    session, page = _session(tmp_path)
    with pytest.raises(ValueError, match="dimensions"):
        asyncio.run(ColorizationStage().process(_input(session, page), Wrong()))


def _decoded(data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data)))


def test_the_request_sends_the_page_then_the_reference_with_the_lora(tmp_path):
    settings = DiffusionColorizerSettings(prompt="trigger", lora_strength=0.8, megapixels=0.25)
    server = _Server()
    reference = Image.new("RGB", (32, 32), PAINT)
    ColorizeProcessor(server, {"steps": 20, "cfg_scale": 5}, settings, tmp_path / "color.safetensors").inference(Image.new("RGB", (600, 900), (255, 255, 255)), reference)

    request = server.requests[0]
    assert request["prompt"] == "trigger"
    assert request["lora"] == [{"path": "color.safetensors", "multiplier": 0.8}]
    assert request["sample_params"]["sample_steps"] == 20
    assert request["sample_params"]["guidance"]["txt_cfg"] == 5
    assert [_decoded(image).size for image in request["ref_images"]] == [(request["width"], request["height"]), (32, 32)]
    assert "init_image" not in request and "mask_image" not in request


def test_the_request_runs_without_a_lora_or_a_reference():
    server = _Server()
    ColorizeProcessor(server, {}, DiffusionColorizerSettings(keep_ink=False)).inference(Image.new("RGB", (64, 64), (255, 255, 255)))
    request = server.requests[0]
    assert request["prompt"] == PROMPT and "lora" not in request and len(request["ref_images"]) == 1


def test_the_result_comes_back_at_the_size_of_the_page():
    result = ColorizeProcessor(_Server(), {}, DiffusionColorizerSettings(megapixels=0.25)).inference(Image.new("RGB", (1600, 2300), (255, 255, 255)))
    assert result.size == (1600, 2300)


def test_keep_ink_restores_solid_ink_and_leaves_tone_to_the_model():
    page = np.array([[[0, 0, 0], [200, 200, 200], [128, 128, 128]]], dtype=np.uint8)
    color = np.full((1, 3, 3), PAINT, dtype=np.uint8)
    pixels = np.asarray(keep_ink(Image.fromarray(page, "RGB"), Image.fromarray(color, "RGB")))
    assert tuple(pixels[0, 0]) == (0, 0, 0)  # ink
    assert tuple(pixels[0, 1]) == PAINT  # paper
    assert tuple(pixels[0, 2]) == PAINT  # screentone takes the color


def _recipe(id: str, url: str = "https://huggingface.co/a/model", features=(), disabled=()) -> Recipe:
    return Recipe(id=id, name=id, model_url=url, features=frozenset(features), disabled=frozenset(disabled))


def test_a_plain_recipe_finds_the_editing_recipe_of_its_weights():
    plain, editing = _recipe("anima"), _recipe("anima-edit", features=["ref_images"])
    other = _recipe("other", "https://huggingface.co/b/other", ["ref_images"])
    assert edit_recipe([plain, editing, other], plain) is editing
    assert edit_recipe([plain, editing], editing) is editing
    assert edit_recipe([plain, other], plain) is None
    assert edit_recipe([_recipe("x", "", ["ref_images"])], _recipe("y", "")) is None


def test_the_provider_accepts_any_model_that_edits_from_references():
    def facts(editor, runtime="diffusion"):
        return ModelFacts(runtime=runtime, repo="a/model", edit_recipe=editor)

    assert DiffusionColorizerProvider.accepts(facts(_recipe("edit", features=["ref_images"]))) == "diffusion"
    assert DiffusionColorizerProvider.accepts(facts(None)) is None
    assert DiffusionColorizerProvider.accepts(facts(_recipe("edit", features=["ref_images"]), "llama")) is None
    # Reference-only recipes that are not image editors (PiD turns ordinary image-to-image off).
    assert DiffusionColorizerProvider.accepts(facts(_recipe("pid", features=["ref_images"], disabled=["img2img"]))) is None


def test_the_provider_names_no_model_of_its_own():
    assert DiffusionColorizerProvider.weight_sources() == ()
    assert DiffusionColorizerProvider.stage is Stage.COLORIZATION


def test_the_provider_can_run_on_a_worker_and_sends_its_lora_there(tmp_path):
    from urakatsu.remote import CLOUD, REMOTE

    assert DiffusionColorizerProvider.runtimes_declared() == ("diffusion", REMOTE, CLOUD)
    lora = tmp_path / "color.safetensors"
    lora.write_bytes(b"x")
    assert DiffusionColorizerProvider({"lora": str(lora)}).remote_files() == {"lora": lora}
    assert DiffusionColorizerProvider({"lora": str(lora), "lora_strength": 0}).remote_files() == {}


def test_a_missing_lora_or_reference_is_reported(tmp_path):
    provider = DiffusionColorizerProvider({"lora": str(tmp_path / "gone.safetensors")})
    with pytest.raises(ProviderUnavailable, match="LoRA"):
        provider._lora()
    provider = DiffusionColorizerProvider({"reference": str(tmp_path / "gone.png")})
    with pytest.raises(ProviderUnavailable, match="color reference"):
        provider.reference_image()


def test_the_server_finds_the_lora_folder(tmp_path):
    lora = tmp_path / "color.safetensors"
    lora.write_bytes(b"x")
    assert DiffusionColorizerProvider({"lora": str(lora), "lora_strength": 0})._lora() is None
    provider = DiffusionColorizerProvider({"lora": str(lora)})

    class Device:
        backend = None

    args = provider._server_args(Device())
    assert args[args.index("--lora-model-dir") + 1] == str(tmp_path)
