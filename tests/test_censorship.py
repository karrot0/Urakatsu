"""The censorship stages, their gating in a run, the checkpoint loader and the LoRA request."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import pickle
import struct
import tarfile

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from fakes import fake_config, fake_processors, fake_registry, page_png
from urakatsu.core import ProgressKind
from urakatsu.core.config import CensorshipConfig, PipelineConfig
from urakatsu.core.errors import ModelLoadError, ProviderUnavailable
from urakatsu.core.stage import Operation, Stage
from urakatsu.pipeline import PagesScope, Pipeline, PipelineError, ProjectStore, Request, RunStatus
from urakatsu.pipeline.images import ImageCache
from urakatsu.pipeline.pipeline import _censorship_stages
from urakatsu.pipeline.stages.base import InpaintingMask, StageInput
from urakatsu.pipeline.stages.censorship import CENSOR_MASK, LAYER_NAME, CensorDetectionStage, DecensorStage, bar_groups, decensor_layer, is_white
from urakatsu.providers.base import CensorDetectionProvider, DecensorProvider
from urakatsu.providers.censor import bars, er0mangaseg, flux2_klein_decensor
from urakatsu.providers.censor.er0manga_net import Er0mangaSegNet
from urakatsu.providers.censor.er0mangaseg import Er0mangaSegProvider
from urakatsu.providers.censor.flux2_klein_decensor import Flux2KleinDecensorProvider, Flux2KleinDecensorSettings, check_lora
from urakatsu.providers.inpainting.flux2_klein import Flux2KleinProcessor
from urakatsu.runtimes import RuntimeManager, models_folder
from urakatsu.runtimes.store import ContainerLayerFile, ModelStore
from urakatsu.scene import At, AssetInput, AssetMetadata, Page, Session, user
from urakatsu.scene import components as c

SIZE = (64, 48)  # width, height


class _Detector:
    id = "fake-detector"

    def __init__(self, mask: np.ndarray):
        self.mask = mask

    async def detect_censorship(self, image):
        return self.mask


class _Reconstructor:
    id = "fake-decensor"

    def __init__(self, color=(10, 200, 30)):
        self.color = color
        self.calls = []

    def _reconstruct(self, image, mask, white):
        self.calls.append((image.size, white))
        return Image.new("RGB", image.size, self.color)


def _png(array: np.ndarray, mode: str) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array, mode).save(buffer, format="PNG")
    return buffer.getvalue()


def _session(tmp_path, array: np.ndarray | None = None) -> tuple[Session, str]:
    session = Session(tmp_path / "scene")
    edit = session.snapshot().edit()
    page = edit.add_page(Page(label="1", width=float(SIZE[0]), height=float(SIZE[1])), At.END)
    pixels = array if array is not None else np.full((SIZE[1], SIZE[0], 3), 120, dtype=np.uint8)
    edit.set_asset(page, "source", AssetInput(data=_png(pixels, "RGB"), media_type="image/png", metadata=AssetMetadata(width=SIZE[0], height=SIZE[1])))
    session.commit(edit.finish())
    return session, page


def _input(session: Session, page: str, **kwargs) -> StageInput:
    return StageInput(session.snapshot(), page, None, None, ImageCache(), **kwargs)


def _bar(value: int = 255) -> np.ndarray:
    mask = np.zeros(SIZE[::-1], dtype=np.uint8)
    mask[10:20, 10:40] = value
    return mask


def _with_mask(tmp_path, mask: np.ndarray | None = None) -> tuple[Session, str]:
    session, page = _session(tmp_path)
    patch = asyncio.run(CensorDetectionStage().process(_input(session, page), _Detector(_bar() if mask is None else mask)))
    session.commit(patch)
    return session, page


def test_stage_order_runs_detection_of_bars_first_and_reconstruction_after_cleanup():
    order = Stage.all()
    assert order.index(Stage.CENSOR_DETECTION) < order.index(Stage.DETECTION)
    assert order.index(Stage.INPAINTING) < order.index(Stage.DECENSOR) < order.index(Stage.SOUND_EFFECTS)
    assert Operation.through(Stage.DECENSOR).resolve() == [Stage.CENSOR_DETECTION, Stage.DECENSOR]


def test_a_full_run_leaves_the_stages_out_until_they_are_enabled():
    full = Operation.full()
    assert not {Stage.CENSOR_DETECTION, Stage.DECENSOR} & set(_censorship_stages(Stage.all(), "off", full))
    review = _censorship_stages(Stage.all(), "detect", full)
    assert Stage.CENSOR_DETECTION in review and Stage.DECENSOR not in review
    assert {Stage.CENSOR_DETECTION, Stage.DECENSOR} <= set(_censorship_stages(Stage.all(), "remove", full))


def test_review_mode_reconstructs_only_when_asked_for_and_off_refuses():
    only = Operation.only(Stage.DECENSOR)
    assert _censorship_stages(only.resolve(), "detect", only) == [Stage.DECENSOR]
    with pytest.raises(ValueError, match="censorship is off"):
        _censorship_stages(only.resolve(), "off", only)


@pytest.mark.parametrize(
    "operation, expected",
    [
        (Operation.select([Stage.DECENSOR]), [Stage.DECENSOR]),  # redoing the step from the Activity panel
        (Operation.select([Stage.CENSOR_DETECTION, Stage.DECENSOR]), [Stage.CENSOR_DETECTION, Stage.DECENSOR]),
        (Operation.through(Stage.DECENSOR), [Stage.CENSOR_DETECTION, Stage.DECENSOR]),
    ],
)
def test_review_mode_honours_every_way_of_asking_for_the_stage(operation, expected):
    assert _censorship_stages(operation.resolve(), "detect", operation) == expected


@pytest.mark.parametrize("operation", [Operation.full(), Operation.through(Stage.SOUND_EFFECTS)])
def test_review_mode_leaves_the_stage_out_of_a_run_that_did_not_name_it(operation):
    assert Stage.DECENSOR not in _censorship_stages(operation.resolve(), "detect", operation)


def test_default_config_keeps_the_feature_off():
    config = PipelineConfig()
    assert config.censorship.mode == "off"
    assert config.provider_for(Stage.CENSOR_DETECTION) == "er0mangaseg"
    assert config.provider_for(Stage.DECENSOR) == "flux2-klein-decensor"


def test_detection_saves_the_mask_and_skips_a_page_that_has_one(tmp_path):
    session, page = _with_mask(tmp_path)
    stage = CensorDetectionStage()
    saved = _input(session, page).images.get(session.snapshot(), page, CENSOR_MASK)
    assert (np.asarray(saved.convert("L")) == _bar()).all()
    assert stage.skip(_input(session, page), None)
    assert not stage.skip(_input(session, page, force=True), None)


def test_decensor_has_nothing_to_do_without_a_mask_or_with_an_empty_one(tmp_path):
    session, page = _session(tmp_path / "none")
    assert DecensorStage().skip(_input(session, page), None)
    session, page = _with_mask(tmp_path / "empty", np.zeros(SIZE[::-1], dtype=np.uint8))
    assert DecensorStage().skip(_input(session, page), None)


def test_decensor_paints_only_the_mask_on_its_own_layer_above_cleanup(tmp_path):
    session, page = _with_mask(tmp_path)
    edit = session.snapshot().edit()
    cleanup = edit.add_entity(page, At.START)
    edit.set(cleanup, c.RasterLayer(origin=user(), name="Cleanup", kind=c.RasterLayerKind.CLEANUP))
    session.commit(edit.finish())

    provider = _Reconstructor()
    session.commit(asyncio.run(DecensorStage().process(_input(session, page), provider)))
    snapshot = session.snapshot()
    layer = decensor_layer(snapshot, page)
    assert snapshot.component(layer, c.RasterLayer).name == LAYER_NAME
    layers = snapshot.raster_layers(page)
    assert layers.index(cleanup) < layers.index(layer)
    overlay = np.asarray(_input(session, page).images.get(snapshot, layer, "source").convert("RGBA"))
    assert (overlay[_bar() > 0, :3] == (10, 200, 30)).all()
    assert (overlay[_bar() == 0, 3] == 0).all()
    assert provider.calls and provider.calls[0][1] is False


def test_soft_mask_edges_become_alpha_so_they_blend(tmp_path):
    mask = _bar()
    mask[9, 10:40] = 100
    session, page = _with_mask(tmp_path, mask)
    session.commit(asyncio.run(DecensorStage().process(_input(session, page), _Reconstructor())))
    snapshot = session.snapshot()
    overlay = np.asarray(_input(session, page).images.get(snapshot, decensor_layer(snapshot, page), "source").convert("RGBA"))
    assert (overlay[9, 10:40, 3] == 100).all() and (overlay[9, 10:40, :3] == (10, 200, 30)).all()


def test_redo_replaces_the_generated_layer_but_never_one_the_user_edited(tmp_path):
    session, page = _with_mask(tmp_path)
    stage = DecensorStage()
    session.commit(asyncio.run(stage.process(_input(session, page), _Reconstructor())))
    assert stage.skip(_input(session, page), None)
    assert stage.done(_input(session, page))
    assert not stage.skip(_input(session, page, force=True), None)

    session.commit(asyncio.run(stage.process(_input(session, page, force=True), _Reconstructor((1, 2, 3)))))
    snapshot = session.snapshot()
    assert len(snapshot.raster_layers(page, c.RasterLayerKind.DECENSOR)) == 1
    overlay = np.asarray(_input(session, page).images.get(snapshot, decensor_layer(snapshot, page), "source").convert("RGBA"))
    assert (overlay[12, 12, :3] == (1, 2, 3)).all()

    stroke = np.zeros(SIZE[::-1], dtype=np.uint8)
    stroke[30:36, 10:20] = 255
    manual = InpaintingMask(page=page, png=_png(stroke, "L"))
    session.commit(asyncio.run(stage.process(_input(session, page, inpainting_mask=manual), _Reconstructor((9, 9, 9)))))
    assert stage.skip(_input(session, page, force=True), None)
    snapshot = session.snapshot()
    overlay = np.asarray(_input(session, page).images.get(snapshot, decensor_layer(snapshot, page), "source").convert("RGBA"))
    assert (overlay[32, 12, :3] == (9, 9, 9)).all() and (overlay[12, 12, :3] == (1, 2, 3)).all()


def test_a_stroke_runs_without_a_saved_mask(tmp_path):
    session, page = _session(tmp_path)
    stroke = InpaintingMask(page=page, png=_png(_bar(), "L"))
    stage = DecensorStage()
    assert not stage.skip(_input(session, page, inpainting_mask=stroke), None)
    session.commit(asyncio.run(stage.process(_input(session, page, inpainting_mask=stroke), _Reconstructor())))
    snapshot = session.snapshot()
    assert c.is_user(snapshot.component(decensor_layer(snapshot, page), c.RasterLayer).origin)


def test_nearby_bars_share_a_crop_and_distant_ones_do_not():
    mask = np.zeros((400, 400), dtype=np.uint8)
    mask[10:14, 10:100] = 255
    mask[24:28, 10:100] = 255  # next to the first
    mask[300:304, 10:100] = 255
    groups = bar_groups(mask)
    assert len(groups) == 2
    assert groups[0][12, 50] == 255 and groups[0][26, 50] == 255 and groups[0][302, 50] == 0
    assert groups[1][302, 50] == 255


def test_group_colour_picks_the_white_or_black_instruction():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    group = np.zeros((20, 20), dtype=np.uint8)
    group[5:10, 5:10] = 255
    assert not is_white(image, group)
    image[5:10, 5:10] = 250
    assert is_white(image, group)


class _Hostile:
    """A pickle that would run a callable if it were loaded the ordinary way."""

    def __reduce__(self):
        return (pytest.fail, ("the checkpoint ran code",))


def test_the_checkpoint_loader_keeps_tensors_and_runs_nothing(tmp_path):
    path = tmp_path / "checkpoint.pth"
    weight = torch.arange(6.0).reshape(2, 3)
    torch.save({"meta": {"hook": _Hostile(), "other": object()}, "state_dict": {"backbone.a": weight, "decode_head.b": weight + 1, "auxiliary_head.c": weight}}, path)
    state = er0mangaseg.load_state(path)
    assert set(state) == {"backbone.a", "decode_head.b"}
    assert torch.equal(state["decode_head.b"], weight + 1)


def test_the_restricted_unpickler_refuses_to_resolve_anything_else():
    unpickler = er0mangaseg._Unpickler(io.BytesIO(pickle.dumps(1)))
    assert unpickler.find_class("os", "system") is er0mangaseg._Inert
    assert unpickler.find_class("builtins", "eval") is er0mangaseg._Inert


def test_network_names_follow_the_checkpoint_and_its_output_is_a_quarter_size():
    net = Er0mangaSegNet().eval()
    names = set(net.state_dict())
    for name in (
        "backbone.downsample_layers.0.0.weight",
        "backbone.downsample_layers.0.1.weight",
        "backbone.downsample_layers.3.1.weight",
        "backbone.stages.2.8.depthwise_conv.weight",
        "backbone.stages.2.8.gamma",
        "backbone.norm3.weight",
        "decode_head.psp_modules.3.1.conv.weight",
        "decode_head.psp_modules.3.1.bn.num_batches_tracked",
        "decode_head.bottleneck.bn.running_mean",
        "decode_head.lateral_convs.2.conv.weight",
        "decode_head.fpn_convs.2.conv.weight",
        "decode_head.fpn_bottleneck.conv.weight",
        "decode_head.conv_seg.bias",
    ):
        assert name in names
    with torch.inference_mode():
        assert net(torch.zeros(1, 3, 64, 64)).shape == (1, 2, 16, 16)


def test_grow_widens_and_blur_softens_the_mask():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[20, 20] = 255
    grown = er0mangaseg.grow_and_blur(mask, 3, 0)
    assert grown[17:24, 17:24].all() and not grown[16, 20]
    soft = er0mangaseg.grow_and_blur(mask, 3, 2.0)
    assert 0 < soft[15, 20] < 255
    assert er0mangaseg.grow_and_blur(np.zeros((4, 4), dtype=np.uint8), 3, 2.0).sum() == 0


def _lora(path, metadata):
    header = json.dumps({"__metadata__": metadata}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header)
    return path


def test_a_lora_for_another_klein_size_is_refused(tmp_path):
    path = _lora(tmp_path / "bars.safetensors", {"modelspec.architecture": "flux2_klein_9b"})
    with pytest.raises(ProviderUnavailable, match="9B"):
        check_lora(path, "unsloth/FLUX.2-klein-4B-GGUF:Q4_K_M")
    check_lora(path, "unsloth/FLUX.2-klein-9B-GGUF:Q8_0")
    check_lora(_lora(tmp_path / "plain.safetensors", {}), "unsloth/FLUX.2-klein-4B-GGUF:Q4_K_M")


def test_decensor_defaults_to_klein_9b_rather_than_the_text_cleanup_4b():
    provider = Flux2KleinDecensorProvider(Flux2KleinDecensorSettings())
    _, recipe, components = provider._components()
    assert recipe == "flux2-klein-9b"
    assert dict(components)["diffusion_model"].filename == "flux-2-klein-9b-Q8_0.gguf"
    assert dict(components)["llm"].filename == "Qwen3-8B-Q4_K_M.gguf"
    assert {role for role, _ in components} == {"diffusion_model", "llm", "vae"}


def _decensor(tmp_path, monkeypatch, **settings) -> Flux2KleinDecensorProvider:
    pinned = tmp_path / "pinned.safetensors"
    pinned.write_bytes(b"lora")
    monkeypatch.setattr(flux2_klein_decensor, "LORA_SHA256", hashlib.sha256(b"lora").hexdigest())
    provider = Flux2KleinDecensorProvider(settings, RuntimeManager("cpu"), "diffusion")
    monkeypatch.setattr(provider, "resolve", lambda source: pinned)
    return provider


@pytest.mark.parametrize("settings", [{}, {"model": "unsloth/FLUX.2-klein-9B-GGUF:Q8_0"}])
def test_klein_9b_gets_the_pinned_lora_without_being_asked(tmp_path, monkeypatch, settings):
    assert _decensor(tmp_path, monkeypatch, **settings)._lora().name == "pinned.safetensors"


def test_the_pinned_lora_stays_off_other_models_and_at_zero_strength(tmp_path, monkeypatch):
    assert _decensor(tmp_path, monkeypatch, model="unsloth/FLUX.2-klein-4B-GGUF:Q4_K_M")._lora() is None
    assert _decensor(tmp_path, monkeypatch, model="unsloth/FLUX.2-klein-9B-GGUF:Q8_0", lora_strength=0)._lora() is None


def test_a_chosen_lora_replaces_the_pinned_one(tmp_path, monkeypatch):
    chosen = tmp_path / "chosen.safetensors"
    chosen.write_bytes(b"other")
    provider = _decensor(tmp_path, monkeypatch, model="unsloth/FLUX.2-klein-9B-GGUF:Q8_0", lora=str(chosen))
    assert provider._lora() == chosen


def test_a_pinned_lora_with_the_wrong_hash_is_refused(tmp_path, monkeypatch):
    provider = _decensor(tmp_path, monkeypatch, model="unsloth/FLUX.2-klein-9B-GGUF:Q8_0")
    monkeypatch.setattr(flux2_klein_decensor, "LORA_SHA256", "0" * 64)
    with pytest.raises(ProviderUnavailable, match="not the LoRA"):
        provider._lora()


KLEIN = "FLUX.2 Klein 9B (about 15.6 GB, under a non-commercial license)"
BAR_LORA = "the bar LoRA (166 MB)"


def _labels(provider, settings, *cached) -> list[str]:
    return [item["label"] for item in provider.pending_downloads(settings, lambda source: source in cached)]


def test_nothing_is_pending_once_it_is_all_on_disk():
    sources = (*Flux2KleinDecensorProvider.weight_sources(), flux2_klein_decensor.LORA)
    assert _labels(Flux2KleinDecensorProvider, {}, *sources) == []
    assert _labels(Er0mangaSegProvider, {}, er0mangaseg.WEIGHTS) == []


def test_a_fresh_install_is_told_about_each_download():
    assert _labels(Flux2KleinDecensorProvider, {}) == [KLEIN, BAR_LORA]
    assert _labels(Er0mangaSegProvider, {}) == ["the bar detector (about 670 MB)"]
    assert [item["model"] for item in Flux2KleinDecensorProvider.pending_downloads({}, lambda source: False)] == [True, False]


def test_what_the_settings_replace_is_not_pending():
    chosen = {"model": "unsloth/FLUX.2-klein-9B-GGUF:Q2_K"}
    assert _labels(Flux2KleinDecensorProvider, chosen) == [BAR_LORA]
    assert _labels(Flux2KleinDecensorProvider, {"model": "unsloth/FLUX.2-klein-4B-GGUF:Q4_K_M"}) == []
    assert _labels(Flux2KleinDecensorProvider, {**chosen, "lora": "D:/mine.safetensors"}) == []
    assert _labels(Flux2KleinDecensorProvider, {**chosen, "lora_strength": 0}) == []
    assert _labels(Er0mangaSegProvider, {"weights": "D:/mine.pth"}) == []


def test_decensor_leaves_this_computer_only_for_a_worker_the_user_picks():
    from urakatsu.remote import CLOUD, REMOTE

    assert Flux2KleinDecensorProvider.runtimes_declared() == ("diffusion", REMOTE, CLOUD)
    assert Flux2KleinDecensorProvider.default_runtime() == "diffusion"  # a worker is only ever picked on purpose
    assert Flux2KleinDecensorProvider.stage is Stage.DECENSOR
    assert "server_url" not in Flux2KleinDecensorSettings.model_fields  # no attached server outside the worker path
    assert not Flux2KleinDecensorProvider(Flux2KleinDecensorSettings()).attached()


def test_only_a_lora_the_user_picked_is_sent_to_a_worker(tmp_path):
    assert Flux2KleinDecensorProvider(Flux2KleinDecensorSettings()).remote_files() == {}  # the worker downloads the pinned one
    lora = tmp_path / "bars.safetensors"
    lora.write_bytes(b"x")
    assert Flux2KleinDecensorProvider(Flux2KleinDecensorSettings(lora=str(lora))).remote_files() == {"lora": lora}
    assert Flux2KleinDecensorProvider(Flux2KleinDecensorSettings(lora=str(lora), lora_strength=0)).remote_files() == {}


class _Server:
    def __init__(self):
        self.request = None

    def generate(self, request):
        self.request = request
        return [Image.new("RGB", (request["width"], request["height"]), (5, 6, 7))]


def test_full_strength_sends_the_reference_alone_with_the_lora():
    server = _Server()
    lora = [{"path": "bars.safetensors", "multiplier": 1.0}]
    processor = Flux2KleinProcessor(server, "Remove the black censor bars.", 8, strength=1.0, lora=lora)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[8:32, 8:32] = 255
    result = processor.inference(Image.new("RGB", (64, 64), (200, 200, 200)), mask)
    assert "init_image" not in server.request and "mask_image" not in server.request and "strength" not in server.request
    assert server.request["ref_images"] and server.request["lora"] == lora
    assert server.request["sample_params"]["sample_steps"] == 8
    assert result.getpixel((16, 16)) == (5, 6, 7) and result.getpixel((60, 60)) == (200, 200, 200)


def test_text_cleanup_requests_are_unchanged():
    server = _Server()
    processor = Flux2KleinProcessor(server, "Remove the text and reconstruct the background.")
    processor.inference(Image.new("RGB", (64, 64)), np.full((64, 64), 255, dtype=np.uint8))
    assert {"init_image", "mask_image", "strength"} <= set(server.request) and "lora" not in server.request
    assert server.request["sample_params"]["sample_steps"] == 4


# ---- through the real pipeline: scheduler, gating, patches committed in order ---------------------


class _FakeCensorDetector(CensorDetectionProvider):
    id = "fake-censor-detector"
    runtimes = ()

    def _detect_censorship(self, image):
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        mask[200:220, 250:350] = 255
        return mask


class _FakeDecensor(DecensorProvider):
    id = "fake-decensor"
    runtimes = ()

    def _reconstruct(self, image, mask, white):
        return Image.new("RGB", image.size, (10, 200, 30))


def _pipeline(mode: str) -> Pipeline:
    registry = fake_registry()
    registry.add(_FakeCensorDetector)
    registry.add(_FakeDecensor)
    config = fake_config()
    config.censorship = CensorshipConfig(mode=mode, detector=_FakeCensorDetector.id, model=_FakeDecensor.id)
    stages = {**fake_processors(), Stage.CENSOR_DETECTION: CensorDetectionStage(), Stage.DECENSOR: DecensorStage()}
    return Pipeline(config, RuntimeManager("cpu"), registry, stages)


@pytest.fixture
def project(isolated_home):
    store = ProjectStore(isolated_home / "projects")
    project_id = store.create("test")
    pages = [store.import_page(project_id, f"p{i}.png", page_png()) for i in range(3)]
    return store.session(project_id), pages


@pytest.mark.asyncio
async def test_a_full_run_with_every_stage_commits_each_page_in_order(project):
    session, pages = project
    finished: list[tuple[str, Stage]] = []
    report = await _pipeline("remove").execute(session, Request(progress=lambda p: finished.append((p.page, p.stage)) if p.kind is ProgressKind.FINISHED else None))
    assert report.status is RunStatus.COMPLETED and report.completed == report.total == 3 * 6
    snapshot = session.snapshot()
    for page in pages:
        order = [stage for owner, stage in finished if owner == page]
        assert order.index(Stage.CENSOR_DETECTION) < order.index(Stage.DETECTION)
        assert order.index(Stage.INPAINTING) < order.index(Stage.DECENSOR)
        assert snapshot.asset(page, CENSOR_MASK) is not None
        layers = snapshot.raster_layers(page)
        assert layers.index(snapshot.raster_layers(page, c.RasterLayerKind.CLEANUP)[0]) < layers.index(decensor_layer(snapshot, page))


@pytest.mark.asyncio
async def test_review_mode_marks_pages_and_a_second_run_reconstructs_on_request(project):
    session, pages = project
    pipeline = _pipeline("detect")
    await pipeline.execute(session, Request())
    snapshot = session.snapshot()
    assert all(snapshot.asset(page, CENSOR_MASK) is not None and decensor_layer(snapshot, page) is None for page in pages)
    await pipeline.execute(session, Request(operation=Operation.only(Stage.DECENSOR)))
    snapshot = session.snapshot()
    assert all(decensor_layer(snapshot, page) is not None for page in pages)


@pytest.mark.asyncio
async def test_off_leaves_a_run_exactly_as_before_and_refuses_the_stage_alone(project):
    session, pages = project
    pipeline = _pipeline("off")
    report = await pipeline.execute(session, Request())
    assert report.total == 3 * 4
    assert all(session.snapshot().asset(page, CENSOR_MASK) is None for page in pages)
    with pytest.raises(PipelineError, match="censorship is off"):
        await pipeline.execute(session, Request(operation=Operation.only(Stage.DECENSOR)))


@pytest.mark.asyncio
async def test_a_brush_stroke_goes_through_the_pipeline_to_the_decensor_stage(project):
    session, pages = project
    stroke = np.zeros((300, 400), dtype=np.uint8)
    stroke[10:30, 10:60] = 255
    mask = InpaintingMask(page=pages[0], png=_png(stroke, "L"))
    await _pipeline("detect").execute(session, Request(operation=Operation.only(Stage.DECENSOR), scope=PagesScope((pages[0],)), inpainting_mask=mask))
    snapshot = session.snapshot()
    assert c.is_user(snapshot.component(decensor_layer(snapshot, pages[0]), c.RasterLayer).origin)
    assert decensor_layer(snapshot, pages[1]) is None


# ---- the real detector: checkpoint load, network forward pass, geometry ----------------------------


def _stacked_page() -> tuple[np.ndarray, np.ndarray]:
    """A page of hatched ink with a stack of black bars, and the levels a checkpoint would score."""
    page = np.full((1200, 900, 3), 235, dtype=np.uint8)
    page[300:700, 200:700][::6] = 120
    levels = np.zeros(page.shape[:2], dtype=np.uint8)
    for top in range(380, 620, 40):
        page[top : top + 16, 300:600] = 0
        levels[top : top + 16, 300:600] = bars.CONFIDENT
    return page, levels


def test_the_real_provider_loads_a_checkpoint_shaped_file_and_runs_the_network(tmp_path, monkeypatch):
    state = dict(Er0mangaSegNet().state_dict())
    state["auxiliary_head.conv_seg.weight"] = torch.zeros(1)
    path = tmp_path / "convnext_1024_iter_400.pth"
    torch.save({"meta": {"hook": _Hostile()}, "state_dict": state}, path)
    monkeypatch.setattr(er0mangaseg, "CHECKPOINT_SHA256", er0mangaseg.sha256(path))

    provider = Er0mangaSegProvider({"weights": str(path)}, RuntimeManager("cpu"), "torch")
    provider._load()
    assert not provider._net.training
    page, _ = _stacked_page()
    levels = provider.scores(page)
    assert levels.shape == page.shape[:2] and set(np.unique(levels)) <= {0, bars.SUPPORTED, bars.CONFIDENT}


def test_a_file_that_is_not_the_pinned_checkpoint_is_refused_before_it_is_read(tmp_path, monkeypatch):
    path = tmp_path / "other.pth"
    path.write_bytes(b"not the checkpoint")
    monkeypatch.setattr(er0mangaseg, "load_state", lambda path: pytest.fail("read an unpinned file"))
    with pytest.raises(ProviderUnavailable, match="not the Er0mangaSeg checkpoint"):
        Er0mangaSegProvider({"weights": str(path)}, RuntimeManager("cpu"), "torch")._load()


def test_a_missing_local_checkpoint_points_back_to_the_download(tmp_path):
    with pytest.raises(ProviderUnavailable, match="clear the path to download"):
        Er0mangaSegProvider({"weights": str(tmp_path / "absent.pth")}, RuntimeManager("cpu"), "torch")._load()


def _layer(name: str, data: bytes) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path, content in (("etc/passwd", b"unrelated"), (name, data)):
            info = tarfile.TarInfo(path)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def _registry(monkeypatch, layer: bytes) -> None:
    def urlopen(request, timeout=None):
        url = request if isinstance(request, str) else request.full_url
        return io.BytesIO(json.dumps({"token": "t"}).encode() if "auth.docker.io" in url else layer)

    monkeypatch.setattr("urakatsu.runtimes.store.urllib.request.urlopen", urlopen)


def test_the_checkpoint_comes_out_of_its_image_layer_by_hash(tmp_path, monkeypatch):
    data = b"weights"
    source = ContainerLayerFile("a/b", "sha256:abc", "home/user/model.pth", hashlib.sha256(data).hexdigest())
    _registry(monkeypatch, _layer("home/user/model.pth", data))
    store = ModelStore(tmp_path, offline=False)
    seen = []
    path = source.resolve(store, lambda label, done, total: seen.append(done))
    assert path.read_bytes() == data and path.name == "model.pth" and seen and store.cached(source)
    assert sorted(p.name for p in path.parent.iterdir()) == ["model.pth"]


def test_a_layer_file_with_the_wrong_hash_is_thrown_away(tmp_path, monkeypatch):
    source = ContainerLayerFile("a/b", "sha256:abc", "home/user/model.pth", "0" * 64)
    _registry(monkeypatch, _layer("home/user/model.pth", b"tampered"))
    store = ModelStore(tmp_path, offline=False)
    with pytest.raises(ModelLoadError, match="unexpected SHA-256"):
        source.resolve(store)
    assert not list(source.local_path(store).parent.iterdir())


def test_a_layer_without_the_file_is_an_error_not_an_empty_download(tmp_path, monkeypatch):
    source = ContainerLayerFile("a/b", "sha256:abc", "home/user/model.pth", "0" * 64)
    _registry(monkeypatch, _layer("other.bin", b"x"))
    with pytest.raises(ModelLoadError, match="not in the layer"):
        source.resolve(ModelStore(tmp_path, offline=False))


def test_scores_become_a_grown_soft_mask_over_the_whole_stack(monkeypatch):
    page, levels = _stacked_page()
    provider = Er0mangaSegProvider({"mask_grow": 6, "mask_blur": 2.0, "white_bars": False}, RuntimeManager("cpu"), "torch")
    provider._net = object()
    monkeypatch.setattr(provider, "scores", lambda rgb: levels)
    mask = provider._detect_censorship(Image.fromarray(page))
    for top in range(380, 620, 40):
        assert mask[top + 8, 450] == 255
    assert mask[375, 450] > 0  # grown past the edge of the bar
    assert mask[100, 100] == 0 and mask[1000, 800] == 0


# ---- real weights: the checkpoint, Klein 9B and sd-server through a real run ------------------------


def _barred_page() -> bytes:
    """A synthetic shape under black bars; nothing in it is explicit."""
    image = Image.new("RGB", (1200, 1700), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 1160, 1660), outline="black", width=8)
    draw.ellipse((300, 500, 900, 1200), fill=(205, 205, 205), outline="black", width=5)
    for offset in range(300, 900, 14):
        draw.line((offset, 500, offset + 200, 1200), fill=(150, 150, 150), width=2)
    for index in range(6):
        y = 700 + index * 52
        draw.line((430, y + 40, 770, y - 40), fill="black", width=30)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


@pytest.mark.asyncio
async def test_a_real_run_finds_the_bars_and_rebuilds_the_art_on_its_own_layer(isolated_home, monkeypatch):
    """The whole path on real weights, with no model picked; set URAKATSU_TEST_HOME to a home holding them."""
    home = os.environ.get("URAKATSU_TEST_HOME")
    if not home:
        pytest.skip("set URAKATSU_TEST_HOME to a model store that holds Er0mangaSeg and Klein 9B")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    store = ProjectStore(isolated_home / "projects")
    project_id = store.create("real")
    page = store.import_page(project_id, "barred.png", _barred_page())
    monkeypatch.setenv("URAKATSU_HOME", home)
    monkeypatch.setattr(models_folder, "_root", None)  # an earlier test's home may be remembered
    models = ModelStore(offline=True)
    if not all(models.cached(source) for source in (er0mangaseg.WEIGHTS, *Flux2KleinDecensorProvider.weight_sources())):
        pytest.skip(f"Er0mangaSeg or Klein 9B is not cached under {home}")

    registry = fake_registry()
    registry.add(Er0mangaSegProvider)
    registry.add(Flux2KleinDecensorProvider)
    config = fake_config()
    config.device = "auto"
    config.censorship = CensorshipConfig(mode="remove", detector=Er0mangaSegProvider.id, model=Flux2KleinDecensorProvider.id)
    stages = {**fake_processors(), Stage.CENSOR_DETECTION: CensorDetectionStage(), Stage.DECENSOR: DecensorStage()}
    pipeline = Pipeline(config, RuntimeManager("auto"), registry, stages)
    session = store.session(project_id)
    try:
        report = await pipeline.execute(session, Request())
    finally:
        pipeline.unload_all()
    assert report.status is RunStatus.COMPLETED, report

    snapshot = session.snapshot()
    images = ImageCache()
    mask = np.asarray(images.get(snapshot, page, CENSOR_MASK).convert("L"))
    entity = decensor_layer(snapshot, page)
    assert (mask >= 127).sum() > 50_000 and entity is not None
    layers = snapshot.raster_layers(page)
    assert layers.index(snapshot.raster_layers(page, c.RasterLayerKind.CLEANUP)[0]) < layers.index(entity)
    assert not c.is_user(snapshot.component(entity, c.RasterLayer).origin)

    rebuilt = np.asarray(images.get(snapshot, entity, "source").convert("RGBA"))
    bars_pixels = (np.asarray(Image.open(io.BytesIO(_barred_page())).convert("L")) < 30) & (mask == 255)
    assert bars_pixels.sum() > 30_000
    assert (rebuilt[..., 3][bars_pixels] == 255).all()  # the layer fully covers the bars
    assert (rebuilt[..., :3].mean(axis=2)[bars_pixels] > 100).all()  # with art, not more black
    assert (rebuilt[..., 3][mask == 0] == 0).all()  # and touches nothing outside the mask
