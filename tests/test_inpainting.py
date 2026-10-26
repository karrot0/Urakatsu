"""Inpainting: ports of the Rust tests that need no weights, plus weight-gated provider smoke tests."""

import asyncio
import io

import numpy as np
import pytest
from PIL import Image

from urakatsu.core.stage import Stage
from urakatsu.pipeline.images import ImageCache
from urakatsu.pipeline.stages import InpaintingMask, StageInput
from urakatsu.pipeline.stages.inpainting import (
    FlatFillRegion,
    InpaintingStage,
    inpaint_tiled,
    inpaint_tiles,
    prepare,
    TILE_SIZE,
    crop_tile_mask,
    fill_region,
)
from urakatsu.providers.inpainting.aot_inpainting import AotInpaintingProvider
from urakatsu.providers.inpainting.common import symmetric_index
from urakatsu.providers.inpainting.flux2_klein import Flux2KleinProvider, apply_overlay, flux2_sigmas, inpaint_strength
from urakatsu.providers.inpainting.lama import LamaProvider, boxes_from_mask, crop_box_for
from urakatsu.providers.inpainting.rorem_mixed import DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPT, RoremMixedProvider, composite, resize_mask
from urakatsu.runtimes.manager import RuntimeManager
from urakatsu.runtimes.store import ModelStore
from urakatsu.scene import AssetInput, AssetMetadata, Page, RasterLayer, RasterLayerKind, Session, user
from urakatsu.providers.base import InpaintingProvider, ProviderSettings


def encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def rectangle_region(bounds):
    left, top, right, bottom = bounds
    return FlatFillRegion(
        bounds=bounds,
        polygon=np.array([(left, top), (right, top), (right, bottom), (left, bottom)], dtype=np.float32),
    )


def noop_inference(calls: list):
    def inference(image, mask):
        calls.append((image.size, mask))
        return Image.new("RGB", image.size, (1, 2, 3))

    return inference


# ---- stage: mask and skip rules (Rust tests in stages/inpainting.rs) ----------------------------


def make_page_session(tmp_path, *, cleanup: bool, mask_size=(8, 8)):
    session = Session(tmp_path)
    patch_edit = session.snapshot().edit()
    page = patch_edit.add_page(Page(label="page", width=8.0, height=8.0))
    patch_edit.set_asset(page, "source", AssetInput(data=encode_png(Image.new("RGB", (8, 8))), media_type="image/png", metadata=AssetMetadata(width=8, height=8)))
    patch_edit.set_asset(page, "text-mask", AssetInput(data=encode_png(Image.new("L", (8, 8), 255)), media_type="image/png", metadata=AssetMetadata(width=8, height=8)))
    if cleanup:
        entity = patch_edit.add_entity(page)
        patch_edit.set(entity, RasterLayer(origin=user(), name="Cleanup", kind=RasterLayerKind.CLEANUP))
        patch_edit.set_asset(entity, "source", AssetInput(data=encode_png(Image.new("RGBA", (8, 8))), media_type="image/png", metadata=AssetMetadata(width=8, height=8)))
    session.commit(patch_edit.finish())
    return session, page


def make_input(snapshot, page, manual_png=None):
    mask = InpaintingMask(page=page, png=manual_png) if manual_png is not None else None
    return StageInput(snapshot, page, None, None, ImageCache(), mask)


def test_transient_inpainting_mask_replaces_persistent_page_masks(tmp_path):
    session, page = make_page_session(tmp_path, cleanup=False)
    transient = Image.new("L", (8, 8), 0)
    transient.putpixel((3, 4), 255)
    inp = make_input(session.snapshot(), page, encode_png(transient))

    prepared = prepare(inp)

    assert prepared.mask[4, 3] == 255
    assert prepared.mask[0, 0] == 0
    assert not prepared.text_mask.any()


def test_automatic_inpainting_skips_committed_cleanup(tmp_path):
    session, page = make_page_session(tmp_path, cleanup=True)
    snapshot = session.snapshot()
    stage = InpaintingStage()
    provider = LamaProvider()

    assert stage.skip(make_input(snapshot, page), provider)
    assert not stage.skip(make_input(snapshot, page, b""), provider)


def test_stage_is_registered_for_inpainting_and_providers_are_listed():
    from urakatsu.core.registry import registry

    assert {cls.id for cls in registry.list(Stage.INPAINTING)} == {"lama", "aot-inpainting", "flux2-klein", "rorem-mixed", "diffusion-model", "nano-banana-2.1"}
    # Every inpainter can also run on a worker (the remote runtime) or on Urakatsu Cloud.
    assert LamaProvider.runtimes_declared() == ("torch", "remote", "cloud")
    assert Flux2KleinProvider.runtimes_declared() == ("diffusion", "remote", "cloud")
    assert RoremMixedProvider.runtimes_declared() == ("diffusion", "remote", "cloud")


# ---- inpaint_tiled (Rust tests in stages/inpainting.rs) ------------------------------------------


def test_uniform_mask_background_is_filled_without_inference():
    image = np.full((96, 96, 3), (240, 241, 242), dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=np.uint8)
    image[40:56, 32:64] = (20, 20, 20)
    mask[40:56, 32:64] = 255
    calls = []

    output = np.asarray(
        inpaint_tiled(Image.fromarray(image), mask, mask, [rectangle_region((0, 0, 96, 96))], noop_inference(calls))
    )

    assert calls == []
    assert tuple(output[48, 48]) == (240, 241, 242)


def test_uniform_bubbles_are_filled_independently_inside_one_textured_tile():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:100, 0:200]
    image[:] = np.where(((xx + yy) % 2 == 0)[..., None], np.array([20, 80, 140], np.uint8), np.array([220, 80, 120], np.uint8))
    image[10:90, 10:90] = (245, 245, 245)
    image[10:90, 110:190] = (250, 240, 220)
    mask = np.zeros((100, 200), dtype=np.uint8)
    for y in range(40, 60):
        image[y, 30:60] = (10, 10, 10)
        mask[y, 30:60] = 255
        image[y, 135:165] = (10, 10, 10)
        mask[y, 135:165] = 255
    calls = []

    output = np.asarray(
        inpaint_tiled(
            Image.fromarray(image),
            mask,
            mask,
            [rectangle_region((10, 10, 90, 90)), rectangle_region((110, 10, 190, 90))],
            noop_inference(calls),
        )
    )

    assert calls == []
    assert tuple(output[50, 45]) == (245, 245, 245)
    assert tuple(output[50, 150]) == (250, 240, 220)


def test_connected_mask_crossing_both_page_grid_axes_is_inferred_once():
    image = Image.new("RGB", (900, 900), (20, 40, 60))
    mask = np.zeros((900, 900), dtype=np.uint8)
    mask[500:525, 500:525] = 255
    calls = []

    def inference(crop, crop_mask):
        calls.append(crop_mask)
        assert int((crop_mask >= 127).sum()) == 25 * 25
        return Image.new("RGB", crop.size, (1, 2, 3))

    output = np.asarray(inpaint_tiled(image, mask, mask, [], inference))

    assert len(calls) == 1
    assert tuple(output[500, 500]) == (1, 2, 3)
    assert tuple(output[524, 524]) == (1, 2, 3)
    assert tuple(output[499, 499]) == (20, 40, 60)


def test_oversized_component_tiles_mask_their_continuation_context():
    mask = np.zeros((300, 1000), dtype=np.uint8)
    mask[100:120, 50:850] = 255

    labels, tiles = inpaint_tiles(mask)

    assert len(tiles) == 2
    for tile in tiles:
        left, top, right, bottom = tile.crop
        assert right - left <= TILE_SIZE + 256
        assert bottom - top <= TILE_SIZE + 256
        label = tile.components[0]
        cl, ct, cr, cb = tile.core
        owned = int((labels[ct:cb, cl:cr] == label).sum())
        masked = int((crop_tile_mask(mask, tile) >= 127).sum())
        assert masked > owned


def test_textured_mask_regions_are_inferred_as_bounded_tiles():
    yy, xx = np.mgrid[0:700, 0:1200]
    image = np.where(((xx + yy) % 2 == 0)[..., None], np.array([0, 40, 80], np.uint8), np.array([255, 215, 175], np.uint8))
    image = image.astype(np.uint8)
    original = image.copy()
    mask = np.zeros((700, 1200), dtype=np.uint8)
    mask[10, 10] = 255
    mask[600, 1100] = 255
    calls = []

    def inference(crop, crop_mask):
        calls.append(crop.size)
        assert crop.width <= 128 * 2 + 1 and crop.height <= 128 * 2 + 1
        return Image.new("RGB", crop.size, (1, 2, 3))

    output = np.asarray(inpaint_tiled(Image.fromarray(image), mask, mask, [], inference))

    assert len(calls) == 2
    assert tuple(output[10, 10]) == (1, 2, 3)
    assert tuple(output[600, 1100]) == (1, 2, 3)
    assert tuple(output[300, 500]) == tuple(original[300, 500])


# ---- shared processor helpers --------------------------------------------------------------------


def test_symmetric_index_mirrors_including_edge():
    assert [symmetric_index(i, 3) for i in range(8)] == [0, 1, 2, 2, 1, 0, 0, 1]


def test_lama_boxes_from_mask_and_crop_box_clamp_to_image():
    mask = np.zeros((200, 300), dtype=np.uint8)
    mask[50:60, 100:120] = 255
    mask[150:155, 10:14] = 255

    boxes = boxes_from_mask(mask)

    assert (100, 50, 120, 60) in boxes
    assert (10, 150, 14, 155) in boxes
    assert crop_box_for(300, 200, (100, 50, 120, 60), 128) == (0, 0, 276, 200)


def test_aot_and_lama_settings_have_no_options():
    assert issubclass(LamaProvider.Settings, ProviderSettings)
    # Only the remote runtime's connection fields, which the Settings page shows for that runtime alone.
    properties = AotInpaintingProvider.info().settings_schema.get("properties", {})
    assert {name: field["runtimes"] for name, field in properties.items()} == {"remote_url": ["remote"], "remote_api_key": ["remote"]}


# ---- FLUX.2 Klein processor (flux2_klein/processor.rs tests and scheduler) ----------------------


def test_flux_overlay_without_a_crop_preserves_unmasked_pixels():
    initial = np.array([[[10, 20, 30], [10, 20, 30]]], dtype=np.uint8)
    generated = np.array([[[200, 210, 220], [200, 210, 220]]], dtype=np.uint8)
    mask = np.array([[0, 255]], dtype=np.uint8)

    output = apply_overlay(mask, initial, generated)

    assert tuple(output[0, 0]) == (10, 20, 30)
    assert tuple(output[0, 1]) == (200, 210, 220)


def test_flux_strength_and_sigmas_match_koharu_defaults():
    # Flux2KleinInpaintOptions::default(): strength 0.8, 4 steps -> boundary 4/4 just below 1.0.
    assert float(inpaint_strength(4, 0.8)) == pytest.approx(0.99999994, abs=1e-8)
    sigmas = flux2_sigmas(1024, 4)
    assert len(sigmas) == 5
    assert sigmas[0] == 1.0 and sigmas[-1] == 0.0
    assert all(a > b for a, b in zip(sigmas, sigmas[1:]))


# ---- RORem processor (rorem_mixed/processor.rs tests) --------------------------------------------


def test_rorem_defaults_match_converted_model_card():
    from urakatsu.providers.inpainting import rorem_mixed as r

    assert (r.RESOLUTION, r.MASK_DILATION, r.NUM_INFERENCE_STEPS, r.GUIDANCE_SCALE, r.STRENGTH, r.SEED) == (
        512,
        0,
        30,
        8.0,
        0.999,
        -1,
    )
    settings = RoremMixedProvider.Settings()
    assert settings.prompt == DEFAULT_PROMPT and settings.negative_prompt == DEFAULT_NEGATIVE_PROMPT
    assert (settings.steps, settings.cfg_scale) == (r.NUM_INFERENCE_STEPS, r.GUIDANCE_SCALE)
    with pytest.raises(ValueError):
        RoremMixedProvider.Settings(steps=0)


def test_rorem_mask_is_binary_and_resized_nearest():
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[256, 256] = 128
    output = resize_mask(mask, 512)
    assert output[256, 256] == 255
    assert output[257, 257] == 0


def test_rorem_compositing_restores_original_size_and_unmasked_pixels():
    original = Image.new("RGB", (3840, 2160), (10, 20, 30))
    generated = Image.new("RGB", (512, 512), (200, 210, 220))
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[:, 256:] = 255

    output = np.asarray(composite(original, mask, generated))

    assert output.shape == (2160, 3840, 3)
    assert tuple(output[1080, 0]) == (10, 20, 30)
    assert tuple(output[1080, 3839]) == (200, 210, 220)


# ---- weight-gated provider smoke tests -----------------------------------------------------------


def _real_store() -> ModelStore:
    """The user's cached weights (``URAKATSU_HOME`` is redirected to a temp dir by conftest)."""
    import os
    from pathlib import Path

    root = os.environ.get("URAKATSU_WEIGHTS_HOME")
    return ModelStore(root=Path(root) if root else Path.home() / ".urakatsu" / "models", offline=True)


def _cached(source) -> bool:
    return _real_store().cached(source)


@pytest.mark.skipif(not _cached(LamaProvider.weight_sources()[0]), reason="LaMa weights are not cached")
def test_lama_loads_strictly_and_keeps_unmasked_pixels():
    runtimes = RuntimeManager("cpu", store=_real_store())
    provider = LamaProvider(None, runtimes)
    import asyncio

    asyncio.run(provider.load())
    try:
        image = Image.new("RGB", (128, 96), (30, 60, 90))
        mask = np.zeros((96, 128), dtype=np.uint8)
        mask[20:40, 30:60] = 255
        output = np.asarray(provider._inpaint(image, mask))
        assert output.shape == (96, 128, 3)
        assert tuple(output[0, 0]) == (30, 60, 90)
    finally:
        provider.unload()


@pytest.mark.skipif(not _cached(AotInpaintingProvider.weight_sources()[0]), reason="AOT weights are not cached")
def test_aot_loads_strictly_and_keeps_unmasked_pixels():
    runtimes = RuntimeManager("cpu", store=_real_store())
    provider = AotInpaintingProvider(None, runtimes)
    import asyncio

    asyncio.run(provider.load())
    try:
        image = Image.new("RGB", (160, 140), (30, 60, 90))
        mask = np.zeros((140, 160), dtype=np.uint8)
        mask[20:40, 30:60] = 255
        output = np.asarray(provider._inpaint(image, mask))
        assert output.shape == (140, 160, 3)
        assert tuple(output[0, 0]) == (30, 60, 90)
    finally:
        provider.unload()


# ---- generative fill (Urakatsu: canvas selection + prompt) ---------------------------------------


class PromptedFake(InpaintingProvider):
    id = "prompted-fake"
    name = "Prompted fake"
    prompted = True

    def __init__(self, settings=None, runtimes=None, runtime=None):
        super().__init__(settings, runtimes, runtime)
        self.calls: list[tuple[tuple[int, int], int, str | None]] = []

    def _fill(self, image, mask, prompt):
        self.calls.append((image.size, int((mask >= 127).sum()), prompt))
        return Image.new("RGB", image.size, (200, 10, 10))


def test_fill_region_paints_the_selection_in_one_crop_with_context():
    image = Image.new("RGB", (1000, 800), (255, 255, 255))
    mask = np.zeros((800, 1000), dtype=np.uint8)
    mask[100:400, 200:900] = 255  # wider than one inpainting tile
    calls = []

    def fill(crop, crop_mask):
        calls.append((crop.size, int((crop_mask >= 127).sum())))
        return Image.new("RGB", crop.size, (9, 9, 9))

    output = np.asarray(fill_region(image, mask, fill))

    # One crop: the 700x300 selection plus half its longer side (350) on every side, clamped to the page.
    assert calls == [((1000, 750), 700 * 300)]
    assert (output[mask >= 127] == 9).all()
    assert (output[mask < 127] == 255).all()


def test_generative_fill_runs_the_prompt_and_commits_cleanup(tmp_path):
    session, page = make_page_session(tmp_path, cleanup=False)
    selection = Image.new("L", (8, 8), 0)
    selection.paste(255, (2, 2, 6, 6))
    inp = StageInput(session.snapshot(), page, None, None, ImageCache(), InpaintingMask(page=page, png=encode_png(selection), prompt="a loud BOOM", generative=True))
    provider = PromptedFake()

    patch = asyncio.run(InpaintingStage().process(inp, provider))
    scene = session.commit(patch)

    assert provider.calls == [((8, 8), 16, "a loud BOOM")]
    cleanup = next(e for e in scene.children(page) if (layer := scene.component(e, RasterLayer)) is not None and layer.kind is RasterLayerKind.CLEANUP)
    overlay = np.asarray(Image.open(io.BytesIO(scene.asset_bytes(cleanup, "source"))).convert("RGBA"))
    assert tuple(overlay[3, 3]) == (200, 10, 10, 255)
    assert overlay[0, 0, 3] == 0


def test_generative_fill_with_a_prompt_needs_a_prompted_model():
    lama = LamaProvider()
    with pytest.raises(ValueError, match="cannot follow a prompt"):
        lama._fill(Image.new("RGB", (8, 8)), np.zeros((8, 8), dtype=np.uint8), "a cat")
    assert not LamaProvider.info().prompted
    assert Flux2KleinProvider.info().prompted and RoremMixedProvider.info().prompted
    # FLUX.2 Klein edits the crop it gets as a reference, so its prompts are instructions; RORem's describe.
    assert Flux2KleinProvider.info().instructed and not RoremMixedProvider.info().instructed


def test_fill_model_takes_the_inpainting_slot_only_while_used():
    from urakatsu.core.config import PipelineConfig
    from urakatsu.core.registry import Registry
    from urakatsu.pipeline.pipeline import Pipeline

    registry = Registry()
    registry._discovered = True
    registry.add(LamaProvider)
    registry.add(PromptedFake)
    config = PipelineConfig()
    config.fill.model = "prompted-fake"
    pipeline = Pipeline(config, runtimes=RuntimeManager(), providers=registry)

    assert pipeline.provider(Stage.INPAINTING, fill=True).id == "prompted-fake"
    pipeline.configure(config.model_copy(deep=True))  # the fill model is still wanted, so it stays
    assert pipeline._providers[Stage.INPAINTING].id == "prompted-fake"
    assert pipeline.provider(Stage.INPAINTING).id == "lama"
    config.fill.model = ""
    assert config.fill_provider() == "lama"
