"""The built-in colorizers: Manga Light Colorizer and Manga Colorization v2 (no weights needed)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from urakatsu.core.config import ColorizationConfig, PipelineConfig
from urakatsu.core.errors import ProviderUnavailable
from urakatsu.core.registry import DEFAULTS, registry
from urakatsu.core.stage import Stage
from urakatsu.providers.colorization import manga_colorization_v2 as v2
from urakatsu.providers.colorization import manga_light_colorizer as light
from urakatsu.remote import CLOUD, REMOTE

PAINT = (40, 120, 200)


class _Session:
    """An ONNX session that records what it was fed and answers with ``answer(feeds)``."""

    def __init__(self, answer):
        self.answer = answer
        self.feeds: list[dict] = []

    def run(self, outputs, feeds):
        self.feeds.append(feeds)
        return self.answer(feeds)


def _page(width: int = 300, height: int = 420) -> Image.Image:
    pixels = np.full((height, width, 3), 255, dtype=np.uint8)
    pixels[40:60, :] = 0  # a solid ink line
    return Image.fromarray(pixels, "RGB")


def test_both_colorizers_are_offered_next_to_the_image_editing_model():
    registry.discover()
    ids = {cls.id for cls in registry.list(Stage.COLORIZATION)}
    assert {"diffusion-colorizer", "manga-light-colorizer", "manga-colorization-v2"} <= ids
    assert DEFAULTS[Stage.COLORIZATION] == "diffusion-colorizer"
    config = PipelineConfig(colorization=ColorizationConfig(mode="colorize", model="manga-light-colorizer"))
    assert config.provider_for(Stage.COLORIZATION) == "manga-light-colorizer"


@pytest.mark.parametrize("cls", [light.MangaLightColorizerProvider, v2.MangaColorizationV2Provider])
def test_the_colorizers_bring_pinned_weights_and_can_run_on_a_worker(cls):
    assert cls.runtimes_declared() == ("onnx", REMOTE, CLOUD)
    assert not cls.uses_catalog()
    for source in cls.weight_sources():
        assert source.describe().startswith("hf://") and len(source.revision) == 40
    info = cls.info()
    assert "model" not in info.settings_schema["properties"] and "keep_ink" in info.settings_schema["properties"]


def test_the_light_colorizer_works_at_the_page_shape_near_its_training_area():
    width, height = light.working_size(960, 1361, 512)
    assert (width, height) == (416, 608)
    assert width % 32 == 0 and height % 32 == 0
    assert light.working_size(2000, 1000, 512)[0] > light.working_size(2000, 1000, 512)[1]
    assert light.working_size(10, 10_000, 512)[0] == 32  # never a side of zero


def test_the_light_colorizer_feeds_both_graphs_and_returns_the_page_size():
    sam = _Session(lambda feeds: [np.zeros((1, 256, 64, 64), np.float32), np.zeros((1, 256, 32, 32), np.float32)])

    def paint(feeds):
        _, _, height, width = feeds["L_bw"].shape
        rgb = (np.array(PAINT, np.float32) / 127.5 - 1.0)[None, :, None, None]
        return [np.broadcast_to(rgb, (1, 3, height, width)).copy()]

    generator = _Session(paint)
    provider = light.MangaLightColorizerProvider()
    provider._generator, provider._sam = generator, sam
    result = provider._colorize(_page())
    assert result.size == (300, 420)
    assert sam.feeds[0]["rgb_input"].shape == (1, 3, 1024, 1024)
    feeds = generator.feeds[0]
    assert feeds["L_bw"].shape == (1, 1, 608, 448) and feeds["L_bw"].min() >= -1.0 and feeds["L_bw"].max() <= 1.0
    assert feeds["wd14_embedding"].shape == (1, 1024) and not feeds["wd14_embedding"].any()
    pixels = np.asarray(result)
    assert tuple(pixels[50, 150]) == (0, 0, 0)  # the page's ink
    assert np.abs(pixels[200, 150].astype(int) - PAINT).max() <= 1  # the model's color


def test_resize_pad_follows_upstream():
    portrait = np.zeros((1000, 700), np.uint8)
    padded, pad = v2.resize_pad(portrait, 576)
    assert padded.shape[1] == 576 and padded.shape[0] % 32 == 0 and pad == (padded.shape[0] - 823, 0)
    assert padded.dtype == np.float32
    # A side that is already a multiple of 32 still gains 32, as upstream.
    assert v2.resize_pad(np.zeros((640, 576), np.float32), 576)[1] == (32, 0)
    landscape = np.zeros((500, 1000), np.float32)
    padded, pad = v2.resize_pad(landscape, 576)
    assert padded.shape[0] == 864 and padded.shape[1] % 32 == 0 and pad[0] == 0
    # The padding repeats each column's brightest pixel.
    column = np.zeros((96, 64), np.float32)
    column[10, 3] = 1.0
    padded, _ = v2.resize_pad(column, 64)
    assert padded[-1, 3] == 1.0 and padded[-1, 4] == 0.0


def _mosaic(image: np.ndarray) -> np.ndarray:
    """The paper's input layer: the four subimages of each 2x2 block, channel by channel."""
    channels, height, width = image.shape
    out = np.zeros((channels * 4, height // 2, width // 2), image.dtype)
    for index, (row, col) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        out[index::4] = image[:, row::2, col::2]
    return out


def test_ffdnet_splits_and_rebuilds_the_image_as_the_checkpoint_expects():
    torch = pytest.importorskip("torch")
    net = v2.ffdnet(torch)().eval()
    first, last = net.layers[0], net.layers[-1]
    image = torch.rand(1, 3, 6, 8)
    seen = {}
    first.register_forward_hook(lambda module, inputs, output: seen.update(input=inputs[0]))  # None keeps the output
    with torch.no_grad():
        last.weight.zero_()
        net(image, 0.1)
    expected = np.concatenate([np.full((3, 3, 4), 0.1, np.float32), _mosaic(image[0].numpy())])
    assert np.allclose(seen["input"][0].numpy(), expected)
    assert len([m for m in net.layers if isinstance(m, torch.nn.Conv2d)]) == v2.FFDNET_LAYERS
    keys = set(net.state_dict())
    assert "layers.0.weight" in keys and "layers.3.running_var" in keys and f"layers.{len(net.layers) - 1}.weight" in keys


def test_v2_feeds_the_gray_page_without_hints_and_returns_the_page_size():
    def paint(feeds):
        _, _, height, width = feeds["input"].shape
        return [np.broadcast_to((np.array(PAINT, np.float32) / 255)[None, :, None, None], (1, 3, height, width)).copy()]

    generator = _Session(paint)
    provider = v2.MangaColorizationV2Provider({"denoise": False, "size": 256})
    provider._generator = generator
    result = provider._colorize(_page())
    assert result.size == (300, 420)
    feeds = generator.feeds[0]["input"]
    assert feeds.shape[:2] == (1, 5) and feeds.shape[3] == 256 and feeds.shape[2] % 32 == 0
    assert not feeds[:, 1:].any()
    pixels = np.asarray(result)
    assert tuple(pixels[50, 150]) == (0, 0, 0)
    assert np.abs(pixels[200, 150].astype(int) - PAINT).max() <= 1


def test_v2_denoises_with_the_blue_channel_of_its_result():
    torch = pytest.importorskip("torch")
    provider = v2.MangaColorizationV2Provider()
    provider._device = torch.device("cpu")
    provider._denoiser = lambda image, sigma: torch.zeros_like(image)  # predicts no noise
    page = np.zeros((5, 7, 3), np.uint8)
    page[..., 2] = 200
    gray = provider.denoised_gray(Image.fromarray(page, "RGB"))
    assert gray.shape == (5, 7) and gray.dtype == np.uint8 and int(gray.max()) in (199, 200)


def test_v2_asks_for_pytorch_only_when_it_denoises(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args: None if name == "torch" else real(name, *args))
    with pytest.raises(ProviderUnavailable, match="PyTorch"):
        v2.MangaColorizationV2Provider()._load()
