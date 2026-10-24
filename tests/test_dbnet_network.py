"""DBNet ResNet34 text detector network (urakatsu/providers/detection/dbnet).

The first tests build the network with random weights on the CPU and need no downloads. The last test
loads the cached manga-image-translator checkpoint strictly and runs it on a page fixture; it is skipped
unless the checkpoint is cached and CUDA is available.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from urakatsu.providers.detection.dbnet.network import DBNetResNet34, forward, load_network
from urakatsu.providers.detection.dbnet.preprocess import prepare
from urakatsu.runtimes.store import GitHubReleaseFile, ModelStore

CHECKPOINT = GitHubReleaseFile("zyddnys/manga-image-translator", "beta-0.3", "detect-20241225.ckpt")
DEFAULT_FIXTURE = Path(
    r"C:\Users\karrot\AppData\Local\Temp\claude\B--Urakatsu\bca97ddd-a10d-4816-bbf1-9861555d82fb\scratchpad\fixtures\1.jpg"
)


def test_network_output_shapes_on_a_random_input():
    net = DBNetResNet34().eval()
    with torch.no_grad():
        db, mask = net(torch.randn(1, 3, 256, 512))
    assert tuple(db.shape) == (1, 2, 256, 512)
    assert tuple(mask.shape) == (1, 1, 128, 256)
    assert float(db[:, 1].min()) >= 0.0 and float(db[:, 1].max()) <= 1.0  # threshold map is a sigmoid
    assert float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0


def test_state_dict_has_exactly_the_checkpoint_layout():
    # The pinned checkpoint holds 437 entries (parameters, BatchNorm buffers and num_batches_tracked).
    keys = set(DBNetResNet34().state_dict())
    assert len(keys) == 437
    for expected in ("backbone.layer4.2.conv2.weight", "down_conv3.conv.7.num_batches_tracked",
                     "upconv7.conv.6.weight", "conv_db.thresh.6.bias", "conv_mask.6.weight"):
        assert expected in keys


def test_prepare_resizes_long_side_and_pads_to_multiples_of_256():
    image = np.zeros((300, 500, 3), dtype=np.uint8)
    prepared = prepare(image, detect_size=2048)
    assert prepared.ratio == pytest.approx(2048 / 500)
    assert (prepared.resized_height, prepared.resized_width) == (1229, 2048)
    assert (prepared.pad_height, prepared.pad_width) == (51, 0)
    assert tuple(prepared.tensor.shape) == (1, 3, 1280, 2048)
    assert prepared.tensor.dtype == torch.float32
    # Padding is black before normalization, so it is -1.0 afterwards.
    assert float(prepared.tensor[0, :, 1229:, :].mean()) == pytest.approx(-1.0)


def _real_store() -> ModelStore:
    """The user's cached weights (``URAKATSU_HOME`` is redirected to a temp dir by conftest)."""
    root = os.environ.get("URAKATSU_WEIGHTS_HOME")
    return ModelStore(root=Path(root) if root else Path.home() / ".urakatsu" / "models", offline=True)


def _cached() -> bool:
    return _real_store().cached(CHECKPOINT)


@pytest.mark.skipif(not _cached(), reason="DBNet detect-20241225.ckpt is not cached")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_dbnet_loads_strictly_and_detects_text_on_the_fixture():
    from PIL import Image

    fixture = Path(os.environ.get("KOHARU_FIXTURE", DEFAULT_FIXTURE))
    if not fixture.exists():
        pytest.skip(f"fixture page not found: {fixture}")
    image = np.asarray(Image.open(fixture).convert("RGB"))

    device = torch.device("cuda")
    net = load_network(CHECKPOINT.local_path(_real_store()), device)
    prepared = prepare(image, detect_size=2048)
    raw = forward(net, prepared, device)

    assert raw.mask.shape == image.shape[:2]
    assert raw.mask.dtype == np.float32
    assert raw.db.shape == (prepared.padded_height, prepared.padded_width)

    text = raw.db[: prepared.resized_height, : prepared.resized_width] > 0.5
    fraction = float(text.mean())
    assert text.any(), "no pixel of the probability map is above 0.5"
    assert fraction < 0.5, f"{fraction:.1%} of the page is text-like"
