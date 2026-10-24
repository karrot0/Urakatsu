"""Network construction for the layout checkpoint.

The network is the upstream RF-DETR module, not a port: the newest ``rfdetr`` builds the same
module tree as the checkpoint's ``load_model.py``, and every one of the checkpoint's 600 tensors
must load, with no unexpected keys; only buffers rfdetr derives from config may be absent
(``DERIVED_BUFFERS``). The loader below follows
``load_model.py``; the Rust counterpart is ``model.rs`` ``Model::new``/``Model::load``.

Precision follows ``backend::set_precision``: bfloat16 on CUDA compute capability >= 8, else float32
(the caller passes the dtype chosen by ``TorchRuntime.dtype()``).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import torch

from urakatsu.core.errors import ModelLoadError

from .config import InferenceConfig
from .processor import NetworkOutput


# Buffers newer rfdetr releases compute from the model config rather than learn; the
# checkpoint predates them. rfdetr 1.11 added ``_kp_active_mask`` (the keypoint schema, empty
# for this detection-only model). Every learned tensor must still load.
DERIVED_BUFFERS = frozenset({"_kp_active_mask"})


def load_network(weights: Path | str, config: InferenceConfig, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    """Builds the upstream LWDETR module, strictly loads the checkpoint, and returns it in eval mode."""
    from rfdetr import RFDETRSeg2XLarge
    from rfdetr.config import PretrainWeightsCompatibilityWarning
    from safetensors.torch import load_file

    class_names = config.class_names()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PretrainWeightsCompatibilityWarning)
        wrapper = RFDETRSeg2XLarge(
            pretrain_weights=None,
            resolution=config.resolution,
            num_select=config.num_select,
            num_classes=len(class_names),
        )
    # load_model.py: the LWDETR module is model.model.model. eval() selects the inference path
    # (only the first group of queries, no sparse segmentation forward).
    module = wrapper.model.model
    incompatible = module.load_state_dict(load_file(str(weights), device="cpu"), strict=False)
    missing = [key for key in incompatible.missing_keys if key not in DERIVED_BUFFERS]
    if missing or incompatible.unexpected_keys:
        raise ModelLoadError(f"incompatible RF-DETR weights: missing {missing}, unexpected {incompatible.unexpected_keys}")
    module.class_names = class_names
    module.eval()
    module.to(device=device, dtype=dtype)
    return module


def run_network(module: torch.nn.Module, pixel_values: torch.Tensor, dtype: torch.dtype) -> NetworkOutput:
    """``inference_with_thresholds`` forward: cast to the network dtype, run, take outputs."""
    output = module(pixel_values.to(dtype))
    return NetworkOutput(
        pred_logits=output["pred_logits"],
        pred_boxes=output["pred_boxes"],
        pred_masks=output["pred_masks"],
    )
