"""DBNet (ResNet34) text detector network: manga-image-translator's default detector, used by
``dbnet_resnet34.py``.

``preprocess`` builds the network input, ``network`` builds and strictly loads the checkpoint and runs
the forward pass. Post-processing (box extraction, thresholds, unclip) is left to the provider.
Torch is imported only when a submodule is accessed, so importing this package stays cheap.
"""

from __future__ import annotations

from typing import Any

__all__ = ["DBNetResNet34", "Prepared", "Raw", "forward", "load_network", "prepare"]

_EXPORTS = {
    "DBNetResNet34": ".network",
    "Raw": ".network",
    "forward": ".network",
    "load_network": ".network",
    "Prepared": ".preprocess",
    "prepare": ".preprocess",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name, __name__), name)
