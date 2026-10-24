"""Providers: interchangeable models and services for each pipeline stage.

Every provider declares the runtimes it
can run on; configuration selects one per provider. Optional dependencies are
imported lazily inside ``_load`` so listing providers never imports Torch,
llama.cpp, or stable-diffusion.cpp.
"""

from .base import (
    DetectionProvider,
    InpaintingProvider,
    OcrProvider,
    Provider,
    ProviderInfo,
    ProviderSettings,
    RuntimeOption,
    TranslationProvider,
)

__all__ = [
    "Provider",
    "ProviderInfo",
    "ProviderSettings",
    "RuntimeOption",
    "DetectionProvider",
    "OcrProvider",
    "InpaintingProvider",
    "TranslationProvider",
]
