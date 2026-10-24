"""Urakatsu: a modular, plug-and-play manga translation pipeline.

The package is organised like into a modular structure.

- ``urakatsu.core``      shared types, stage enum, configuration, provider registry
- ``urakatsu.runtimes``  hardware discovery, devices, model store, inference runtimes
- ``urakatsu.providers`` detection / OCR / inpainting / rendering / translation providers
- ``urakatsu.pipeline``  document model, per-page scheduler, stage orchestration
- ``urakatsu.service``   gRPC + gRPC-web + HTTP service, CLI entry point
"""

__version__ = "0.1.0"
