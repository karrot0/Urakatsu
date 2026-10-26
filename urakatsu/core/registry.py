"""Provider registry: the plug-and-play surface.

Built-in providers mirror the upstream processor catalog (``config.rs``)
and translator providers (``provider.rs``), in display order;
the first of each stage is the default. Third-party packages add providers through
the ``urakatsu.providers`` entry-point group.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
from typing import TYPE_CHECKING, Any, Iterable

from .stage import Stage

if TYPE_CHECKING:  # pragma: no cover
    from urakatsu.providers.base import Provider

log = logging.getLogger(__name__)

BUILTIN_MODULES = (
    # detection: Koharu layout first (the default), then the opt-in text detectors
    "urakatsu.providers.detection.koharu_layout_rfdetr_seg_2xl",
    "urakatsu.providers.detection.comic_text_detector",
    "urakatsu.providers.detection.comic_text_bubble_detector",
    "urakatsu.providers.detection.dbnet_resnet34",
    # OCR, in display order
    "urakatsu.providers.ocr.paddle_ocr_vl",
    "urakatsu.providers.ocr.manga_ocr",
    "urakatsu.providers.ocr.baberu_ocr",
    "urakatsu.providers.ocr.hayai_ocr",
    # inpainting
    "urakatsu.providers.inpainting.lama",
    "urakatsu.providers.inpainting.aot_inpainting",
    "urakatsu.providers.inpainting.flux2_klein",
    "urakatsu.providers.inpainting.rorem_mixed",
    "urakatsu.providers.inpainting.diffusion_recipe",
    "urakatsu.providers.inpainting.nano_banana",
    # translation (local llama.cpp + hosted providers)
    "urakatsu.providers.translation",
)

# config.rs: the first entry of each list is the stage default.
DEFAULTS: dict[Stage, str] = {
    Stage.DETECTION: "koharu-layout-rfdetr-seg-2xl",
    Stage.OCR: "paddleocr-vl-1.6",
    Stage.TRANSLATION: "local",
    Stage.INPAINTING: "lama",
}


class Registry:
    def __init__(self) -> None:
        self._providers: dict[Stage, dict[str, type["Provider"]]] = {stage: {} for stage in Stage}
        self._discovered = False

    def add(self, cls: type["Provider"]) -> type["Provider"]:
        stage = Stage.parse(cls.stage)
        if not cls.id:
            raise ValueError(f"{cls.__name__} has no id")
        existing = self._providers[stage].get(cls.id)
        if existing is not None and existing is not cls:
            log.warning("provider %s/%s re-registered by %s", stage.value, cls.id, cls.__module__)
        self._providers[stage][cls.id] = cls
        return cls

    def remove(self, stage: Stage | str, provider_id: str) -> None:
        self._providers[Stage.parse(stage)].pop(provider_id, None)

    def get(self, stage: Stage | str, provider_id: str) -> type["Provider"]:
        self.discover()
        stage = Stage.parse(stage)
        if stage is Stage.SOUND_EFFECTS:  # its model is an inpainting provider that follows prompts
            stage = Stage.INPAINTING
        try:
            return self._providers[stage][provider_id]
        except KeyError as error:
            known = ", ".join(self._providers[stage]) or "(none)"
            raise KeyError(f"no {stage.value} provider named {provider_id!r}; known: {known}") from error

    def list(self, stage: Stage | str | None = None) -> list[type["Provider"]]:
        self.discover()
        stages = [Stage.parse(stage)] if stage is not None else Stage.all()
        return [cls for s in stages for cls in self._providers[s].values()]

    def info(self, stage: Stage | str | None = None) -> list[Any]:
        return [cls.info() for cls in self.list(stage)]

    def discover(self, extra_modules: Iterable[str] = ()) -> None:
        if self._discovered:
            return
        self._discovered = True
        for module in (*BUILTIN_MODULES, *extra_modules):
            try:
                importlib.import_module(module)
            except Exception as error:  # a broken optional provider must not take the app down
                log.warning("skipping provider module %s: %s", module, error)
        # Display order is the module list's, whatever order imports registered them in (a
        # provider built on another's class registers that one first).
        order = {module: index for index, module in enumerate(BUILTIN_MODULES)}
        for stage, providers in self._providers.items():
            self._providers[stage] = dict(sorted(providers.items(), key=lambda item: order.get(item[1].__module__, len(order))))
        try:
            entry_points = importlib.metadata.entry_points(group="urakatsu.providers")
        except TypeError:  # pragma: no cover
            entry_points = []
        for entry_point in entry_points:
            try:
                self.add(entry_point.load())
            except Exception as error:
                log.warning("skipping provider entry point %s: %s", entry_point.name, error)


registry = Registry()


def register(cls: type["Provider"]) -> type["Provider"]:
    """Class decorator that adds a provider to the global registry."""
    return registry.add(cls)
