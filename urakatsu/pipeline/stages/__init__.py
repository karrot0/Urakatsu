"""Stage processors: the stage logic over the scene, one module per stage."""

from __future__ import annotations

from urakatsu.core.stage import Stage

from .base import InpaintingMask, StageInput, StageProcessor, finish, generation


def processors() -> dict[Stage, StageProcessor]:
    from .detection import DetectionStage
    from .inpainting import InpaintingStage
    from .ocr import OcrStage
    from .sound_effects import SoundEffectsStage
    from .translation import TranslationStage

    return {
        Stage.DETECTION: DetectionStage(),
        Stage.OCR: OcrStage(),
        Stage.TRANSLATION: TranslationStage(),
        Stage.INPAINTING: InpaintingStage(),
        Stage.SOUND_EFFECTS: SoundEffectsStage(),
    }


__all__ = ["InpaintingMask", "StageInput", "StageProcessor", "finish", "generation", "processors"]
