"""Pipeline stages and the operations that select them.

A stage runs on one page: detection, then OCR and translation, or detection, then
inpainting. Rendering is not a stage: the renderer prepares a frame from the
current scene snapshot on demand.

``SOUND_EFFECTS`` is not in Koharu: with sound effects set to ``redraw`` it redraws each
translated sound effect with an image model, so it waits for translation and inpainting.
In any other mode a run leaves it out (``pipeline.py``).

``WATERMARKS`` is not in Koharu either: with watermarks set to ``remove`` it removes repeated
watermarks before detection reads the page (``pipeline/stages/watermarks.py``), and a run that
detects gets it in front. Otherwise a run leaves it out.

``CENSOR_DETECTION`` and ``DECENSOR`` are not in Koharu either (``pipeline/stages/censorship.py``).
With censorship set to ``detect`` a run maps the bars into a reviewable mask; with ``remove`` it
also reconstructs them. Censor detection runs before detection, because both write the page's
assets; Decensor runs after inpainting, over the cleaned page. With ``detect``, Decensor runs only
when asked for by itself. With ``off`` a run leaves both out.

``COLORIZATION`` is not in Koharu either (``pipeline/stages/colorization.py``). It runs after
Decensor, over the cleaned page, and before sound effects, so a redraw sees the colors. With
colorization ``off`` a run leaves it out.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable


class Stage(str, Enum):
    WATERMARKS = "watermarks"
    CENSOR_DETECTION = "censor_detection"
    DETECTION = "detection"
    OCR = "ocr"
    TRANSLATION = "translation"
    INPAINTING = "inpainting"
    DECENSOR = "decensor"
    COLORIZATION = "colorization"
    SOUND_EFFECTS = "sound_effects"

    @classmethod
    def all(cls) -> list["Stage"]:
        return [cls.WATERMARKS, cls.CENSOR_DETECTION, cls.DETECTION, cls.OCR, cls.TRANSLATION, cls.INPAINTING, cls.DECENSOR, cls.SOUND_EFFECTS]

    @classmethod
    def parse(cls, value: "str | Stage") -> "Stage":
        if isinstance(value, Stage):
            return value
        try:
            return cls(value.strip().lower())
        except ValueError as error:
            names = [s.value for s in cls]
            raise ValueError(f"unknown stage {value!r}; expected one of {names}") from error


# scheduler.rs ``prerequisite``: OCR and inpainting need detection, translation needs OCR.
# Detection also waits for the watermarks stage when a run has it.
PREREQUISITES: dict[Stage, tuple[Stage, ...]] = {
    Stage.WATERMARKS: (),
    Stage.CENSOR_DETECTION: (Stage.WATERMARKS,),
    Stage.DETECTION: (Stage.WATERMARKS, Stage.CENSOR_DETECTION),
    Stage.OCR: (Stage.DETECTION,),
    Stage.TRANSLATION: (Stage.OCR,),
    Stage.INPAINTING: (Stage.DETECTION,),
    Stage.DECENSOR: (Stage.CENSOR_DETECTION, Stage.INPAINTING),
    Stage.COLORIZATION: (Stage.INPAINTING, Stage.DECENSOR),
    Stage.SOUND_EFFECTS: (Stage.TRANSLATION, Stage.INPAINTING, Stage.DECENSOR, Stage.COLORIZATION),
}


class Operation:
    """``Operation``: full | through(stage) | only(stage) | stages([...])."""

    __slots__ = ("kind", "stage", "stages")

    def __init__(self, kind: str = "full", stage: Stage | None = None, stages: Iterable[Stage] | None = None):
        self.kind = kind
        self.stage = stage
        self.stages = list(stages or [])

    @classmethod
    def full(cls) -> "Operation":
        return cls("full")

    @classmethod
    def through(cls, stage: Stage) -> "Operation":
        return cls("through", stage=stage)

    @classmethod
    def only(cls, stage: Stage) -> "Operation":
        return cls("only", stage=stage)

    @classmethod
    def select(cls, stages: Iterable[Stage]) -> "Operation":
        return cls("stages", stages=stages)

    def resolve(self) -> list[Stage]:
        if self.kind == "full":
            selected = Stage.all()
        elif self.kind == "through":
            selected = {
                Stage.WATERMARKS: [Stage.WATERMARKS],
                Stage.CENSOR_DETECTION: [Stage.CENSOR_DETECTION],
                Stage.DETECTION: [Stage.DETECTION],
                Stage.OCR: [Stage.DETECTION, Stage.OCR],
                Stage.TRANSLATION: [Stage.DETECTION, Stage.OCR, Stage.TRANSLATION],
                Stage.INPAINTING: [Stage.DETECTION, Stage.INPAINTING],
                Stage.DECENSOR: [Stage.CENSOR_DETECTION, Stage.DECENSOR],
                Stage.COLORIZATION: [Stage.DETECTION, Stage.INPAINTING, Stage.COLORIZATION],
                Stage.SOUND_EFFECTS: Stage.all(),
            }[Stage.parse(self.stage)]  # type: ignore[arg-type]
        elif self.kind == "only":
            selected = [Stage.parse(self.stage)]  # type: ignore[arg-type]
        elif self.kind == "stages":
            selected = [stage for stage in Stage.all() if stage in self.stages]
        else:
            raise ValueError(f"unknown operation {self.kind!r}")
        if not selected:
            raise ValueError("at least one pipeline stage must be selected")
        return selected

    def __repr__(self) -> str:  # pragma: no cover
        return f"Operation({self.kind}, stage={self.stage}, stages={self.stages})"
