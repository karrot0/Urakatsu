"""Pipeline stages and the operations that select them.

A stage runs on one page: detection, then OCR and translation, or detection, then
inpainting. Rendering is not a stage: the renderer prepares a frame from the
current scene snapshot on demand.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable


class Stage(str, Enum):
    DETECTION = "detection"
    OCR = "ocr"
    TRANSLATION = "translation"
    INPAINTING = "inpainting"

    @classmethod
    def all(cls) -> list["Stage"]:
        return [cls.DETECTION, cls.OCR, cls.TRANSLATION, cls.INPAINTING]

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
PREREQUISITES: dict[Stage, tuple[Stage, ...]] = {
    Stage.DETECTION: (),
    Stage.OCR: (Stage.DETECTION,),
    Stage.TRANSLATION: (Stage.OCR,),
    Stage.INPAINTING: (Stage.DETECTION,),
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
                Stage.DETECTION: [Stage.DETECTION],
                Stage.OCR: [Stage.DETECTION, Stage.OCR],
                Stage.TRANSLATION: [Stage.DETECTION, Stage.OCR, Stage.TRANSLATION],
                Stage.INPAINTING: [Stage.DETECTION, Stage.INPAINTING],
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
