"""Progress events emitted while a request runs.

Koharu reports a stage only as started and finished. Urakatsu adds ``STEP`` events with a
live preview of a diffusion model's sampling steps (as ComfyUI's latent previews do). A
provider never learns who listens: the pipeline opens ``reporting_steps`` around a stage,
and whatever runs inside it (the stage, then the runtime) calls ``report_step``.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Callable, Iterator, Optional

from pydantic import BaseModel, ConfigDict, Field

from .stage import Stage

if TYPE_CHECKING:
    from PIL import Image


class ProgressKind(str, Enum):
    STARTED = "started"
    LOADING = "loading"
    RUNNING = "running"
    STEP = "step"
    FINISHED = "finished"
    SKIPPED = "skipped"
    FAILED = "failed"
    STOPPED = "stopped"
    COMPLETED = "completed"


class Progress(BaseModel):
    model_config = ConfigDict(ser_json_bytes="base64")

    kind: ProgressKind
    page: Optional[str] = None
    stage: Optional[Stage] = None
    model: Optional[str] = None
    elapsed_ms: Optional[float] = None
    message: str = ""
    completed: int = 0
    total: int = 0
    pages: list[str] = Field(default_factory=list)
    stages: list[Stage] = Field(default_factory=list)
    # STEP only: the sampling step, and a JPEG of the result so far placed at ``preview_box``
    # (x, y, width, height in page pixels). Previews are transient and never stored.
    step: int = 0
    steps: int = 0
    preview: Optional[bytes] = None
    preview_box: Optional[tuple[int, int, int, int]] = None


ProgressSink = Callable[[Progress], None]


@dataclass(frozen=True)
class StepPreview:
    step: int
    total: int
    image: Optional["Image.Image"] = None
    box: Optional[tuple[int, int, int, int]] = None

    def placed(self, image: "Image.Image", box: tuple[int, int, int, int]) -> "StepPreview":
        return replace(self, image=image, box=box)


StepSink = Callable[[StepPreview], None]

_steps: ContextVar[Optional[StepSink]] = ContextVar("urakatsu_steps", default=None)


@contextmanager
def reporting_steps(sink: Optional[StepSink]) -> Iterator[None]:
    """Routes ``report_step`` calls made in this context (and threads started from it) to ``sink``."""
    token = _steps.set(sink)
    try:
        yield
    finally:
        _steps.reset(token)


def step_sink() -> Optional[StepSink]:
    return _steps.get()


def report_step(preview: StepPreview) -> None:
    sink = _steps.get()
    if sink is not None:
        sink(preview)
