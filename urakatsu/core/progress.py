"""Progress events emitted while a request runs."""

from __future__ import annotations

from enum import Enum
from typing import Callable, Optional

from pydantic import BaseModel, Field

from .stage import Stage


class ProgressKind(str, Enum):
    STARTED = "started"
    LOADING = "loading"
    RUNNING = "running"
    FINISHED = "finished"
    SKIPPED = "skipped"
    FAILED = "failed"
    STOPPED = "stopped"
    COMPLETED = "completed"


class Progress(BaseModel):
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


ProgressSink = Callable[[Progress], None]
