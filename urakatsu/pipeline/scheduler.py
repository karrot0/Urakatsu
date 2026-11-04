"""Per-page stage scheduler.

Pages enter in project order. A stage on a page is eligible once its
prerequisites on that page have finished, and at most one job per stage runs at
a time.

Every accelerated job here takes the one admission lane, so nothing overlaps and the
interleaving only switches models more often and finishes pages later. Here a
page's stages all finish before the next page starts (D1 R1 I1 T1, D2 ...).
Measured on four pages with the models resident (detection, OCR, LaMa): the
same total time as stage-by-stage, the first page done at ~24 s instead of
~35 s. Automations keep their stage-major batches (``pipeline/batch.py``) for
models too large to sit together.
"""

from __future__ import annotations

from enum import Enum

from urakatsu.core.stage import PREREQUISITES, Stage


class WorkState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"


class _PageWork:
    def __init__(self, page: str, stages: list[Stage]):
        self.page = page
        self.stages = list(stages)
        self.state = {stage: WorkState.PENDING for stage in stages}

    def started(self) -> bool:
        return any(state is not WorkState.PENDING for state in self.state.values())

    def finished(self) -> bool:
        return all(state is WorkState.FINISHED for state in self.state.values())

    def ready(self, stage: Stage) -> bool:
        for prerequisite in PREREQUISITES[stage]:
            # A prerequisite outside this run is assumed satisfied by earlier work.
            if prerequisite in self.state and self.state[prerequisite] is not WorkState.FINISHED:
                return False
        return True


class Scheduler:
    def __init__(self, pages: list[str], stages: list[Stage]):
        self._pages = [_PageWork(page, stages) for page in pages]
        self._index = {work.page: i for i, work in enumerate(self._pages)}
        self._window = 1  # pages in flight: page by page, see the module docstring
        self._active_pages = 0
        self._head = 0
        self.total = len(pages) * len(stages)
        self.completed = 0

    def start_next(self, busy: set[Stage]) -> tuple[str, Stage] | None:
        for index in range(self._head, len(self._pages)):
            work = self._pages[index]
            started = work.started()
            if not started and self._active_pages >= self._window:
                break
            for stage in work.stages:
                if work.state[stage] is WorkState.PENDING and stage not in busy and work.ready(stage):
                    if not started:
                        self._active_pages += 1
                    work.state[stage] = WorkState.RUNNING
                    return work.page, stage
        return None

    def complete(self, page: str, stage: Stage) -> bool:
        """Marks a stage finished; returns True when the whole page just finished."""
        index = self._index.get(page)
        if index is None:
            return False
        work = self._pages[index]
        was_finished = work.finished()
        if work.state.get(stage) is not WorkState.FINISHED:
            work.state[stage] = WorkState.FINISHED
            self.completed += 1
        page_finished = not was_finished and work.finished()
        if page_finished:
            self._active_pages = max(0, self._active_pages - 1)
            while self._head < len(self._pages) and self._pages[self._head].finished():
                self._head += 1
        return page_finished

    def fail_page(self, page: str) -> list[Stage]:
        """Finishes every pending stage of a page after a failure; returns the stages dropped."""
        index = self._index.get(page)
        if index is None:
            return []
        work = self._pages[index]
        dropped = [stage for stage, state in work.state.items() if state is WorkState.PENDING]
        for stage in dropped:
            self.complete(page, stage)
        return dropped

    def done(self) -> bool:
        return all(work.finished() for work in self._pages)

    def running(self) -> bool:
        return any(state is WorkState.RUNNING for work in self._pages for state in work.state.values())
