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

Next (an Urakatsu addition; Koharu's scheduler has no way to reorder a run): the user
can put a page first in a running job (``prioritize``). It starts once the stage running
now finishes, ahead of the page in flight, which waits part done. Its stages run again
when they had finished (forced, as Redo), it joins the run when it was not in it, and
pages put first run in the order they were asked for. When they are done the run goes
on in project order where it left off.
"""

from __future__ import annotations

from enum import Enum

from urakatsu.core.stage import PREREQUISITES, Stage

_ORDER = {stage: i for i, stage in enumerate(Stage)}


class WorkState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"


class _PageWork:
    def __init__(self, page: str, stages: list[Stage]):
        self.page = page
        self.stages = list(stages)
        self.state = {stage: WorkState.PENDING for stage in stages}
        self.active = False  # counted in the scheduler's pages in flight
        self.forced: set[Stage] = set()  # stages put back by a forced Next (Redo)
        self.again: set[Stage] = set()  # running stages to run once more when they finish

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
        self._next: list[_PageWork] = []  # pages put first (``prioritize``), in the order asked
        self.total = len(pages) * len(stages)
        self.completed = 0

    def start_next(self, busy: set[Stage]) -> tuple[str, Stage] | None:
        self._next = [work for work in self._next if not work.finished()]
        if self._next:
            # A page put first holds the lane until it is done; the page in flight waits.
            return self._take(self._next[0], busy)
        for index in range(self._head, len(self._pages)):
            work = self._pages[index]
            if not work.active and work.started():
                continue  # finished, or put first and already done
            if not work.active and self._active_pages >= self._window:
                break
            job = self._take(work, busy)
            if job is not None:
                return job
        return None

    def _take(self, work: _PageWork, busy: set[Stage]) -> tuple[str, Stage] | None:
        for stage in work.stages:
            if work.state[stage] is WorkState.PENDING and stage not in busy and work.ready(stage):
                if not work.active:
                    work.active = True
                    self._active_pages += 1
                work.state[stage] = WorkState.RUNNING
                return work.page, stage
        return None

    def forced(self, page: str, stage: Stage) -> bool:
        """Whether this stage on this page was put back by a forced Next."""
        index = self._index.get(page)
        return index is not None and stage in self._pages[index].forced

    def prioritize(self, page: str, stages: list[Stage], force: bool) -> list[Stage]:
        """Puts ``page`` first: its ``stages`` run as soon as the running stage finishes.
        A finished stage runs again, a stage running now runs once more after it, and a page
        outside the run joins it. ``force`` redoes stages whose results are there (Redo).
        Returns the stages queued again or added (counted in ``total``)."""
        index = self._index.get(page)
        if index is None:
            work = _PageWork(page, [])
            self._index[page] = len(self._pages)
            self._pages.append(work)
        else:
            work = self._pages[index]
        queued: list[Stage] = []
        for stage in stages:
            state = work.state.get(stage)
            if state is None:
                work.stages.append(stage)
                work.state[stage] = WorkState.PENDING
                queued.append(stage)
            elif state is WorkState.FINISHED:
                work.state[stage] = WorkState.PENDING
                queued.append(stage)
            elif state is WorkState.RUNNING and force and stage not in work.again:
                work.again.add(stage)
                queued.append(stage)
            if force:
                work.forced.add(stage)
        work.stages.sort(key=_ORDER.__getitem__)
        self.total += len(queued)
        if work not in self._next and not work.finished():
            self._next.append(work)
        return queued

    def complete(self, page: str, stage: Stage) -> bool:
        """Marks a stage finished; returns True when the whole page just finished."""
        index = self._index.get(page)
        if index is None:
            return False
        work = self._pages[index]
        if stage in work.again:  # put back while it ran: it runs once more
            work.again.discard(stage)
            work.state[stage] = WorkState.PENDING
            self.completed += 1
            return False
        if work.state.get(stage) is not WorkState.FINISHED:
            work.state[stage] = WorkState.FINISHED
            self.completed += 1
        page_finished = work.active and work.finished()
        if page_finished:
            work.active = False
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
        work.again.clear()
        dropped = [stage for stage, state in work.state.items() if state is WorkState.PENDING]
        for stage in dropped:
            self.complete(page, stage)
        return dropped

    def done(self) -> bool:
        return all(work.finished() for work in self._pages)

    def running(self) -> bool:
        return any(state is WorkState.RUNNING for work in self._pages for state in work.state.values())
