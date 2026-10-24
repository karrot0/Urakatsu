"""Keeping the computer awake for unattended work, and waking it for scheduled work.

``keep_awake`` holds off system sleep while a batch runs (the display may still turn off).
``WakeTask`` keeps one Windows Task Scheduler entry per data folder whose trigger is the
next scheduled automation that asked to wake the computer; it wakes the machine shortly
before that time and, when given a launch command, starts Urakatsu if it is not running.
Waking is Windows-only; elsewhere ``WakeTask.supported`` is False.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional
from xml.sax.saxutils import escape

from .config import home

log = logging.getLogger(__name__)

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001

# The task fires this long before the automation, so the service is up when it is due.
WAKE_LEAD_SECONDS = 90

_holds = 0
_holds_lock = threading.Lock()
_caffeinate: Optional[subprocess.Popen] = None


def _windows() -> bool:
    return sys.platform == "win32"


def _hold(on: bool) -> None:
    global _caffeinate
    if _windows():
        # The request belongs to the calling thread; batches run on the service's event loop thread.
        flags = _ES_CONTINUOUS | (_ES_SYSTEM_REQUIRED if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags))
    elif sys.platform == "darwin":
        if on and _caffeinate is None:
            try:
                _caffeinate = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
            except OSError as error:
                log.warning("cannot keep the computer awake: %s", error)
        elif not on and _caffeinate is not None:
            _caffeinate.terminate()
            _caffeinate = None


@contextmanager
def keep_awake() -> Iterator[None]:
    """Keeps the system from sleeping while the block runs; nested holds are counted."""
    global _holds
    with _holds_lock:
        _holds += 1
        if _holds == 1:
            _hold(True)
    try:
        yield
    finally:
        with _holds_lock:
            _holds -= 1
            if _holds == 0:
                _hold(False)


def task_xml(when: datetime, command: str, arguments: str) -> str:
    """A Task Scheduler definition: one time trigger at ``when`` (local time) that may wake the computer."""
    start = when.replace(microsecond=0).isoformat()
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Wakes the computer for a scheduled Urakatsu automation.</Description>
  </RegistrationInfo>
  <Triggers>
    <TimeTrigger>
      <StartBoundary>{start}</StartBoundary>
      <Enabled>true</Enabled>
    </TimeTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <WakeToRun>true</WakeToRun>
    <ExecutionTimeLimit>PT10M</ExecutionTimeLimit>
    <Enabled>true</Enabled>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{escape(command)}</Command>
      <Arguments>{escape(arguments)}</Arguments>
    </Exec>
  </Actions>
</Task>
"""


_UNKNOWN = object()


class WakeTask:
    """The Task Scheduler entry that wakes the computer for the next automation."""

    def __init__(self, root: Optional[Path] = None):
        root = Path(root or home()).resolve()
        digest = hashlib.sha1(str(root).lower().encode("utf-8")).hexdigest()[:8]
        self.name = f"Urakatsu\\Automations-{digest}"
        self.root = root
        # The program and arguments that start Urakatsu in the background; None only wakes.
        self.launch: Optional[tuple[str, list[str]]] = None
        # What the task was last set to; unknown at first, so a task left from before is replaced.
        self._scheduled: object = _UNKNOWN

    @property
    def supported(self) -> bool:
        return _windows()

    def action(self) -> tuple[str, str]:
        """The task's command line. ``cmd /c start`` detaches the app, so the task itself ends at once."""
        comspec = os.environ.get("COMSPEC", "cmd.exe")
        if self.launch is None:
            return comspec, "/c exit 0"
        program, args = self.launch
        quoted = " ".join(subprocess.list2cmdline([part]) for part in [program, *args])
        return comspec, f'/c set "URAKATSU_HOME={self.root}" && start "" {quoted}'

    def schedule(self, when: Optional[float]) -> None:
        """Sets the task to fire before ``when`` (epoch seconds), or removes it for None."""
        if not self.supported:
            return
        if when is not None:
            when = max(when - WAKE_LEAD_SECONDS, 0.0)
        if when == self._scheduled:
            return
        try:
            if when is None:
                self._run(["schtasks", "/Delete", "/TN", self.name, "/F"], quiet=True)
            else:
                command, arguments = self.action()
                definition = task_xml(datetime.fromtimestamp(when), command, arguments)
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "task.xml"
                    path.write_text(definition, encoding="utf-16")
                    self._run(["schtasks", "/Create", "/TN", self.name, "/XML", str(path), "/F"])
            self._scheduled = when
        except (OSError, subprocess.CalledProcessError) as error:
            log.warning("cannot update the wake task %s: %s", self.name, error)

    def _run(self, args: list[str], quiet: bool = False) -> None:
        completed = subprocess.run(args, capture_output=True, text=True, creationflags=0x08000000)  # CREATE_NO_WINDOW
        if completed.returncode != 0 and not quiet:
            raise subprocess.CalledProcessError(completed.returncode, args, completed.stdout, completed.stderr)


__all__ = ["WAKE_LEAD_SECONDS", "WakeTask", "keep_awake", "task_xml"]
