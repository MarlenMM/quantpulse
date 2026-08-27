"""Starting and watching a data refresh from the UI (Section 12).

Nothing refreshes on a schedule, so the button on the Settings page is the
trigger. That makes *how* the job is started a real design question rather than
plumbing, and two obvious-looking options are both wrong here:

* **Calling `run()` inline** blocks the Streamlit script run for the length of a
  full refresh -- minutes on a daily run, hours on a weekly one -- with no
  output until it ends.
* **Calling it in a background thread** breaks the job outright. `refresh_data`
  gives every step a timeout built on `signal.alarm`, which only works in the
  main thread of the main interpreter; off it, every step would raise instead of
  running and the whole refresh would collapse into a pile of failed steps. A
  thread would also load the refresh's ~2.5 GB of model weights into the
  Streamlit process and never give them back.

So the refresh runs as a subprocess -- the same command the workflow runs -- and
this module owns the handle. Its output is drained on a reader thread into a
bounded buffer so the page can tail it live, and the runner is a *process-wide*
singleton rather than per-session state: two browser tabs are still one SQLite
file, and SQLite would not thank us for two concurrent writers.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "refresh_data.py"

# Enough to see what a run is doing and where it stopped, bounded so a job that
# logs a line per ticker cannot grow the Streamlit process without limit.
MAX_LOG_LINES = 500


@dataclass(frozen=True)
class RefreshState:
    """An immutable snapshot of the runner, safe to read during a script run."""

    running: bool
    started_at: datetime | None
    finished_at: datetime | None
    returncode: int | None
    log: tuple[str, ...]

    @property
    def never_run(self) -> bool:
        return self.started_at is None

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


class RefreshRunner:
    """Owns at most one `scripts/refresh_data.py` subprocess at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._log: deque[str] = deque(maxlen=MAX_LOG_LINES)
        self._started_at: datetime | None = None
        self._finished_at: datetime | None = None
        self._returncode: int | None = None

    def start(self, *, weekly: bool = False, ignore_market_calendar: bool = False) -> bool:
        """Start a refresh. Returns False if one is already running.

        The two flags are `refresh_data.py`'s own, passed straight through
        rather than reinterpreted here: the script is the single place that
        decides what "weekly" and "closed today" mean.
        """
        with self._lock:
            if self._is_running():
                return False
            command = [sys.executable, str(SCRIPT)]
            if weekly:
                command.append("--force-weekly")
            if ignore_market_calendar:
                command.append("--ignore-market-calendar")
            # `cwd` is pinned to the repo root rather than inherited: the default
            # `DATABASE_URL` is the *relative* `sqlite:///./quantpulse.db`, so a
            # Streamlit process launched from anywhere else would refresh a
            # different (probably empty, freshly created) database than the one
            # the pages read. stderr is folded into stdout because
            # `configure_logging` installs a `StreamHandler` -- the run's entire
            # narration arrives on stderr, and merging keeps it in order.
            self._process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self._log.clear()
            self._started_at = datetime.now()
            self._finished_at = None
            self._returncode = None
            process = self._process
        threading.Thread(target=self._drain, args=(process,), daemon=True).start()
        return True

    def state(self) -> RefreshState:
        with self._lock:
            return RefreshState(
                running=self._is_running(),
                started_at=self._started_at,
                finished_at=self._finished_at,
                returncode=self._returncode,
                log=tuple(self._log),
            )

    def stop(self) -> bool:
        """Ask a running refresh to stop. Returns False if none is running."""
        with self._lock:
            process = self._process
            if not self._is_running() or process is None:
                return False
        # Outside the lock: the drain thread needs it to record the exit, and
        # holding it across the terminate would deadlock against that.
        process.terminate()
        return True

    def _is_running(self) -> bool:
        # Keyed on the recorded exit code, deliberately not on `Popen.poll()`.
        # `poll()` flips to "not running" the instant the child exits, which is
        # a moment *before* the drain thread has read the tail of its output and
        # recorded the return code -- so a page rendering in that window would
        # report a finished run with no outcome and a truncated log. The drain
        # thread always records an exit (its `finally` runs even if reading the
        # pipe raises), so this cannot latch on forever.
        return self._process is not None and self._returncode is None

    def _drain(self, process: subprocess.Popen[str]) -> None:
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    with self._lock:
                        # A newer run has replaced this one; stop writing into
                        # its buffer rather than interleaving two runs' logs.
                        if self._process is not process:
                            return
                        self._log.append(line.rstrip())
        finally:
            returncode = process.wait()
            with self._lock:
                if self._process is process:
                    self._returncode = returncode
                    self._finished_at = datetime.now()


@st.cache_resource(show_spinner=False)
def get_runner() -> RefreshRunner:
    """The one runner shared by every session in this Streamlit process."""
    return RefreshRunner()
