import logging
import os
import uuid

_RUN_ID: str | None = None
_LOG_FORMAT = "%(asctime)s [%(levelname)s] [run=%(run_id)s] %(name)s: %(message)s"


def get_run_id() -> str:
    """A stable correlation ID for this process's run (Section 6.14).

    Prefers the CI run identifier so every log line from one nightly job can
    be pulled together after the fact; falls back to a generated id locally.
    Computed once and reused for the life of the process.
    """
    global _RUN_ID
    if _RUN_ID is None:
        _RUN_ID = os.environ.get("GITHUB_RUN_ID") or uuid.uuid4().hex[:12]
    return _RUN_ID


class _RunIdFilter(logging.Filter):
    def __init__(self, run_id: str) -> None:
        super().__init__()
        self._run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self._run_id
        return True


def configure_logging(level: str | int = "INFO", run_id: str | None = None) -> str:
    """Install a root handler that tags every line with the run ID. Idempotent.

    Returns the run ID in use so a caller can echo it once at startup.
    """
    resolved_run_id = run_id or get_run_id()
    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.addFilter(_RunIdFilter(resolved_run_id))

    # Replace any handler we previously installed so re-invocation (e.g. tests,
    # or a script calling this twice) doesn't stack duplicate output.
    root.handlers = [h for h in root.handlers if not getattr(h, "_quantpulse", False)]
    handler._quantpulse = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    return resolved_run_id
