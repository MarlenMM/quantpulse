import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Literal

State = Literal["closed", "open", "half_open"]


class CircuitOpenError(RuntimeError):
    """Raised instead of making a call while a source's circuit is open."""

    def __init__(self, name: str) -> None:
        super().__init__(f"circuit breaker for {name!r} is open; skipping call")
        self.name = name


class CircuitBreaker:
    """Stops hammering a data source that is sustainably failing (Section 6.12).

    A single 429 is a rate-limit blip the retry layer handles; a *pattern* of
    failures means the source is down, and retrying it 500 times across 500
    tickers would burn the whole job's time budget for nothing. After
    `failure_threshold` consecutive failures the circuit opens and further
    calls short-circuit (raising `CircuitOpenError`) until `reset_timeout`
    elapses, at which point one trial call is allowed: success closes the
    circuit, failure re-opens it.

    Thread-safe: one instance is shared across the nightly job's thread pool.
    The clock is injectable so tests don't sleep.
    """

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        reset_timeout_seconds: float = 300.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        import time

        self.name = name
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout_seconds
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._failure_count = 0
        self._opened_at: float | None = None
        self._state: State = "closed"

    @property
    def state(self) -> State:
        with self._lock:
            return self._current_state()

    def _current_state(self) -> State:
        if self._state == "open" and self._opened_at is not None:
            if self._clock() - self._opened_at >= self._reset_timeout:
                self._state = "half_open"
        return self._state

    def before_call(self) -> None:
        with self._lock:
            if self._current_state() == "open":
                raise CircuitOpenError(self.name)

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._opened_at = None
            self._state = "closed"

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            # A failure in half-open (the trial call) re-opens immediately;
            # in closed, open once the threshold of consecutive failures is hit.
            if self._state == "half_open" or self._failure_count >= self._failure_threshold:
                self._state = "open"
                self._opened_at = self._clock()

    @contextmanager
    def guard(self) -> Iterator[None]:
        """Wrap a call: raise if open, record the outcome, re-raise on failure."""
        self.before_call()
        try:
            yield
        except Exception:
            self.record_failure()
            raise
        else:
            self.record_success()


_registry: dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_breaker(name: str) -> CircuitBreaker:
    """Return the shared circuit breaker for a data source, creating it on first use."""
    with _registry_lock:
        breaker = _registry.get(name)
        if breaker is None:
            breaker = CircuitBreaker(name)
            _registry[name] = breaker
        return breaker


def reset_all_breakers() -> None:
    """Clear the registry — used to keep tests isolated from each other."""
    with _registry_lock:
        _registry.clear()
