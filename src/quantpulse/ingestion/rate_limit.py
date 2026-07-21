import threading
import time
from collections.abc import Callable


class SimpleRateLimiter:
    """Guarantees at least `min_interval_seconds` between consecutive `wait()` calls.

    The conservative choice for sources where the safe move is to never burst:
    an unofficial endpoint (yfinance) or a "fair-use, no published number"
    source (SEC EDGAR). Thread-safe so callers can share one instance across
    a thread pool.
    """

    def __init__(self, min_interval_seconds: float) -> None:
        self._min_interval = min_interval_seconds
        self._lock = threading.Lock()
        self._last_call: float | None = None

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._last_call is not None:
                remaining = self._min_interval - (now - self._last_call)
                if remaining > 0:
                    time.sleep(remaining)
            self._last_call = time.monotonic()


class TokenBucketRateLimiter:
    """Classic token bucket matched to a provider's documented per-window limit.

    Refills at `capacity / per_seconds` tokens per second up to `capacity`.
    Unlike a fixed min-interval, this lets a caller spend its full quota in a
    burst when the bucket is full (e.g. blow through a batch of Finnhub calls
    up to 60/min) and only then throttle to the sustainable rate — using the
    whole free allowance rather than artificially serializing every call —
    while still never exceeding the window limit. Thread-safe.

    Exposes `wait()` with the same contract as SimpleRateLimiter so the two
    are interchangeable at call sites (and equally easy to neutralize in tests).
    """

    def __init__(
        self,
        capacity: int,
        per_seconds: float,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if per_seconds <= 0:
            raise ValueError("per_seconds must be positive")
        self._capacity = float(capacity)
        self._refill_per_second = capacity / per_seconds
        self._tokens = float(capacity)
        self._clock = clock
        self._updated_at = clock()
        self._lock = threading.Lock()
        self._sleep = sleep

    def _refill(self, now: float) -> None:
        elapsed = now - self._updated_at
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_per_second)
            self._updated_at = now

    def wait(self) -> None:
        """Block until a token is available, then consume it."""
        with self._lock:
            self._refill(self._clock())
            if self._tokens < 1.0:
                deficit = 1.0 - self._tokens
                self._sleep(deficit / self._refill_per_second)
                self._refill(self._clock())
            self._tokens -= 1.0
