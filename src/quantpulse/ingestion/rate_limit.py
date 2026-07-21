import threading
import time


class SimpleRateLimiter:
    """Guarantees at least `min_interval_seconds` between consecutive `wait()` calls.

    Deliberately basic: one shared cooldown per instance, no per-endpoint
    token bucket or circuit breaker. Enough to keep a single client under
    its provider's documented rate limit; thread-safe so callers can share
    one instance across a thread pool.
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
