import threading
import time

from quantpulse.ingestion.rate_limit import SimpleRateLimiter


def test_wait_enforces_minimum_interval() -> None:
    limiter = SimpleRateLimiter(min_interval_seconds=0.2)
    start = time.monotonic()

    limiter.wait()
    limiter.wait()

    assert time.monotonic() - start >= 0.2


def test_wait_does_not_block_once_interval_has_already_elapsed() -> None:
    limiter = SimpleRateLimiter(min_interval_seconds=0.05)
    limiter.wait()
    time.sleep(0.1)

    start = time.monotonic()
    limiter.wait()

    assert time.monotonic() - start < 0.05


def test_concurrent_waits_still_serialize_to_the_minimum_interval() -> None:
    # `min_interval_seconds`'s docstring claims "callers can share one
    # instance across a thread pool" -- release every thread at once so they
    # all race `wait()`'s read-then-write of `_last_call` simultaneously. If
    # that weren't lock-protected, several threads could read the same stale
    # `_last_call` and pass straight through, and the whole batch would
    # finish in well under the spacing this asserts.
    min_interval = 0.05
    limiter = SimpleRateLimiter(min_interval_seconds=min_interval)
    n_threads = 8
    barrier = threading.Barrier(n_threads)

    def worker() -> None:
        barrier.wait()
        limiter.wait()

    start = time.monotonic()
    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - start

    assert elapsed >= min_interval * (n_threads - 1)
