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
