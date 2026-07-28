import threading

import pytest

from quantpulse.ingestion.rate_limit import TokenBucketRateLimiter


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_rejects_non_positive_config() -> None:
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(capacity=0, per_seconds=60)
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(capacity=10, per_seconds=0)


def test_full_bucket_lets_a_burst_through_without_sleeping() -> None:
    slept: list[float] = []
    clock = _FakeClock()
    bucket = TokenBucketRateLimiter(capacity=5, per_seconds=5, sleep=slept.append, clock=clock)

    for _ in range(5):
        bucket.wait()

    assert slept == []  # a full bucket spends its whole allowance immediately


def test_empty_bucket_sleeps_for_one_refill_interval() -> None:
    slept: list[float] = []
    clock = _FakeClock()
    # capacity 2 over 2s -> 1 token/sec refill.
    bucket = TokenBucketRateLimiter(capacity=2, per_seconds=2, sleep=slept.append, clock=clock)

    bucket.wait()  # 2 -> 1
    bucket.wait()  # 1 -> 0
    bucket.wait()  # empty: must wait ~1s for one token at 1 token/sec

    assert len(slept) == 1
    assert slept[0] == pytest.approx(1.0)


def test_tokens_refill_over_elapsed_time() -> None:
    slept: list[float] = []
    clock = _FakeClock()
    bucket = TokenBucketRateLimiter(capacity=2, per_seconds=2, sleep=slept.append, clock=clock)

    bucket.wait()
    bucket.wait()  # bucket now empty
    clock.now += 5.0  # plenty of time passes

    bucket.wait()  # refilled -> no sleep
    assert slept == []


def test_concurrent_waits_never_double_spend_the_same_token() -> None:
    # A frozen clock means zero refill for the whole test: exactly one of the
    # bucket's single starting token is available. Without `wait`'s lock,
    # two threads could both read `_tokens == 1.0` before either decrements
    # it, letting more than one caller through without sleeping -- the token
    # gets spent twice. `len(slept)` is a purely public-API-observable
    # count of how many callers did NOT get a free pass, so this doesn't
    # need to peek at `_tokens` directly.
    clock = _FakeClock()
    slept: list[float] = []
    n_threads = 30
    bucket = TokenBucketRateLimiter(capacity=1, per_seconds=1000, sleep=slept.append, clock=clock)
    barrier = threading.Barrier(n_threads)

    def worker() -> None:
        barrier.wait()
        bucket.wait()

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(slept) == n_threads - 1
