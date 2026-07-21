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
