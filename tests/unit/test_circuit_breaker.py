import pytest

from quantpulse.ingestion.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    get_breaker,
    reset_all_breakers,
)


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _breaker(clock: _FakeClock, **kw: object) -> CircuitBreaker:
    return CircuitBreaker("test", clock=clock, **kw)  # type: ignore[arg-type]


def test_stays_closed_below_the_failure_threshold() -> None:
    clock = _FakeClock()
    breaker = _breaker(clock, failure_threshold=3)

    breaker.record_failure()
    breaker.record_failure()

    assert breaker.state == "closed"
    breaker.before_call()  # does not raise


def test_opens_after_consecutive_failures_and_short_circuits() -> None:
    clock = _FakeClock()
    breaker = _breaker(clock, failure_threshold=3)

    for _ in range(3):
        breaker.record_failure()

    assert breaker.state == "open"
    with pytest.raises(CircuitOpenError):
        breaker.before_call()


def test_a_success_resets_the_failure_count() -> None:
    clock = _FakeClock()
    breaker = _breaker(clock, failure_threshold=3)

    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()

    assert breaker.state == "closed"  # count restarted, threshold not reached


def test_transitions_to_half_open_after_the_reset_timeout() -> None:
    clock = _FakeClock()
    breaker = _breaker(clock, failure_threshold=1, reset_timeout_seconds=60)

    breaker.record_failure()
    assert breaker.state == "open"

    clock.now += 61
    assert breaker.state == "half_open"
    breaker.before_call()  # half-open allows a trial call through


def test_half_open_success_closes_and_failure_reopens() -> None:
    clock = _FakeClock()
    breaker = _breaker(clock, failure_threshold=1, reset_timeout_seconds=60)

    breaker.record_failure()
    clock.now += 61
    assert breaker.state == "half_open"
    breaker.record_success()
    assert breaker.state == "closed"

    breaker.record_failure()  # open again
    clock.now += 61
    assert breaker.state == "half_open"
    breaker.record_failure()  # trial failed -> straight back to open
    assert breaker.state == "open"


def test_guard_records_success_on_clean_exit() -> None:
    clock = _FakeClock()
    breaker = _breaker(clock, failure_threshold=1)
    with breaker.guard():
        pass
    assert breaker.state == "closed"


def test_guard_records_failure_and_reraises() -> None:
    clock = _FakeClock()
    breaker = _breaker(clock, failure_threshold=1)

    with pytest.raises(ValueError):
        with breaker.guard():
            raise ValueError("boom")

    assert breaker.state == "open"


def test_registry_returns_the_same_instance_per_name() -> None:
    reset_all_breakers()
    assert get_breaker("finnhub") is get_breaker("finnhub")
    assert get_breaker("finnhub") is not get_breaker("fred")


def test_reset_all_breakers_clears_registry_state() -> None:
    first = get_breaker("finnhub")
    reset_all_breakers()
    assert get_breaker("finnhub") is not first
