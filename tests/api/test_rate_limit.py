import pytest

from hrm.api.rate_limit import RateLimitedError, RateLimiter


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_a_burst_up_to_the_limit_is_allowed_then_refused() -> None:
    limiter = RateLimiter(3, clock=Clock())
    for _ in range(3):
        limiter.check(1)
    with pytest.raises(RateLimitedError) as exc:
        limiter.check(1)
    assert exc.value.retry_after_s == 20  # one token per 20 s at 3/minute


def test_tokens_refill_over_time() -> None:
    clock = Clock()
    limiter = RateLimiter(3, clock=clock)
    for _ in range(3):
        limiter.check(1)
    clock.now = 20
    limiter.check(1)  # one token back


def test_users_have_separate_buckets() -> None:
    limiter = RateLimiter(1, clock=Clock())
    limiter.check(1)
    limiter.check(2)
    with pytest.raises(RateLimitedError):
        limiter.check(1)
