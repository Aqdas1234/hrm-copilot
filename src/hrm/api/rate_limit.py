"""Per-user rate limit for the chat endpoints: a token bucket kept in memory.

Why: one chat message costs several LLM calls, and the Groq free tier allows about 30
requests per minute in total. Without a limit, one user (or a stuck client) could use
the quota for everyone.

A bucket holds up to `per_minute` tokens and refills continuously; each message takes one.
Short bursts are fine, a sustained flood is not.
In memory, so it is per process: running several API workers would need a shared store
(e.g. Redis). One process is the deployment here.
"""

import math
import time
from collections.abc import Callable
from threading import Lock

from hrm.core.errors import HRMError


class RateLimitedError(HRMError):
    def __init__(self, retry_after_s: int) -> None:
        super().__init__(f"Too many messages. Please wait {retry_after_s} s and try again.")
        self.retry_after_s = retry_after_s


class RateLimiter:
    def __init__(self, per_minute: int, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._capacity = float(per_minute)
        self._refill_per_s = per_minute / 60
        self._clock = clock
        self._buckets: dict[int, tuple[float, float]] = {}  # key -> (tokens, last update)
        self._lock = Lock()  # endpoints run in a thread pool

    def check(self, key: int) -> None:
        """Take one token for `key`, or raise RateLimitedError if none is left."""
        with self._lock:
            now = self._clock()
            tokens, last = self._buckets.get(key, (self._capacity, now))
            tokens = min(self._capacity, tokens + (now - last) * self._refill_per_s)
            if tokens < 1:
                self._buckets[key] = (tokens, now)
                raise RateLimitedError(math.ceil((1 - tokens) / self._refill_per_s))
            self._buckets[key] = (tokens - 1, now)
