"""Shared HTTP helper: GET with retries and exponential backoff.

Retries only what is likely to be temporary: connection problems, timeouts and 5xx
responses. 4xx answers are returned at once, because asking again will not change them.
"""

import time
from collections.abc import Callable

import httpx

from hrm.core.logging import get_logger

log = get_logger(__name__)


class ServiceUnavailableError(Exception):
    """The service could not be reached, or kept failing, after all attempts."""


def get_with_retries(
    client: httpx.Client,
    url: str,
    *,
    attempts: int = 3,
    backoff_s: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    for attempt in range(1, attempts + 1):
        try:
            response = client.get(url)
            if response.status_code < 500:
                return response
            problem = f"HTTP {response.status_code}"
        except httpx.TransportError as exc:  # connection refused, timeout, DNS, ...
            problem = type(exc).__name__
        log.warning("http.retry", url=url, attempt=attempt, of=attempts, problem=problem)
        if attempt < attempts:
            sleep(backoff_s * 2 ** (attempt - 1))  # 0.5 s, 1 s, 2 s, ...
    raise ServiceUnavailableError(f"{url} failed after {attempts} attempts ({problem})")
