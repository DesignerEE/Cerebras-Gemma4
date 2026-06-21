"""Exception hierarchy for cfire.

The split between RetryableError and NonRetryableError is the contract that
RetryPolicy uses to decide whether to retry — fixes bug #1 from the plan
(the legacy _request_with_retry at cerebras_race_client.py:380 retried
*every* exception including auth failures and the circuit-breaker's own
OPEN state).

RequestTimeoutError multi-inherits from the builtin TimeoutError so callers
who write `except TimeoutError:` (the common Python idiom) still catch it.
"""

from __future__ import annotations


class CerebrasError(Exception):
    """Base class for every error cfire raises."""


# --- Retryable -----------------------------------------------------------

class RetryableError(CerebrasError):
    """Marker base: RetryPolicy may retry subclasses of this."""


class RateLimitError(RetryableError):
    """HTTP 429 from the API. Always retryable (with backoff)."""

    def __init__(self, message: str = "rate limited", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ServerError(RetryableError):
    """HTTP 5xx from the API. Retryable."""

    def __init__(self, status_code: int, message: str = ""):
        super().__init__(f"server error {status_code}: {message}" if message else f"server error {status_code}")
        self.status_code = status_code


class RequestTimeoutError(RetryableError, TimeoutError):
    """httpx.TimeoutException surfaced as a cfire error.

    Multi-inherits the builtin TimeoutError so callers using
    `except TimeoutError:` (the Python idiom) still catch it.
    """


# --- Non-retryable -------------------------------------------------------

class NonRetryableError(CerebrasError):
    """Marker base: RetryPolicy must NOT retry subclasses of this."""


class AuthError(NonRetryableError):
    """HTTP 401 / 403. Retrying with the same key cannot succeed."""


class BadRequestError(NonRetryableError):
    """HTTP 400. The request payload itself is wrong; retry won't help."""


class CircuitOpenError(NonRetryableError):
    """CircuitBreaker refused — too many recent failures.

    Not retryable by the same breaker; the Router may still fail over
    to a different backend.
    """


class ConfigError(NonRetryableError):
    """Missing api key, malformed base_url, etc."""


__all__ = [
    "CerebrasError",
    "RetryableError",
    "RateLimitError",
    "ServerError",
    "RequestTimeoutError",
    "NonRetryableError",
    "AuthError",
    "BadRequestError",
    "CircuitOpenError",
    "ConfigError",
]
