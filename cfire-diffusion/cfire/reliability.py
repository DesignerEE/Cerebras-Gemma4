"""Reliability primitives: CircuitBreaker, RetryPolicy, DualRateLimiter.

Ports the three classes from cerebras_race_client.py and fixes the two
reliability bugs verified during exploration:

  Bug #1 — _request_with_retry retried EVERY exception.
           Fix: RetryPolicy.should_retry classifies by exception type.
           Retryable: RateLimitError, ServerError, RequestTimeoutError.
           Never retried: AuthError, BadRequestError, CircuitOpenError.

  Bug #6 — DualRateLimiter spin-waited with 10ms asyncio.sleep.
           Fix: asyncio.Condition. Waiters block on cond.wait(); the
           acquire path notify_all()s on each successful take so other
           waiters whose budget now fits can proceed without polling.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Awaitable, Callable, TypeVar

from .exceptions import (
    CircuitOpenError,
    RateLimitError,
    RequestTimeoutError,
    RetryableError,
    ServerError,
)

log = logging.getLogger("cfire.reliability")

T = TypeVar("T")


# --- CircuitBreaker ------------------------------------------------------

class CircuitBreaker:
    """Three-state circuit breaker.

    State machine:
      closed    -> open     after `threshold` consecutive failures
      open      -> half-open after `cooldown` seconds have elapsed
      half-open -> closed   on next success
      half-open -> open     on next failure

    Improvements over legacy (cerebras_race_client.py:201-232):
      - Raises CircuitOpenError (NonRetryableError) instead of RuntimeError,
        so RetryPolicy never accidentally retries the open state.
      - Logs every state transition.
    """

    def __init__(
        self,
        threshold: int = 5,
        cooldown: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.threshold = threshold
        self.cooldown = cooldown
        self.failures = 0
        self.last_failure = 0.0
        self.state: str = "closed"
        self._clock = clock
        self._lock = asyncio.Lock()

    async def call(self, fn: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        # Check + maybe transition open -> half-open
        async with self._lock:
            if self.state == "open":
                if self._clock() - self.last_failure > self.cooldown:
                    self._transition("half-open")
                else:
                    raise CircuitOpenError(
                        f"circuit open ({self.failures} failures, "
                        f"cooldown {self.cooldown:.0f}s)"
                    )

        # Call outside the lock so concurrent callers aren't serialized
        try:
            result = await fn(*args, **kwargs)
        except Exception as exc:
            async with self._lock:
                self.failures += 1
                self.last_failure = self._clock()
                if self.failures >= self.threshold and self.state != "open":
                    self._transition("open")
            raise

        async with self._lock:
            if self.state == "half-open":
                self._transition("closed")
            self.failures = 0
        return result

    def _transition(self, new_state: str) -> None:
        """Call under self._lock."""
        old = self.state
        if old == new_state:
            return
        self.state = new_state
        log.info(
            "circuit %s -> %s (failures=%d, threshold=%d)",
            old, new_state, self.failures, self.threshold,
        )


# --- RetryPolicy ---------------------------------------------------------

# Default retryable exception types. RetryPolicy.should_retry returns True
# only for these (or subclasses).
DEFAULT_RETRYABLE: tuple[type[Exception], ...] = (
    RateLimitError,
    ServerError,
    RequestTimeoutError,
)


class RetryPolicy:
    """Decides whether to retry, and for how long.

    Fix #1: legacy _request_with_retry (cerebras_race_client.py:371-389)
    caught bare `Exception` and retried — including AuthError, BadRequestError,
    even RuntimeError('circuit breaker OPEN'). This class restricts retries
    to the RetryableError hierarchy.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 10.0,
        retryable: tuple[type[Exception], ...] = DEFAULT_RETRYABLE,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retryable = retryable

    def should_retry(self, exc: Exception, attempt: int) -> bool:
        """True iff exc is a RetryableError AND we haven't exhausted attempts."""
        if attempt >= self.max_retries:
            return False
        return isinstance(exc, self.retryable)

    def delay_for(self, exc: Exception, attempt: int) -> float:
        """Compute backoff. Honors RateLimitError.retry_after when present."""
        if isinstance(exc, RateLimitError) and exc.retry_after is not None:
            return max(0.0, float(exc.retry_after))
        # Exponential backoff + jitter (matches legacy formula at line 386)
        backoff = self.base_delay * (2 ** attempt)
        return min(self.max_delay, backoff) + random.random()


# --- DualRateLimiter -----------------------------------------------------

class DualRateLimiter:
    """Sliding-window rate limiter enforcing both req/min and tok/min.

    Fix #6: legacy (cerebras_race_client.py:117-150) spin-waited with a
    fixed 10ms asyncio.sleep, burning CPU under sustained load. This
    version uses asyncio.Condition so waiters block efficiently and wake
    the moment budget frees up.
    """

    def __init__(
        self,
        req_per_min: float = 1000.0,
        tok_per_min: float = 1_000_000.0,
        window_sec: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.req_per_min = req_per_min
        self.tok_per_min = tok_per_min
        self.window_sec = window_sec
        self._req_log: list[float] = []
        self._tok_log: list[tuple[float, int]] = []
        self._clock = clock
        self._cond = asyncio.Condition()

    async def acquire(self, tokens: int) -> None:
        """Block until both req and token budgets have room for this call."""
        async with self._cond:
            while True:
                now = self._clock()
                self._prune(now)

                current_req = len(self._req_log)
                current_tok = sum(n for _, n in self._tok_log)

                if (current_req + 1 <= self.req_per_min
                        and current_tok + tokens <= self.tok_per_min):
                    self._req_log.append(now)
                    self._tok_log.append((now, tokens))
                    # Wake other waiters — they might now fit too
                    self._cond.notify_all()
                    return

                # Compute the soonest moment budget could free up:
                # the oldest timestamp in either log + window.
                oldest = self._oldest_timestamp()
                wait_for = max(0.05, oldest + self.window_sec - now + 0.005)
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=wait_for)
                except asyncio.TimeoutError:
                    # Timed out — loop and recheck (entries may have pruned)
                    continue

    def _prune(self, now: float) -> None:
        """Drop entries older than the window. Caller holds the lock."""
        cutoff = now - self.window_sec
        self._req_log = [t for t in self._req_log if t > cutoff]
        self._tok_log = [(t, n) for t, n in self._tok_log if t > cutoff]

    def _oldest_timestamp(self) -> float:
        """Earliest timestamp across both logs. Caller holds the lock."""
        candidates: list[float] = list(self._req_log)
        if self._tok_log:
            candidates.append(self._tok_log[0][0])
        return min(candidates) if candidates else 0.0

    def snapshot(self) -> dict[str, float | int]:
        """Current budget snapshot (for metrics / dashboards). No lock."""
        return {
            "req_window": len(self._req_log),
            "tok_window": sum(n for _, n in self._tok_log),
            "req_per_min": self.req_per_min,
            "tok_per_min": self.tok_per_min,
        }


__all__ = [
    "CircuitBreaker",
    "RetryPolicy",
    "DualRateLimiter",
    "DEFAULT_RETRYABLE",
]
