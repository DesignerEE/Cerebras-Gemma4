"""Tests for cfire.reliability.

Covers the three classes ported from cerebras_race_client.py and the
two bug fixes they implement:

  Fix #1 — RetryPolicy.should_retry classifies by exception type.
           Retryable: RateLimitError, ServerError, RequestTimeoutError.
           Never retried: AuthError, BadRequestError, CircuitOpenError.

  Fix #6 — DualRateLimiter uses asyncio.Condition (no spin-wait).
           Under-the-budget callers proceed instantly; over-budget callers
           block until budget frees (verified via timing, not sleeps).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from cfire.exceptions import (
    AuthError,
    BadRequestError,
    CircuitOpenError,
    RateLimitError,
    RequestTimeoutError,
    ServerError,
)
from cfire.reliability import (
    CircuitBreaker,
    DEFAULT_RETRYABLE,
    DualRateLimiter,
    RetryPolicy,
)


# --- RetryPolicy ------------------------------------------------------------

def test_retry_policy_retries_rate_limit_error():
    policy = RetryPolicy(max_retries=3)
    assert policy.should_retry(RateLimitError("429"), attempt=0) is True


def test_retry_policy_retries_server_error():
    policy = RetryPolicy()
    assert policy.should_retry(ServerError(503), attempt=0) is True


def test_retry_policy_retries_timeout_error():
    """RequestTimeoutError multi-inherits builtin TimeoutError."""
    policy = RetryPolicy()
    assert policy.should_retry(RequestTimeoutError(), attempt=0) is True
    # And the pythonic idiom catches it too
    try:
        raise RequestTimeoutError()
    except TimeoutError:
        pass


def test_retry_policy_does_not_retry_auth_error():
    """Fix #1 regression: legacy code retried ALL exceptions."""
    policy = RetryPolicy()
    assert policy.should_retry(AuthError("401"), attempt=0) is False


def test_retry_policy_does_not_retry_bad_request():
    policy = RetryPolicy()
    assert policy.should_retry(BadRequestError("400"), attempt=0) is False


def test_retry_policy_does_not_retry_circuit_open():
    """Fix #1 regression: legacy retried RuntimeError('circuit breaker OPEN')."""
    policy = RetryPolicy()
    assert policy.should_retry(CircuitOpenError("open"), attempt=0) is False


def test_retry_policy_stops_after_max_retries():
    policy = RetryPolicy(max_retries=2)
    assert policy.should_retry(RateLimitError(), attempt=2) is False
    assert policy.should_retry(RateLimitError(), attempt=1) is True


def test_retry_policy_honors_rate_limit_retry_after():
    """RateLimitError.retry_after should override exponential backoff."""
    policy = RetryPolicy(base_delay=1.0, max_delay=10.0)
    err = RateLimitError("429", retry_after=5.0)
    delay = policy.delay_for(err, attempt=0)
    assert delay == 5.0


def test_retry_policy_exponential_backoff_within_bounds():
    """Without retry_after, delay is 2^attempt * base_delay + jitter in [0,1)."""
    policy = RetryPolicy(base_delay=0.5, max_delay=10.0)
    for attempt in range(4):
        delay = policy.delay_for(ServerError(503), attempt)
        expected_min = 0.5 * (2 ** attempt)
        assert expected_min <= delay < expected_min + 1.0


def test_retry_policy_caps_at_max_delay():
    """At very high attempt counts, delay is capped at max_delay + jitter < 1."""
    policy = RetryPolicy(base_delay=1.0, max_delay=4.0)
    delay = policy.delay_for(ServerError(503), attempt=20)
    assert 4.0 <= delay < 5.0


def test_default_retryable_tuple_has_three_types():
    assert set(DEFAULT_RETRYABLE) == {RateLimitError, ServerError, RequestTimeoutError}


# --- CircuitBreaker ---------------------------------------------------------

async def test_circuit_breaker_starts_closed():
    cb = CircuitBreaker(threshold=3, cooldown=10.0)
    assert cb.state == "closed"
    assert cb.failures == 0


async def test_circuit_breaker_opens_after_threshold_failures():
    """3 consecutive failures should open the circuit."""
    cb = CircuitBreaker(threshold=3, cooldown=10.0)

    async def _fail():
        raise ServerError(500)

    for _ in range(3):
        with pytest.raises(ServerError):
            await cb.call(_fail)
    assert cb.state == "open"


async def test_circuit_breaker_raises_circuit_open_error_when_open():
    """Fix #1 contract: open circuit raises CircuitOpenError (NonRetryableError),
    not the legacy bare RuntimeError."""
    cb = CircuitBreaker(threshold=1, cooldown=1000.0)  # long cooldown

    async def _fail():
        raise ServerError(500)

    with pytest.raises(ServerError):
        await cb.call(_fail)
    assert cb.state == "open"

    with pytest.raises(CircuitOpenError):
        await cb.call(_fail)


async def test_circuit_breaker_resets_failures_on_success():
    """A success after partial failures resets the counter."""
    cb = CircuitBreaker(threshold=3, cooldown=10.0)
    calls = {"n": 0}

    async def _fail_then_succeed():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ServerError(500)
        return "ok"

    with pytest.raises(ServerError):
        await cb.call(_fail_then_succeed)
    assert cb.failures == 1
    result = await cb.call(_fail_then_succeed)
    assert result == "ok"
    assert cb.failures == 0
    assert cb.state == "closed"


async def test_circuit_breaker_half_open_to_closed_on_success():
    """Open -> half-open after cooldown -> closed on next success."""
    cb = CircuitBreaker(threshold=1, cooldown=0.0)  # cooldown elapses immediately

    async def _fail():
        raise ServerError(500)

    async def _succeed():
        return "recovered"

    with pytest.raises(ServerError):
        await cb.call(_fail)
    assert cb.state == "open"

    # Next call should transition open -> half-open -> closed
    result = await cb.call(_succeed)
    assert result == "recovered"
    assert cb.state == "closed"


# --- DualRateLimiter --------------------------------------------------------

async def test_rate_limiter_allows_under_budget():
    """Under-the-budget calls acquire instantly."""
    rl = DualRateLimiter(req_per_min=10, tok_per_min=1000)
    start = time.perf_counter()
    await rl.acquire(tokens=100)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.05


async def test_rate_limiter_blocks_when_request_budget_exhausted():
    """Fix #6 contract: callers block (via asyncio.Condition) instead of spin-polling.

    We exhaust the request budget, then verify a second acquire blocks until
    the window elapses.
    """
    rl = DualRateLimiter(req_per_min=1, tok_per_min=10_000, window_sec=0.05)
    await rl.acquire(tokens=1)
    start = time.perf_counter()
    await rl.acquire(tokens=1)
    elapsed = time.perf_counter() - start
    assert elapsed >= 0.03


async def test_rate_limiter_concurrent_callers_serialize():
    """Multiple concurrent acquires don't exceed the budget."""
    rl = DualRateLimiter(req_per_min=5, tok_per_min=10_000, window_sec=10.0)
    await asyncio.gather(*(rl.acquire(tokens=1) for _ in range(5)))
    snap = rl.snapshot()
    assert snap["req_window"] == 5


def test_rate_limiter_snapshot_shape():
    rl = DualRateLimiter(req_per_min=100, tok_per_min=2000)
    snap = rl.snapshot()
    assert snap["req_per_min"] == 100
    assert snap["tok_per_min"] == 2000
    assert snap["req_window"] == 0
    assert snap["tok_window"] == 0
