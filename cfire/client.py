"""AsyncCfire — the primary async facade.

Composes a Backend, Cache, reliability primitives, and Metrics into a
single user-facing client. Replaces CerebrasRaceClient
(cerebras_race_client.py:240-474).

Reliability layering (in order, per request):
  1. Cache lookup      — short-circuits on hit (latency=0, cached=True)
  2. DualRateLimiter   — pessimistic max_completion_tokens reservation
  3. Semaphore         — bounded concurrency (default 16, the benchmark
                         sweet spot)
  4. CircuitBreaker    — refuses if too many recent failures
  5. RetryPolicy       — classifies exception, decides retry vs raise
  6. Backend.complete  — actual HTTP call via Transport

Fix #5: the legacy bulk_complete (cerebras_race_client.py:441-471) acquired
BOTH self.semaphore (via complete()) AND a local semaphore per-batch. The
local one was redundant. AsyncCfire.bulk() uses a single shared semaphore.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Iterable

from .backends import Backend, CerebrasBackend
from .cache import Cache, MemoryLRU, TieredCache, cache_key
from .config import (
    CACHE_MAXSIZE,
    CACHE_TTL_SEC,
    CIRCUIT_COOLDOWN,
    CIRCUIT_THRESHOLD,
    DEFAULT_CONCURRENCY,
    DEFAULT_MODEL,
    DEFAULT_REQ_PER_MIN,
    DEFAULT_TOK_PER_MIN,
    REDIS_URL,
)
from .exceptions import CircuitOpenError
from .metrics import Metrics
from .models import ChatRequest, ChatResponse, StreamChunk
from .reliability import CircuitBreaker, DualRateLimiter, RetryPolicy
from .streaming import parse_sse_stream


class AsyncCfire:
    """Primary async client. Composes backend + cache + reliability."""

    def __init__(
        self,
        backend: Backend | None = None,
        cache: Cache | None = ...,  # sentinel: build default if not provided
        *,
        model: str = DEFAULT_MODEL,
        concurrency: int = DEFAULT_CONCURRENCY,
        req_per_min: float = DEFAULT_REQ_PER_MIN,
        tok_per_min: float = DEFAULT_TOK_PER_MIN,
        retry_policy: RetryPolicy | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        rate_limiter: DualRateLimiter | None = None,
        metrics: Metrics | None = None,
        enable_cache: bool = True,
    ):
        # Backend
        self.backend: Backend = backend or CerebrasBackend()
        # Some backends (Router in Phase 3) won't have base_url; default to ""
        self.base_url = getattr(self.backend, "base_url", "")

        # Cache: default is MemoryLRU, optionally tiered with Redis
        if not enable_cache:
            self.cache: Cache | None = None
        elif cache is ... or cache is None:
            memory = MemoryLRU(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL_SEC)
            if REDIS_URL:
                from .cache import RedisCache
                self.cache = TieredCache(memory, RedisCache(REDIS_URL))
            else:
                self.cache = memory
        else:
            self.cache = cache

        # Reliability
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
        self.limiter = rate_limiter or DualRateLimiter(req_per_min, tok_per_min)
        self.circuit = circuit_breaker or CircuitBreaker(
            threshold=CIRCUIT_THRESHOLD, cooldown=CIRCUIT_COOLDOWN,
        )
        self.retry_policy = retry_policy or RetryPolicy()
        self.metrics = metrics or Metrics()

        # Default model used when caller doesn't supply one in ChatRequest
        self.model = model

    # --- Lifecycle ------------------------------------------------------

    async def __aenter__(self) -> "AsyncCfire":
        await self.open()
        return self

    async def __aexit__(self, *args) -> None:
        await self.aclose()

    async def open(self) -> None:
        """Open the backend (and any resources it owns)."""
        opener = getattr(self.backend, "open", None)
        if opener is not None and asyncio.iscoroutinefunction(opener):
            await opener()

    async def aclose(self) -> None:
        closer = getattr(self.backend, "aclose", None)
        if closer is not None and asyncio.iscoroutinefunction(closer):
            await closer()
        if self.cache is not None:
            await self.cache.aclose()

    # --- Cache key ------------------------------------------------------

    def _cache_key(self, request: ChatRequest) -> str:
        # Two requests that differ only in stream=True vs False must NOT
        # share a cache slot — normalize before hashing.
        normalized = request.model_copy(update={"stream": False})
        return cache_key(normalized)

    # --- Single request -------------------------------------------------

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Single non-streaming request.

        Walks the reliability stack: cache -> rate limit -> semaphore ->
        circuit -> retry -> backend.
        """
        if not request.model:
            request = request.model_copy(update={"model": self.model})

        # 1. Cache check (skip if caller wants streaming)
        if self.cache is not None and not request.stream:
            key = self._cache_key(request)
            cached = await self.cache.get(key)
            if cached is not None:
                self.metrics.record_response(cached)
                return cached

        # 2. Rate limit (pessimistic token reservation)
        await self.limiter.acquire(request.max_completion_tokens)

        # 3. Concurrency + 4. Circuit + 5. Retry + 6. Backend
        async with self.semaphore:
            response = await self._request_with_retry(request)

        # 7. Cache write (skip if streaming)
        if self.cache is not None and not request.stream:
            await self.cache.put(key, response)  # type: ignore[possibly-undefined]

        # 8. Metrics
        self.metrics.record_response(response)
        return response

    async def _request_with_retry(self, request: ChatRequest) -> ChatResponse:
        """Run backend.complete via circuit breaker + retry policy.

        Fix #1: only retry RetryableError subclasses.
        """
        import asyncio as _asyncio

        attempt = 0
        while True:
            try:
                # CircuitBreaker raises CircuitOpenError (NonRetryableError)
                # if the circuit is open — RetryPolicy won't retry it.
                result = await self.circuit.call(self.backend.complete, request)
                return result
            except Exception as exc:
                self.metrics.record_error(str(exc))
                if not self.retry_policy.should_retry(exc, attempt):
                    raise
                delay = self.retry_policy.delay_for(exc, attempt)
                self.metrics.record_retry()
                await _asyncio.sleep(delay)
                attempt += 1

    # --- Streaming ------------------------------------------------------

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Streaming request via SSE. Never cached, never retried."""
        if not request.model:
            request = request.model_copy(update={"model": self.model})
        request = request.model_copy(update={"stream": True})

        # Rate limit (still pessimistic — streaming doesn't change token budget)
        await self.limiter.acquire(request.max_completion_tokens)

        async with self.semaphore:
            # stream() returns an async generator — we yield through it.
            # Circuit/retry intentionally NOT applied here: streaming errors
            # mid-flight can't be cleanly retried to the caller.
            streamer = self.backend.stream(request)
            async for chunk in streamer:
                yield chunk

    # --- Bulk ------------------------------------------------------------

    async def bulk(
        self,
        requests: Iterable[ChatRequest],
        *,
        progress_queue: asyncio.Queue | None = None,
    ) -> list[ChatResponse]:
        """Max-throughput batch path.

        Fix #5: the legacy bulk_complete (cerebras_race_client.py:441-471)
        acquired BOTH self.semaphore AND a local `sem = asyncio.Semaphore(
        self.concurrency)` per batch. The local one was redundant. We rely
        solely on the shared self.semaphore inside complete().
        """
        request_list = list(requests)
        results: list[ChatResponse | None] = [None] * len(request_list)

        async def worker(idx: int, req: ChatRequest) -> None:
            r = await self.complete(req)
            results[idx] = r
            if progress_queue is not None:
                await progress_queue.put({
                    "type": "request",
                    "idx": idx,
                    "latency": r.latency,
                    "tokens": r.usage.completion_tokens,
                    "cached": r.cached,
                    "compressed": r.compressed,
                })

        # asyncio.gather preserves order — results[i] is set by worker(i, ...)
        await asyncio.gather(*[worker(i, r) for i, r in enumerate(request_list)])
        return [r for r in results if r is not None]  # type: ignore[list-item]

    # --- Snapshot -------------------------------------------------------

    def report(self) -> dict[str, Any]:
        """Metrics snapshot for dashboards. Same shape as legacy."""
        return self.metrics.report()


__all__ = ["AsyncCfire"]
