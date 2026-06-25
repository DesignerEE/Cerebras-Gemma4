"""Tiered cache for cfire.

Splits the monolithic PromptCache (cerebras_race_client.py:153-198) into
composable pieces and fixes both cache bugs verified during exploration:

  Bug #2 — PromptCache._memory was an unbounded dict (line 157).
          Long-running processes would grow it without limit.
          Fix: MemoryLRU(maxsize, ttl) — bounded OrderedDict + per-entry TTL.

  Bug #3 — Redis round-trip dropped 3 of 7 CompletionResult fields
          (lines 191-196 stored only text/tokens; lost cached/compressed/
          reasoning_tokens).
          Fix: RedisCache uses ChatResponse.model_dump_json() on put and
          ChatResponse.model_validate_json() on get — zero field loss.

Cache key: SHA-256 of canonicalized ChatRequest. Same payload -> same key.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from typing import Awaitable, Callable, Protocol

from .models import ChatRequest, ChatResponse


# --- Key derivation ------------------------------------------------------

def cache_key(request: ChatRequest) -> str:
    """SHA-256 of canonicalized request. Stable across runs."""
    # Use model_dump with mode=json so the serialization is identical for
    # identical requests (no Python-object ambiguity in nested fields).
    payload = request.model_dump(mode="json")
    normalized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(normalized.encode()).hexdigest()


# --- Cache Protocol ------------------------------------------------------

class Cache(Protocol):
    """Async cache interface. Both MemoryLRU and RedisCache conform."""
    async def get(self, key: str) -> ChatResponse | None: ...
    async def put(self, key: str, value: ChatResponse, ttl: float | None = None) -> None: ...
    async def aclose(self) -> None: ...


# --- MemoryLRU -----------------------------------------------------------

class MemoryLRU:
    """Bounded in-memory LRU with per-entry TTL.

    Fix for bug #2: the legacy PromptCache._memory grew unbounded.
    """
    def __init__(
        self,
        maxsize: int = 1024,
        ttl: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.maxsize = maxsize
        self.default_ttl = ttl
        self._clock = clock
        self._store: OrderedDict[str, ChatResponse] = OrderedDict()
        self._expiry: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> ChatResponse | None:
        async with self._lock:
            # Lazy expiration
            exp = self._expiry.get(key)
            if exp is not None and exp < self._clock():
                self._store.pop(key, None)
                self._expiry.pop(key, None)
                return None
            value = self._store.get(key)
            if value is None:
                return None
            # Mark as recently used
            self._store.move_to_end(key)
            # Mark as cache hit on a copy so callers don't mutate stored
            cached = value.model_copy(deep=True)
            cached.cached = True
            cached.latency = 0.0
            return cached

    async def put(self, key: str, value: ChatResponse, ttl: float | None = None) -> None:
        async with self._lock:
            # Evict if at capacity and this is a new key
            if key not in self._store and len(self._store) >= self.maxsize:
                evicted_key, _ = self._store.popitem(last=False)  # FIFO eviction
                self._expiry.pop(evicted_key, None)
            self._store[key] = value
            self._store.move_to_end(key)
            self._expiry[key] = self._clock() + (ttl if ttl is not None else self.default_ttl)

    async def aclose(self) -> None:
        async with self._lock:
            self._store.clear()
            self._expiry.clear()

    def __len__(self) -> int:
        return len(self._store)


# --- RedisCache ----------------------------------------------------------

class RedisCache:
    """Optional Redis-backed cache. Preserves ALL ChatResponse fields.

    Fix for bug #3: legacy PromptCache stored only 4 fields (text/tokens)
    in Redis, dropping cached/compressed/reasoning_tokens. We use Pydantic's
    JSON round-trip so the entire ChatResponse shape survives.
    """
    def __init__(self, redis_url: str, default_ttl: float = 3600.0, prefix: str = "cfire:"):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.prefix = prefix
        self._redis = None  # lazily connected on first use
        self._connect_lock = asyncio.Lock()

    async def _ensure_client(self):
        if self._redis is not None:
            return self._redis
        async with self._connect_lock:
            if self._redis is not None:
                return self._redis
            try:
                import redis.asyncio as aioredis  # type: ignore[import-untyped]
            except ImportError as e:
                raise RuntimeError(
                    "RedisCache requires `pip install cfire[redis]` "
                    "(or `pip install redis>=5.0`)"
                ) from e
            self._redis = aioredis.from_url(self.redis_url)
        return self._redis

    async def get(self, key: str) -> ChatResponse | None:
        try:
            client = await self._ensure_client()
            data = await client.get(self.prefix + key)
            if not data:
                return None
            # data is bytes; ChatResponse.model_validate_json accepts bytes/str
            value = ChatResponse.model_validate_json(data)
            value.cached = True
            value.latency = 0.0
            return value
        except Exception:
            # Fail open — a Redis error shouldn't kill the request
            return None

    async def put(self, key: str, value: ChatResponse, ttl: float | None = None) -> None:
        try:
            client = await self._ensure_client()
            await client.set(
                self.prefix + key,
                value.model_dump_json(),
                ex=int(ttl if ttl is not None else self.default_ttl),
            )
        except Exception:
            # Fail open — write failure shouldn't kill the request
            pass

    async def aclose(self) -> None:
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception:
                pass
            self._redis = None


# --- Tiered composition --------------------------------------------------

class TieredCache:
    """Composes multiple caches into a tiered lookup.

    get() walks tiers in order; on a hit in tier N, asynchronously writes
    the value back to tiers 0..N-1 (read-repair). put() writes to all tiers.
    """
    def __init__(self, *tiers: Cache):
        if not tiers:
            raise ValueError("TieredCache needs at least one tier")
        self.tiers: tuple[Cache, ...] = tiers

    async def get(self, key: str) -> ChatResponse | None:
        for i, tier in enumerate(self.tiers):
            value = await tier.get(key)
            if value is not None:
                # Read-repair: backfill earlier tiers in parallel
                if i > 0:
                    await asyncio.gather(
                        *(t.put(key, value) for t in self.tiers[:i]),
                        return_exceptions=True,
                    )
                return value
        return None

    async def put(self, key: str, value: ChatResponse, ttl: float | None = None) -> None:
        await asyncio.gather(
            *(t.put(key, value, ttl) for t in self.tiers),
            return_exceptions=True,
        )

    async def aclose(self) -> None:
        await asyncio.gather(
            *(t.aclose() for t in self.tiers),
            return_exceptions=True,
        )


__all__ = [
    "cache_key",
    "Cache",
    "MemoryLRU",
    "RedisCache",
    "TieredCache",
]
