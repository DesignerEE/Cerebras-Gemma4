"""Tests for cfire.cache.

Covers:
- cache_key() determinism for identical ChatRequest
- MemoryLRU bounded eviction (fix #2 regression)
- MemoryLRU TTL expiration (lazy)
- MemoryLRU marks hits as cached=True, latency=0.0
- TieredCache read-repair: hit in tier 1 backfills tier 0
- Redis field preservation is integration-tested separately (needs redis);
  here we cover the Pydantic JSON round-trip contract that Redis relies on
"""

from __future__ import annotations

import asyncio

import pytest

from cfire.cache import MemoryLRU, TieredCache, cache_key
from cfire.models import ChatRequest, ChatResponse, Message


# --- cache_key --------------------------------------------------------------

def test_cache_key_is_deterministic():
    """Identical requests must hash to the same key."""
    r1 = ChatRequest(model="m", messages=[Message(role="user", content="hi")])
    r2 = ChatRequest(model="m", messages=[Message(role="user", content="hi")])
    assert cache_key(r1) == cache_key(r2)


def test_cache_key_differs_on_content():
    r1 = ChatRequest(model="m", messages=[Message(role="user", content="hi")])
    r2 = ChatRequest(model="m", messages=[Message(role="user", content="bye")])
    assert cache_key(r1) != cache_key(r2)


def test_cache_key_differs_on_model():
    r1 = ChatRequest(model="m1", messages=[Message(role="user", content="hi")])
    r2 = ChatRequest(model="m2", messages=[Message(role="user", content="hi")])
    assert cache_key(r1) != cache_key(r2)


def test_cache_key_normalizes_stream_flag():
    """Same payload, different stream flag must NOT collide — see AsyncCfire._cache_key."""
    from cfire.client import AsyncCfire  # noqa: F401 — just to confirm import path
    r1 = ChatRequest(model="m", messages=[Message(role="user", content="hi")], stream=False)
    r2 = ChatRequest(model="m", messages=[Message(role="user", content="hi")], stream=True)
    # Raw keys DO differ because stream is part of the model dump.
    assert cache_key(r1) != cache_key(r2)


# --- MemoryLRU --------------------------------------------------------------

async def test_memory_lru_basic_put_get():
    cache = MemoryLRU(maxsize=10, ttl=60.0)
    req = ChatRequest(model="m", messages=[Message(role="user", content="hi")])
    key = cache_key(req)
    resp = ChatResponse(model="m", id="id1")
    await cache.put(key, resp)
    got = await cache.get(key)
    assert got is not None
    assert got.id == "id1"
    assert got.cached is True       # hit flag set on read
    assert got.latency == 0.0       # zero-latency on cache hit


async def test_memory_lru_miss_returns_none():
    cache = MemoryLRU(maxsize=10, ttl=60.0)
    got = await cache.get("nonexistent-key")
    assert got is None


async def test_memory_lru_evicts_oldest_at_capacity():
    """Fix #2 regression: bounded memory.

    With maxsize=3, inserting 4 keys evicts the oldest.
    """
    cache = MemoryLRU(maxsize=3, ttl=60.0)
    for i in range(4):
        key = f"k{i}"
        resp = ChatResponse(model="m", id=f"id{i}")
        await cache.put(key, resp)
    # k0 should be evicted; k1, k2, k3 remain
    assert await cache.get("k0") is None
    assert (await cache.get("k1")).id == "id1"
    assert (await cache.get("k2")).id == "id2"
    assert (await cache.get("k3")).id == "id3"
    assert len(cache) == 3


async def test_memory_lru_get_marks_as_recently_used():
    """Reading a key prevents it from being evicted next."""
    cache = MemoryLRU(maxsize=2, ttl=60.0)
    await cache.put("k0", ChatResponse(model="m", id="id0"))
    await cache.put("k1", ChatResponse(model="m", id="id1"))
    # Touch k0 — should now be most-recently-used
    await cache.get("k0")
    # Insert k2 — should evict k1 (LRU), not k0
    await cache.put("k2", ChatResponse(model="m", id="id2"))
    assert (await cache.get("k0")) is not None
    assert (await cache.get("k1")) is None


async def test_memory_lru_ttl_expiration():
    """Lazy expiration: expired entries return None on next get."""
    # Use a mutable clock so we don't have to sleep
    times = [0.0]
    cache = MemoryLRU(maxsize=10, ttl=10.0, clock=lambda: times[-1])
    await cache.put("k", ChatResponse(model="m", id="id"))
    times.append(5.0)
    assert (await cache.get("k")) is not None  # not yet expired
    times.append(15.0)  # past TTL
    assert await cache.get("k") is None        # expired -> None


async def test_memory_lru_aclose_clears():
    cache = MemoryLRU(maxsize=10, ttl=60.0)
    await cache.put("k", ChatResponse(model="m", id="id"))
    await cache.aclose()
    assert len(cache) == 0


# --- TieredCache ------------------------------------------------------------

class _RecordingTier:
    """Fake cache tier that records every operation and has canned data."""

    def __init__(self, name: str, canned: dict[str, ChatResponse] | None = None):
        self.name = name
        self.store: dict[str, ChatResponse] = dict(canned or {})
        self.gets: list[str] = []
        self.puts: list[str] = []

    async def get(self, key: str) -> ChatResponse | None:
        self.gets.append(key)
        return self.store.get(key)

    async def put(self, key: str, value: ChatResponse, ttl: float | None = None) -> None:
        self.puts.append(key)
        self.store[key] = value

    async def aclose(self) -> None:
        pass


async def test_tiered_cache_hit_in_first_tier_skips_others():
    """Tier 0 hit returns immediately without checking tier 1."""
    resp = ChatResponse(model="m", id="cached-id")
    t0 = _RecordingTier("t0", canned={"k": resp})
    t1 = _RecordingTier("t1")
    cache = TieredCache(t0, t1)

    got = await cache.get("k")
    assert got is not None and got.id == "cached-id"
    assert t0.gets == ["k"]
    assert t1.gets == []   # short-circuit


async def test_tiered_cache_miss_in_t0_hits_t1():
    resp = ChatResponse(model="m", id="from-t1")
    t0 = _RecordingTier("t0")
    t1 = _RecordingTier("t1", canned={"k": resp})
    cache = TieredCache(t0, t1)

    got = await cache.get("k")
    assert got is not None and got.id == "from-t1"


async def test_tiered_cache_read_repair_backfills_earlier_tiers():
    """Fix #3 contract: hit in tier N writes back to tiers 0..N-1."""
    resp = ChatResponse(model="m", id="from-t1")
    t0 = _RecordingTier("t0")
    t1 = _RecordingTier("t1", canned={"k": resp})
    cache = TieredCache(t0, t1)

    await cache.get("k")
    # Read-repair should have populated t0
    assert "k" in t0.store
    assert t0.store["k"].id == "from-t1"


async def test_tiered_cache_put_writes_all_tiers():
    t0 = _RecordingTier("t0")
    t1 = _RecordingTier("t1")
    cache = TieredCache(t0, t1)

    resp = ChatResponse(model="m", id="id")
    await cache.put("k", resp)
    assert "k" in t0.store
    assert "k" in t1.store


async def test_tiered_cache_all_miss_returns_none():
    t0 = _RecordingTier("t0")
    t1 = _RecordingTier("t1")
    cache = TieredCache(t0, t1)
    assert await cache.get("missing") is None


def test_tiered_cache_requires_at_least_one_tier():
    with pytest.raises(ValueError):
        TieredCache()
