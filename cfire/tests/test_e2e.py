"""End-to-end smoke for cfire.

Phase 2 coverage:
- AsyncCfire.complete() walks the full reliability stack against FakeBackend
- Cache hit short-circuits (second call doesn't touch the backend)
- Cache miss → backend call → cache write → cache hit on retry
- AsyncCfire.bulk() parallelizes N requests (fix #5: single shared semaphore)
- AsyncCfire.stream() yields through to the caller
- Sync Cfire wrapper mirrors async behavior via background daemon loop
- Metrics counters tick correctly per call
"""

from __future__ import annotations

import asyncio

import pytest

from cfire import AsyncCfire, Cfire
from cfire.backends import MockBackend
from cfire.cache import MemoryLRU
from cfire.exceptions import RateLimitError
from cfire.metrics import Metrics
from cfire.models import ChatRequest, ChatResponse, Message, StreamChunk

from .conftest import FakeBackend


# --- Single complete() -----------------------------------------------------

async def test_complete_walks_full_pipeline(fake_backend):
    """FakeBackend -> AsyncCfire.complete() returns the canned response."""
    fake_backend.respond_text("hello world", completion_tokens=3)
    async with AsyncCfire(backend=fake_backend, enable_cache=False) as client:
        req = ChatRequest(model="m", messages=[Message(role="user", content="hi")])
        resp = await client.complete(req)
    assert resp.text == "hello world"
    assert resp.usage.completion_tokens == 3
    assert fake_backend.complete_calls == 1


async def test_complete_injects_default_model_when_unset(fake_backend):
    """ChatRequest with model='' should fall back to client.model."""
    async with AsyncCfire(backend=fake_backend, model="default-model", enable_cache=False) as client:
        req = ChatRequest(model="", messages=[Message(role="user", content="hi")])
        await client.complete(req)
    # The backend saw the injected model name
    assert fake_backend.calls[-1].model == "default-model"


async def test_complete_records_metrics(fake_backend):
    """Metrics counters should tick once per request."""
    fake_backend.respond_text("ok", completion_tokens=10, prompt_tokens=5)
    async with AsyncCfire(backend=fake_backend, enable_cache=False) as client:
        req = ChatRequest(model="m", messages=[Message(role="user", content="hi")])
        await client.complete(req)
        report = client.report()
    assert report["requests"] == 1
    assert report["total_tokens"] == 15


# --- Cache integration -----------------------------------------------------

async def test_complete_cache_hit_short_circuits_backend(fake_backend):
    """Second identical call hits cache — backend called only once."""
    fake_backend.respond_text("cached reply", completion_tokens=2)
    cache = MemoryLRU(maxsize=10, ttl=60.0)
    async with AsyncCfire(backend=fake_backend, cache=cache, enable_cache=True) as client:
        req = ChatRequest(model="m", messages=[Message(role="user", content="ping")])
        first = await client.complete(req)
        second = await client.complete(req)
    assert first.text == "cached reply"
    assert second.text == "cached reply"
    assert second.cached is True        # hit flag set by MemoryLRU
    assert second.latency == 0.0        # cache hits are zero-latency
    assert fake_backend.complete_calls == 1


async def test_complete_cache_miss_does_not_short_circuit(fake_backend):
    """Different request content bypasses cache."""
    fake_backend.respond_fn(
        lambda req: ChatResponse(
            id="id", model="m",
            choices=[{"index": 0, "message": {"role": "assistant", "content": req.messages[-1].content},
                      "finish_reason": "stop"}],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
    )
    async with AsyncCfire(backend=fake_backend, enable_cache=True) as client:
        await client.complete(ChatRequest(model="m", messages=[Message(role="user", content="A")]))
        await client.complete(ChatRequest(model="m", messages=[Message(role="user", content="B")]))
    assert fake_backend.complete_calls == 2


# --- bulk() ----------------------------------------------------------------

async def test_bulk_runs_all_requests(fake_backend):
    """Fix #5 contract: single shared semaphore, no double-acquire deadlock."""
    fake_backend.respond_text("ok", completion_tokens=5)
    async with AsyncCfire(backend=fake_backend, enable_cache=False) as client:
        reqs = [
            ChatRequest(model="m", messages=[Message(role="user", content=f"q{i}")])
            for i in range(10)
        ]
        results = await client.bulk(reqs)
    assert len(results) == 10
    assert fake_backend.complete_calls == 10


async def test_bulk_progress_queue_receives_one_event_per_request(fake_backend):
    fake_backend.respond_text("ok", completion_tokens=5)
    queue: asyncio.Queue = asyncio.Queue()
    async with AsyncCfire(backend=fake_backend, enable_cache=False) as client:
        reqs = [
            ChatRequest(model="m", messages=[Message(role="user", content=f"q{i}")])
            for i in range(5)
        ]
        await client.bulk(reqs, progress_queue=queue)

    # Drain the queue — should have 5 events
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    assert len(events) == 5
    assert all(e["type"] == "request" for e in events)
    # Indexes should cover all 5 (in some order)
    assert sorted(e["idx"] for e in events) == [0, 1, 2, 3, 4]


# --- stream() --------------------------------------------------------------

async def test_stream_yields_through(fake_backend):
    """Streaming path passes backend chunks straight through."""
    fake_backend.stream_chunks([
        StreamChunk(delta="Hel"),
        StreamChunk(delta="lo"),
        StreamChunk(delta="!", finish_reason="stop"),
    ])
    async with AsyncCfire(backend=fake_backend, enable_cache=False) as client:
        req = ChatRequest(model="m", messages=[Message(role="user", content="hi")])
        chunks = []
        async for c in client.stream(req):
            chunks.append(c)
    assert len(chunks) == 3
    assert chunks[0].delta == "Hel"
    assert "".join(c.delta for c in chunks) == "Hello!"


# --- Retry integration -----------------------------------------------------

async def test_retry_recovers_from_rate_limit(fake_backend):
    """RetryPolicy retries RateLimitError; FakeBackend raises once then succeeds."""
    call_count = {"n": 0}

    def _respond(req):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RateLimitError("429", retry_after=0.01)  # tiny retry_after for speed
        return ChatResponse(
            id="id", model="m",
            choices=[{"index": 0, "message": {"role": "assistant", "content": "ok"},
                      "finish_reason": "stop"}],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    fake_backend.respond_fn(_respond)
    async with AsyncCfire(backend=fake_backend, enable_cache=False) as client:
        req = ChatRequest(model="m", messages=[Message(role="user", content="hi")])
        resp = await client.complete(req)
    assert resp.text == "ok"
    assert call_count["n"] == 2  # first failed, second succeeded


# --- Sync wrapper ----------------------------------------------------------

def test_sync_cfire_mirrors_async_pipeline():
    """The sync Cfire wrapper should produce the same result as AsyncCfire."""
    backend = FakeBackend()
    backend.respond_text("from sync", completion_tokens=4)
    with Cfire(backend=backend, enable_cache=False) as client:
        req = ChatRequest(model="m", messages=[Message(role="user", content="hi")])
        resp = client.complete(req)
        report = client.report()
    assert resp.text == "from sync"
    assert resp.usage.completion_tokens == 4
    assert report["requests"] == 1


def test_sync_cfire_bulk_works():
    """Sync Cfire.bulk() returns the same number of results as inputs."""
    backend = FakeBackend()
    backend.respond_text("ok", completion_tokens=1)
    with Cfire(backend=backend, enable_cache=False) as client:
        reqs = [
            ChatRequest(model="m", messages=[Message(role="user", content=f"q{i}")])
            for i in range(3)
        ]
        results = client.bulk(reqs)
    assert len(results) == 3


# --- Metrics observer ------------------------------------------------------

def test_metrics_callback_fires_on_record():
    """Metrics.on_event callbacks should fire on every record_response call."""
    metrics = Metrics()
    events = []
    metrics.on_event(events.append)
    metrics.record_response(ChatResponse(
        id="x", model="m",
        choices=[{"index": 0, "message": {"role": "assistant", "content": "hi"},
                  "finish_reason": "stop"}],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        latency=0.1,
    ))
    assert len(events) == 1
    assert events[0].kind == "response"
    assert events[0].tokens == 2


# --- MockBackend end-to-end (covers _compat path used by web_demo) ---------

async def test_mock_backend_via_compat_shim():
    """The legacy CerebrasRaceClient(mock=True) path must still work."""
    from cfire._compat import CerebrasRaceClient

    async with CerebrasRaceClient(mock=True) as client:
        result = await client.complete("hi", max_completion_tokens=50)
    assert result.text == "Mock response."
    assert result.completion_tokens >= 10  # max(10, ratio*max)
    assert result.latency == 0.0


async def test_compat_client_bulk_works():
    from cfire._compat import CerebrasRaceClient, CompletionResult

    async with CerebrasRaceClient(mock=True) as client:
        results = await client.bulk_complete(["q1", "q2", "q3"], max_completion_tokens=20)
    assert len(results) == 3
    assert all(isinstance(r, CompletionResult) for r in results)
