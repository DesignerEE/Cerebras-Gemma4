"""Tests for cfire.router."""

from __future__ import annotations

import re
from typing import AsyncIterator

import pytest

from cfire.backends import Backend
from cfire.exceptions import AuthError, RateLimitError, RetryableError, ServerError
from cfire.models import ChatRequest, Message, StreamChunk
from cfire.router import Router, RoutingPolicy

from .conftest import FakeBackend


# --- RoutingPolicy -------------------------------------------------------

def test_routing_policy_defaults():
    p = RoutingPolicy()
    assert p.cerebras_first is True
    assert p.route_tools_to_diffusiongemma is True
    assert p.max_retries_per_backend == 1
    assert p.failover_on == [RetryableError]


def test_routing_policy_custom_patterns():
    p = RoutingPolicy(
        prefer_local_for=[re.compile(r"local-only")],
        code_keywords=["rust", "go"],
    )
    assert len(p.prefer_local_for) == 1
    assert "rust" in p.code_keywords


# --- Router construction -------------------------------------------------

def test_router_requires_at_least_one_backend():
    with pytest.raises(ValueError, match="at least one backend"):
        Router({})


def test_router_conforms_to_backend_protocol():
    cb = FakeBackend()
    dg = FakeBackend()
    router = Router({"cerebras": cb, "diffusiongemma": dg})
    assert isinstance(router, Backend)


# --- Backend selection ---------------------------------------------------

def test_router_selects_cerebras_by_default():
    cb = FakeBackend()
    dg = FakeBackend()
    router = Router({"cerebras": cb, "diffusiongemma": dg})
    req = ChatRequest(messages=[Message(role="user", content="hello")])
    name, backend = router._select_backend(req)
    assert name == "cerebras"
    assert backend is cb


def test_router_selects_diffusiongemma_for_tools():
    cb = FakeBackend()
    dg = FakeBackend()
    router = Router({"cerebras": cb, "diffusiongemma": dg})
    req = ChatRequest(
        messages=[Message(role="user", content="call a function")],
        tools=[{
            "type": "function",
            "function": {"name": "foo", "parameters": {"type": "object"}},
        }],
    )
    name, backend = router._select_backend(req)
    assert name == "diffusiongemma"
    assert backend is dg


def test_router_selects_diffusiongemma_for_code_keywords():
    cb = FakeBackend()
    dg = FakeBackend()
    router = Router({"cerebras": cb, "diffusiongemma": dg})
    req = ChatRequest(messages=[Message(role="user", content="write a python function")])
    name, backend = router._select_backend(req)
    assert name == "diffusiongemma"
    assert backend is dg


def test_router_selects_diffusiongemma_for_prefer_local_pattern():
    cb = FakeBackend()
    dg = FakeBackend()
    router = Router(
        {"cerebras": cb, "diffusiongemma": dg},
        policy=RoutingPolicy(prefer_local_for=[re.compile(r"local-only")]),
    )
    req = ChatRequest(messages=[Message(role="user", content="this is local-only stuff")])
    name, backend = router._select_backend(req)
    assert name == "diffusiongemma"
    assert backend is dg


def test_router_respects_cerebras_first_false():
    cb = FakeBackend()
    dg = FakeBackend()
    router = Router(
        {"cerebras": cb, "diffusiongemma": dg},
        policy=RoutingPolicy(cerebras_first=False),
    )
    req = ChatRequest(messages=[Message(role="user", content="hello")])
    name, backend = router._select_backend(req)
    assert name == "diffusiongemma"
    assert backend is dg


def test_router_falls_back_to_first_available_if_primary_missing():
    dg = FakeBackend()
    router = Router({"diffusiongemma": dg})
    req = ChatRequest(messages=[Message(role="user", content="hello")])
    name, backend = router._select_backend(req)
    assert name == "diffusiongemma"
    assert backend is dg


# --- Failover ------------------------------------------------------------

async def test_router_failover_on_rate_limit():
    cb = FakeBackend()
    cb.error_fn(lambda req: RateLimitError("429"))
    dg = FakeBackend()
    dg.respond_text("from dg", completion_tokens=1)

    router = Router({"cerebras": cb, "diffusiongemma": dg})
    req = ChatRequest(messages=[Message(role="user", content="hello")])
    result = await router.complete(req)

    assert result.text == "from dg"
    assert cb.complete_calls == 2  # initial + 1 retry before failover
    assert dg.complete_calls == 1


async def test_router_no_failover_on_auth_error():
    cb = FakeBackend()
    cb.error_fn(lambda req: AuthError("bad key"))
    dg = FakeBackend()

    router = Router({"cerebras": cb, "diffusiongemma": dg})
    req = ChatRequest(messages=[Message(role="user", content="hello")])
    with pytest.raises(AuthError):
        await router.complete(req)
    assert dg.complete_calls == 0


async def test_router_retries_before_failover():
    cb = FakeBackend()
    cb.error_fn(lambda req: RateLimitError("429"))
    dg = FakeBackend()
    dg.respond_text("from dg", completion_tokens=1)

    router = Router(
        {"cerebras": cb, "diffusiongemma": dg},
        policy=RoutingPolicy(max_retries_per_backend=2),
    )
    req = ChatRequest(messages=[Message(role="user", content="hello")])
    result = await router.complete(req)

    assert result.text == "from dg"
    assert cb.complete_calls == 3  # initial + 2 retries
    assert dg.complete_calls == 1


async def test_router_re_raises_when_all_exhausted():
    cb = FakeBackend()
    cb.error_fn(lambda req: RateLimitError("429"))
    dg = FakeBackend()
    dg.error_fn(lambda req: ServerError(500, "dg down"))

    router = Router({"cerebras": cb, "diffusiongemma": dg})
    req = ChatRequest(messages=[Message(role="user", content="hello")])
    with pytest.raises(ServerError, match="dg down"):
        await router.complete(req)


# --- Stream failover -----------------------------------------------------

async def test_router_stream_failover_on_initial_error():
    class _FailingStreamBackend:
        base_url = "failing://"
        complete_calls = 0

        async def complete(self, request: ChatRequest) -> ChatResponse:
            raise NotImplementedError

        async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
            raise RateLimitError("429")
            yield  # type: ignore[unreachable]

        async def aclose(self) -> None:
            pass

    cb = _FailingStreamBackend()
    dg = FakeBackend()
    dg.stream_chunks([StreamChunk(delta="from dg", finish_reason="stop")])

    router = Router({"cerebras": cb, "diffusiongemma": dg})
    req = ChatRequest(messages=[Message(role="user", content="hello")])
    chunks = [c async for c in router.stream(req)]

    assert len(chunks) == 1
    assert chunks[0].delta == "from dg"


# --- Lifecycle -----------------------------------------------------------

async def test_router_opens_and_closes_backends():
    class _LifecycleBackend:
        base_url = "lifecycle://"

        def __init__(self):
            self.opened = False
            self.closed = False

        async def open(self) -> None:
            self.opened = True

        async def aclose(self) -> None:
            self.closed = True

        async def complete(self, request: ChatRequest) -> ChatResponse:
            raise NotImplementedError

        async def stream(self, request: ChatRequest):
            raise NotImplementedError

    cb = _LifecycleBackend()
    dg = _LifecycleBackend()
    async with Router({"cerebras": cb, "diffusiongemma": dg}) as router:
        assert cb.opened
        assert dg.opened
    assert cb.closed
    assert dg.closed
