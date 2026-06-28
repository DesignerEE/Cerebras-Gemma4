"""Pytest configuration and shared fixtures for cfire.

FakeBackend is the workhorse — an in-memory Backend implementation that
records every call, returns deterministic responses, and can be
programmed to raise specific exceptions. Every Phase 2 test uses it
instead of touching the network.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Callable

import pytest

from cfire.backends import Backend
from cfire.config import DEFAULT_MODEL
from cfire.models import (
    ChatRequest,
    ChatResponse,
    Choice,
    Message,
    StreamChunk,
    Usage,
)


class FakeBackend:
    """In-memory Backend for tests. Records calls; returns deterministic data.

    Usage:
        backend = FakeBackend()
        backend.respond_text("hello back", completion_tokens=3)
        # or:
        backend.error_fn(lambda req: RateLimitError("429"))
        # or for streaming:
        backend.stream_chunks([StreamChunk(delta="hel"), StreamChunk(delta="lo")])
    """

    base_url = "fake://"

    def __init__(self) -> None:
        self.calls: list[ChatRequest] = []
        self.complete_calls: int = 0
        self.stream_calls: int = 0
        self._responder: Callable[[ChatRequest], ChatResponse] | None = None
        self._error_fn: Callable[[ChatRequest], Exception] | None = None
        self._stream_chunks: list[StreamChunk] | None = None

    # --- Configuration --------------------------------------------------

    def respond_text(self, text: str, completion_tokens: int = 10, **response_fields: Any) -> None:
        """Program the next complete() to return this text."""
        def _r(req: ChatRequest) -> ChatResponse:
            return _build_response(text=text, completion_tokens=completion_tokens, **response_fields)
        self._responder = _r

    def respond_fn(self, fn: Callable[[ChatRequest], ChatResponse]) -> None:
        """Program with a function of the request."""
        self._responder = fn

    def error_fn(self, fn: Callable[[ChatRequest], Exception]) -> None:
        """Program complete() to raise. fn is called per-request."""
        self._error_fn = fn

    def stream_chunks(self, chunks: list[StreamChunk]) -> None:
        """Program stream() to yield these chunks (one per call)."""
        self._stream_chunks = chunks

    # --- Backend Protocol -----------------------------------------------

    async def __aenter__(self) -> "FakeBackend":
        return self

    async def __aexit__(self, *args) -> None:
        pass

    async def open(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self.calls.append(request)
        self.complete_calls += 1
        if self._error_fn is not None:
            raise self._error_fn(request)
        if self._responder is not None:
            return self._responder(request)
        # Default: echo the prompt back with a fixed token count
        msg_text = ""
        if request.messages:
            msg_text = f"echo: {request.messages[-1].content[:32]}"
        return _build_response(text=msg_text, completion_tokens=10)

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        self.calls.append(request)
        self.stream_calls += 1
        chunks = self._stream_chunks or [
            StreamChunk(delta="hello"),
            StreamChunk(delta=" world", finish_reason="stop"),
        ]
        for c in chunks:
            yield c


def _build_response(
    *,
    text: str = "fake response",
    completion_tokens: int = 10,
    prompt_tokens: int = 5,
    reasoning_tokens: int = 0,
    model: str = "fake-model",
    latency: float = 0.0,
    cached: bool = False,
    compressed: bool = False,
) -> ChatResponse:
    """Helper that builds a fully-populated ChatResponse."""
    return ChatResponse(
        id="fake-id",
        model=model,
        choices=[
            Choice(index=0, message=Message(role="assistant", content=text), finish_reason="stop"),
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            reasoning_tokens=reasoning_tokens,
        ),
        time_info=None,
        latency=latency,
        cached=cached,
        compressed=compressed,
    )


# --- Fixtures -------------------------------------------------------------

@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def canned_request() -> ChatRequest:
    return ChatRequest(
        model=DEFAULT_MODEL,
        messages=[Message(role="user", content="What is 2+2?")],
        max_completion_tokens=100,
    )


@pytest.fixture
def canned_response() -> ChatResponse:
    return _build_response(text="4", completion_tokens=1, prompt_tokens=10)


@pytest.fixture
def build_response() -> Callable[..., ChatResponse]:
    """Expose the helper so tests can build variants."""
    return _build_response


@pytest.fixture
def fast_clock() -> list[float]:
    """A mutable clock for testing time-based logic.

    Append a timestamp; the clock returns the last value. Used by
    DualRateLimiter / CircuitBreaker tests to advance time without sleeping.
    """
    times: list[float] = [0.0]

    def _now() -> float:
        return times[-1]

    times.now = _now  # type: ignore[attr-defined]
    return times
