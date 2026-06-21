"""Tests for cfire.streaming.

Fix #4 regression: the legacy complete_stream at cerebras_race_client.py:424-439
set stream=True on the payload but then called resp.json() and yielded the
whole text at once. This module parses SSE properly.
"""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from cfire.models import StreamChunk
from cfire.streaming import parse_chunk_obj, parse_sse_stream


# --- parse_chunk_obj --------------------------------------------------------

def test_parse_chunk_obj_extracts_delta_content():
    obj = {"choices": [{"delta": {"content": "Hello"}}]}
    chunk = parse_chunk_obj(obj)
    assert chunk is not None
    assert chunk.delta == "Hello"
    assert chunk.finish_reason is None


def test_parse_chunk_obj_extracts_finish_reason():
    obj = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    chunk = parse_chunk_obj(obj)
    assert chunk is not None
    assert chunk.delta == ""
    assert chunk.finish_reason == "stop"


def test_parse_chunk_obj_extracts_usage():
    obj = {
        "choices": [{"finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    chunk = parse_chunk_obj(obj)
    assert chunk is not None
    assert chunk.usage is not None
    assert chunk.usage.total_tokens == 15


def test_parse_chunk_obj_returns_none_for_empty_payload():
    """First chunk in a stream often has only role: no content, no finish, no usage."""
    obj = {"choices": [{"delta": {"role": "assistant"}}]}
    assert parse_chunk_obj(obj) is None


def test_parse_chunk_obj_handles_no_choices():
    """Tolerant of malformed responses."""
    assert parse_chunk_obj({}) is None


def test_parse_chunk_obj_uses_message_when_no_delta():
    """Some endpoints send `message` instead of `delta`."""
    obj = {"choices": [{"message": {"content": "hi"}}]}
    chunk = parse_chunk_obj(obj)
    assert chunk is not None
    assert chunk.delta == "hi"


# --- parse_sse_stream -------------------------------------------------------

class _FakeSSEResponse:
    """Fake httpx.Response that yields canned text chunks via aiter_text()."""

    def __init__(self, text_chunks: list[str]):
        self._chunks = text_chunks

    async def aiter_text(self) -> AsyncIterator[str]:
        for c in self._chunks:
            yield c


async def test_parse_sse_stream_yields_one_chunk_per_data_block():
    """Standard happy path: well-formed SSE stream, one block per chunk."""
    text = (
        'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":", world"}}]}\n\n'
        'data: {"choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n\n'
        'data: [DONE]\n\n'
    )
    resp = _FakeSSEResponse([text])
    chunks = [c async for c in parse_sse_stream(resp)]

    assert len(chunks) == 3
    assert chunks[0].delta == "Hello"
    assert chunks[1].delta == ", world"
    assert chunks[2].finish_reason == "stop"
    assert chunks[2].usage is not None
    assert chunks[2].usage.total_tokens == 3


async def test_parse_sse_stream_stops_cleanly_at_done():
    """Nothing after [DONE] should be yielded."""
    text = (
        'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        'data: [DONE]\n\n'
        'data: {"choices":[{"delta":{"content":"should-not-see"}}]}\n\n'
    )
    resp = _FakeSSEResponse([text])
    chunks = [c async for c in parse_sse_stream(resp)]
    assert len(chunks) == 1
    assert chunks[0].delta == "a"


async def test_parse_sse_stream_handles_split_blocks_across_chunks():
    """A single SSE block split across HTTP text chunks must still parse as one."""
    http_chunks = [
        'data: {"choices":[{"delta":{"content":"He"}}]}\n\n',
        'data: {"choices":[{"delta":{"content":"llo"}}]}\n\n'
        'data: [DONE]\n\n',
    ]
    resp = _FakeSSEResponse(http_chunks)
    chunks = [c async for c in parse_sse_stream(resp)]
    assert len(chunks) == 2
    assert chunks[0].delta == "He"
    assert chunks[1].delta == "llo"


async def test_parse_sse_stream_ignores_comments_and_event_lines():
    """SSE allows `: comment` lines and `event: foo` lines — we ignore both."""
    text = (
        ': this is a heartbeat comment\n\n'
        'event: ping\n'
        'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
        'data: [DONE]\n\n'
    )
    resp = _FakeSSEResponse([text])
    chunks = [c async for c in parse_sse_stream(resp)]
    assert len(chunks) == 1
    assert chunks[0].delta == "x"


async def test_parse_sse_stream_handles_empty_stream():
    """No data: lines at all — yields nothing."""
    resp = _FakeSSEResponse([""])
    chunks = [c async for c in parse_sse_stream(resp)]
    assert chunks == []


async def test_parse_sse_stream_skips_non_json_data():
    """A `data:` line with invalid JSON is silently dropped, not raised."""
    text = (
        'data: not-json\n\n'
        'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
        'data: [DONE]\n\n'
    )
    resp = _FakeSSEResponse([text])
    chunks = [c async for c in parse_sse_stream(resp)]
    assert len(chunks) == 1
    assert chunks[0].delta == "ok"


async def test_parse_sse_stream_handles_block_without_trailing_newline():
    """Trailing block (no \\n\\n after it) is still flushed."""
    text = (
        'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        'data: {"choices":[{"finish_reason":"stop"}]}'
        # No final \n\n
    )
    resp = _FakeSSEResponse([text])
    chunks = [c async for c in parse_sse_stream(resp)]
    assert len(chunks) == 2
    assert chunks[1].finish_reason == "stop"
