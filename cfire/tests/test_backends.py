"""Tests for cfire.backends.

Phase 2 coverage:
- CerebrasBackend accepts custom base_url (constructor + .rstrip("/"))
- CerebrasBackend.complete parses choices/usage/time_info correctly
- CerebrasBackend._parse_response is tolerant of missing fields
- MockBackend mirrors the legacy mock behavior (sleep + random tokens + latency=0)
- Backend Protocol check: both CerebrasBackend and MockBackend conform
"""

from __future__ import annotations

import asyncio
import time

import pytest

from cfire.backends import Backend, CerebrasBackend, DiffusionGemmaBackend, MockBackend
from cfire.config import DEFAULT_MODEL
from cfire.models import ChatRequest, Message


# --- CerebrasBackend construction ------------------------------------------

def test_cerebras_backend_defaults_to_cerebras_cloud():
    """With no kwargs, base_url points at api.cerebras.ai."""
    b = CerebrasBackend(api_key="csk-test")
    assert b.base_url == "https://api.cerebras.ai/v1"


def test_cerebras_backend_accepts_custom_base_url():
    """Custom CDN endpoint requirement: base_url is first-class configurable."""
    b = CerebrasBackend(base_url="https://my-cdn.example.com/v1/", api_key="k")
    assert b.base_url == "https://my-cdn.example.com/v1"  # trailing slash stripped


def test_cerebras_backend_lazy_api_key_until_open():
    """api_key=None is held until open() so import doesn't fail without env."""
    b = CerebrasBackend(api_key=None)
    assert b._api_key is None


def test_cerebras_backend_conforms_to_protocol():
    """runtime_checkable Protocol — isinstance check should pass."""
    b = CerebrasBackend(api_key="k")
    assert isinstance(b, Backend)


# --- Response parsing -------------------------------------------------------

def test_parse_response_happy_path():
    """Full Cerebras response shape with usage + time_info."""
    data = {
        "id": "chat-xyz",
        "model": DEFAULT_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "4"},
                "finish_reason": "stop",
            },
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 1,
            "total_tokens": 11,
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
        "time_info": {
            "queue_time": 0.001,
            "prompt_time": 0.005,
            "completion_time": 0.010,
            "total_time": 0.016,
        },
    }
    r = CerebrasBackend._parse_response(data, latency=0.05, compressed=False)
    assert r.id == "chat-xyz"
    assert r.model == DEFAULT_MODEL
    assert r.text == "4"
    assert r.usage.prompt_tokens == 10
    assert r.usage.completion_tokens == 1
    assert r.usage.reasoning_tokens == 0
    assert r.latency == 0.05
    assert r.compressed is False
    assert r.time_info is not None
    assert r.time_info.queue_time == 0.001
    assert r.time_info.total_time == 0.016


def test_parse_response_tolerates_missing_fields():
    """Some endpoints omit usage, time_info, or even choices."""
    # No usage, no time_info, empty choices
    r = CerebrasBackend._parse_response({"id": "x", "model": "m"}, latency=0.0)
    assert r.id == "x"
    assert r.text == ""              # no choices -> empty
    assert r.usage.prompt_tokens == 0
    assert r.usage.completion_tokens == 0
    assert r.time_info is None


def test_parse_response_uses_delta_when_no_message():
    """Streaming-style payloads use `delta` instead of `message`."""
    data = {
        "choices": [{"delta": {"content": "stream-text"}}],
    }
    r = CerebrasBackend._parse_response(data, latency=0.0)
    assert r.text == "stream-text"


def test_parse_response_carries_reasoning_tokens():
    """OpenAI reasoning-model extension: completion_tokens_details.reasoning_tokens."""
    data = {
        "choices": [{"message": {"role": "assistant", "content": "thinking..."}}],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 100,
            "total_tokens": 105,
            "completion_tokens_details": {"reasoning_tokens": 80},
        },
    }
    r = CerebrasBackend._parse_response(data, latency=0.0)
    assert r.usage.reasoning_tokens == 80
    # Back-compat alias still works
    assert r.reasoning_tokens == 80


# --- MockBackend ------------------------------------------------------------

async def test_mock_backend_returns_text():
    """MockBackend mirrors legacy mock — returns 'Mock response.'."""
    backend = MockBackend(latency_range=(0.0, 0.0))  # no sleep for speed
    req = ChatRequest(model="m", messages=[Message(role="user", content="hi")])
    r = await backend.complete(req)
    assert r.text == "Mock response."
    assert r.usage.prompt_tokens == 50  # legacy default


async def test_mock_backend_random_token_count_in_range():
    """completion_tokens is max * ratio in [token_ratio[0], token_ratio[1])."""
    backend = MockBackend(
        latency_range=(0.0, 0.0),
        token_ratio=(0.9, 0.95),
        prompt_tokens=10,
    )
    req = ChatRequest(
        model="m",
        messages=[Message(role="user", content="hi")],
        max_completion_tokens=100,
    )
    r = await backend.complete(req)
    assert 90 <= r.usage.completion_tokens <= 100
    assert r.usage.total_tokens == r.usage.completion_tokens + 10


async def test_mock_backend_latency_is_zero_legacy_behavior():
    """Legacy mock set latency=0.0 — Metrics totals stayed at 0 in mock races."""
    backend = MockBackend(latency_range=(0.05, 0.05))
    req = ChatRequest(model="m", messages=[Message(role="user", content="hi")])
    r = await backend.complete(req)
    assert r.latency == 0.0


async def test_mock_backend_stream_yields_single_chunk():
    backend = MockBackend()
    req = ChatRequest(model="m", messages=[Message(role="user", content="hi")])
    chunks = []
    async for c in backend.stream(req):
        chunks.append(c)
    assert len(chunks) == 1
    assert chunks[0].delta == "Mock response."
    assert chunks[0].finish_reason == "stop"


def test_mock_backend_conforms_to_protocol():
    assert isinstance(MockBackend(), Backend)


async def test_mock_backend_async_context_manager():
    """async with MockBackend() as b: ... works as no-op lifecycle."""
    async with MockBackend() as b:
        req = ChatRequest(model="m", messages=[Message(role="user", content="hi")])
        r = await b.complete(req)
        assert r.text == "Mock response."


async def test_cerebras_backend_stream_uses_transport_stream_chat():
    """Regression: stream() must call Transport.stream_chat(), not _client directly."""
    from unittest.mock import AsyncMock, MagicMock, patch

    backend = CerebrasBackend(api_key="k")

    async def _mock_aiter_text():
        yield 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        yield 'data: [DONE]\n\n'

    mock_cm = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.aiter_text = _mock_aiter_text
    mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_cm.__aexit__ = AsyncMock(return_value=None)

    with patch.object(backend._transport, "stream_chat", return_value=mock_cm) as mock_stream_chat:
        req = ChatRequest(model="m", messages=[Message(role="user", content="hello")], stream=True)
        chunks = [c async for c in backend.stream(req)]

    mock_stream_chat.assert_called_once()
    assert any(c.delta == "hi" for c in chunks)


# --- DiffusionGemmaBackend -------------------------------------------------

def test_diffusiongemma_backend_defaults():
    b = DiffusionGemmaBackend()
    assert b.base_url == "https://api.cerebras.ai/v1"
    assert b._default_model() == "nvidia/diffusiongemma-26B-A4B-it-NVFP4"


def test_diffusiongemma_backend_accepts_custom_base_url():
    b = DiffusionGemmaBackend(base_url="http://localhost:9999/v1/")
    assert b.base_url == "http://localhost:9999/v1"


def test_diffusiongemma_backend_conforms_to_protocol():
    assert isinstance(DiffusionGemmaBackend(), Backend)


def test_diffusiongemma_backend_no_auth_required():
    """Local DiffusionGemma4 needs no API key; constructor must not fail."""
    b = DiffusionGemmaBackend()
    assert b._api_key == ""
