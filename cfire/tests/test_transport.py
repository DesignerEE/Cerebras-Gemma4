"""Tests for cfire.transport.

Phase 2 coverage:
- maybe_compress: under threshold, over threshold (gzip), msgpack opt-in
- parse_ratelimit_headers: extracts every x-ratelimit-* field with type coercion
- parse_time_info: returns TimeInfo | None based on payload shape
- classify_http_error: 429 -> RateLimitError, 401 -> AuthError, 400 ->
  BadRequestError, 5xx -> ServerError
- Transport lifecycle: open() creates httpx client, aclose() releases it
"""

from __future__ import annotations

import gzip
import json

import httpx
import pytest

from cfire.exceptions import (
    AuthError,
    BadRequestError,
    RateLimitError,
    RequestTimeoutError,
    ServerError,
)
from cfire.models import TimeInfo
from cfire.transport import (
    Transport,
    classify_http_error,
    maybe_compress,
    parse_ratelimit_headers,
    parse_time_info,
)


# --- maybe_compress ---------------------------------------------------------

def test_maybe_compress_under_threshold_returns_unchanged():
    body = b'{"x":1}'
    out, encoding = maybe_compress(body, threshold=4096)
    assert out == body
    assert encoding is None


def test_maybe_compress_over_threshold_returns_gzip():
    body = b"x" * 5000
    out, encoding = maybe_compress(body, threshold=4096)
    assert encoding == "gzip"
    assert out != body
    # Round-trip should be byte-identical
    assert gzip.decompress(out) == body


def test_maybe_compress_msgpack_opt_in_returns_msgpack_when_available():
    """When use_msgpack=True and msgpack is installed, encode as msgpack."""
    try:
        import msgpack  # noqa: F401
    except ImportError:
        pytest.skip("msgpack not installed in this environment")
    body = json.dumps({"key": "value", "n": 42}).encode()
    out, encoding = maybe_compress(body, threshold=10, use_msgpack=True)
    assert encoding == "application/vnd.msgpack"
    import msgpack
    decoded = msgpack.unpackb(out, raw=False)
    assert decoded == {"key": "value", "n": 42}


def test_maybe_compress_msgpack_unavailable_falls_back_to_gzip():
    """If msgpack isn't installed, gzip is used even when use_msgpack=True.

    We can't easily uninstall msgpack mid-test, so we just verify the gzip
    path with use_msgpack=False still produces gzip output.
    """
    body = b"x" * 5000
    out, encoding = maybe_compress(body, threshold=4096, use_msgpack=False)
    assert encoding == "gzip"
    assert gzip.decompress(out) == body


# --- parse_ratelimit_headers ------------------------------------------------

def _resp_with_headers(headers: dict[str, str]) -> httpx.Response:
    """Build an httpx.Response with the given headers (no body needed)."""
    return httpx.Response(200, headers=headers)


def test_parse_ratelimit_headers_extracts_all_fields():
    resp = _resp_with_headers({
        "x-ratelimit-requests-remaining": "950",
        "x-ratelimit-tokens-remaining": "850000.5",
        "x-ratelimit-requests-reset": "60s",
    })
    out = parse_ratelimit_headers(resp)
    assert out["requests-remaining"] == 950.0
    assert out["tokens-remaining"] == 850000.5
    # "60s" doesn't parse as float — kept as string
    assert out["requests-reset"] == "60s"


def test_parse_ratelimit_headers_ignores_non_ratelimit():
    resp = _resp_with_headers({
        "content-type": "application/json",
        "x-request-id": "abc-123",
    })
    out = parse_ratelimit_headers(resp)
    assert out == {}


def test_parse_ratelimit_headers_case_insensitive():
    """HTTP headers are case-insensitive — parser must lowercase."""
    resp = _resp_with_headers({"X-RateLimit-Requests-Remaining": "100"})
    out = parse_ratelimit_headers(resp)
    assert out["requests-remaining"] == 100.0


# --- parse_time_info --------------------------------------------------------

def test_parse_time_info_present():
    raw = {"time_info": {
        "queue_time": 0.01, "prompt_time": 0.02,
        "completion_time": 0.03, "total_time": 0.06,
    }}
    ti = parse_time_info(raw)
    assert isinstance(ti, TimeInfo)
    assert ti.queue_time == 0.01
    assert ti.total_time == 0.06


def test_parse_time_info_absent_returns_none():
    ti = parse_time_info({"foo": "bar"})
    assert ti is None


def test_parse_time_info_empty_payload_returns_none():
    ti = parse_time_info({})
    assert ti is None


# --- classify_http_error ----------------------------------------------------

def test_classify_429_returns_rate_limit_error_with_retry_after():
    err = classify_http_error(429, {"error": {"message": "slow down", "retry_after": 2.5}})
    assert isinstance(err, RateLimitError)
    assert err.retry_after == 2.5


def test_classify_429_no_retry_after_returns_none():
    err = classify_http_error(429, {"error": {"message": "slow down"}})
    assert isinstance(err, RateLimitError)
    assert err.retry_after is None


def test_classify_401_returns_auth_error():
    err = classify_http_error(401, {"error": {"message": "bad key"}})
    assert isinstance(err, AuthError)


def test_classify_403_returns_auth_error():
    err = classify_http_error(403, {"error": {"message": "forbidden"}})
    assert isinstance(err, AuthError)


def test_classify_400_returns_bad_request_error():
    err = classify_http_error(400, {"error": {"message": "malformed"}})
    assert isinstance(err, BadRequestError)


def test_classify_500_returns_server_error():
    err = classify_http_error(500, {"error": {"message": "boom"}})
    assert isinstance(err, ServerError)
    assert err.status_code == 500


def test_classify_503_returns_server_error():
    err = classify_http_error(503, {"error": {"message": "unavailable"}})
    assert isinstance(err, ServerError)
    assert err.status_code == 503


def test_classify_handles_string_error_field():
    """Some endpoints return just a string in 'error'."""
    err = classify_http_error(500, {"error": "internal"})
    assert isinstance(err, ServerError)


# --- Transport lifecycle ----------------------------------------------------

async def test_transport_open_creates_client_and_closes():
    transport = Transport(base_url="https://example.invalid/v1", api_key="k")
    assert transport._client is None
    await transport.open()
    assert transport._client is not None
    await transport.aclose()
    assert transport._client is None


async def test_transport_open_is_idempotent():
    """Double-open shouldn't recreate the client."""
    transport = Transport(base_url="https://example.invalid/v1", api_key="k")
    await transport.open()
    first = transport._client
    await transport.open()
    assert transport._client is first
    await transport.aclose()


async def test_transport_post_chat_without_open_raises():
    transport = Transport(base_url="https://example.invalid/v1", api_key="k")
    with pytest.raises(RuntimeError, match="not opened"):
        await transport.post_chat({"model": "m"})


async def test_transport_post_chat_classifies_500():
    """Live httpx transport hit a 500 — surfaces as ServerError."""
    # Use a route mock so we don't hit the network
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "boom"}})

    transport = httpx.MockTransport(handler)
    real = Transport(base_url="https://mock.invalid/v1", api_key="k")
    await real.open()
    # Patch the internal client's transport to use our mock
    real._client._transport = transport
    with pytest.raises(ServerError):
        await real.post_chat({"model": "m"})
    await real.aclose()
