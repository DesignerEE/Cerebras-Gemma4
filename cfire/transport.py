"""HTTP transport for cfire.

Wraps httpx.AsyncClient with the specifics the legacy client got right, and
adds the pieces it missed:

Existing (kept):
  - HTTP/2 enabled
  - max_keepalive=concurrency*2, max_connections=concurrency*4
    (cerebras_race_client.py:271-274)
  - Preflight GET /models to warm TLS+HTTP/2 (cerebras_race_client.py:283)
  - gzip compression above 4096-byte threshold

New (Phase 2):
  - parse_ratelimit_headers(resp) -> dict of x-ratelimit-* values
  - parse_time_info(resp_json) -> TimeInfo | None
  - request() raises proper cfire exceptions instead of bare RuntimeError
  - Smart error classification by HTTP status code
"""

from __future__ import annotations

import gzip
import json
import time
from typing import Any

import httpx

from .exceptions import (
    AuthError,
    BadRequestError,
    RateLimitError,
    RequestTimeoutError,
    ServerError,
)
from .models import TimeInfo

# Optional msgpack — backend-conditional, off by default
try:
    import msgpack  # type: ignore[import-untyped]
    HAS_MSGPACK = True
except Exception:
    HAS_MSGPACK = False


DEFAULT_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


def maybe_compress(
    body: bytes,
    threshold: int = 4096,
    use_msgpack: bool = False,
) -> tuple[bytes, str | None]:
    """Return (body, content_encoding).

    - Under threshold: (body, None)
    - Over threshold + msgpack available + use_msgpack=True: (msgpack, "vnd.msgpack")
    - Over threshold + otherwise: (gzip, "gzip")

    Legacy behavior (cerebras_race_client.py:311-317) was gzip-only because
    Cerebras docs at the time said msgpack unsupported. The current docs
    (2026) accept `application/vnd.msgpack` so we expose it as opt-in.
    """
    if len(body) < threshold:
        return body, None
    if use_msgpack and HAS_MSGPACK:
        # The original payload is JSON; msgpack-encode the *parsed* object
        # so we don't double-wrap strings-in-strings.
        try:
            obj = json.loads(body)
            return msgpack.packb(obj, use_bin_type=True), "application/vnd.msgpack"
        except Exception:
            pass  # fall through to gzip
    return gzip.compress(body), "gzip"


def parse_ratelimit_headers(resp: httpx.Response) -> dict[str, Any]:
    """Pull every x-ratelimit-* header into a dict.

    Cerebras returns these on every response — they're the source of truth
    for adaptive throttling. Returns floats where parseable, strings otherwise.
    """
    out: dict[str, Any] = {}
    for name, value in resp.headers.multi_items():
        if not name.lower().startswith("x-ratelimit-"):
            continue
        key = name.lower()[len("x-ratelimit-"):]
        # Try float, fall back to string
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            out[key] = value
    return out


def parse_time_info(resp_json: dict[str, Any]) -> TimeInfo | None:
    """Extract Cerebras-unique time_info if present."""
    return TimeInfo.from_api(resp_json.get("time_info"))


def classify_http_error(status_code: int, body: Any) -> Exception:
    """Convert HTTP status + parsed body into the right cfire exception.

    Replaces the bare `RuntimeError(f"API error: {code}")` at
    cerebras_race_client.py:351 with proper exception classes so RetryPolicy
    can decide correctly.
    """
    msg = ""
    retry_after: float | None = None
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            msg = str(err.get("message") or err.get("code") or "")
            ra = err.get("retry_after")
            if ra is not None:
                try:
                    retry_after = float(ra)
                except (TypeError, ValueError):
                    pass
        elif isinstance(err, str):
            msg = err

    if status_code == 429:
        return RateLimitError(msg or "rate limited", retry_after=retry_after)
    if status_code in (401, 403):
        return AuthError(msg or f"auth error {status_code}")
    if 400 <= status_code < 500:
        return BadRequestError(f"{status_code}: {msg}" if msg else f"bad request {status_code}")
    if 500 <= status_code < 600:
        return ServerError(status_code, msg)
    return ServerError(status_code, f"unexpected status {status_code}: {msg}")


class Transport:
    """httpx.AsyncClient wrapper with preflight warmup.

    Holds connection pool, applies compression, classifies errors.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        concurrency: int = 16,
        timeout: httpx.Timeout | None = None,
        compress_threshold: int = 4096,
        use_msgpack: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.concurrency = concurrency
        self.timeout = timeout or DEFAULT_TIMEOUT
        self.compress_threshold = compress_threshold
        self.use_msgpack = use_msgpack
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "Transport":
        await self.open()
        return self

    async def __aexit__(self, *args) -> None:
        await self.aclose()

    async def open(self) -> None:
        """Create the httpx client + preflight GET /models to warm TLS+HTTP/2."""
        if self._client is not None:
            return
        limits = httpx.Limits(
            max_keepalive_connections=self.concurrency * 2,
            max_connections=self.concurrency * 4,
            keepalive_expiry=120.0,
        )
        self._client = httpx.AsyncClient(
            http2=True,
            timeout=self.timeout,
            limits=limits,
        )
        # Preflight warmup — same intent as cerebras_race_client.py:283.
        # Swallow errors so a dead /models endpoint doesn't block real work.
        try:
            await self._client.get(f"{self.base_url}/models", headers=self._auth_headers())
        except Exception:
            pass

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def post_chat(
        self,
        payload: dict[str, Any],
        *,
        stream: bool = False,
    ) -> httpx.Response:
        """POST /chat/completions. Returns the raw httpx.Response.

        Raises cfire exceptions on non-2xx. Caller is responsible for parsing
        the body (JSON for non-stream, SSE for stream).
        """
        if self._client is None:
            raise RuntimeError("Transport not opened — use `async with Transport(...)`")

        headers = self._auth_headers()
        body_bytes = json.dumps(payload).encode()
        body, encoding = maybe_compress(body_bytes, self.compress_threshold, self.use_msgpack)
        if encoding:
            if encoding == "gzip":
                headers["Content-Encoding"] = "gzip"
            else:
                headers["Content-Type"] = encoding  # application/vnd.msgpack

        start = time.perf_counter()
        try:
            resp = await self._client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                content=body,
            )
        except httpx.TimeoutException as e:
            raise RequestTimeoutError(f"chat request timed out after {self.timeout.read}s") from e
        except httpx.ConnectError as e:
            raise ServerError(503, f"connect error: {e}") from e

        # Latency instrumentation lives on the response object for callers
        resp.elapsed_client = time.perf_counter() - start  # type: ignore[attr-defined]

        if resp.status_code >= 400:
            try:
                err_body = resp.json()
            except Exception:
                err_body = {"error": resp.text}
            raise classify_http_error(resp.status_code, err_body)

        return resp


__all__ = [
    "Transport",
    "maybe_compress",
    "parse_ratelimit_headers",
    "parse_time_info",
    "classify_http_error",
    "DEFAULT_TIMEOUT",
    "HAS_MSGPACK",
]
