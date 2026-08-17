"""Pluggable backends for cfire.

Every backend implements the Backend Protocol so the Router (Phase 3) can
fail over between them transparently. All accept base_url + api_key,
satisfying the "custom CDN + custom Cerebras endpoints" requirement at the
constructor level.

Phase 2 ships only CerebrasBackend (port of the existing client's HTTP
path with proper Pydantic parsing + cfire exception hierarchy).

Phase 3 will add:
  - LocalBackend  — OpenAI-compatible local server (default 127.0.0.1:8123)
  - CDNBackend    — Edge proxy that forwards to an inner backend
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator, Protocol, runtime_checkable

from .config import (
    CEREBRAS_BASE_URL,
    COMPRESS_THRESHOLD_BYTES,
    DEFAULT_CONCURRENCY,
    DIFFUSIONGEMMA_API_KEY,
    DIFFUSIONGEMMA_BASE_URL,
    DIFFUSIONGEMMA_MODEL,
    get_api_key,
)
from .models import ChatRequest, ChatResponse, StreamChunk
from .streaming import parse_sse_stream
from .transport import Transport, parse_time_info


@runtime_checkable
class Backend(Protocol):
    """Every backend conforms to this. The Router speaks to backends
    through this interface only, so they're interchangeable."""
    base_url: str

    async def complete(self, request: ChatRequest) -> ChatResponse: ...
    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]: ...
    async def aclose(self) -> None: ...


class CerebrasBackend:
    """Primary backend — POST {base_url}/chat/completions.

    Defaults match the values cfire.config reads from env. Override via
    kwargs to plug in a custom Cerebras-compatible endpoint (staging,
    preview, enterprise dedicated tier, self-hosted proxy, etc.).
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        concurrency: int = DEFAULT_CONCURRENCY,
        compress_threshold: int = COMPRESS_THRESHOLD_BYTES,
        use_msgpack: bool = False,
        transport: Transport | None = None,
    ):
        self.base_url = (base_url or CEREBRAS_BASE_URL).rstrip("/")
        self._api_key = api_key  # lazy-resolved on open() if None
        self.concurrency = concurrency
        self._transport = transport or Transport(
            base_url=self.base_url,
            api_key=api_key or "",  # placeholder; replaced on open()
            concurrency=concurrency,
            compress_threshold=compress_threshold,
            use_msgpack=use_msgpack,
        )
        self._owns_transport = transport is None

    async def __aenter__(self) -> "CerebrasBackend":
        await self.open()
        return self

    async def __aexit__(self, *args) -> None:
        await self.aclose()

    async def open(self) -> None:
        # Resolve API key on first open() so constructors don't fail at
        # import time if env isn't set yet.
        if self._api_key is None:
            self._api_key = get_api_key()
            self._transport.api_key = self._api_key
        await self._transport.open()

    async def aclose(self) -> None:
        if self._owns_transport:
            await self._transport.aclose()

    # --- Request building -----------------------------------------------

    def _payload(self, request: ChatRequest) -> dict[str, Any]:
        """Convert ChatRequest to the wire payload.

        Uses exclude_none=True so optional fields (prompt_cache_key etc.)
        don't appear when unset. The non-None defaults like service_tier
        and reasoning_effort are always sent — Cerebras accepts them.
        """
        return request.model_dump(mode="json", exclude_none=True)

    # --- Backend Protocol -----------------------------------------------

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Single non-streaming request. Raises cfire exceptions on error."""
        # Force stream=False on the wire regardless of the request flag —
        # stream() handles the streaming path separately.
        payload = self._payload(request)
        payload["stream"] = False

        resp = await self._transport.post_chat(payload, stream=False)
        data = resp.json()
        return self._parse_response(data, getattr(resp, "elapsed_client", 0.0), compressed=_payload_compressed(resp))

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Streaming request via SSE. Yields StreamChunk until [DONE]."""
        payload = self._payload(request)
        payload["stream"] = True

        # Use the transport's streaming entry point instead of reaching into
        # its private httpx client. Compression and auth headers are applied
        # inside Transport.stream_chat(); we handle SSE parsing and errors here.
        async with self._transport.stream_chat(payload) as resp:
            if resp.status_code >= 400:
                # Materialize the body for classification, then raise
                await resp.aread()
                from .transport import classify_http_error
                try:
                    err_body = resp.json()
                except Exception:
                    err_body = {"error": resp.text}
                raise classify_http_error(resp.status_code, err_body)
            async for chunk in parse_sse_stream(resp):
                yield chunk

    # --- Response parsing -----------------------------------------------

    @staticmethod
    def _parse_response(
        data: dict[str, Any],
        latency: float,
        compressed: bool = False,
    ) -> ChatResponse:
        """Build a ChatResponse from a Cerebras /chat/completions body.

        Tolerant of missing fields — `usage` may be absent on early stream
        chunks, `time_info` is Cerebras-only.
        """
        from .models import Choice, Message, Usage

        choices_raw = data.get("choices") or []
        choices: list[Choice] = []
        for c in choices_raw:
            msg_raw = c.get("message") or c.get("delta") or {}
            choices.append(Choice(
                index=c.get("index", 0),
                message=Message(
                    role=msg_raw.get("role", "assistant"),
                    content=msg_raw.get("content", ""),
                ),
                finish_reason=c.get("finish_reason"),
            ))

        return ChatResponse(
            id=data.get("id", ""),
            model=data.get("model", ""),
            choices=choices,
            usage=Usage.from_api(data.get("usage")),
            time_info=parse_time_info(data),
            latency=latency,
            cached=False,
            compressed=compressed,
        )


def _payload_compressed(resp: Any) -> bool:
    """Best-effort: was the request gzipped? Transport doesn't currently
    surface this on the response, so we always return False here. The
    Metrics observer records compressed=False unless a future transport
    exposes request encoding on the response object."""
    return False


class DiffusionGemmaBackend(CerebrasBackend):
    """Local DiffusionGemma4 inference server.

    Targets the OpenAI-compatible endpoint served by vLLM on
    ``CFIRE_DIFFUSIONGEMMA_BASE_URL`` (default ``https://api.cerebras.ai/v1``).
    DiffusionGemma4 is treated as a coding model: the Router sends tool-calling
    and code-keyword requests here by default.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url or self._default_base_url(),
            api_key=api_key if api_key is not None else DIFFUSIONGEMMA_API_KEY,
            **kwargs,
        )

    def _default_base_url(self) -> str:
        return DIFFUSIONGEMMA_BASE_URL

    def _default_model(self) -> str:
        return DIFFUSIONGEMMA_MODEL

    async def open(self) -> None:
        # Local inference servers typically require no auth. Avoid calling
        # get_api_key() unless the caller explicitly passed api_key=None.
        if self._api_key is None:
            self._api_key = DIFFUSIONGEMMA_API_KEY
            self._transport.api_key = self._api_key
        await self._transport.open()


class MockBackend:
    """Returns canned responses with no network. For tests + dashboard mock races.

    Mirrors the legacy mock path at cerebras_race_client.py:320-329:
      - sleeps uniform(latency_range) to simulate request time
      - emits `Mock response.` text
      - completion_tokens is a random ratio of max_completion_tokens
      - latency=0.0 (legacy behavior — metrics total stays at 0 in mock)
    """

    base_url = "mock://"

    def __init__(
        self,
        latency_range: tuple[float, float] = (0.05, 0.4),
        token_ratio: tuple[float, float] = (0.7, 0.95),
        prompt_tokens: int = 50,
    ):
        self.latency_range = latency_range
        self.token_ratio = token_ratio
        self.prompt_tokens = prompt_tokens

    async def __aenter__(self) -> "MockBackend":
        return self

    async def __aexit__(self, *args) -> None:
        pass

    async def open(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

    async def complete(self, request: ChatRequest) -> ChatResponse:
        import asyncio
        import random
        await asyncio.sleep(random.uniform(*self.latency_range))
        tokens = max(
            10,
            int(request.max_completion_tokens * random.uniform(*self.token_ratio)),
        )
        return ChatResponse(
            id="mock",
            model=request.model or "mock",
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": "Mock response."},
                "finish_reason": "stop",
            }],
            usage={
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": tokens,
                "total_tokens": tokens + self.prompt_tokens,
            },
            time_info=None,
            latency=0.0,
            cached=False,
            compressed=False,
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(delta="Mock response.", finish_reason="stop")


__all__ = ["Backend", "CerebrasBackend", "DiffusionGemmaBackend", "MockBackend"]
