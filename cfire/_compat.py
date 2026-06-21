"""Legacy compatibility shim.

Provides the old `CompletionResult` dataclass and `CerebrasRaceClient`
class names that wrap the new cfire primitives, so news_agents.py and
web_demo.py keep working unchanged during Phase 4 migration.

This module is the only place that knows about the old API shape. Phase 4
will delete it once every consumer has migrated to direct cfire imports.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator

from .backends import CerebrasBackend, MockBackend
from .cache import MemoryLRU
from .client import AsyncCfire
from .config import (
    DEFAULT_CONCURRENCY,
    DEFAULT_MODEL,
    DEFAULT_REQ_PER_MIN,
    DEFAULT_TOK_PER_MIN,
    get_api_key,
)
from .models import ChatRequest, ChatResponse


# --- Legacy CompletionResult -------------------------------------------

@dataclass
class CompletionResult:
    """Drop-in for the old cerebras_race_client.CompletionResult.

    Field-for-field identical so any consumer that destructures or
    attribute-reads the legacy result keeps working.
    """
    text: str
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    latency: float
    cached: bool = False
    compressed: bool = False
    reasoning_tokens: int = 0

    @classmethod
    def from_response(cls, r: ChatResponse) -> "CompletionResult":
        return cls(
            text=r.text,
            completion_tokens=r.usage.completion_tokens,
            prompt_tokens=r.usage.prompt_tokens,
            total_tokens=r.usage.total_tokens,
            latency=r.latency,
            cached=r.cached,
            compressed=r.compressed,
            reasoning_tokens=r.usage.reasoning_tokens,
        )


# --- Legacy CerebrasRaceClient -----------------------------------------

def _messages_from_prompt(prompt: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return list(prompt)


class CerebrasRaceClient:
    """Drop-in for the old cerebras_race_client.CerebrasRaceClient.

    Constructor signature, method shapes, and attribute mutation surface
    (semaphore, concurrency) match the legacy class so web_demo.py's
    RaceManager can mutate them between sweep iterations without changes.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        concurrency: int = 24,  # legacy default was 24, not 16
        req_per_min: float = DEFAULT_REQ_PER_MIN,
        tok_per_min: float = DEFAULT_TOK_PER_MIN,
        redis_url: str | None = None,
        enable_compression: bool = True,
        enable_cache: bool = True,
        mock: bool = False,
    ):
        self._mock = mock

        if mock:
            backend: Any = MockBackend()
        else:
            backend = CerebrasBackend(
                base_url=base_url,
                api_key=api_key,
                concurrency=concurrency,
                use_msgpack=False,
            )

        cache: Any = MemoryLRU() if enable_cache else None

        self._client = AsyncCfire(
            backend=backend,
            cache=cache,
            model=model,
            concurrency=concurrency,
            req_per_min=req_per_min,
            tok_per_min=tok_per_min,
            enable_cache=enable_cache,
        )

        # Legacy attribute surface — web_demo.py mutates these between
        # sweep iterations to retune the client without re-creating it.
        # We expose them as properties proxying to self._client.

    # --- Legacy attribute surface (mutable) -----------------------------

    @property
    def api_key(self) -> str | None:
        return getattr(self._client.backend, "_api_key", None)

    @property
    def model(self) -> str:
        return self._client.model

    @model.setter
    def model(self, value: str) -> None:
        self._client.model = value

    @property
    def base_url(self) -> str:
        return getattr(self._client.backend, "base_url", "")

    @property
    def concurrency(self) -> int:
        return self._client.concurrency

    @concurrency.setter
    def concurrency(self, value: int) -> None:
        self._client.concurrency = value
        self._client.semaphore = asyncio.Semaphore(value)

    @property
    def semaphore(self) -> asyncio.Semaphore:
        return self._client.semaphore

    @semaphore.setter
    def semaphore(self, value: asyncio.Semaphore) -> None:
        self._client.semaphore = value

    @property
    def metrics(self):
        return self._client.metrics

    @property
    def cache(self):
        return self._client.cache

    @property
    def limiter(self):
        return self._client.limiter

    @property
    def circuit(self):
        return self._client.circuit

    @property
    def enable_compression(self) -> bool:
        return getattr(self._client.backend, "use_msgpack", False) or True  # legacy always true unless explicitly off

    @property
    def enable_cache(self) -> bool:
        return self._client.cache is not None

    @property
    def mock(self) -> bool:
        return self._mock

    @property
    def client(self):
        """Legacy httpx.AsyncClient accessor. Returns None in cfire
        (the transport is encapsulated). web_demo.py uses this only for
        truthiness checks — None breaks that, so we return a sentinel."""
        return self._client.backend if not self._mock else None

    # --- Async context manager ------------------------------------------

    async def __aenter__(self) -> "CerebrasRaceClient":
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *args) -> None:
        await self._client.__aexit__(*args)

    # --- Legacy methods -------------------------------------------------

    async def complete(
        self,
        prompt: str | list[dict[str, Any]],
        max_completion_tokens: int = 1000,
        temperature: float = 0.3,
        top_p: float = 1.0,
        reasoning_effort: str = "low",
        service_tier: str = "default",
        stream: bool = False,
    ) -> CompletionResult:
        request = ChatRequest(
            model=self.model,
            messages=_messages_from_prompt(prompt),
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
            service_tier=service_tier,  # type: ignore[arg-type]
            stream=stream,
        )
        response = await self._client.complete(request)
        return CompletionResult.from_response(response)

    async def complete_stream(
        self,
        prompt: str | list[dict[str, Any]],
        max_completion_tokens: int = 1000,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Legacy streaming. The original was fake (yielded whole text once);
        this version actually streams per-chunk text from the SSE parser."""
        request = ChatRequest(
            model=self.model,
            messages=_messages_from_prompt(prompt),
            max_completion_tokens=max_completion_tokens,
            stream=True,
            **{k: v for k, v in kwargs.items() if k in {"temperature", "top_p", "reasoning_effort"}},
        )
        async for chunk in self._client.stream(request):
            if chunk.delta:
                yield chunk.delta

    async def bulk_complete(
        self,
        prompts: list[str],
        max_completion_tokens: int = 1000,
        progress_queue: asyncio.Queue | None = None,
    ) -> list[CompletionResult]:
        requests = [
            ChatRequest(
                model=self.model,
                messages=_messages_from_prompt(p),
                max_completion_tokens=max_completion_tokens,
                reasoning_effort="low",
            )
            for p in prompts
        ]
        responses = await self._client.bulk(requests, progress_queue=progress_queue)
        return [CompletionResult.from_response(r) for r in responses]

    def report(self) -> dict[str, Any]:
        return self._client.report()


__all__ = ["CompletionResult", "CerebrasRaceClient"]
