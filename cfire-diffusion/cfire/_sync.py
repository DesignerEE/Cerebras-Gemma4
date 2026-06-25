"""Sync Cfire wrapper.

Cfire is a thin sync facade over AsyncCfire. Lets notebook / CLI / non-async
callers use the library without managing their own event loop.

Implementation: a background daemon thread runs a single asyncio event loop.
Each sync method submits a coroutine to it via run_coroutine_threadsafe and
blocks on the resulting Future. This is the canonical pattern and works
even when called from inside an async context (which `asyncio.run` cannot).

Limitation: streaming is async-only. Sync callers should use bulk() and
process results after. (A cross-thread generator bridge is possible but
the API ergonomics are worse than just using AsyncCfire.)
"""

from __future__ import annotations

import asyncio
import queue as _queue
import threading
from typing import Any, Iterable, Iterator

from .client import AsyncCfire
from .models import ChatRequest, ChatResponse, StreamChunk

_SENTINEL = object()


class _BackgroundLoop:
    """One daemon thread running one event loop, for the lifetime of a Cfire."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="cfire-bg")
        self.thread.start()
        self._ready.wait(timeout=5.0)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self._ready.set()
        self.loop.run_forever()

    def submit(self, coro):
        """Schedule coro on the bg loop, block until done, return result."""
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()  # blocks calling thread; raises on coro exception

    def shutdown(self) -> None:
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=2.0)


class Cfire:
    """Sync wrapper around AsyncCfire.

    Usage:
        with Cfire() as client:
            r = client.complete(ChatRequest(messages=[{"role":"user","content":"hi"}]))
            print(r.text)
    """

    def __init__(self, **kwargs: Any):
        self._async = AsyncCfire(**kwargs)
        self._bg = _BackgroundLoop()
        try:
            # Open the underlying async client
            self._bg.submit(self._async.__aenter__())
        except BaseException:
            self._bg.shutdown()
            raise

    # --- Sync surface ---------------------------------------------------

    def complete(self, request: ChatRequest) -> ChatResponse:
        return self._bg.submit(self._async.complete(request))

    def bulk(self, requests: Iterable[ChatRequest]) -> list[ChatResponse]:
        return self._bg.submit(self._async.bulk(list(requests)))

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        """Sync iterator bridge for the async stream.

        Cross-thread queue: the bg loop produces, the caller thread consumes.
        Errors surface as raises on the next __next__ call.
        """
        q: _queue.Queue = _queue.Queue()
        async_gen = self._async.stream(request)

        async def _produce():
            try:
                async for chunk in async_gen:
                    q.put(chunk)
            except BaseException as e:
                q.put(e)
            finally:
                q.put(_SENTINEL)

        self._bg.submit(_produce())

        while True:
            item = q.get()
            if item is _SENTINEL:
                return
            if isinstance(item, Exception):
                raise item
            yield item  # type: ignore[misc]

    def report(self) -> dict[str, Any]:
        return self._async.report()

    # --- Lifecycle ------------------------------------------------------

    def close(self) -> None:
        try:
            self._bg.submit(self._async.__aexit__(None, None, None))
        except Exception:
            pass
        self._bg.shutdown()

    def __enter__(self) -> "Cfire":
        return self

    def __exit__(self, *args) -> None:
        self.close()


__all__ = ["Cfire"]
