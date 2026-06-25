"""Small local router for cfire.

Policy-driven, not heuristic-heavy. No ML routing — deterministic rules.
Satisfies the "small local router for fallback" constraint.

The Router itself conforms to the Backend Protocol so ``AsyncCfire`` can wrap
it transparently. Backends are registered by name (e.g. ``"cerebras"``,
``"diffusiongemma"``) so the policy can refer to them unambiguously.

Decision order in ``Router._select_backend(request)``:
  1. If request has tools and ``route_tools_to_diffusiongemma`` is True
     → diffusiongemma
  2. If request content matches any code keyword → diffusiongemma
  3. If request content matches any ``prefer_local_for`` pattern → diffusiongemma
  4. Else use the primary backend (``cerebras_first`` determines which)

Failover order in ``Router.complete()``:
  1. Try selected backend up to ``max_retries_per_backend`` times.
  2. If it raises a ``failover_on`` exception, move to the next backend.
  3. If every backend is exhausted, re-raise the most recent error.
"""

from __future__ import annotations

import logging
import re
from typing import Any, AsyncIterator

from pydantic import BaseModel, ConfigDict, Field

from .backends import Backend
from .exceptions import RetryableError
from .models import ChatRequest, ChatResponse, StreamChunk

log = logging.getLogger("cfire.router")


class RoutingPolicy(BaseModel):
    """Deterministic routing rules. No ML — pattern matching + heuristics."""

    model_config = ConfigDict(extra="allow")

    # Failover configuration
    failover_on: list[type[Exception]] = Field(
        default_factory=lambda: [RetryableError],
    )
    max_retries_per_backend: int = 1

    # Preference rules (checked in order)
    prefer_local_for: list[re.Pattern] = Field(default_factory=list)
    code_keywords: list[str] = Field(
        default_factory=lambda: [
            "code",
            "python",
            "javascript",
            "typescript",
            "rust",
            "go",
            "function",
            "debug",
            "refactor",
            "implement",
            "write a script",
            "shell",
            "bash",
            "sql",
            "regex",
            "algorithm",
        ]
    )

    # Backend ordering
    cerebras_first: bool = True

    # Heuristic: route requests with tools to DiffusionGemma4
    route_tools_to_diffusiongemma: bool = True


class Router:
    """Backend that routes across named backends based on a ``RoutingPolicy``.

    Conforms to the Backend Protocol so ``AsyncCfire`` can wrap it directly.
    """

    base_url = "router://"

    def __init__(
        self,
        backends: dict[str, Backend],
        policy: RoutingPolicy | None = None,
    ):
        if not backends:
            raise ValueError("Router requires at least one backend")
        self.backends = backends
        self.policy = policy or RoutingPolicy()
        self._primary_name: str = ""
        self._fallback_names: list[str] = []
        self._recompute_order()

    def _recompute_order(self) -> None:
        """Determine primary and fallback order from policy."""
        names = list(self.backends.keys())
        if self.policy.cerebras_first:
            self._primary_name = "cerebras" if "cerebras" in names else names[0]
        else:
            self._primary_name = (
                "diffusiongemma" if "diffusiongemma" in names else names[0]
            )
        self._fallback_names = [n for n in names if n != self._primary_name]

    # --- Backend Protocol ------------------------------------------------

    async def __aenter__(self) -> "Router":
        await self.open()
        return self

    async def __aexit__(self, *args) -> None:
        await self.aclose()

    async def open(self) -> None:
        for backend in self.backends.values():
            opener = getattr(backend, "open", None)
            if opener is not None:
                await opener()

    async def aclose(self) -> None:
        for backend in self.backends.values():
            closer = getattr(backend, "aclose", None)
            if closer is not None:
                await closer()

    # --- Routing logic ---------------------------------------------------

    def _select_backend(self, request: ChatRequest) -> tuple[str, Backend]:
        """Return ``(name, backend)`` for this request."""
        dg = self.backends.get("diffusiongemma")

        # Heuristic 1: Tool calling
        if self.policy.route_tools_to_diffusiongemma and request.tools and dg is not None:
            return "diffusiongemma", dg

        content = " ".join(
            str(m.content) for m in request.messages if m.content is not None
        ).lower()

        # Heuristic 2: Content contains code keywords
        if dg is not None:
            for kw in self.policy.code_keywords:
                if kw in content:
                    return "diffusiongemma", dg

        # Heuristic 3: prefer_local_for regex patterns
        if dg is not None:
            for pattern in self.policy.prefer_local_for:
                if pattern.search(content):
                    return "diffusiongemma", dg

        # Default to primary backend
        primary = self.backends.get(self._primary_name)
        if primary is not None:
            return self._primary_name, primary

        # Fallback to first available
        name, backend = next(iter(self.backends.items()))
        return name, backend

    def _is_failover_exception(self, exc: Exception) -> bool:
        """Check if exception type matches ``failover_on``."""
        for exc_type in self.policy.failover_on:
            if isinstance(exc, exc_type):
                return True
        return False

    # --- complete with failover ------------------------------------------

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Route to selected backend, with failover on configured exceptions."""
        order = [self._select_backend(request)]
        # Append fallbacks that were not selected first
        for name in self._fallback_names:
            if name != order[0][0]:
                order.append((name, self.backends[name]))

        last_error: Exception | None = None
        for name, backend in order:
            for attempt in range(self.policy.max_retries_per_backend + 1):
                try:
                    return await backend.complete(request)
                except Exception as exc:
                    last_error = exc
                    if not self._is_failover_exception(exc):
                        raise
                    if attempt < self.policy.max_retries_per_backend:
                        log.warning(
                            "Router: %s failed (attempt %d/%d): %s",
                            name,
                            attempt + 1,
                            self.policy.max_retries_per_backend + 1,
                            exc,
                        )
                        continue
                    # Exhausted retries on this backend; move to next
                    break

        if last_error is not None:
            raise last_error
        raise RuntimeError("Router: no backends available")

    # --- stream with failover --------------------------------------------

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Stream from selected backend. Failover on initial connection errors.

        Mid-stream failures cannot be cleanly retried, so we only failover when
        the initial ``backend.stream()`` call raises a failover-eligible exception.
        """
        order = [self._select_backend(request)]
        for name in self._fallback_names:
            if name != order[0][0]:
                order.append((name, self.backends[name]))

        last_error: Exception | None = None
        for name, backend in order:
            try:
                async for chunk in backend.stream(request):
                    yield chunk
                return
            except Exception as exc:
                last_error = exc
                if not self._is_failover_exception(exc):
                    raise
                log.warning("Router: stream failed on %s, trying fallback: %s", name, exc)

        if last_error is not None:
            raise last_error
        raise RuntimeError("Router: no backends available")


__all__ = ["Router", "RoutingPolicy"]
