"""Metrics observer for cfire.

Replaces the unbounded-list Metrics class (cerebras_race_client.py:77-114)
with a callback-based observer pattern plus reservoir-sampled percentiles.

Why reservoir sampling: the legacy code stored every latency in
`self.latencies: list[float]`. In a long-running bulk_complete that did
10k requests, this list grew to 10k floats — wasteful and slow to sort.
A reservoir of N=1024 samples gives unbiased p50/p99 with constant memory.

Why callbacks: the legacy class had no event surface — the F1 dashboard
had to poll `.report()`. Callbacks let dashboards push updates live.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

from .models import ChatResponse


@dataclass
class MetricEvent:
    """One event emitted by Metrics.record_response."""
    kind: str  # "response" | "error" | "retry"
    latency: float = 0.0
    tokens: int = 0
    cached: bool = False
    compressed: bool = False
    error: str | None = None


class Metrics:
    """Accumulates counters + reservoir-sampled latency percentiles.

    Counters are sync-safe under a single asyncio event loop because Python
    doesn't preempt between `await` points — so `self.x += 1` is atomic from
    the loop's perspective. No lock needed.
    """

    def __init__(self, reservoir_size: int = 1024):
        self.reservoir_size = reservoir_size
        self.requests: int = 0
        self.cached: int = 0
        self.compressed: int = 0
        self.errors: int = 0
        self.retries: int = 0
        self.total_tokens: int = 0

        self._seen: int = 0
        self._reservoir: list[float] = []
        self._callbacks: list[Callable[[MetricEvent], None]] = []

    # --- Callbacks ------------------------------------------------------

    def on_event(self, cb: Callable[[MetricEvent], None]) -> None:
        """Register a callback invoked on every record_* call. Sync only."""
        self._callbacks.append(cb)

    def _emit(self, event: MetricEvent) -> None:
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception:
                pass  # never let a metrics callback kill the request

    # --- Recording ------------------------------------------------------

    def record_response(self, response: ChatResponse) -> None:
        """Record a successful (possibly cached) response."""
        self.requests += 1
        if response.cached:
            self.cached += 1
        if response.compressed:
            self.compressed += 1
        self.total_tokens += response.usage.total_tokens
        self._observe_latency(response.latency)
        self._emit(MetricEvent(
            kind="response",
            latency=response.latency,
            tokens=response.usage.total_tokens,
            cached=response.cached,
            compressed=response.compressed,
        ))

    def record_error(self, error: str | None = None) -> None:
        """Record a failed request."""
        self.errors += 1
        self._emit(MetricEvent(kind="error", error=error))

    def record_retry(self) -> None:
        """Record that a retry was attempted."""
        self.retries += 1
        self._emit(MetricEvent(kind="retry"))

    # --- Reservoir sampling --------------------------------------------

    def _observe_latency(self, latency: float) -> None:
        """Add to reservoir using Algorithm R (Vitter 1985).

        Unbiased estimator: every sample has equal probability of being
        retained, regardless of total stream length.
        """
        self._seen += 1
        if len(self._reservoir) < self.reservoir_size:
            self._reservoir.append(latency)
            return
        # Replace a random slot with probability reservoir_size / seen
        idx = random.randint(0, self._seen - 1)
        if idx < self.reservoir_size:
            self._reservoir[idx] = latency

    def percentile(self, p: float) -> float:
        """Estimate the p-th percentile (0..1) from the reservoir."""
        if not self._reservoir:
            return 0.0
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"percentile must be 0..1, got {p}")
        sorted_lat = sorted(self._reservoir)
        # Linear interpolation between adjacent reservoir entries
        k = p * (len(sorted_lat) - 1)
        lo = int(k)
        hi = min(lo + 1, len(sorted_lat) - 1)
        frac = k - lo
        return sorted_lat[lo] * (1 - frac) + sorted_lat[hi] * frac

    # --- Snapshot -------------------------------------------------------

    def report(self) -> dict[str, Any]:
        """Dict snapshot — same shape as legacy Metrics.report() so the
        F1 dashboard doesn't need to change."""
        avg = (sum(self._reservoir) / len(self._reservoir)) if self._reservoir else 0.0
        return {
            "requests": self.requests,
            "cached": self.cached,
            "compressed": self.compressed,
            "errors": self.errors,
            "retries": self.retries,
            "total_tokens": self.total_tokens,
            "avg_latency": avg,
            "p50": self.percentile(0.50),
            "p99": self.percentile(0.99),
            "cache_hit_rate": (self.cached / self.requests) if self.requests else 0.0,
        }

    def reset(self) -> None:
        """Clear all counters and the reservoir."""
        self.requests = 0
        self.cached = 0
        self.compressed = 0
        self.errors = 0
        self.retries = 0
        self.total_tokens = 0
        self._seen = 0
        self._reservoir.clear()


__all__ = ["Metrics", "MetricEvent"]
