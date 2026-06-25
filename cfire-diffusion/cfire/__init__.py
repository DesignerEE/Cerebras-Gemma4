"""cfire — Fast-inference library for Cerebras Inference and DiffusionGemma4.

Public surface (Phase 3 complete):
  - Models:       ChatRequest, ChatResponse, Message, Usage, TimeInfo, StreamChunk
  - Clients:      AsyncCfire, Cfire (sync wrapper)
  - Backends:     CerebrasBackend, DiffusionGemmaBackend, MockBackend, Backend (Protocol)
  - Router:       Router, RoutingPolicy
  - Reliability:  CircuitBreaker, RetryPolicy, DualRateLimiter
  - Cache:        MemoryLRU, RedisCache, TieredCache, cache_key
  - Exceptions:   CerebrasError + Retryable/NonRetryable hierarchy
  - Metrics:      Metrics, MetricEvent
  - Transport:    Transport, maybe_compress, classify_http_error
  - Streaming:    parse_sse_stream, parse_chunk_obj
"""

from __future__ import annotations

__version__ = "0.2.0"

# Re-export the public symbols. Local imports (no `.`) keep the names
# stable for callers regardless of internal module layout.
from .exceptions import (
    AuthError,
    BadRequestError,
    CerebrasError,
    CircuitOpenError,
    ConfigError,
    NonRetryableError,
    RateLimitError,
    RequestTimeoutError,
    RetryableError,
    ServerError,
)
from .models import (
    ChatRequest,
    ChatResponse,
    Choice,
    Message,
    PredictedOutput,
    ResponseFormat,
    StreamChunk,
    TimeInfo,
    Usage,
)
from .config import get_api_key
from .metrics import MetricEvent, Metrics
from .cache import Cache, MemoryLRU, RedisCache, TieredCache, cache_key
from .reliability import (
    CircuitBreaker,
    DEFAULT_RETRYABLE,
    DualRateLimiter,
    RetryPolicy,
)
from .transport import (
    Transport,
    classify_http_error,
    maybe_compress,
    parse_ratelimit_headers,
    parse_time_info,
)
from .streaming import parse_chunk_obj, parse_sse_stream
from .backends import (
    Backend,
    CerebrasBackend,
    DiffusionGemmaBackend,
    MockBackend,
    OpenAICompatibleBackend,
)
from .client import AsyncCfire
from ._sync import Cfire
from .router import Router, RoutingPolicy

__all__ = [
    "__version__",
    # Models
    "ChatRequest", "ChatResponse", "Choice", "Message",
    "PredictedOutput", "ResponseFormat", "StreamChunk", "TimeInfo", "Usage",
    # Clients
    "AsyncCfire", "Cfire",
    # Backends
    "Backend", "CerebrasBackend", "DiffusionGemmaBackend",
    "MockBackend", "OpenAICompatibleBackend",
    # Router
    "Router", "RoutingPolicy",
    # Reliability
    "CircuitBreaker", "RetryPolicy", "DualRateLimiter", "DEFAULT_RETRYABLE",
    # Cache
    "Cache", "MemoryLRU", "RedisCache", "TieredCache", "cache_key",
    # Exceptions
    "CerebrasError",
    "RetryableError", "RateLimitError", "ServerError", "RequestTimeoutError",
    "NonRetryableError", "AuthError", "BadRequestError",
    "CircuitOpenError", "ConfigError",
    # Metrics
    "Metrics", "MetricEvent",
    # Transport
    "Transport", "maybe_compress", "classify_http_error",
    "parse_ratelimit_headers", "parse_time_info",
    # Streaming
    "parse_sse_stream", "parse_chunk_obj",
    # Config
    "get_api_key",
]
