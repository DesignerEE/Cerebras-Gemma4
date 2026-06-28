# Cerebras Best-Speed Guide

Derived from the `cfire` test suite, race benchmarks, and web demo runs in this repo.

## Recommended Library

Use **`cfire.AsyncCfire`**.

- Built on `httpx[http2]` with HTTP/2 multiplexing and a tuned connection pool.
- Default pool limits: `max_keepalive=concurrency*2`, `max_connections=concurrency*4`.
- The repo README claims sustained **4,843 tok/s** on `gemma-4-31b`.
- Avoid the sync `Cfire` wrapper for bulk throughput — it adds a background-thread bridge.

## Mandatory Dependency

HTTP/2 is required for best speed and correctness.

```bash
.venv/bin/pip install httpx[http2]   # ensures h2 is present
```

Run scripts with the venv Python so `h2` is available:

```bash
.venv/bin/python cerebras_race_advanced.py
```

## Optimal Request Parameters

```python
from cfire import AsyncCfire, ChatRequest, Message

async with AsyncCfire(concurrency=24, enable_cache=False) as client:
    resp = await client.complete(ChatRequest(
        model="gemma-4-31b",
        messages=[Message(role="user", content=prompt)],
        max_completion_tokens=1500,   # 1000-1500 for max tok/s
        reasoning_effort="low",       # fastest reasoning setting
        service_tier="priority",      # fastest queue (costs more)
        temperature=0.3,
        top_p=1.0,
        # Do NOT send clear_thinking=True or tool_choice="auto"
        # unless you are actually using those features.
    ))
```

## Concurrency Sweet Spot

Real (non-mock, non-cached) benchmark results from `results/`:

| Concurrency | Max tokens | Best tok/s | req/s |
|-------------|-----------|------------|-------|
| 24          | 1000      | **15,791** | 15.79 |
| 16          | 1500      | **15,170** | 10.11 |
| 24          | 750       | **13,566** | 18.09 |
| 16          | 1000      | 7,505      | 7.51  |
| 8           | 200       | 5,653      | 33.75 |

- **Default: 16** (`cfire.config.DEFAULT_CONCURRENCY`).
- **Peak tok/s: 24 concurrency** with 1000–1500 tokens.
- Concurrency 32+ can queue or hit rate limits and is not consistently faster.

## Token Count Trade-off

- **Higher `max_completion_tokens`** → higher tok/s (amortizes network overhead).
- **Lower `max_completion_tokens`** → higher req/s, lower latency per request.
- For raw throughput: **1000–1500 tokens**.
- For latency-sensitive probes: **100–250 tokens**.

## Caching and Compression

| Goal | Setting |
|------|---------|
| Throughput benchmark / race | `enable_cache=False` |
| Production with repeated prompts | `enable_cache=True` (memory LRU + optional Redis) |
| Large prompts (>4 KB) | `enable_compression=True` (gzip; msgpack if `msgpack` installed) |

## Dashboard Presets

The F1 dashboard strategy modes map to these configs:

| Mode | Concurrency | Max tokens | Use case |
|------|-------------|------------|----------|
| 1 Eco | 8 | 250 | Low-latency probe |
| 2 Race | 16 | 1000 | Balanced sustained race |
| 3 Qualy | 24 | 1500 | Max tok/s qualifying lap |
| DRS Boost | 32 | 1000 | Push limits (may queue) |

## Rate Limits

Default env ceilings (Developer tier):

```bash
CFIRE_REQ_PER_MIN=1000
CFIRE_TOK_PER_MIN=1000000
```

For higher tiers, raise these before increasing concurrency further.

## Full Example

```python
import asyncio
from cfire import AsyncCfire, ChatRequest, Message

async def main():
    async with AsyncCfire(
        concurrency=24,
        req_per_min=1000,
        tok_per_min=1_000_000,
        enable_cache=False,
    ) as client:
        req = ChatRequest(
            model="gemma-4-31b",
            messages=[Message(role="user", content="Explain async I/O in detail.")],
            max_completion_tokens=1500,
            reasoning_effort="low",
            service_tier="priority",
        )
        resp = await client.complete(req)
        print(resp.text)
        print(resp.time_info)

asyncio.run(main())
```

## News Agents Tuning

The commander-scout news pipeline in `news_agents.py` now uses the same optimized defaults:

```python
REASONING_EFFORT = "low"
SERVICE_TIER = "priority"
SCOUT_MAX_TOKENS = 1000
SCOUT_TEMPERATURE = 0.2
COMMANDER_MAX_TOKENS = 1200
COMMANDER_TEMPERATURE = 0.3
```

And the web demo creates the news client with:

```python
CerebrasRaceClient(
    concurrency=12,   # enough for 3 scouts + commander parallelism
    enable_cache=False,
    enable_compression=True,
)
```

This lowers queue time and generation latency for each scout summary and the final commander synthesis.

## What to Avoid

1. **System Python without `h2`** — `httpx` will raise `ImportError` on `http2=True`.
2. **`clear_thinking=True` on `gemma-4-31b`** — returns `400 clear_thinking not supported for this model`.
3. **`tool_choice="auto"` without `tools`** — returns `400 Tools were requested but no tools found`.
4. **Unbounded concurrency** — >32 can degrade throughput due to queueing and rate limits.
5. **Cache in throughput benchmarks** — cached hits distort tok/s measurements.
