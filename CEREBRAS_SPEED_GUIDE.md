# Cerebras Speed Guide

How to get maximum throughput out of `gemma-4-31b` on Cerebras Inference using `cfire`. Derived from this repo's race benchmarks, the `cfire` test suite, and live dashboard runs.

## TL;DR

```python
async with AsyncCfire(
    concurrency=24,          # peak tok/s in benchmarks
    enable_cache=False,      # required for honest throughput measurement
    req_per_min=1000,        # Developer-tier ceiling
    tok_per_min=1_000_000,
) as client:
    resp = await client.complete(ChatRequest(
        model="gemma-4-31b",
        messages=[Message(role="user", content=prompt)],
        max_completion_tokens=1500,   # 1000-1500 amortizes overhead best
        reasoning_effort="low",
        service_tier="priority",
    ))
```

Peak: **15,791 tok/s** at `concurrency=24 × max_tokens=1000`. Developer-tier rate limits kick in above ~32 concurrent or under burst loads — see [Rate-limit ceiling](#rate-limit-ceiling).

## Mandatory dependency

HTTP/2 is **required** for `cfire` to hit peak speed. Without `h2` installed, `httpx` raises `ImportError` on startup.

```bash
.venv/bin/pip install 'httpx[http2]'
```

Run everything with the venv Python so `h2` resolves:

```bash
.venv/bin/python web_demo.py
```

## Why `cfire.AsyncCfire` and not the raw SDK

- Built on `httpx[http2]` with HTTP/2 multiplexing and a tuned connection pool (`max_keepalive=concurrency*2`, `max_connections=concurrency*4`).
- Adaptive concurrency driven by `x-ratelimit-*` response headers.
- Exponential backoff, circuit breaker, and tiered cache (LRU + Redis).
- Avoid the sync `cfire.Cfire` wrapper for bulk throughput — it bridges through a background thread.

## Two throughput numbers — don't confuse them

The dashboard produces two distinct metrics. Knowing the difference prevents wrong conclusions:

| Metric | What it is | When it matters |
|---|---|---|
| **Sweep peak** | Single batched request per config; pure API max capability | Comparing configs, finding the theoretical ceiling |
| **Sustained race** | Auto-sized ~20s race at the sweep winner's settings | Real-world steady-state throughput under rate limits |

A config that wins the sweep can still trip the circuit breaker during the sustained race if its burst volume is too high. See [Rate-limit ceiling](#rate-limit-ceiling).

## Concurrency sweet spot

Real (non-mock, non-cached) results from `results/` — these are **sweep-peak** numbers:

| Concurrency | Max tokens | Best tok/s | req/s |
|---|---|---|---|
| **24** | **1000** | **15,791** | 15.79 |
| 16 | 1500 | 15,170 | 10.11 |
| 24 | 750  | 13,566 | 18.09 |
| 16 | 1000 |  7,505 |  7.51 |
| 8  | 200  |  5,653 | 33.75 |

- **Default: 16** (`cfire.config.DEFAULT_CONCURRENCY`).
- **Peak tok/s: 24 concurrency** with 1000–1500 tokens.
- Concurrency 32+ can queue or hit rate limits and is not consistently faster.

## Rate-limit ceiling

Verified live against the Developer tier — the limit is on **burst concurrency × sustained volume**, not concurrency alone:

| Pattern | Result |
|---|---|
| `c16/c24/c32` single request (sweep, 1 each) | ✅ All succeed (~650–885 tok/s per request) |
| `c16` burst of 15 requests | ❌ Trips circuit after ~11 successes → 12 failures |
| `c48` / `c64` burst | ❌ Immediate 6 failures → circuit opens 30s |
| **`c4` sustained (~4.4 req/s for 20s)** | ✅ **0 errors, 2,113 tok/s sustained over 88 requests** |

Practical guidance:

- **For honest benchmark sweeps:** use `race_requests=1` per config. You'll see the API ceiling cleanly.
- **For sustained real traffic:** stay at `concurrency ≤ 8` or expect circuit-breaker trips on the Developer tier. Higher tiers raise the ceiling proportionally.
- The circuit breaker opens at **6 failures** with a **30 s cooldown** and is not retried (`CircuitOpenError`). See [Troubleshooting](#troubleshooting).

## Token count trade-off

- **Higher `max_completion_tokens`** → higher tok/s (amortizes network + queue overhead).
- **Lower `max_completion_tokens`** → higher req/s, lower per-request latency.
- **Raw throughput:** 1000–1500 tokens.
- **Latency-sensitive probes:** 100–250 tokens.

## Caching and compression

| Goal | Setting |
|---|---|
| Throughput benchmark / race | `enable_cache=False` (cached hits distort tok/s) |
| Production with repeated prompts | `enable_cache=True` (memory LRU + optional Redis) |
| Large prompts (>4 KB) | `enable_compression=True` (gzip; msgpack if installed) |

## Dashboard presets

The F1-themed dashboard modes map to these configs:

| Mode | Concurrency | Max tokens | Use case |
|---|---|---|---|
| 1 Eco     | 8  | 250  | Low-latency probe |
| 2 Race    | 16 | 1000 | Balanced sustained race |
| 3 Qualy   | 24 | 1500 | **Max tok/s qualifying lap** |
| DRS Boost | 32 | 1000 | Push limits (may queue) |

## Reading `time_info`

Every `cfire` response carries a `time_info` object exposing Cerebras's per-response telemetry:

```python
resp.time_info.queue_time       # seconds spent waiting in the Cerebras queue
resp.time_info.prompt_time      # seconds processing the prompt
resp.time_info.completion_time  # seconds generating tokens
resp.time_info.total_time       # end-to-end
```

Interpretation:

- **`queue_time` climbing** → congestion; lower `concurrency` or `service_tier`.
- **`prompt_time` dominant** → prompt is large; consider `enable_compression=True` or trim context.
- **`completion_time` dominant** → expected for max-throughput runs.
- **`total_time` jittery across requests** → rate-limit throttling; check `x-ratelimit-*` headers in headroom.

## Troubleshooting

### `CircuitOpenError: circuit open (N failures, cooldown 30s)`

The `cfire` reliability layer opened the breaker after 6+ failures. **Wait 30 s** for cooldown — no retry will succeed during this window. Root causes by frequency:

1. **Rate-limit 429s** from too-aggressive burst concurrency. Drop `concurrency` and `race_requests`.
2. **Auth failures (401)** — verify `CEREBRAS_API_KEY` in `.env`.
3. **Network errors** — check connectivity to `https://api.cerebras.ai/v1`.

### `400 clear_thinking not supported for this model`

`gemma-4-31b` does not support `clear_thinking=True`. Drop the parameter.

### `400 Tools were requested but no tools found`

You sent `tool_choice="auto"` without a `tools=[...]` array. Either provide tools or drop `tool_choice`.

### `ImportError: No module named 'h2'`

Install HTTP/2 support: `pip install 'httpx[http2]'`. Run scripts with `.venv/bin/python` so the venv resolves.

### Throughput far below the benchmark

In order of likelihood:

1. **Cache is on** — `enable_cache=True` returns cached responses instantly, which *looks* slow because no actual generation happened. Use `enable_cache=False` for benchmarks.
2. **On a lower tier** — Developer tier throttles aggressively above ~32 concurrency.
3. **Wrong base URL** — `CFIRE_CEREBRAS_BASE_URL` must include `/v1`; `CEREBRAS_BASE_URL` must not. (The SDK appends `/v1` itself; `cfire`'s direct HTTP path does not.)
4. **HTTP/2 not negotiated** — verify `pip install 'httpx[http2]'` succeeded and you're using `AsyncCfire`, not the sync wrapper.

## What to avoid

1. **System Python without `h2`** — `httpx` raises on `http2=True`.
2. **`clear_thinking=True` on `gemma-4-31b`** — returns `400 clear_thinking not supported`.
3. **`tool_choice="auto"` without `tools`** — returns `400 Tools were requested but no tools found`.
4. **Unbounded concurrency** — `>32` degrades throughput via queueing and rate limits.
5. **Cache in throughput benchmarks** — cached hits distort tok/s measurements.
6. **`enable_cache=False` in production** — you lose the LRU/Redis benefit; use only for benchmarks.

## Appendix: News agents tuning

The commander-scout news pipeline in `news_agents.py` uses these optimized defaults:

```python
REASONING_EFFORT       = "low"
SERVICE_TIER           = "priority"
SCOUT_MAX_TOKENS       = 1000
SCOUT_TEMPERATURE      = 0.2
COMMANDER_MAX_TOKENS   = 1200
COMMANDER_TEMPERATURE  = 0.3
```

And the web demo creates the news client with:

```python
CerebrasRaceClient(
    concurrency=12,            # enough for 3 scouts + commander parallelism
    enable_cache=False,
    enable_compression=True,
)
```

This lowers queue time and generation latency for each scout summary and the final commander synthesis.
