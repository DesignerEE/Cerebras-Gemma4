# cfire

**Fast-inference library for Cerebras WSE-3 and DiffusionGemma4.**

`cfire` provides a thin, OpenAI-compatible client for [Cerebras Inference](https://inference.cerebras.ai) and local DiffusionGemma4 servers,
with first-class support for the things that make Cerebras different:

- **`time_info` telemetry** — per-response `queue_time` / `prompt_time` / `completion_time` / `total_time`
- **Adaptive concurrency** driven by `x-ratelimit-*` headers
- **Server-side `prompt_cache_key`** management for multi-turn conversations
- **`predicted_output`** for regeneration-heavy workloads (code editing, agentic loops)
- **`service_tier`** (`flex` / `default` / `auto` / `priority`) — queue priority control
- **`reasoning_effort`** (`low` / `medium` / `high`) for reasoning models like `gpt-oss-120b`

Plus the pieces you need to build resilient apps on top:

- **Tiered cache** — bounded in-memory LRU + optional Redis (zero field loss)
- **Smart router** — Cerebras primary, local model fallback (Ollama / Qwen / vLLM)
- **Pluggable backends** — custom Cerebras endpoints, custom CDN, anything OpenAI-compatible
- **Sync + async APIs** — `cfire.Cfire` and `cfire.AsyncCfire`

## Quick start

```python
from cfire import AsyncCfire, ChatRequest

async with AsyncCfire() as client:
    response = await client.complete(ChatRequest(
        model="gpt-oss-120b",
        messages=[{"role": "user", "content": "Hello, Cerebras."}],
    ))
    print(response.choices[0].message.content)
    print(response.time_info)  # Cerebras-unique latency breakdown
```

With a local fallback:

```python
from cfire import AsyncCfire, CerebrasBackend, LocalBackend, Router, RoutingPolicy

client = AsyncCfire(router=Router(
    primary=CerebrasBackend(),
    fallbacks=[LocalBackend(base_url="http://127.0.0.1:8123")],
    policy=RoutingPolicy(),
))
```

## Status

Alpha. Extracted from `cerebras-bench` where it sustained **4,843 tok/s** on `gpt-oss-120b`.
See `pyproject.toml` for dependencies and the `cfire/` package for source.
