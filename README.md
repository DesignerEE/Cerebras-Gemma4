# VisionOps — Cerebras × Google DeepMind Hackathon

> **Fast inference observability reimagined for the image-aware agent era.**

This repo powers our submission for the **Cerebras × Google DeepMind** 24-hour virtual hackathon.

| | |
|---|---|
| **Date** | June 28–29, 2026 |
| **Theme** | Image-aware agents: navigate UIs, interpret documents/diagrams, act on visual input |
| **Model** | Gemma 4 31B on Cerebras Inference (~1,500 TPS, multimodal, Apache 2.0) |
| **Tracks** | Best Enterprise Use Case, Best Inference Speed Demo |

## What we built

- **`cfire`** — a fast, OpenAI-compatible Python client for Cerebras Inference.
- **Live dashboard** — race Cerebras configs in real time, run news-scout agents, and monitor inference telemetry.
- **VisionOps tab** — image-aware command center that analyzes screenshots, dashboards, diagrams, and artwork via Gemma 4 31B vision reasoning.
- **Benchmark memory** — index and query historical race results to find the fastest configs.
- **`DESIGN.md`** — a unified control-room design system for the dashboard.

## Demo

```bash
# 1. Set your Cerebras API key
export CEREBRAS_API_KEY=csk-...

# 2. Run the dashboard
python web_demo.py

# 3. Open http://localhost:8000
#    - RACE tab: sweep configs and run a sustained race
#    - NEWS tab: deploy commander-scout research agents
#    - VISION tab: upload an image for structured analysis
```

Use a different port if `8000` is taken:

```bash
python web_demo.py --port 8080
```

## Quick start

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Configure

Create a `.env` file in the project root (it is gitignored):

```bash
CEREBRAS_API_KEY=csk-...
# Cerebras Python SDK uses the host below and appends /v1 itself
CEREBRAS_BASE_URL=https://api.cerebras.ai
# cfire's direct httpx path expects the full /v1 path
CFIRE_CEREBRAS_BASE_URL=https://api.cerebras.ai/v1
```

### Run a smoke test

```bash
python cerebras_smoke_test.py
```

### Use `cfire` in code

```python
import asyncio
from cfire import AsyncCfire, ChatRequest

async def main():
    async with AsyncCfire() as client:
        response = await client.complete(ChatRequest(
            model="gemma-4-31b",
            messages=[{"role": "user", "content": "Hello, Cerebras."}],
        ))
        print(response.choices[0].message.content)
        print(response.time_info)  # queue_time / prompt_time / completion_time / total_time

asyncio.run(main())
```

### VisionOps agent

```python
import asyncio
from vision_ops import VisionOpsAgent

async def main():
    with open("screenshot.png", "rb") as f:
        image_bytes = f.read()

    agent = VisionOpsAgent()
    diagnosis = await agent.analyze(image_bytes, mime_type="image/png")
    print(diagnosis.summary)
    print(diagnosis.severity)
    for action in diagnosis.actions:
        print(action.name, action.command)

asyncio.run(main())
```

## `cfire` library

`cfire` provides a thin, OpenAI-compatible client for [Cerebras Inference](https://inference.cerebras.ai) with first-class support for the things that make Cerebras different:

- **`time_info` telemetry** — per-response `queue_time` / `prompt_time` / `completion_time` / total_time
- **Adaptive concurrency** driven by `x-ratelimit-*` headers
- **Server-side `prompt_cache_key`** management for multi-turn conversations
- **`predicted_output`** for regeneration-heavy workloads
- **`service_tier`** (`flex` / `default` / `auto` / `priority`) — queue priority control
- **`reasoning_effort`** (`low` / `medium` / `high`) for reasoning models
- **Tiered cache** — bounded in-memory LRU + optional Redis
- **Smart router** — Cerebras primary, local model fallback
- **Sync + async APIs** — `cfire.Cfire` and `cfire.AsyncCfire`

## Speed guide

Real benchmark results from `results/` on `gemma-4-31b`:

| Concurrency | Max tokens | Best tok/s | req/s |
|-------------|-----------|------------|-------|
| 24          | 1000      | **15,791** | 15.79 |
| 16          | 1500      | **15,170** | 10.11 |
| 24          | 750       | **13,566** | 18.09 |
| 16          | 1000      | 7,505      | 7.51  |
| 8           | 200       | 5,653      | 33.75 |

Dashboard presets:

| Mode | Concurrency | Max tokens | Use case |
|------|-------------|------------|----------|
| 1 Eco | 8 | 250 | Low-latency probe |
| 2 Race | 16 | 1000 | Balanced sustained race |
| 3 Qualy | 24 | 1500 | Max tok/s qualifying lap |
| DRS Boost | 32 | 1000 | Push limits (may queue) |

**Peak throughput** is around **24 concurrency** with **1000–1500 tokens**. Concurrency above 32 tends to queue or hit rate limits on the Developer tier.

See `CEREBRAS_SPEED_GUIDE.md` for the full tuning guide.

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `CEREBRAS_API_KEY` | Required Cerebras API key | — |
| `CEREBRAS_BASE_URL` | Host for the Cerebras Python SDK | `https://api.cerebras.ai` |
| `CFIRE_CEREBRAS_BASE_URL` | Full `/v1` URL for cfire's direct HTTP path | `https://api.cerebras.ai/v1` |
| `CFIRE_MODEL` | Default model | `gemma-4-31b` |
| `CFIRE_CONCURRENCY` | Default concurrency | `16` |
| `CFIRE_REQ_PER_MIN` | Request-rate ceiling | `1000` |
| `CFIRE_TOK_PER_MIN` | Token-rate ceiling | `1_000_000` |
| `VISIONOPS_MODEL` | Model for VisionOps | `gemma-4-31b` |

## API endpoints

The dashboard exposes these endpoints:

- `GET  /` — dashboard HTML
- `POST /api/start` — start a sweep + race
- `POST /api/stop` — stop the current run
- `GET  /api/stream` — SSE events for live telemetry
- `POST /api/news/start` — start news scout pipeline
- `POST /api/news/stop` — stop news pipeline
- `GET  /api/news/stream` — news SSE events
- `POST /api/vision/analyze` — analyze an uploaded image
- `GET  /api/headroom/status` — compression snapshot
- `GET  /api/benchmark/memory/insights` — aggregate benchmark insights

## Tests

```bash
pytest cfire/tests tests/ -q
```

To also run the `cfire-diffusion` tests (must be run from that directory because of import path overlap):

```bash
cd cfire-diffusion
pytest cfire/tests -q
```

## Project structure

```
.
├── cfire/                  # core fast-inference library
├── cfire-diffusion/        # variant with DiffusionGemma4 backend
├── vision_ops/             # VisionOps image-analysis agent
├── web_demo.py             # FastAPI dashboard
├── cerebras_smoke_test.py  # quick API connectivity check
├── cerebras_race_optimizer.py
├── cerebras_race_advanced.py
├── news_agents.py          # commander-scout news pipeline
├── benchmark_memory.py     # result indexing and insights
├── headroom.py             # proxy.log parser/monitor
├── static/index.html       # dashboard frontend
├── CEREBRAS_SPEED_GUIDE.md
├── DESIGN.md
└── README.md
```

## Status

Alpha. Extracted from `cerebras-bench` where it sustained **15,791 tok/s** on `gemma-4-31b`.

## License

MIT — see `pyproject.toml`.
