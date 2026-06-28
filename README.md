<div align="center">

# VisionOps

**See the problem. Fix the problem.**

Image-aware SRE agent dashboard for the **Cerebras × Google DeepMind** hackathon.

`cfire` drives **15,791 tok/s** on `gema-4-31b` — Gemma 4 31B on Cerebras WSE-3.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Model — Gemma 4 31B](https://img.shields.io/badge/model-gemma--4--31b-ff4d6d.svg)](https://www.deepmind.com/models/gemma)
[![Inference — Cerebras](https://img.shields.io/badge/inference-Cerebras%20WSE--3-4ecdc4.svg)](https://www.cerebras.ai/)
[![Status — Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#status)

<br/>

<!-- Screenshot placeholder — see "Screenshots" section below for capture guide -->
<p align="center">
  <em>📸 Screenshot slot — see <a href="#screenshots">capture guide</a>. The dashboard runs locally on <code>localhost:8000</code>.</em>
</p>

</div>

---

## Table of contents

1. [Why VisionOps wins](#why-visionops-wins)
2. [Architecture](#architecture)
3. [Features](#features)
4. [Quick start](#quick-start)
5. [Use `cfire` in code](#use-cfire-in-code)
6. [Use the VisionOps agent](#use-the-visionops-agent)
7. [Dashboard API](#dashboard-api)
8. [Benchmarks](#benchmarks)
9. [Environment variables](#environment-variables)
10. [Project structure](#project-structure)
11. [Tests](#tests)
12. [Status](#status)
13. [Roadmap](#roadmap)
14. [License](#license)
15. [Screenshots & demo assets](#screenshots)

---

## Why VisionOps wins

Multiverse-agents pitch in three bullets:

- **Multimodal** — point any camera at a failing system; Gemma 4 31B sees the issue and prescribes a fix. Handles screenshots, dashboards, terminal output, and physical hardware photos.
- **Multi-agent** — autonomous news scouts fan out for live context while the vision diagnosis runs; a commander agent synthesizes everything into a cited action plan.
- **Physically aware** — the dashboard acts in the physical world: real-time Cerebras inference race telemetry drives the SRE control-room UI, not the other way around.

The unifying thesis: an image-aware SRE agent that reads the screen, gathers context, and pushes back into the physical world — with Cerebras-class speed as the connective tissue.

## Architecture

```
                 ┌────────────────────────────────────────────┐
                 │              Browser (SPA)                 │
                 │   static/index.html — vanilla JS, no build │
                 │   EventSource SSE × 3 streams              │
                 └──────────────────┬─────────────────────────┘
                                    │  fetch() + SSE
                                    ▼
                 ┌────────────────────────────────────────────┐
                 │       web_demo.py  (FastAPI + uvicorn)     │
                 │       17 endpoints • 3 SSE channels        │
                 │       RaceManager · NewsManager            │
                 └──┬──────────┬──────────┬──────────┬────────┘
                    │          │          │          │
              ┌─────▼───┐ ┌────▼────┐ ┌───▼────┐ ┌───▼────────────┐
              │ RACE    │ │ NEWS    │ │ VISION │ │ HEADROOM/MEMORY│
              │ sweep + │ │ scouts →│ │ Gemma  │ │ rate-limit +   │
              │ sustain │ │ command │ │ 4 31B  │ │ cross-run best │
              └────┬────┘ └────┬────┘ └───┬────┘ └───┬────────────┘
                   │           │          │          │
                   └───────────┴──────────┴──────────┘
                               │
                               ▼
                 ┌────────────────────────────────────────────┐
                 │              cfire (this repo)              │
                 │  OpenAI-compatible client for Cerebras      │
                 │  retry · circuit · rate-limit · SSE stream  │
                 │  tiered cache (LRU + Redis) · smart router  │
                 └────────────────────────────────────────────┘
                               │
                               ▼
                      Cerebras Inference API
                      gemma-4-31b · ~1,500 TPS
```

## Features

| Area | What you get |
|---|---|
| **Race dashboard** | Sweep concurrency × token configs, watch sustained tok/s live, deterministic post-race report + LLM analysis |
| **Vision agent** | `vision_ops.VisionOpsAgent` — multimodal Gemma 4 31B diagnosis with a robust 3-stage response parser (delimited → JSON → free-text salvage) |
| **News scouts** | `news_agents.NewsAgentTeam` — parallel scouts → commander synthesis with citations |
| **`cfire` library** | Sync + async, OpenAI-compatible, `time_info` telemetry, `prompt_cache_key`, `predicted_output`, `service_tier`, `reasoning_effort` |
| **Reliability** | Adaptive concurrency from `x-ratelimit-*` headers, exponential backoff, circuit breaker, tiered cache |
| **Benchmark memory** | Index historical races, query by metric, compare configs — `benchmark_memory.BenchmarkMemory` |
| **Telemetry** | Live rate-limit headroom + sparkline histogram of sustained throughput |

## Quick start

```bash
# 1. Clone and enter
git clone https://github.com/DesignerEE/Cerebras-Gemma4.git
cd Cerebras-Gemma4

# 2. Create venv and install
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 3. Add your Cerebras key (the .env file is gitignored)
cat > .env <<'EOF'
CEREBRAS_API_KEY=csk-...
CEREBRAS_BASE_URL=https://api.cerebras.ai
CFIRE_CEREBRAS_BASE_URL=https://api.cerebras.ai/v1
EOF

# 4. Smoke test
python cerebras_smoke_test.py

# 5. Launch the dashboard
python web_demo.py
# → http://localhost:8000
```

Use a different port if `8000` is taken:

```bash
python web_demo.py --port 8080
```

Tabs in the UI:
- **RACE** — sweep configs, run a sustained race, watch live telemetry
- **NEWS** — deploy commander-scout research agents on any query
- **VISION** — upload a screenshot/diagram/photo for structured analysis

## Use `cfire` in code

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
        # Per-response Cerebras telemetry:
        print(response.time_info)  # queue_time / prompt_time / completion_time / total_time

asyncio.run(main())
```

### Why `cfire` and not the raw SDK

- **`time_info` telemetry** — per-response `queue_time` / `prompt_time` / `completion_time`
- **Adaptive concurrency** driven by `x-ratelimit-*` headers
- **Server-side `prompt_cache_key`** for multi-turn conversations
- **`predicted_output`** for regeneration-heavy workloads
- **`service_tier`** (`flex` / `default` / `auto` / `priority`) — queue priority control
- **`reasoning_effort`** (`low` / `medium` / `high`) for reasoning models
- **Tiered cache** — bounded in-memory LRU + optional Redis
- **Smart router** — Cerebras primary, local model fallback (Phase 3)
- **Sync + async** — `cfire.Cfire` and `cfire.AsyncCfire`

## Use the VisionOps agent

```python
import asyncio
from vision_ops import VisionOpsAgent

async def main():
    with open("screenshot.png", "rb") as f:
        image_bytes = f.read()

    agent = VisionOpsAgent()
    diagnosis = await agent.analyze(image_bytes, mime_type="image/png")
    print(diagnosis.summary)
    print(diagnosis.severity)            # info | warning | critical
    for action in diagnosis.actions:
        print(action.name, action.command, action.safe_to_run)

asyncio.run(main())
```

Returns a typed `Diagnosis` Pydantic model. The parser handles whatever Gemma 4 emits — fenced JSON, XML-tagged JSON, delimited blocks, or free-text — and degrades gracefully to a severity-keyword salvage path.

## Dashboard API

The FastAPI server exposes 17 endpoints. Highlights:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Dashboard HTML |
| `POST` | `/api/start` | Start a sweep + sustained race |
| `POST` | `/api/stop` | Abort the current race |
| `GET` | `/api/stream` | SSE: live race telemetry |
| `POST` | `/api/news/start` | Spawn news scouts + commander |
| `GET` | `/api/news/stream` | SSE: scout progress + report |
| `POST` | `/api/vision/analyze` | Multimodal image diagnosis |
| `GET` | `/api/headroom/status` | Rate-limit headroom snapshot |
| `GET` | `/api/benchmark/memory/insights` | Aggregate historical insights |

Full list in the source (`web_demo.py`).

## Benchmarks

Real runs against `gemma-4-31b` on the Developer tier:

| Concurrency | Max tokens | Best tok/s | req/s |
|-------------|-----------|------------|-------|
| **24**      | **1000**  | **15,791** | 15.79 |
| 16          | 1500      | 15,170     | 10.11 |
| 24          | 750       | 13,566     | 18.09 |
| 16          | 1000      |  7,505     |  7.51 |
| 8           | 200       |  5,653     | 33.75 |

Dashboard presets:

| Mode | Concurrency | Max tokens | Use case |
|------|-------------|------------|----------|
| 1 Eco       | 8  | 250  | Low-latency probe |
| 2 Race      | 16 | 1000 | Balanced sustained race |
| 3 Qualy     | 24 | 1500 | **Max tok/s qualifying lap** |
| DRS Boost   | 32 | 1000 | Push limits (may queue) |

**Peak:** `24 concurrency × 1000–1500 tokens` on the Developer tier. Concurrency above 32 tends to queue or hit rate limits.

See [`CEREBRAS_SPEED_GUIDE.md`](CEREBRAS_SPEED_GUIDE.md) for the full tuning guide.

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

## Project structure

```
.
├── cfire/                       # core fast-inference library (Phase 3 in progress)
├── cfire-diffusion/             # variant with DiffusionGemma4 backend
├── vision_ops/                  # multimodal SRE diagnosis agent
├── web_demo.py                  # FastAPI dashboard (17 endpoints, 3 SSE streams)
├── static/index.html            # terminal/F1-styled SPA frontend
├── cerebras_smoke_test.py       # quick API connectivity check
├── cerebras_race_optimizer.py   # race tuning helpers
├── news_agents.py               # commander-scout news pipeline
├── benchmark_memory.py          # historical race indexing
├── headroom.py                  # rate-limit headroom monitor
├── build_demo_video.py          # ffmpeg assembler for 60s hackathon demo
├── DESIGN.md                    # control-room design system
├── demo_storyboard.md           # 60-second demo shot list
├── CEREBRAS_SPEED_GUIDE.md      # full benchmark + tuning guide
├── BENCHMARK_MEMORY.md          # benchmark-memory service docs
└── LICENSE                      # MIT
```

## Tests

```bash
# Main suite
pytest cfire/tests tests/ -q

# DiffusionGemma4 variant (run from its own dir due to import path overlap)
cd cfire-diffusion && pytest cfire/tests -q
```

Coverage: 89% on `cfire/` — 101 passed, 1 xfailed (router Phase 3 placeholder).

## Status

**Alpha.** Extracted from `cerebras-bench` where `cfire` sustained **15,791 tok/s** on `gemma-4-31b`.

## Roadmap

- **Phase 3** (in progress) — `cfire` smart router with `LocalBackend` (local-Qwen fallback) and `CDNBackend` (edge proxy). `Router` is itself a `Backend` so it composes under `AsyncCfire`.
- **Phase 4** — migrate `news_agents.py`, `web_demo.py`, `cerebras_smoke_test.py` off the `_compat` shim onto native `cfire.AsyncCfire`. Delete the shim afterwards.

See [`cfire/`](cfire/) for the in-progress router design.

## License

[MIT](LICENSE) — © 2026 DesignerEE.

## Screenshots

The dashboard runs locally — there is no hosted demo. To capture screenshots for this README:

1. **Launch the dashboard:** `python web_demo.py` → open `http://localhost:8000`
2. **Browser zoom** to ~125% for crisp 1080p captures (Ctrl/Cmd + `+`)
3. **Hide browser chrome** with F11 or a clean browser profile
4. **For the RACE tab:** press START, wait for the sustained race phase (~20s), capture mid-run with gauges peaked
5. **For the VISION tab:** upload a screenshot of an error message / failing device; capture the diagnosis card
6. **For the NEWS tab:** enter a query like `"latest Cerebras inference updates"`, deploy scouts, capture mid-pipeline

Drop captures into `docs/` (create the dir) and embed with:

```markdown
![Race tab at peak](docs/race_peak.png)
```

For the 60-second hackathon demo video, follow the shot list in [`demo_storyboard.md`](demo_storyboard.md) and assemble with [`build_demo_video.py`](build_demo_video.py).
