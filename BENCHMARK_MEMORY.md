# Benchmark Memory

> **Remembers the fastest setup for next time.**

A lightweight local indexing and query engine over the `results/` directory. Every Cerebras race the dashboard runs gets persisted as JSON; `benchmark_memory` loads them at server startup and answers questions about historical performance — without calling any external service.

## Why it matters

Without benchmark memory, every dashboard restart is amnesia. With it:

- **Faster decisions** — query "what won last time on `gemma-4-31b`?" instead of re-running a 20-second race.
- **Cross-run trends** — see how a config performs on average, not just on its best single run.
- **Honest rankings** — pathological cache-hit outliers (near-zero latency inflating tok/s) are filtered out automatically.
- **Head-to-head comparisons** — average two configs' histories to settle "is c16/1000 actually better than c24/750?".

## What it does

| Step | Operation |
|---|---|
| 1. Load | Reads every `results/*.json` file at server startup. |
| 2. Normalize | Reconciles config shapes across legacy and advanced result formats. |
| 3. Index | Sweep points, best configs, sustained race results. |
| 4. Filter | Drops cache-hit outliers (`avg_latency == 0` or `tok_per_sec > 1,000,000`) from rankings. |
| 5. Rank | Sorts by `tok/s` or `req/s`. |
| 6. Search | Keyword match over config names, phases, metrics. |
| 7. Compare | Averages two configs' historical records. |

## Endpoints

All endpoints return JSON. Base URL: `http://localhost:8000`.

### `GET /api/benchmark/memory/insights`

Aggregate stats and top performers across all loaded records.

```bash
curl http://localhost:8000/api/benchmark/memory/insights | jq
```

Sample response (truncated):

```json
{
  "ok": true,
  "total_records": 56,
  "files_loaded": ["race_20260620_184409.json", "..."],
  "models": ["gemma-4-31b"],
  "summary": {
    "mean_tok_per_sec": 3626.73,
    "median_tok_per_sec": 2667.12,
    "configs_tested": 18
  },
  "top_tok_per_sec":              { "config": { "name": "c24_t1000" }, "tok_per_sec": 15791.79 },
  "top_req_per_sec":              { "config": { "name": "c8_t200"  }, "req_per_sec":    57.59 },
  "top_sustained_tok_per_sec":    { /* ... */ }
}
```

### `GET /api/benchmark/memory/search?q=<query>`

Keyword search over config names and phases.

```bash
curl 'http://localhost:8000/api/benchmark/memory/search?q=c16' | jq
```

### `GET /api/benchmark/memory/compare?a=<config>&b=<config>`

Head-to-head comparison averaging each config's historical records.

```bash
curl 'http://localhost:8000/api/benchmark/memory/compare?a=c8_t100&b=c16_t200' | jq
```

### `POST /api/benchmark/memory/test`

Upload a single benchmark result JSON for one-off insights. The file is parsed in memory and **not** persisted to disk — useful for evaluating an exported run without polluting `results/`.

```bash
curl -X POST -F "file=@results/race_20260620_184409.json" \
  http://localhost:8000/api/benchmark/memory/test | jq
```

## Dashboard integration

The four endpoints are wired into `web_demo.py` (lines 22–25) and exposed under `/api/benchmark/memory/*`. The service is **API-only** at this time — there is no dedicated UI tab yet. The dashboard tabs are `[RACE]`, `[NEWS]`, and `[VISION]`. Future work may surface memory insights inside the existing RACE tab as a "previous winners" panel.

You can drive everything from the command line while the server runs:

```bash
# What's the all-time peak config?
curl -s http://localhost:8000/api/benchmark/memory/insights | jq '.top_tok_per_sec'

# Is c16/1000 actually better than c24/750?
curl -s 'http://localhost:8000/api/benchmark/memory/compare?a=c16_t1000&b=c24_t750' | jq
```

## Files

| Path | Role |
|---|---|
| `benchmark_memory.py` | Indexing and query engine |
| `tests/test_benchmark_memory.py` | Unit tests |
| `web_demo.py` | API endpoint handlers |
| `results/*.json` | Source-of-truth race artifacts (gitignored) |

## Caveats

- **Index built once at startup.** New runs require a server restart to appear in memory queries.
- **Outlier filtering** excludes records with `avg_latency == 0` or `tok_per_sec > 1,000,000` from rankings, so cache-heavy runs don't dominate.
- **No persistence layer.** Everything lives in process memory; there's no on-disk index file.

## Related

- [`CEREBRAS_SPEED_GUIDE.md`](CEREBRAS_SPEED_GUIDE.md) — what the benchmarks measure (sweep-peak vs sustained-race).
- [`README.md`](README.md) — overall project architecture and quick start.
