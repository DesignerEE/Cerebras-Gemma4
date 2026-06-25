# Benchmark Memory

A lightweight, local indexing and query engine over the `results/` directory.
It applies context-mode-style algorithms (load, normalize, rank, filter,
search, compare) to past Cerebras benchmark runs so the dashboard can answer
questions without calling external services.

## What it does

1. **Loads** every `results/*.json` file at server startup.
2. **Normalizes** config shapes across legacy and advanced result formats.
3. **Indexes** sweep points, best configs, and sustained race results.
4. **Filters** pathological cache-hit outliers (near-zero latency causing
   inflated tok/s) from rankings.
5. **Ranks** records by tok/s or req/s.
6. **Searches** configs by name/phase/metric keyword.
7. **Compares** two configs by averaging their historical records.

## Endpoints

### `GET /api/benchmark/memory/insights`

Aggregate stats and top performers.

```json
{
  "ok": true,
  "total_records": 56,
  "files_loaded": ["race_20260620_184409.json", ...],
  "models": ["gpt-oss-120b"],
  "summary": {
    "mean_tok_per_sec": 3626.73,
    "median_tok_per_sec": 2667.12,
    "configs_tested": 18
  },
  "top_tok_per_sec": { "config": {"name": "c24_t1000"}, "tok_per_sec": 15791.79, ... },
  "top_req_per_sec": { "config": {"name": "c8_t200"}, "req_per_sec": 57.59, ... },
  "top_sustained_tok_per_sec": { ... }
}
```

### `GET /api/benchmark/memory/search?q=<query>`

Keyword search over config names and phases.

```bash
curl 'http://localhost:8000/api/benchmark/memory/search?q=c16'
```

### `GET /api/benchmark/memory/compare?a=<config>&b=<config>`

Compare two configs by averaging their records.

```bash
curl 'http://localhost:8000/api/benchmark/memory/compare?a=c8_t100&b=c16_t200'
```

### `POST /api/benchmark/memory/test`

Upload a single benchmark result JSON file for one-off insights. The file is
parsed in memory and not persisted to disk.

```bash
curl -X POST -F "file=@results/race_20260620_184409.json" \
  http://localhost:8000/api/benchmark/memory/test
```

## Dashboard

Open the `[MEMORY]` tab in the dashboard to see:

- Aggregate insights (records, models, mean/median tok/s)
- Top tok/s, top req/s, and top sustained config
- Search box for config names
- Compare box for head-to-head config comparison
- File upload box for one-off test-run analysis

## Files

- `benchmark_memory.py` — indexing and query engine
- `tests/test_benchmark_memory.py` — unit tests
- `web_demo.py` — API endpoints
- `static/index.html` — `[MEMORY]` tab UI

## Notes

- The index is built once at server startup. New runs require a server restart
  to appear in memory.
- Outlier filtering excludes records with `avg_latency == 0` and
  `tok_per_sec > 1,000,000` from rankings, so cache-heavy runs don't dominate.
