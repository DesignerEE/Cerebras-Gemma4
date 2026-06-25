# Headroom Integration

The dashboard includes a live **Headroom** panel that instruments Headroom's
proxy compression logs (`~/.headroom/logs/proxy.log`) and displays the same
metrics as [`headroom-meter`](https://github.com/RonnieTheTester/headroom-meter).

## What it shows

| Metric | Meaning |
|--------|---------|
| **Live Savings Speedometer** | `tok_saved / tok_before` for the latest completed request |
| **NOW** | Latest request savings percentage |
| **Odometer** | Cumulative tokens saved |
| **Token Regen** | Aggregate compression rate across all requests |
| **Recent Request Pulses** | Sparkline of recent per-request savings |
| **Frame Compression** | WebSocket frame byte reduction |
| **Saved Tokens** | Total / last-request / average saved |
| **Requests** | Completed count + average saved |
| **Cache Battery** | Weighted average cache hit percentage |
| **Optimize Time** | Average optimization/compression time in ms |
| **Transforms** | Active transforms on the latest request |
| **TOIN** | Token/object indexing patterns, compressions, retrievals, retrieval rate |

## How it works

```
~/.headroom/logs/proxy.log
        ↓
headroom.py (parser + monitor thread)
        ↓
web_demo.py  →  GET /api/headroom/stream  →  static/index.html [HEADROOM] tab
```

1. `headroom.py` tails the log file using stdlib only.
2. It parses `PERF`, `frame compressed`, and `TOIN:` lines.
3. Aggregated metrics are broadcast to SSE subscribers.
4. The dashboard renders block gauges, sparklines, and stat rows in the
   terminal-brutalist style.

## Enabling it

The monitor starts automatically when `web_demo.py` starts. It watches:

```
~/.headroom/logs/proxy.log
```

You can override the path with the `HEADROOM_LOG` environment variable:

```bash
HEADROOM_LOG=/var/log/headroom/proxy.log .venv/bin/python web_demo.py
```

If the log file does not exist, the panel shows `STATE: waiting_for_log` and
starts displaying data as soon as the file appears and Headroom begins writing
to it.

## Testing without a live Headroom proxy

Create a synthetic log and append lines while the dashboard is running:

```bash
mkdir -p ~/.headroom/logs
cat > ~/.headroom/logs/proxy.log <<'EOF'
2026-06-25 07:13:45,123 PERF req_id=abc tok_before=60229 tok_after=57678 tok_saved=2551 cache_read=120 cache_write=5 cache_hit_pct=98 opt_ms=45 total_ms=120 transforms=text,smart_crusher
2026-06-25 07:13:46,456 frame compressed 15432 bytes to 14512 bytes (120 tokens saved)
2026-06-25 07:13:47,789 TOIN: 181 patterns, 283 compressions, 0 retrievals, 0.0% retrieval rate
EOF
```

Then append more lines to see live updates:

```bash
echo '2026-06-25 07:14:00,111 PERF req_id=def tok_before=100000 tok_after=90000 tok_saved=10000 cache_read=500 cache_write=50 cache_hit_pct=90 opt_ms=200 total_ms=500 transforms=text,smart_crusher,kompress' >> ~/.headroom/logs/proxy.log
```

## Files

- `headroom.py` — parser, `Meter`, and `HeadroomMonitor`
- `web_demo.py` — SSE endpoint and monitor lifecycle
- `static/index.html` — `[HEADROOM]` tab and rendering
- `tests/test_headroom.py` — unit tests

## Notes

- The monitor is read-only; it never modifies Headroom configuration or sends
  data anywhere.
- It tolerates missing log files and log rotation.
- No extra Python packages are required beyond the project dependencies.
