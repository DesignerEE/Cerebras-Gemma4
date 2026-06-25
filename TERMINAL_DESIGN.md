# Terminal-Brutalist Dashboard Design

Applied to `static/index.html` for the Cerebras benchmark dashboard.

## Palette

| Role | Hex | CSS Variable | Usage |
|------|-----|--------------|-------|
| Background | `#F8F8F8` | `--bg` | Terminal canvas (light theme, matching the reference screenshot) |
| Primary / Headers | `#4ECDC4` | `--primary` | Section titles, brand, system prefixes |
| Active / Positive | `#00FF41` | `--active` | Bars, sparklines, battery fill, status, progress |
| Real-time Highlight | `#FFD700` | `--highlight` | "NOW" badge, live deltas, current request values, scout prefixes |
| Compression / Secondary | `#9B59B6` | `--compression` | Secondary data series, DiffusionGemma accent, commander prefixes |
| Secondary Text | `#AAAAAA` | `--muted` | Labels, timestamps, file paths, scale markers |
| Body Text | `#1A1A1A` | `--text` | Default readable text |
| Error | `#FF2800` | `--error` | Error states and `[ERR]` log prefix |
| Border / Rules | `#D0D0D0` | `--border` | Panel borders and ASCII separators |

## Typography

- **Font stack:** `Menlo, Monaco, Consolas, 'Roboto Mono', monospace`
- **Headers:** ALL CAPS, bold weight, letter-spaced
- **Labels:** ALL CAPS, regular weight, left-aligned
- **Values:** Mixed case, right-aligned or inline with labels
- **Scale markers:** Small caps for percentage markers

## Layout

- High information density, minimal whitespace
- Two-column grid: 340px sidebar + flexible main, collapsing to single column at 900px
- Rectangular panels with `1px solid #333` borders, no border-radius, no decorative shadows
- ASCII-native separators: `=` for major breaks, `-` for minor rules
- Strict left-edge alignment for labels

## Components

### Header
```
HEADROOM REGEN DASHBOARD          [RACE] [NEWS]
```

### Status Line
```
user@cerebras-bench:~$ status: RUNNING
```

### Mode Selector
```
[P] [1] [2] [3] [DRS]
```
Active mode uses inverse video: green background, black text.

### Block Gauges
```
REQ/S  [████████░░░░░░░░░░░░]  8.33/16.67
TOK/S  [████████████████░░░░]  13542/16667
LAT    [░░░░░░░░░░░░░░░░░░░░]  0.234s
```
Fill: `█` (`--active`); empty: `░` (`--muted`).

### Progress Meter
```
[=====>                ] 24%
```

### Timing Tower
```
#  CONFIG       REQ/S   TOK/S   LATENCY
1  c16_t1000    4.05    4050    0.234s
```

### Terminal Log
```
04:02:15 [ENG] Preflight check complete
04:02:16 [SYS] Race started
04:02:18 [ERR] Stream connection failed
```

Prefix colors:
- `[ENG]` — `--active` (green)
- `[SYS]` — `--primary` (cyan)
- `[ERR]` — `--error` (red)
- `[SCOUT]` — `--highlight` (gold)
- `[CMDR]` — `--compression` (purple)

### Sparkline Histogram
Canvas-based vertical block bars. Primary series in `--active`, secondary series in `--compression`. 60-point rolling window.

### News Worker Map
```
scout-0  [RUNNING]  4/4 sources  [████░░░░░░]
```

### Boot Overlay
Line-by-line terminal boot animation:
```
BOOTING HEADROOM REGEN DASHBOARD...
MOUNT /dev/cerebras0... OK
LOAD telemetry modules... OK
READY
user@cerebras-bench:~$
```

## Files

- `static/index.html` — full implementation
- `F1_REDESIGN_PLAN.md` — previous F1 dashboard concept (superseded by this design)
