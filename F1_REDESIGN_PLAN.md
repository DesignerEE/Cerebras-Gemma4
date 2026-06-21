# F1-Style Dashboard Redesign Plan

## Core Metaphor
Treat the benchmark as an F1 race car with a steering-wheel dashboard:
- **Driver** = user
- **Engine** = Cerebras WSE-3
- **Gearbox** = concurrency + token preset modes
- **Telemetry** = live req/s, tok/s, latency gauges
- **Pit Wall** = commander-scout news agents
- **Team Radio** = log console

## 1. Strategy Modes (Gear Selector)
A row of F1-style mode buttons that instantly switch the benchmark config:

| Mode | Icon | Concurrency | Max Tokens | Purpose |
|------|------|-------------|------------|---------|
| P    | Pit  | 0 (idle)    | —          | Stop / reset |
| 1    | Eco  | 8           | 250        | Low latency probe |
| 2    | Race | 16          | 1000       | Balanced race pace |
| 3    | Qualy| 24          | 1500       | Max qualifying lap |
| DRS  | Boost| 32          | 1000       | Overtake — push both rate limits |

Selecting a mode updates the config form and applies a visual "gear engaged" animation.

## 2. Steering-Wheel Telemetry Gauges
Three SVG gauges in the header/main area:
- **RPM gauge (left)** — req/s vs 16.67 req/s theoretical ceiling
- **Speedometer (center)** — tok/s vs 16,667 tok/s ceiling
- **Turbo/ERS gauge (right)** — current latency vs target <1s

Needles animate smoothly with CSS transitions on each update.

## 3. Race Track Progress
A horizontal "track" bar at the top of the results area:
- Start/finish line
- Sector markers (25%, 50%, 75%)
- A small F1 car SVG that moves from left to right as requests complete
- Lap time display (batch elapsed time)

## 4. Start Lights Sequence
Five red light circles that illuminate one-by-one, then go out — the traditional F1 start sequence — before the benchmark begins.

## 5. Team Radio Log
Restyle the log console as team radio messages:
- Prefix messages with "Engineer: " or "Driver: "
- Use radio static / crackle visual cues
- Color-code: green = OK, yellow = caution, red = problem

## 6. F1 Timing Tower Results
Replace the results table with a vertical timing tower:
- Position numbers (1, 2, 3...)
- Driver-style config names (e.g., "Ricciardo" for c16_t1000)
- Gap to leader
- Highlight fastest lap (best config) in purple

## 7. News Agents Strategy Map
For the News Agents tab, draw a pit-wall strategy map:
- Three scout "cars" on parallel lanes
- Each advances as it completes sources
- Commander car follows behind
- Final synthesis = chequered flag

## 8. Color Palette & Typography
- Background: carbon fiber dark (#0a0a0f) with subtle grid
- Primary accent: Ferrari red (#ff2800) + neon cyan (#00f0ff)
- Secondary: McLaren papaya (#ff8000), Mercedes teal (#00d2be)
- Font: F1-style condensed sans-serif (use system fonts: Inter / Roboto Condensed via Google Fonts)
- Sharp corners, technical lines, telemetry numbers

## Implementation Files
- `static/index.html` — full UI rewrite
- `static/f1_theme.css` — optional separated theme
- `web_demo.py` — no backend changes needed (same endpoints)

## Next Steps
1. Rewrite HTML/CSS with F1 theme
2. Add SVG gauges with needle animation
3. Add gear selector and mode presets
4. Add start lights overlay
5. Add race track progress bar
6. Restyle logs as team radio
7. Restyle results as timing tower
8. Test real benchmark
