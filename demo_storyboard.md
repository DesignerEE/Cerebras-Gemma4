# VisionOps Demo Storyboard

**Target length:** 60 seconds  
**Format:** Pre-recorded video from screenshots (silent)  
**Primary prize angle:** Multiverse Agents (multi-agent + multimodal)  
**Secondary angle:** Enterprise Impact (SRE/observability)  
**Output file:** `demo_video.mp4`  

## Asset Checklist

Before you run `build_demo_video.py`, collect these screenshots in `demo_shots/`:

| Asset | Filename | Source |
|-------|----------|--------|
| Boot/title shot | `demo_shots/01_boot.png` | Screenshot of dashboard loading + title card |
| Vision upload | `demo_shots/02_vision_upload.png` | VISION tab with image selected |
| Vision diagnosis | `demo_shots/03_vision_diagnosis.png` | VISION tab showing Gemma 4 diagnosis |
| News scouts | `demo_shots/04_news_scouts.png` | NEWS tab with scouts deploying |
| News report | `demo_shots/05_news_report.png` | NEWS tab commander report complete |
| Race start | `demo_shots/06_race_start.png` | RACE tab idle before start |
| Race peak | `demo_shots/07_race_peak.png` | RACE tab gauges at peak tok/s |
| Headroom panel | `demo_shots/08_headroom.png` | HEADROOM tab / panel |
| Benchmark memory | `demo_shots/09_benchmark_memory.png` | BENCHMARK MEMORY ranking panel |
| Physical alert | `demo_shots/10_physical_alert.png` | Dashboard + red light/phone screen |
| End card | `demo_shots/11_end_card.png` | Logo + tagline |

## Shot Breakdown

### 01 — Boot / Title Card
- **Duration:** 4s
- **Visual:** Black screen → terminal dashboard boot overlay → "VisionOps" title
- **Caption:** "VisionOps — See the problem. Fix the problem."
- **File:** `01_boot.png`

### 02 — Vision Upload
- **Duration:** 5s
- **Visual:** VISION tab, image of a failing device (router LEDs / server rack / error screen) being uploaded
- **Caption:** "Point any camera at the failing system."
- **File:** `02_vision_upload.png`

### 03 — Vision Diagnosis
- **Duration:** 8s
- **Visual:** Diagnosis card appears: severity, root cause, recommended actions
- **Caption:** "Gemma 4 31B sees the issue and prescribes a fix."
- **File:** `03_vision_diagnosis.png`

### 04 — News Scouts Deploy
- **Duration:** 6s
- **Visual:** NEWS tab, three scout cards spinning up, RSS feeds loading
- **Caption:** "While it diagnoses, autonomous agents gather live context."
- **File:** `04_news_scouts.png`

### 05 — News Commander Report
- **Duration:** 7s
- **Visual:** Commander report with citations, CVEs, outage context
- **Caption:** "A commander agent synthesizes everything into an action plan."
- **File:** `05_news_report.png`

### 06 — Race Start
- **Duration:** 4s
- **Visual:** RACE tab, start button pressed, gauges waking up
- **Caption:** "When speed matters, cfire saturates Cerebras inference."
- **File:** `06_race_start.png`

### 07 — Race Peak
- **Duration:** 7s
- **Visual:** RACE tab, tok/s counter climbing, sparklines peaking
- **Caption:** "Thousands of tokens per second, live."
- **File:** `07_race_peak.png`

### 08 — Headroom Telemetry
- **Duration:** 5s
- **Visual:** HEADROOM panel with live compression/perf metrics
- **Caption:** "Real-time telemetry from the edge."
- **File:** `08_headroom.png`

### 09 — Benchmark Memory
- **Duration:** 5s
- **Visual:** BENCHMARK MEMORY panel ranking the fastest configs
- **Caption:** "It remembers the fastest setup for next time."
- **File:** `09_benchmark_memory.png`

### 10 — Physical Alert
- **Duration:** 5s
- **Visual:** Dashboard + phone/LED turning red or alarm indicator
- **Caption:** "The dashboard acts in the physical world."
- **File:** `10_physical_alert.png`

### 11 — End Card
- **Duration:** 4s
- **Visual:** Logo + "VisionOps — Multiverse Agents" + hackathon badge
- **Caption:** "Multi-agent. Multimodal. Physically aware."
- **File:** `11_end_card.png`

## Timing Summary

```
01  0:00  Boot              4s
02  0:04  Vision upload     5s
03  0:09  Diagnosis         8s
04  0:17  Scouts deploy     6s
05  0:23  Commander report  7s
06  0:30  Race start        4s
07  0:34  Race peak         7s
08  0:41  Headroom          5s
09  0:46  Benchmark memory  5s
10  0:51  Physical alert    5s
11  0:56  End card          4s
    ─────────────────────────
    Total                60s
```

## Capture Tips

1. Use browser zoom (Ctrl/Cmd +) to make text crisp at 1080p.
2. Use dark mode if the dashboard supports it — the terminal design looks best dark.
3. Hide browser toolbars with F11 or use a clean Chromium profile.
4. For the diagnosis shot, replace any sensitive hostnames/IPs with placeholders.
5. For the physical alert shot, point a phone flashlight at a red object if you do not have a smart bulb.

## Submission Notes

- Keep `web_demo.py` running during the live pitch in case judges ask for a live walkthrough.
- Upload `demo_video.mp4` to YouTube / Loom / Google Drive as a backup.
- For People's Choice, post a 15-second vertical cut (9:16) of shots 02 → 03 → 07 → 11 on X/LinkedIn/TikTok.
