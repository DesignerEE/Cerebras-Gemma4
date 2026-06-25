# Lessons learned

## 2026-06-22 — DiffusionGemma4 test gauge feedback

[Mistake pattern]: When a user asks for a live gauge animation during an inference request, I implemented a synthetic sine-wave pulse that moved the needles randomly.

[Rule to prevent recurrence]: For "live" inference metrics, stream the actual model output and update the gauges with real data: token speed (tok/s), requests in flight, and elapsed latency. Synthetic oscillations feel random and do not satisfy the request.
