# cfire — Adaptive Throughput SDK

**Positioning:** `pip install cfire` → bulk LLM jobs automatically hold the maximum sustainable
speed on Cerebras-class inference, where the bottleneck is rate limits, not latency.
Zero 429s. No manual concurrency tuning.

**One-liner for the README:**
*"Autopilot for API throughput: your bulk jobs run at the maximum sustainable speed your
tier allows — measured, not guessed."*

> Status: spec (2026-08-17). Execution tracker: [`tasks/todo.md`](tasks/todo.md).
> Derived from a full code audit of this repo (both `cfire/` and `cfire-diffusion/` trees,
> tests executed, all listed bugs reproduced).

---

## 1. Product

### Target user
Developers running bulk jobs (classification, extraction, corpus generation, RAG warm-up)
through Cerebras / OpenAI-compatible APIs.

**Pain:** static concurrency either under-uses the tier or triggers a 429 storm; limits vary
by tier and over time; hand-tuned tables don't transfer between accounts.

### The core (the one new feature = closing the loop)

```
response ──> parse_ratelimit_headers() ──> {rem_req, rem_tok, limits, reset}
                                                    │
                                                    v
        DualRateLimiter.update(feedback)  <──  AIMD controller (adaptive concurrency)
              ^                                     │
              └── refund actual usage <── response.usage
```

1. **Observe** — every response carries `x-ratelimit-*` (the parser already exists and is
   tested in `cfire/transport.py:77` — it has never been called).
2. **Update** — static rpm/tpm replaced by measured values; token budget reserved
   pessimistically, refunded from actual `usage`.
3. **Control** — AIMD: additive increase without throttling, multiplicative decrease on
   429 / approaching the limit. Target = dynamic concurrency level.
4. **Protect** — circuit breaker on 5xx/timeout clusters; capped retries; queue, don't drop.
5. **Report** — honest metrics: sustained tok/s, utilization % vs measured tier ceiling,
   429 count.

### API (MVP)

```python
import cfire

async with cfire.open() as client:                     # CEREBRAS_API_KEY from env
    results = await client.map(prompts, concurrency="auto")   # autopilot
    # or explicit: client.map(prompts, target_rps=8)

print(client.report())
# {"sustained_toks": 2113, "utilization": 0.91, "errors_429": 0, "p50_ms": ..., "p99_ms": ...}
```

A sync mirror ships too (after the `_sync.py` fix — see Phase 1, B1).

### Calibration CLI (turns the race engine into a utility)

```bash
cfire calibrate --prompts 200
# measures YOUR real ceiling on YOUR tier
# → writes ~/.cfire/calibration.json (autopilot seed) + prints a SPEED_GUIDE-style table
```

Seed priors come from `CEREBRAS_SPEED_GUIDE.md` (c4 sustained = 0 errors / 2,113 tok/s /
88 reqs) — the controller starts near the optimum instead of from zero.

---

## 2. Scope and non-goals

**In scope:** Cerebras + any OpenAI-compatible endpoint; async + sync; `map()` bulk;
autopilot + manual mode; calibration; metrics.

**Non-goals (explicit):**
- ❌ Learned/ML routing (that's the personal-infra path — later, on this same substrate)
- ❌ Dashboard as a product (the race UI stays a demo showcase)
- ❌ All-in-one multi-provider orchestration (narrow positioning is the strength)
- ❌ VisionOps as a product (demoted to example #2: a multimodal use case for the SDK)

**Monetization (later, non-blocking):** open-core — SDK stays MIT; hosted
calibration/telemetry for teams.

---

## 3. Execution order

Full checklist with acceptance criteria lives in [`tasks/todo.md`](tasks/todo.md):

| Phase | Days | Content |
|---|---|---|
| 0 — Truth & hygiene | 1–2 | merge `cfire`/`cfire-diffusion`, README truth pass, packaging extras, `.env.example`, examples, CI — no new code |
| 1 — Reliability core | 3–4 | fix B1–B7 (verified audit bugs: sync streaming, transport exception hole, retry caps, cache poisoning, SSE CRLF, half-open breaker) |
| 2 — Autopilot loop | 5–8 | THE PRODUCT: wire `parse_ratelimit_headers` → `DualRateLimiter.update()` + AIMD + usage refund; `map()`; calibration |
| 3 — Packaging & launch | 9–10 | PyPI alpha, product README, SPEED_GUIDE → docs page, landing |

---

## 4. Done-when

1. **Autopilot on a real key:** 200-prompt bulk job — after warm-up, 0×429 errors,
   sustained ≥ 85% of the ceiling measured by `cfire calibrate`, p99 stable for 10 minutes
2. **Adversarial test:** limit drops mid-run (simulated) → controller re-converges ≤ 30 s,
   no 429 cascade
3. **Stranger test:** clean machine → pip install → one env var → example runs in < 5 min
4. CI green, coverage ≥ 85% source-only (excluding `cfire/tests`), fork deleted
5. **Zero phantom claims:** every README feature maps to a file that exists

## 5. Risks (honest)

| Risk | Response |
|---|---|
| `x-ratelimit-*` headers absent/inconsistent | Fallback: pure AIMD on 429s — this mode is mandatory and must work |
| Cerebras ships an official adaptive client | Moat: cross-provider support + calibration data + SPEED_GUIDE empirics as content; move fast |
| Crowded space (openai-sdk + tenacity) | Narrow positioning: "for inference where limits — not latency — are the bottleneck" |
| `usage` incomplete in streams | Verify on gemma-4-31b first; fallback = tokenizer-based estimate |
