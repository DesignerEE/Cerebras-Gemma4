# TODO — cfire: Adaptive Throughput SDK (public product)

> **Resume context (2026-08-17).** A full audit of this repo was run on a fresh clone:
> both trees reviewed, the test suite executed (real result: **130 passed, 1 skipped,
> 1 xfailed** after manually installing undeclared demo deps), and every bug below was
> reproduced by execution, not inferred. Spec: [`PRODUCT_SPEC.md`](../PRODUCT_SPEC.md).
> This file is the execution tracker — work top-down, phases are ordered by dependency.
> **First action on resume:** Phase 2, item 1 (verify `x-ratelimit-*` + stream `usage`
> on gemma-4-31b) — it decides which feedback mode the autopilot loop is built on.

---

## Phase 0 — Truth & hygiene (days 1–2) — no new code
- [ ] Merge `cfire/` + `cfire-diffusion/`: promote `Router` (239 lines, 16 tests) and
      `OpenAICompatibleBackend` into the root package; delete the duplicate tree; one
      `pytest.ini`, one CI. *(The fork already contradicts root: `DIFFUSIONGEMMA_BASE_URL`
      defaults differ by a trailing `/v1`)*
- [ ] README truth pass: remove AgentSession / `cfire/agent.py` (does not exist),
      150→130 tests, "6 fails"→5 (breaker threshold is 5: `reliability.py:57`,
      `config.py:71`), rewrite the headroom claim (it tails a proxy log, it does not
      parse `x-ratelimit-*`)
- [ ] `pyproject`: add extras `[demo]` (fastapi, uvicorn, feedparser, beautifulsoup4,
      lxml) — `web_demo.py`/`news_agents.py` currently fail on import; drop unused
      `anyio`; exclude `cfire/tests` from the wheel; remove the private LAN IP default
      (`192.168.10.100:1235` in `cfire/config.py:37-39`) → env-only
- [ ] `.env.example` + an env-var matrix (~16 vars, see `cfire/config.py:7-25`)
- [ ] `examples/`: 01_complete.py, 02_stream.py, 03_bulk_autopilot.py — each runnable
      in one command
- [ ] CI: pytest on 3.11/3.12 + ruff + a check for private defaults

## Phase 1 — Reliability core (days 3–4) — audit bugs, all reproduced
- [ ] **B1** `cfire/_sync.py:100` — schedule the producer as a task; it currently blocks
      via `run_coroutine_threadsafe(...).result()` until the whole stream is consumed
      (sync "streaming" = full buffering, TTFT = full generation time)
- [ ] **B5** `cfire/transport.py:229-232` — catch `httpx.HTTPError` in `post_chat` AND
      `stream_chat` (currently only Timeout/Connect are mapped; ~9 of 11 transport
      errors leak raw and are never retried)
- [ ] **B6** cap `retry_after` at `max_delay` (`reliability.py:150-151`) and move the
      retry sleep OUTSIDE the semaphore (`client.py:149-181`) — one hostile
      `Retry-After: 3600` parks a slot for an hour
- [ ] **B4** `cfire/cache.py:96` — `model_copy(deep=True)` in `put()` (callers mutate
      cached responses; `get()` deep-copies, `put()` does not)
- [ ] **B3** `cfire/streaming.py:66,105` — CRLF-tolerant SSE framing + concatenate
      multiple `data:` lines per block (spec-compliant; proxies/CDNs emit CRLF)
- [ ] **B2** `cfire/reliability.py:71-96` — half-open must admit a single probe;
      re-open on probe failure regardless of the reset counter
- [ ] RedisCache — stop swallowing all exceptions silently: log + degradation counter
- [ ] Metrics — count cache hits separately from full requests; make `compressed`
      real (`_payload_compressed()` in `backends.py:179-184` always returns False)

## Phase 2 — The autopilot loop (days 5–8) — THE PRODUCT
- [ ] **Verify first:** does gemma-4-31b return `x-ratelimit-*` headers and complete
      `usage` in streams? → decides header-driven vs 429-driven feedback
- [ ] Call `parse_ratelimit_headers` (`transport.py:77` — written, tested, never wired)
      on every response, `post_chat` + `stream_chat`
- [ ] `DualRateLimiter.update(feedback)` + refund actual `usage` (kill the permanent
      `max_completion_tokens` reservation)
- [ ] AIMD controller for adaptive concurrency; `notify_all()` only when budget is
      returned (removes the thundering herd at `reliability.py:198`)
- [ ] `client.map(prompts, concurrency="auto" | int)`
- [ ] FakeBackend → server emulator: scriptable headers + 429 scenarios (limit drops
      mid-run)
- [ ] `cfire/calibration.py`: seed tables from `CEREBRAS_SPEED_GUIDE.md`
- [ ] CLI `cfire calibrate` (the race engine becomes a utility)
- [ ] Fallback mode without headers: pure AIMD on 429s (mandatory, must work)

## Phase 3 — Packaging & launch (days 9–10)
- [ ] PyPI alpha 0.2.0, `py.typed`
- [ ] README → product README: one chart "static vs autopilot sustained tok/s" from a
      real run
- [ ] SPEED_GUIDE → docs page "Understanding your rate limits"
- [ ] Landing draft + announcement (demo-video pipeline already exists)

---

## Done-when
1. Bulk 200 prompts on a real key: after warm-up 0×429, sustained ≥ 85% of the
   `cfire calibrate` ceiling, p99 stable for 10 min
2. Adversarial: limit drops mid-run → re-converges ≤ 30 s, no cascade
3. Stranger test: clean machine → pip install → one env var → example runs in < 5 min
4. CI green; coverage ≥ 85% source-only; fork deleted
5. Zero phantom claims in the README

## Non-goals (hold the line)
❌ learned/ML routing · ❌ dashboard as a product · ❌ all-in-one multi-provider ·
❌ VisionOps as a product (→ example #2)

## Final review (fill in on completion)
- [ ] Shipped vs plan: ___
- [ ] Deviations and why: ___
- [ ] Lessons → `tasks/lessons.md`: ___
