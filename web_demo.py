#!/usr/bin/env python3
"""
Cerebras Racing Web Demo — live dashboard for the advanced race client.

Run:
    .venv/bin/python web_demo.py
    open http://localhost:8000

Endpoints:
    GET  /                   dashboard
    POST /api/start          {metric, sweep_requests, race_requests, concurrency, max_tokens}
    POST /api/stop           stop current run
    GET  /api/status         current status
    GET  /api/stream         SSE events
    GET  /api/race/report    last deterministic report
    POST /api/race/report/llm  LLM narrative report
    POST /api/news/start     {query, angles}
    POST /api/news/stop      stop news search
    GET  /api/news/stream    news SSE events
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx

from cerebras_race_client import CerebrasRaceClient, CompletionResult
from news_agents import NewsAgentTeam
import cfire

STATIC_DIR = Path(__file__).parent / "static"


def long_prompt(topic: str, words: int) -> str:
    return (
        f"Write a detailed, continuous technical explanation of '{topic}' "
        f"that is at least {words} words long. Do not use lists or headers; "
        f"use flowing paragraphs only. Start immediately with the content."
    )


TOPICS = [
    "quantum computing", "neural network backpropagation", "asyncio event loops",
    "container orchestration", "distributed consensus protocols", "HTTP/2 multiplexing",
    "transformer architecture", "vector databases", "GPU memory hierarchy",
    "reinforcement learning",
]


# --- News API contract models ----------------------------------------------
# Typed request bodies — FastAPI validates automatically and generates OpenAPI.

class NewsStartRequest(BaseModel):
    """Body for POST /api/news/start. Query is required."""
    query: str
    angles: list[str] | None = None


class RaceManager:
    def __init__(self):
        self.task: asyncio.Task | None = None
        self.queue: asyncio.Queue = asyncio.Queue()
        self.status: dict[str, Any] = {"state": "idle", "message": "Ready"}
        self.stop_event = asyncio.Event()
        self.results: dict[str, Any] | None = None
        self.client: CerebrasRaceClient | None = None

    async def _emit(self, event: dict):
        await self.queue.put(event)

    def _make_prompts(self, count: int, target_words: int) -> list[str]:
        return [long_prompt(random.choice(TOPICS), target_words) for _ in range(count)]

    def _summary(self, completions: list[CompletionResult], batch_time: float) -> dict[str, Any]:
        ok = [c for c in completions if not c.cached]
        latencies = [c.latency for c in completions]
        total_tok = sum(c.completion_tokens for c in completions)
        return {
            "num_requests": len(completions),
            "batch_time": batch_time,
            "req_per_sec": len(completions) / batch_time if batch_time > 0 else 0,
            "tok_per_sec": total_tok / batch_time if batch_time > 0 else 0,
            "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
            "cache_hits": sum(1 for c in completions if c.cached),
            "compressed": sum(1 for c in completions if c.compressed),
        }

    @staticmethod
    def _score(summary: dict[str, Any], metric: str) -> float:
        if metric == "tok/s":
            return summary["tok_per_sec"]
        if metric == "req/s":
            return summary["req_per_sec"]
        # balanced — weighted combo normalised to per-minute ceilings
        return summary["req_per_sec"] / (1000 / 60) + summary["tok_per_sec"] / (1_000_000 / 60)

    def _build_race_report(
        self,
        metric: str,
        summaries: list[dict[str, Any]],
        race_summary: dict[str, Any],
    ) -> dict[str, Any]:
        """Deterministic post-race report. No LLM call.

        Picks winner & runner-up by metric, computes margin, and templated
        verdict. Used by the 'Show Report' button; the LLM Analysis button
        hits /api/race/report/llm for a narrative version.
        """
        ranked = sorted(summaries, key=lambda s: self._score(s, metric), reverse=True)
        winner = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        win_score = self._score(winner, metric)
        runner_score = self._score(runner_up, metric) if runner_up else 0.0
        margin_pct = (((win_score - runner_score) / runner_score) * 100
                      if runner_score > 0 else float("inf"))

        if margin_pct == float("inf"):
            verdict_strength = "was the only config tested and"
        elif margin_pct > 25:
            verdict_strength = "dominated"
        elif margin_pct > 10:
            verdict_strength = "clearly beat"
        elif margin_pct > 3:
            verdict_strength = "edged out"
        else:
            verdict_strength = "essentially tied with"

        runner_phrase = (
            f"the field on {metric}: {win_score:.1f} vs runner-up's "
            f"{runner_score:.1f} ({margin_pct:+.1f}%)"
            if runner_up else
            f"the field on {metric}: {win_score:.1f} (no runner-up)"
        )
        verdict = (
            f"Config {winner['config']['name']} "
            f"(concurrency={winner['config']['concurrency']}, "
            f"max_tokens={winner['config']['max_completion_tokens']}) "
            f"{verdict_strength} {runner_phrase}. "
            f"Over {race_summary.get('actual_duration_sec', 0):.1f}s of sustained racing "
            f"it held {race_summary.get('tok_per_sec', 0):.1f} tok/s "
            f"({race_summary.get('num_requests', 0)} requests, "
            f"avg latency {race_summary.get('avg_latency', 0) * 1000:.0f}ms)."
        )

        return {
            "metric": metric,
            "winner": {
                "name": winner["config"]["name"],
                "concurrency": winner["config"]["concurrency"],
                "max_tokens": winner["config"]["max_completion_tokens"],
                "score": win_score,
                "tok_per_sec": winner["tok_per_sec"],
                "req_per_sec": winner["req_per_sec"],
                "avg_latency_ms": winner["avg_latency"] * 1000,
            },
            "runner_up": ({
                "name": runner_up["config"]["name"],
                "concurrency": runner_up["config"]["concurrency"],
                "max_tokens": runner_up["config"]["max_completion_tokens"],
                "score": runner_score,
            } if runner_up else None),
            "margin_pct": margin_pct if margin_pct != float("inf") else None,
            "race_actual_duration_sec": race_summary.get("actual_duration_sec",
                                                         race_summary.get("batch_time", 0)),
            "race_total_requests": race_summary.get("num_requests", 0),
            "race_avg_latency_ms": race_summary.get("avg_latency", 0) * 1000,
            "all_configs": [
                {
                    "name": s["config"]["name"],
                    "concurrency": s["config"]["concurrency"],
                    "max_tokens": s["config"]["max_completion_tokens"],
                    "score": self._score(s, metric),
                    "tok_per_sec": s["tok_per_sec"],
                    "req_per_sec": s["req_per_sec"],
                }
                for s in ranked
            ],
            "verdict": verdict,
        }

    async def run(
        self,
        metric: str,
        sweep_requests: int,
        race_requests: int,
        concurrency: list[int],
        max_tokens: list[int],
    ):
        self.stop_event.clear()
        self.results = None
        self.status = {"state": "running", "message": "Connecting..."}
        await self._emit({"type": "status", **self.status})

        try:
            async with CerebrasRaceClient(
                concurrency=8,
                enable_cache=False,  # cache would pollute throughput measurements
                enable_compression=True,
            ) as client:
                self.client = client

                # Preflight
                await self._emit({"type": "log", "level": "info", "text": "Preflight check..."})
                pre = await client.complete("Say hello world.", max_completion_tokens=50)
                await self._emit({"type": "log", "level": "info", "text": f"Preflight OK: {pre.completion_tokens} tokens in {pre.latency:.3f}s"})

                # Sweep
                await self._emit({"type": "log", "level": "info", "text": f"Running {len(concurrency)*len(max_tokens)} scouts..."})
                summaries = []
                for c in concurrency:
                    for t in max_tokens:
                        if self.stop_event.is_set():
                            raise asyncio.CancelledError()
                        client.concurrency = c
                        client.semaphore = asyncio.Semaphore(c)
                        prompts = self._make_prompts(sweep_requests, t)
                        start = asyncio.get_event_loop().time()
                        completions = await client.bulk_complete(prompts, max_completion_tokens=t, progress_queue=self.queue)
                        batch_time = asyncio.get_event_loop().time() - start
                        summary = {
                            "config": {"name": f"c{c}_t{t}", "concurrency": c, "max_completion_tokens": t},
                            **self._summary(completions, batch_time),
                        }
                        summaries.append(summary)
                        await self._emit({"type": "sweep_point", "summary": summary})

                await self._emit({"type": "sweep_complete", "summaries": summaries})

                # Pick best
                def score(s):
                    if metric == "tok/s":
                        return s["tok_per_sec"]
                    if metric == "req/s":
                        return s["req_per_sec"]
                    return s["req_per_sec"] / (1000 / 60) + s["tok_per_sec"] / (1_000_000 / 60)

                best = max(summaries, key=score)
                await self._emit({"type": "best_config", "summary": best})

                # Race — time-bound: random target in [20, 35]s, chunked so we
                # can check the deadline between batches. Total wall time may
                # overshoot by up to one chunk's duration (one wave of requests).
                race_duration_sec = random.uniform(20, 35)
                await self._emit({"type": "log", "level": "info",
                                  "text": f"Racing with {best['config']['name']} for ~{race_duration_sec:.0f}s..."})
                client.concurrency = best["config"]["concurrency"]
                client.semaphore = asyncio.Semaphore(client.concurrency)
                max_t = best["config"]["max_completion_tokens"]
                chunk_size = max(client.concurrency, 8)

                all_completions: list[CompletionResult] = []
                start = asyncio.get_event_loop().time()
                batch_idx = 0
                while True:
                    elapsed = asyncio.get_event_loop().time() - start
                    if elapsed >= race_duration_sec:
                        break
                    if self.stop_event.is_set():
                        raise asyncio.CancelledError()
                    prompts = self._make_prompts(chunk_size, max_t)
                    chunk = await client.bulk_complete(
                        prompts, max_completion_tokens=max_t, progress_queue=self.queue,
                    )
                    all_completions.extend(chunk)
                    batch_idx += 1
                    await self._emit({"type": "log", "level": "debug",
                                      "text": f"Wave {batch_idx}: {len(all_completions)} reqs in {elapsed:.1f}s"})

                batch_time = asyncio.get_event_loop().time() - start
                race_summary = {
                    "config": best["config"],
                    **self._summary(all_completions, batch_time),
                    "target_duration_sec": race_duration_sec,
                    "actual_duration_sec": batch_time,
                }
                await self._emit({"type": "race_complete", "summary": race_summary})

                # Build deterministic post-race report
                report = self._build_race_report(metric, summaries, race_summary)
                await self._emit({"type": "race_report", "report": report})

                self.results = {
                    "metric": metric,
                    "best": best,
                    "race": race_summary,
                    "summaries": summaries,
                    "client_metrics": client.report(),
                    "report": report,
                }
                self.status = {"state": "done", "message": "Benchmark complete"}
        except asyncio.CancelledError:
            self.status = {"state": "stopped", "message": "Stopped by user"}
            await self._emit({"type": "status", **self.status})
            raise
        except Exception as e:
            self.status = {"state": "error", "message": str(e)}
            await self._emit({"type": "error", "error": str(e), "traceback": traceback.format_exc()})
        finally:
            self.client = None
            await self._emit({"type": "status", **self.status})

    def start(self, **kwargs):
        if self.task and not self.task.done():
            raise RuntimeError("Benchmark already running")
        self.task = asyncio.create_task(self.run(**kwargs))

    def stop(self):
        if self.task and not self.task.done():
            self.stop_event.set()
            self.task.cancel()
            self.status = {"state": "stopping", "message": "Stopping..."}
            return True
        return False


class NewsManager:
    def __init__(self):
        self.task: asyncio.Task | None = None
        self.queue: asyncio.Queue = asyncio.Queue()
        self.status: dict[str, Any] = {"state": "idle", "message": "Ready"}
        self.stop_event = asyncio.Event()
        self.results: dict[str, Any] | None = None

    async def _emit(self, event: dict):
        await self.queue.put(event)

    async def run(self, query: str, angles: list[str] | None = None):
        self.stop_event.clear()
        self.results = None
        self.status = {"state": "running", "message": f"Searching: {query}"}
        await self._emit({"type": "status", **self.status})

        try:
            async with CerebrasRaceClient(
                concurrency=12,  # tuned for 3 scouts + commander parallelism
                enable_cache=False,
                enable_compression=True,
            ) as client:
                team = NewsAgentTeam(client, angles=angles)
                report = await team.search(query, progress_queue=self.queue)
                self.results = report.model_dump()
                self.status = {"state": "done", "message": "Search complete"}
        except asyncio.CancelledError:
            self.status = {"state": "stopped", "message": "Stopped by user"}
            await self._emit({"type": "status", **self.status})
            raise
        except Exception as e:
            self.status = {"state": "error", "message": str(e)}
            await self._emit({"type": "error", "error": str(e), "traceback": traceback.format_exc()})
        finally:
            await self._emit({"type": "status", **self.status})

    def start(self, query: str, angles: list[str] | None = None):
        if self.task and not self.task.done():
            raise RuntimeError("News search already running")
        self.task = asyncio.create_task(self.run(query, angles))

    def stop(self):
        if self.task and not self.task.done():
            self.stop_event.set()
            self.task.cancel()
            self.status = {"state": "stopping", "message": "Stopping..."}
            asyncio.create_task(self._emit({"type": "status", **self.status}))
            return True
        return False


race_manager = RaceManager()
news_manager = NewsManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if race_manager.task and not race_manager.task.done():
        race_manager.stop()
    if news_manager.task and not news_manager.task.done():
        news_manager.stop()


app = FastAPI(title="Cerebras Racing Demo", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=(STATIC_DIR / "index.html").read_text())


@app.post("/api/start")
async def start_benchmark(request: Request):
    data = await request.json()
    try:
        race_manager.start(
            metric=data.get("metric", "tok/s"),
            sweep_requests=int(data.get("sweep_requests", 10)),
            race_requests=int(data.get("race_requests", 50)),
            concurrency=[int(x) for x in data.get("concurrency", [8, 16, 24])],
            max_tokens=[int(x) for x in data.get("max_tokens", [500, 1000])],
        )
        return {"ok": True, "status": race_manager.status}
    except Exception as e:
        return {"ok": False, "error": str(e)}, 400


@app.post("/api/stop")
async def stop_benchmark():
    ok = race_manager.stop()
    return {"ok": ok, "status": race_manager.status}


@app.post("/api/race/report/llm")
async def race_report_llm():
    """Generate an LLM narrative analysis of the last race's results.

    Uses a fresh single-shot CerebrasRaceClient (the race client is closed
    by the time the user clicks the button). Reads from race_manager.results.
    """
    if not race_manager.results:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "no race results yet — run a race first"},
        )

    r = race_manager.results
    metric = r.get("metric", "tok/s")
    summaries = r.get("summaries", [])
    race = r.get("race", {})

    def _fmt(s):
        return (f"- {s['config']['name']} (c={s['config']['concurrency']}, "
                f"t={s['config']['max_completion_tokens']}): "
                f"{s.get('tok_per_sec', 0):.1f} tok/s, "
                f"{s.get('req_per_sec', 0):.1f} req/s, "
                f"avg latency {s.get('avg_latency', 0)*1000:.0f}ms")

    sweep_lines = "\n".join(_fmt(s) for s in summaries) or "- (no sweep data)"
    winner_name = r.get("best", {}).get("config", {}).get("name", "unknown")

    prompt = (
        f"You are a benchmark race analyst. In 2-3 tight paragraphs, explain why "
        f"the winning config won this Cerebras inference race.\n\n"
        f"Optimization metric: {metric}\n"
        f"Race duration: {race.get('actual_duration_sec', race.get('batch_time', 0)):.1f}s\n\n"
        f"Configs tested during sweep:\n{sweep_lines}\n\n"
        f"Winner: {winner_name}\n"
        f"Sustained race result: {race.get('tok_per_sec', 0):.1f} tok/s, "
        f"{race.get('num_requests', 0)} requests completed, "
        f"avg latency {race.get('avg_latency', 0)*1000:.0f}ms.\n\n"
        f"Cover:\n"
        f"1. Why the winner won — what concurrency/token tradeoff did it strike?\n"
        f"2. What the runner-up configs revealed about the API's behavior under load\n"
        f"3. One concrete recommendation for the next race\n"
    )

    try:
        async with CerebrasRaceClient(
            concurrency=1, enable_cache=False, enable_compression=True,
        ) as client:
            result = await client.complete(
                prompt, max_completion_tokens=800, temperature=0.4,
            )
        return {"ok": True, "report": result.text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.get("/api/race/report")
async def race_report():
    """Return the deterministic post-race report for the last completed race.

    Lets the 'Show Report' button work even if the SSE race_report event was
    missed (e.g., page reload, stream reconnect).
    """
    if not race_manager.results or "report" not in race_manager.results:
        return JSONResponse(
            status_code=404,
            content={"ok": False, "error": "no race results yet — run a race first"},
        )
    return {"ok": True, "report": race_manager.results["report"]}


@app.get("/api/status")
async def get_status():
    return {
        "race": {"status": race_manager.status, "results": race_manager.results},
        "news": {"status": news_manager.status, "results": news_manager.results},
    }


@app.get("/api/stream")
async def event_stream():
    async def generator():
        yield f"data: {json.dumps({'type': 'status', **race_manager.status})}\n\n"
        while True:
            try:
                event = await asyncio.wait_for(race_manager.queue.get(), timeout=30.0)
                yield f"data: {json.dumps(event, default=str)}\n\n"
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
            if race_manager.status.get("state") in ("done", "error", "stopped"):
                # Drain any events emitted after state change
                while not race_manager.queue.empty():
                    event = await race_manager.queue.get()
                    yield f"data: {json.dumps(event, default=str)}\n\n"
                await asyncio.sleep(1)
                break

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.post("/api/news/start")
async def start_news(req: NewsStartRequest):
    """Schema-validated: FastAPI returns 422 if `query` missing or empty."""
    query = req.query.strip()
    if not query:
        return {"ok": False, "error": "query required"}, 400
    try:
        news_manager.start(query, angles=req.angles)
        return {"ok": True, "status": news_manager.status}
    except Exception as e:
        return {"ok": False, "error": str(e)}, 400


@app.post("/api/news/stop")
async def stop_news():
    ok = news_manager.stop()
    return {"ok": ok, "status": news_manager.status}


@app.get("/api/news/stream")
async def news_event_stream():
    async def generator():
        yield f"data: {json.dumps({'type': 'status', **news_manager.status})}\n\n"
        while True:
            try:
                event = await asyncio.wait_for(news_manager.queue.get(), timeout=30.0)
                yield f"data: {json.dumps(event, default=str)}\n\n"
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
            if news_manager.status.get("state") in ("done", "error", "stopped"):
                # Drain any events emitted after state change
                while not news_manager.queue.empty():
                    event = await news_manager.queue.get()
                    yield f"data: {json.dumps(event, default=str)}\n\n"
                await asyncio.sleep(1)
                break

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.get("/api/diffusiongemma/stream_test")
async def stream_test_diffusiongemma():
    """Streaming inference test against the LAN DiffusionGemma4 server.

    Targets ``192.168.10.100:1235`` (the vLLM DiffusionGemma4 instance).
    Because this diffusion model returns the full response in a single SSE
    chunk, the token count is estimated from the generated text length so the
    dashboard Speed gauge shows a realistic tok/s value.
    """

    async def generator():
        start = time.perf_counter()
        text = ""
        try:
            backend = cfire.DiffusionGemmaBackend()
            model = backend._default_model()
            url = f"{backend.base_url}/chat/completions"
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "Count from one to ten."}],
                "max_completion_tokens": 50,
                "stream": True,
            }
            async with httpx.AsyncClient(timeout=60.0) as http_client:
                async with http_client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers={"Accept": "text/event-stream"},
                ) as resp:
                    if resp.status_code >= 400:
                        body = await resp.aread()
                        raise Exception(f"{resp.status_code}: {body.decode('utf-8', errors='replace')[:200]}")
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            obj = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        choice = (obj.get("choices") or [{}])[0]
                        delta = (choice.get("delta") or {}).get("content", "")
                        finish = choice.get("finish_reason")
                        if delta:
                            text += delta
                        elapsed = time.perf_counter() - start
                        tokens = len(text)
                        done = bool(finish)
                        yield f"data: {json.dumps({
                            'elapsed_ms': round(elapsed * 1000, 1),
                            'tokens': tokens,
                            'text': text,
                            'tok_per_sec': round(tokens / elapsed, 1) if elapsed > 0 else 0,
                            'done': done,
                        })}\n\n"
                        if done:
                            return
        except Exception as e:
            elapsed = time.perf_counter() - start
            yield f"data: {json.dumps({
                'error': str(e),
                'elapsed_ms': round(elapsed * 1000, 1),
                'done': True,
            })}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
