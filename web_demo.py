#!/usr/bin/env python3
"""
Cerebras Racing Web Demo — live dashboard for the advanced race client.

Run:
    .venv/bin/python web_demo.py
    open http://localhost:8000

Endpoints:
    GET  /              dashboard
    POST /api/start     {metric, sweep_requests, race_requests, concurrency, max_tokens, mock}
    POST /api/stop      stop current run
    GET  /api/status    current status
    GET  /api/stream    SSE events
"""

from __future__ import annotations

import asyncio
import json
import random
import traceback
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from cerebras_race_client import CerebrasRaceClient, CompletionResult
from news_agents import NewsAgentTeam

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


class NewsDemoRequest(BaseModel):
    """Body for POST /api/news/demo. All fields optional — mocked pipeline."""
    query: str = "OpenAI GPT-5 launch"
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

    async def run(
        self,
        metric: str,
        sweep_requests: int,
        race_requests: int,
        concurrency: list[int],
        max_tokens: list[int],
        mock: bool,
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
                mock=mock,
            ) as client:
                self.client = client

                # Preflight
                if not mock:
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

                # Race
                await self._emit({"type": "log", "level": "info", "text": f"Racing with {best['config']['name']}..."})
                client.concurrency = best["config"]["concurrency"]
                client.semaphore = asyncio.Semaphore(client.concurrency)
                prompts = self._make_prompts(race_requests, best["config"]["max_completion_tokens"])
                start = asyncio.get_event_loop().time()
                completions = await client.bulk_complete(prompts, max_completion_tokens=best["config"]["max_completion_tokens"], progress_queue=self.queue)
                batch_time = asyncio.get_event_loop().time() - start
                race_summary = {
                    "config": best["config"],
                    **self._summary(completions, batch_time),
                }
                await self._emit({"type": "race_complete", "summary": race_summary})

                self.results = {
                    "metric": metric,
                    "best": best,
                    "race": race_summary,
                    "summaries": summaries,
                    "client_metrics": client.report(),
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
                concurrency=6,
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

    async def run_demo(self, query: str = "OpenAI GPT-5 launch", angles: list[str] | None = None):
        """Mocked end-to-end run — no network, no API key, no cost.
        Emits the same event shapes as run() so the dashboard renders it identically.
        Used by the pytest e2e test and the /api/news/demo button.
        """
        from unittest.mock import patch
        from cerebras_race_client import CompletionResult

        self.stop_event.clear()
        self.results = None
        self.status = {"state": "running", "message": f"DEMO: {query}"}
        await self._emit({"type": "status", **self.status})

        CANNED_RSS = {
            "entries": [
                {"title": "OpenAI launches GPT-5", "link": "https://example.com/gpt5",
                 "published": "2026-06-20", "summary": "OpenAI announced GPT-5 with major reasoning gains."},
                {"title": "Industry reacts to GPT-5", "link": "https://example.com/react",
                 "published": "2026-06-20", "summary": "Competitors and analysts weigh in."},
            ],
        }
        CANNED_PAGES = {
            "https://example.com/gpt5": "OpenAI today launched GPT-5. It scores 95% on MMLU. Price is $5/Mtok in, $15/Mtok out.",
            "https://example.com/react": "Anthropic, Google, and Meta all responded. Anthropic emphasized safety. Google noted speed.",
        }

        async def _fake_feed(url, timeout=15.0):
            return CANNED_RSS
        async def _fake_page(url, timeout=12.0):
            return CANNED_PAGES.get(url, "Generic article body text.")

        class _FakeClient:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def complete(self, prompt, max_completion_tokens=1000, **kw):
                p = prompt.lower()
                if "you are the commander" in p or "synthesize" in p:
                    text = ("Based on scout reports, OpenAI launched GPT-5 with 95% MMLU. "
                            "Sources disagree on pricing impact. See https://example.com/gpt5.")
                else:
                    text = "Scout summary: GPT-5 launched; strong benchmarks; mixed industry reaction."
                return CompletionResult(text=text, completion_tokens=30, prompt_tokens=100,
                                         total_tokens=130, latency=0.05)

        try:
            with patch("news_agents.fetch_feed", side_effect=_fake_feed), \
                 patch("news_agents.fetch_page", side_effect=_fake_page):
                team = NewsAgentTeam(_FakeClient(), angles=angles or ["latest breaking news"])
                report = await team.search(query, progress_queue=self.queue)
                self.results = report.model_dump()
                self.status = {"state": "done", "message": "Demo complete"}
        except asyncio.CancelledError:
            self.status = {"state": "stopped", "message": "Stopped by user"}
            await self._emit({"type": "status", **self.status})
            raise
        except Exception as e:
            self.status = {"state": "error", "message": str(e)}
            await self._emit({"type": "error", "error": str(e), "traceback": traceback.format_exc()})
        finally:
            await self._emit({"type": "status", **self.status})

    def start_demo(self, query: str = "OpenAI GPT-5 launch", angles: list[str] | None = None):
        if self.task and not self.task.done():
            raise RuntimeError("News search already running")
        self.task = asyncio.create_task(self.run_demo(query, angles))

    def stop(self):
        if self.task and not self.task.done():
            self.stop_event.set()
            self.task.cancel()
            self.status = {"state": "stopping", "message": "Stopping..."}
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
            mock=bool(data.get("mock", False)),
        )
        return {"ok": True, "status": race_manager.status}
    except Exception as e:
        return {"ok": False, "error": str(e)}, 400


@app.post("/api/stop")
async def stop_benchmark():
    ok = race_manager.stop()
    return {"ok": ok, "status": race_manager.status}


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
                # Drain any events emitted after state change (e.g. fast mocked runs)
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


@app.post("/api/news/demo")
async def demo_news(req: NewsDemoRequest):
    """Run the mocked news pipeline. Zero API cost. Events flow through /api/news/stream."""
    try:
        query = (req.query or "OpenAI GPT-5 launch").strip() or "OpenAI GPT-5 launch"
        angles = req.angles or ["latest breaking news"]
        news_manager.start_demo(query, angles=angles)
        return {"ok": True, "status": news_manager.status, "demo": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}, 400


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
                # Drain any events emitted after state change (e.g. fast mocked runs)
                while not news_manager.queue.empty():
                    event = await news_manager.queue.get()
                    yield f"data: {json.dumps(event, default=str)}\n\n"
                await asyncio.sleep(1)
                break

    return StreamingResponse(generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
