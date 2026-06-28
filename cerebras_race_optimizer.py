#!/usr/bin/env python3
"""
Cerebras Racing Optimizer — Commander/Scout throughput auto-tuner.

Goal: maximize throughput (tok/s or req/s) on Cerebras API within rate limits.
Strategy:
  1. Scout preflight: verify auth, model, token ratio
  2. Scout sweep: test (concurrency, max_tokens) combos in parallel
  3. Commander: fit Pareto frontier and pick winning config
  4. Final race: sustained run with winning config

Usage:
    .venv/bin/python cerebras_race_optimizer.py --metric tok/s --budget-sec 120
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

# Optional uvloop for faster event loop on Linux
try:
    import uvloop
    uvloop.install()
except Exception:
    pass

API_KEY = os.environ.get("CEREBRAS_API_KEY")
MODEL = "gemma-4-31b"
BASE_URL = "https://api.cerebras.ai/v1"

# Cerebras published limits (competition ceiling)
REQ_PER_MIN_LIMIT = 1000
TOK_PER_MIN_LIMIT = 1_000_000


@dataclass
class BenchmarkConfig:
    name: str
    concurrency: int
    max_completion_tokens: int
    reasoning_effort: str = "low"
    stream: bool = False
    temperature: float = 0.3
    top_p: float = 1.0
    http2: bool = True


@dataclass
class RequestResult:
    latency: float
    tokens: int
    prompt_tokens: int
    total_tokens: int
    error: str | None = None
    timestamp: float = field(default_factory=time.perf_counter)
    status_code: int | None = None


@dataclass
class RunSummary:
    config: BenchmarkConfig
    num_requests: int
    successful: int
    failed: int
    batch_time: float
    req_per_sec: float
    tok_per_sec: float
    total_tokens: int
    avg_latency: float
    p50_latency: float
    p99_latency: float
    max_latency: float
    min_latency: float
    errors: dict[str, int] = field(default_factory=dict)

    def score(self, metric: str) -> float:
        if metric == "tok/s":
            return self.tok_per_sec
        if metric == "req/s":
            return self.req_per_sec
        if metric == "balanced":
            # Normalize both vs theoretical ceiling and sum
            r_score = self.req_per_sec / (REQ_PER_MIN_LIMIT / 60)
            t_score = self.tok_per_sec / (TOK_PER_MIN_LIMIT / 60)
            return r_score + t_score
        raise ValueError(f"Unknown metric: {metric}")


def long_prompt(topic: str, words: int) -> str:
    """Prompt engineered to elicit ~words tokens of output."""
    return (
        f"Write a detailed, continuous technical explanation of '{topic}' "
        f"that is at least {words} words long. Do not use lists or headers; "
        f"use flowing paragraphs only. Start immediately with the content."
    )


TOPICS = [
    "quantum computing",
    "neural network backpropagation",
    "asyncio event loops",
    "container orchestration",
    "distributed consensus protocols",
    "HTTP/2 multiplexing",
    "transformer architecture",
    "vector databases",
    "GPU memory hierarchy",
    "reinforcement learning",
]


class CerebrasRacer:
    def __init__(
        self,
        api_key: str = API_KEY,
        model: str = MODEL,
        base_url: str = BASE_URL,
        metric: str = "tok/s",
        verbose: bool = True,
    ):
        if not api_key:
            raise RuntimeError(
                "CEREBRAS_API_KEY not set. Export it before running this script."
            )
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.metric = metric
        self.verbose = verbose
        self.client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        limits = httpx.Limits(
            max_keepalive_connections=100,
            max_connections=200,
            keepalive_expiry=120.0,
        )
        self.client = httpx.AsyncClient(
            http2=True,
            timeout=httpx.Timeout(120.0, connect=10.0),
            limits=limits,
        )
        # Pre-warm connection
        try:
            await self.client.get(f"{self.base_url}/models")
        except Exception:
            pass
        return self

    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()

    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    def _make_prompt(self, target_words: int) -> str:
        return long_prompt(random.choice(TOPICS), target_words)

    async def _send_one(
        self,
        config: BenchmarkConfig,
        prompt: str,
        attempt: int = 0,
    ) -> RequestResult:
        start = time.perf_counter()
        try:
            resp = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": config.max_completion_tokens,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "reasoning_effort": config.reasoning_effort,
                    "stream": config.stream,
                },
            )
            elapsed = time.perf_counter() - start
            data = resp.json()

            if "error" in data:
                err = data["error"]
                code = err.get("code") if isinstance(err, dict) else str(err)
                return RequestResult(
                    latency=elapsed,
                    tokens=0,
                    prompt_tokens=0,
                    total_tokens=0,
                    error=code or "unknown",
                    status_code=resp.status_code,
                )

            usage = data.get("usage", {})
            return RequestResult(
                latency=elapsed,
                tokens=usage.get("completion_tokens", 0),
                prompt_tokens=usage.get("prompt_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                status_code=resp.status_code,
            )

        except Exception as e:
            elapsed = time.perf_counter() - start
            return RequestResult(
                latency=elapsed,
                tokens=0,
                prompt_tokens=0,
                total_tokens=0,
                error=type(e).__name__,
            )

    async def preflight(self) -> dict[str, Any]:
        self._log("\n[SCOUT] Preflight — verifying API and token ratio...")
        config = BenchmarkConfig(
            name="preflight",
            concurrency=1,
            max_completion_tokens=200,
        )
        results = []
        for target in [50, 100, 200, 500]:
            prompt = self._make_prompt(target)
            r = await self._send_one(config, prompt)
            if r.error:
                self._log(f"  ERROR: {r.error}")
                raise RuntimeError(f"Preflight failed: {r.error}")
            results.append({"target_words": target, "requested": config.max_completion_tokens, "actual": r.tokens, "latency": r.latency})
            self._log(f"  target_words={target:4} requested={config.max_completion_tokens:4} actual={r.tokens:4} latency={r.latency:.3f}s")

        # Estimate actual/requested ratio
        ratios = [r["actual"] / r["requested"] for r in results]
        avg_ratio = statistics.mean(ratios)
        self._log(f"  Average actual/requested token ratio: {avg_ratio:.2f}")
        return {"token_ratio": avg_ratio, "samples": results}

    async def run_config(
        self,
        config: BenchmarkConfig,
        num_requests: int,
        progress_queue: asyncio.Queue | None = None,
    ) -> RunSummary:
        semaphore = asyncio.Semaphore(config.concurrency)
        results: list[RequestResult] = []

        async def send_one(idx: int):
            async with semaphore:
                prompt = self._make_prompt(config.max_completion_tokens)
                r = await self._send_one(config, prompt)
                results.append(r)
                if progress_queue:
                    await progress_queue.put({
                        "type": "request",
                        "config": config.name,
                        "idx": idx,
                        "latency": r.latency,
                        "tokens": r.tokens,
                        "error": r.error,
                    })
                return r

        batch_start = time.perf_counter()
        await asyncio.gather(*[send_one(i) for i in range(num_requests)])
        batch_time = time.perf_counter() - batch_start

        return self._summarize(config, results, num_requests, batch_time)

    def _summarize(
        self,
        config: BenchmarkConfig,
        results: list[RequestResult],
        num_requests: int,
        batch_time: float,
    ) -> RunSummary:
        successful = [r for r in results if r.error is None]
        failed = [r for r in results if r.error is not None]
        latencies = [r.latency for r in successful]

        errors: dict[str, int] = {}
        for r in failed:
            errors[r.error] = errors.get(r.error, 0) + 1

        total_tokens = sum(r.tokens for r in successful)
        summary = RunSummary(
            config=config,
            num_requests=num_requests,
            successful=len(successful),
            failed=len(failed),
            batch_time=batch_time,
            req_per_sec=len(successful) / batch_time if batch_time > 0 else 0,
            tok_per_sec=total_tokens / batch_time if batch_time > 0 else 0,
            total_tokens=total_tokens,
            avg_latency=statistics.mean(latencies) if latencies else 0,
            p50_latency=statistics.median(latencies) if latencies else 0,
            p99_latency=sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0,
            max_latency=max(latencies) if latencies else 0,
            min_latency=min(latencies) if latencies else 0,
            errors=errors,
        )
        return summary

    async def scout_sweep(
        self,
        configs: list[BenchmarkConfig],
        requests_per_config: int,
        progress_queue: asyncio.Queue | None = None,
    ) -> list[RunSummary]:
        self._log(f"\n[COMMANDER] Launching {len(configs)} scouts × {requests_per_config} requests...")
        # Run each scout as a separate task; they manage their own semaphores
        tasks = [self.run_config(c, requests_per_config, progress_queue) for c in configs]
        summaries = await asyncio.gather(*tasks)

        self._log("\n[SCOUT RESULTS]")
        self._log(f"{'Config':<20} {'C':>3} {'T':>5} {'Req/s':>8} {'Tok/s':>10} {'AvgLat':>8} {'Fail':>5}")
        for s in summaries:
            self._log(
                f"{s.config.name:<20} {s.config.concurrency:>3} {s.config.max_completion_tokens:>5} "
                f"{s.req_per_sec:>8.2f} {s.tok_per_sec:>10.1f} {s.avg_latency:>8.3f} {s.failed:>5}"
            )
        return summaries

    def find_best(self, summaries: list[RunSummary]) -> RunSummary:
        valid = [s for s in summaries if s.successful > 0]
        if not valid:
            raise RuntimeError("No successful runs in sweep")
        best = max(valid, key=lambda s: s.score(self.metric))
        self._log(f"\n[COMMANDER] Winning config for metric '{self.metric}': {best.config.name}")
        self._log(f"  concurrency={best.config.concurrency} max_tokens={best.config.max_completion_tokens}")
        self._log(f"  req/s={best.req_per_sec:.2f} tok/s={best.tok_per_sec:.1f} score={best.score(self.metric):.3f}")
        return best

    async def race(
        self,
        config: BenchmarkConfig,
        num_requests: int,
        progress_queue: asyncio.Queue | None = None,
    ) -> RunSummary:
        self._log(f"\n[RACE] Sustained run: {num_requests} requests with {config.name}...")
        summary = await self.run_config(config, num_requests, progress_queue)
        self._log(f"\n[RACE RESULT]")
        self._log(f"  Successful: {summary.successful}/{summary.num_requests}")
        self._log(f"  Batch time: {summary.batch_time:.2f}s")
        self._log(f"  Requests/s: {summary.req_per_sec:.2f}")
        self._log(f"  Tokens/s:   {summary.tok_per_sec:.1f}")
        self._log(f"  Avg latency:{summary.avg_latency:.3f}s")
        return summary

    def save_results(
        self,
        summaries: list[RunSummary],
        best: RunSummary,
        race: RunSummary | None,
        out_dir: Path,
    ):
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON
        json_path = out_dir / f"race_{stamp}.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    "metric": self.metric,
                    "model": self.model,
                    "timestamp": datetime.now().isoformat(),
                    "sweep": [asdict(s) for s in summaries],
                    "best_config": asdict(best),
                    "race": asdict(race) if race else None,
                },
                f,
                indent=2,
                default=str,
            )

        # CSV
        csv_path = out_dir / f"race_{stamp}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "name", "concurrency", "max_tokens", "successful", "failed",
                "req_per_sec", "tok_per_sec", "avg_latency", "p99_latency",
            ])
            for s in summaries:
                writer.writerow([
                    s.config.name,
                    s.config.concurrency,
                    s.config.max_completion_tokens,
                    s.successful,
                    s.failed,
                    f"{s.req_per_sec:.3f}",
                    f"{s.tok_per_sec:.1f}",
                    f"{s.avg_latency:.3f}",
                    f"{s.p99_latency:.3f}",
                ])

        self._log(f"\n[SAVE] Results written to {out_dir}")


async def main():
    parser = argparse.ArgumentParser(description="Cerebras Racing Optimizer")
    parser.add_argument("--metric", choices=["tok/s", "req/s", "balanced"], default="tok/s")
    parser.add_argument("--sweep-requests", type=int, default=20, help="Requests per scout config")
    parser.add_argument("--race-requests", type=int, default=100, help="Requests in final race")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[8, 16, 24, 32], help="Concurrency levels to sweep")
    parser.add_argument("--max-tokens", nargs="+", type=int, default=[500, 1000, 1500], help="Max tokens to sweep")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    configs = []
    for c in args.concurrency:
        for t in args.max_tokens:
            configs.append(BenchmarkConfig(
                name=f"c{c}_t{t}",
                concurrency=c,
                max_completion_tokens=t,
            ))

    async with CerebrasRacer(metric=args.metric) as racer:
        if not args.skip_preflight:
            await racer.preflight()

        summaries = await racer.scout_sweep(configs, args.sweep_requests)
        best = racer.find_best(summaries)

        race_summary = await racer.race(best.config, args.race_requests)
        racer.save_results(summaries, best, race_summary, args.out_dir)


if __name__ == "__main__":
    asyncio.run(main())
