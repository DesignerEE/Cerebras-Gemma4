#!/usr/bin/env python3
"""
Cerebras Race Advanced — quota-saturation benchmark using the dual-rate-limited client.

Features:
  - Dual-constraint sliding-window rate limiter
  - Exact-match prompt cache
  - Compression for large payloads
  - Exponential backoff + circuit breaker
  - Controlled concurrency

Usage:
    .venv/bin/python cerebras_race_advanced.py --race-requests 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from cerebras_race_client import CerebrasRaceClient, CompletionResult


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


@dataclass
class RunResult:
    concurrency: int
    max_completion_tokens: int
    num_requests: int
    batch_time: float
    req_per_sec: float
    tok_per_sec: float
    avg_latency: float
    p99_latency: float
    max_latency: float
    cache_hits: int
    compressed: int
    errors: int
    retries: int


def summarize_results(
    concurrency: int,
    max_tokens: int,
    results: list[CompletionResult],
    batch_time: float,
) -> RunResult:
    latencies = [r.latency for r in results]
    latencies.sort()
    n = len(latencies)
    total_tok = sum(r.completion_tokens for r in results)
    return RunResult(
        concurrency=concurrency,
        max_completion_tokens=max_tokens,
        num_requests=len(results),
        batch_time=batch_time,
        req_per_sec=len(results) / batch_time if batch_time > 0 else 0,
        tok_per_sec=total_tok / batch_time if batch_time > 0 else 0,
        avg_latency=statistics.mean(latencies) if latencies else 0,
        p99_latency=latencies[int(n * 0.99)] if n > 1 else (latencies[0] if latencies else 0),
        max_latency=max(latencies) if latencies else 0,
        cache_hits=sum(1 for r in results if r.cached),
        compressed=sum(1 for r in results if r.compressed),
        errors=sum(1 for r in results if r.total_tokens == 0 and not r.cached),
        retries=0,  # tracked globally on client
    )


async def sweep(
    client: CerebrasRaceClient,
    concurrency_levels: list[int],
    token_levels: list[int],
    requests_per_config: int,
) -> list[RunResult]:
    results = []
    print("\n[ADVANCED SWEEP]")
    print(f"{'Config':<12} {'C':>3} {'T':>6} {'Req/s':>8} {'Tok/s':>10} {'AvgLat':>8} {'Cache':>6} {'Comp':>5}")

    for c in concurrency_levels:
        for t in token_levels:
            prompts = [long_prompt(random.choice(TOPICS), t) for _ in range(requests_per_config)]
            # Reconfigure client concurrency for this scout
            client.concurrency = c
            client.semaphore = asyncio.Semaphore(c)

            start = time.perf_counter()
            completions = await client.bulk_complete(prompts, max_completion_tokens=t)
            batch_time = time.perf_counter() - start

            run = summarize_results(c, t, completions, batch_time)
            results.append(run)
            print(
                f"{'c'+str(c)+'_t'+str(t):<12} {c:>3} {t:>6} "
                f"{run.req_per_sec:>8.2f} {run.tok_per_sec:>10.1f} "
                f"{run.avg_latency:>8.3f} {run.cache_hits:>6} {run.compressed:>5}"
            )
    return results


async def race(
    client: CerebrasRaceClient,
    concurrency: int,
    max_tokens: int,
    num_requests: int,
) -> RunResult:
    client.concurrency = concurrency
    client.semaphore = asyncio.Semaphore(concurrency)
    prompts = [long_prompt(random.choice(TOPICS), max_tokens) for _ in range(num_requests)]

    print(f"\n[ADVANCED RACE] concurrency={concurrency} max_tokens={max_tokens} requests={num_requests}")
    start = time.perf_counter()
    completions = await client.bulk_complete(prompts, max_completion_tokens=max_tokens)
    batch_time = time.perf_counter() - start

    run = summarize_results(concurrency, max_tokens, completions, batch_time)
    print(f"  Batch time: {run.batch_time:.2f}s")
    print(f"  Requests/s: {run.req_per_sec:.2f}")
    print(f"  Tokens/s:   {run.tok_per_sec:.1f}")
    print(f"  Avg latency:{run.avg_latency:.3f}s")
    print(f"  Cache hits: {run.cache_hits}")
    print(f"  Compressed: {run.compressed}")
    return run


async def main():
    parser = argparse.ArgumentParser(description="Cerebras Advanced Race Benchmark")
    parser.add_argument("--metric", choices=["tok/s", "req/s", "balanced"], default="tok/s")
    parser.add_argument("--sweep-requests", type=int, default=12)
    parser.add_argument("--race-requests", type=int, default=100)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[8, 16, 24, 32])
    parser.add_argument("--max-tokens", nargs="+", type=int, default=[500, 1000, 1500])
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--req-per-min", type=float, default=1000)
    parser.add_argument("--tok-per-min", type=float, default=1_000_000)
    args = parser.parse_args()

    async with CerebrasRaceClient(
        concurrency=8,  # default, overridden per-scout
        req_per_min=args.req_per_min,
        tok_per_min=args.tok_per_min,
        enable_cache=False,  # cache would pollute throughput measurements
        enable_compression=True,
    ) as client:
        results = await sweep(client, args.concurrency, args.max_tokens, args.sweep_requests)

        def score(r: RunResult) -> float:
            if args.metric == "tok/s":
                return r.tok_per_sec
            if args.metric == "req/s":
                return r.req_per_sec
            r_score = r.req_per_sec / (args.req_per_min / 60)
            t_score = r.tok_per_sec / (args.tok_per_min / 60)
            return r_score + t_score

        best = max(results, key=score)
        print(f"\n[BEST CONFIG] concurrency={best.concurrency} max_tokens={best.max_completion_tokens}")
        print(f"  {args.metric}={score(best):.2f} (req/s={best.req_per_sec:.2f}, tok/s={best.tok_per_sec:.1f})")

        race_result = await race(client, best.concurrency, best.max_completion_tokens, args.race_requests)

        args.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = args.out_dir / f"advanced_race_{stamp}.json"
        with open(path, "w") as f:
            json.dump({
                "metric": args.metric,
                "sweep": [asdict(r) for r in results],
                "best": asdict(best),
                "race": asdict(race_result),
                "client_metrics": client.report(),
            }, f, indent=2)
        print(f"\n[SAVE] {path}")


if __name__ == "__main__":
    asyncio.run(main())
