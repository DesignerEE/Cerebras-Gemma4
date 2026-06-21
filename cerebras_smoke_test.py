#!/usr/bin/env python3
"""
Cerebras smoke test — small, fast verification of SDK + API connectivity.
Based on cerebras_max_speed.py but with minimal request count and token budget.
"""

import os
import time
import asyncio
import statistics
from datetime import datetime

API_KEY = os.environ.get("CEREBRAS_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "CEREBRAS_API_KEY not set. Export it before running this script."
    )
MODEL = "gpt-oss-120b"
BASE_URL = "https://api.cerebras.ai/v1"

OPTIMAL_CONFIG = {
    "max_completion_tokens": 100,
    "temperature": 0.3,
    "top_p": 1,
    "reasoning_effort": "low",
    "stream": False,
}

CONCURRENCY = 2
NUM_REQUESTS = 5

TEST_PROMPTS = [
    "Say hello world.",
    "What is 2+2?",
    "Name a color.",
    "Explain asyncio in one sentence.",
    "What is Python?",
]


def sync_smoke_test(n=2):
    from cerebras.cloud.sdk import Cerebras
    client = Cerebras(api_key=API_KEY)

    print("\n" + "=" * 50)
    print("  SYNC SMOKE TEST")
    print("=" * 50)

    times = []
    for i in range(n):
        start = time.perf_counter()
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": "Say hello world."}],
            model=MODEL,
            max_completion_tokens=50,
            temperature=0.3,
            reasoning_effort="low",
            stream=False,
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Req {i+1}: {elapsed:.3f}s | {resp.usage.completion_tokens} tokens")

    print(f"  Avg latency: {statistics.mean(times):.3f}s")


async def async_smoke_test():
    import httpx

    semaphore = asyncio.Semaphore(CONCURRENCY)
    results = []

    print("\n" + "=" * 50)
    print("  ASYNC SMOKE TEST")
    print("=" * 50)
    print(f"  Concurrency: {CONCURRENCY} | Requests: {NUM_REQUESTS}")

    async with httpx.AsyncClient(http2=True, timeout=60.0) as client:
        async def send_request(idx):
            async with semaphore:
                prompt = TEST_PROMPTS[idx % len(TEST_PROMPTS)]
                req_start = time.perf_counter()
                try:
                    resp = await client.post(
                        f"{BASE_URL}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": MODEL,
                            "messages": [{"role": "user", "content": prompt}],
                            **OPTIMAL_CONFIG,
                        },
                    )
                    data = resp.json()
                    if "error" in data:
                        return {"error": data["error"]}
                    tokens = data.get("usage", {}).get("completion_tokens", 0)
                except Exception as e:
                    return {"error": str(e)}
                elapsed = time.perf_counter() - req_start
                return {"latency": elapsed, "tokens": tokens}

        batch_start = time.perf_counter()
        responses = await asyncio.gather(*[send_request(i) for i in range(NUM_REQUESTS)])
        batch_end = time.perf_counter()

    successful = [r for r in responses if "error" not in r]
    failed = [r for r in responses if "error" in r]

    batch_time = batch_end - batch_start
    total_tokens = sum(r["tokens"] for r in successful)
    latencies = [r["latency"] for r in successful]

    print(f"  Successful: {len(successful)}/{NUM_REQUESTS}")
    if failed:
        print(f"  Failed: {len(failed)}")
        print(f"  First error: {failed[0]['error']}")

    print(f"  Batch time: {batch_time:.2f}s")
    print(f"  Requests/sec: {len(successful)/batch_time:.2f}" if successful else "  N/A")
    print(f"  Tokens/sec: {total_tokens/batch_time:.1f}" if successful else "  N/A")
    if latencies:
        print(f"  Avg latency: {statistics.mean(latencies):.3f}s")


async def main():
    print("╔" + "═" * 48 + "╗")
    print("║" + "  CEREBRAS SMOKE TEST".center(48) + "║")
    print("║" + f"  Started: {datetime.now().strftime('%H:%M:%S')}".center(48) + "║")
    print("╚" + "═" * 48 + "╝")

    sync_smoke_test(n=2)
    await async_smoke_test()

    print("\n" + "=" * 50)
    print("  ✅ SMOKE TEST COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
