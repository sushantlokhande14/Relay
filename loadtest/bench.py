"""Reproducible benchmark driver. Fires a fixed number of chat requests at a
running gateway with a mix of exact repeats, paraphrases, and unique prompts,
measures client-side latency, and reads the server's own metrics.

Start the gateway first (python -m relay), then:
    python loadtest/bench.py --n 4000 --concurrency 24

For the no-cache baseline, start the gateway with RELAY_CACHE_DISABLED=1 and run
the same command. Compare the two."""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompts  # noqa: E402


async def warm(client: httpx.AsyncClient) -> None:
    for q in prompts.all_canonicals():
        await client.post(
            "/v1/chat/completions",
            json={"model": "mock", "route": "bench", "messages": [{"role": "user", "content": q}]},
        )


async def one(client, sem, latencies, p_exact, p_semantic):
    prompt, _ = prompts.pick(p_exact, p_semantic)
    body = {"model": "mock", "route": "bench", "messages": [{"role": "user", "content": prompt}]}
    async with sem:
        start = time.perf_counter()
        r = await client.post("/v1/chat/completions", json=body)
        latencies.append((time.perf_counter() - start) * 1000)
        return r.status_code


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--p-exact", type=float, default=0.45)
    ap.add_argument("--p-semantic", type=float, default=0.35)
    ap.add_argument("--no-warm", action="store_true")
    args = ap.parse_args()

    async with httpx.AsyncClient(base_url=args.host, timeout=60) as client:
        health = (await client.get("/healthz")).json()
        cache_on = "default" in health.get("routes", [])
        if not args.no_warm:
            await warm(client)
        await client.post("/metrics/reset")

        sem = asyncio.Semaphore(args.concurrency)
        latencies: list[float] = []
        wall_start = time.perf_counter()
        tasks = [one(client, sem, latencies, args.p_exact, args.p_semantic) for _ in range(args.n)]
        codes = await asyncio.gather(*tasks)
        wall = time.perf_counter() - wall_start

        snap = (await client.get("/metrics.json")).json()

    arr = np.asarray(latencies)
    ok = sum(1 for c in codes if c == 200)
    print("=" * 60)
    print(f"requests: {args.n}  ok: {ok}  concurrency: {args.concurrency}")
    print(f"throughput: {args.n / wall:.0f} req/s over {wall:.1f}s")
    print(f"cache: {'ON' if snap['counters']['exact'] or snap['counters']['semantic'] else 'OFF / cold'}")
    print("-" * 60)
    print(f"hit rate:        {snap['hit_rate'] * 100:5.1f}%  "
          f"(exact {snap['exact_rate'] * 100:.1f}%, semantic {snap['semantic_rate'] * 100:.1f}%)")
    print(f"client latency:  p50 {np.percentile(arr, 50):7.1f} ms   p95 {np.percentile(arr, 95):7.1f} ms")
    sm = snap["latency_ms"]
    print(f"server all:      p50 {sm['all']['p50']:7.1f} ms   p95 {sm['all']['p95']:7.1f} ms")
    if sm["hit"]["p50"] is not None:
        print(f"server hit:      p50 {sm['hit']['p50']:7.1f} ms   p95 {sm['hit']['p95']:7.1f} ms")
    if sm["miss"]["p50"] is not None:
        print(f"server miss:     p50 {sm['miss']['p50']:7.1f} ms   p95 {sm['miss']['p95']:7.1f} ms")
    print("-" * 60)
    print(f"tokens saved:    {snap['tokens_saved']:,}")
    print(f"cost spent:      ${snap['cost_spent']:.6f}")
    print(f"cost saved:      ${snap['cost_saved']:.6f}  (${snap['cost_saved_per_mtok']:.2f} per 1M tokens served)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
