"""End-to-end check of the gateway through the real app, driven in-process with
httpx so it needs no running server. Run from the repo root:
    python scripts/smoke.py
Uses a random nonce in the prompts so the first call is always a fresh miss
regardless of what's already in the on-disk cache."""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from relay.app import app


async def chat(client, text, stream=False):
    body = {"model": "mock", "messages": [{"role": "user", "content": text}], "stream": stream}
    r = await client.post("/v1/chat/completions", json=body)
    return r


async def main() -> None:
    nonce = uuid.uuid4().hex[:8]
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as c:
        r = await c.get("/healthz")
        print("health:", r.json())

        q = f"[{nonce}] what is the capital of France?"
        r1 = await chat(c, q)
        print("first   :", r1.headers.get("x-relay-cache"), "| provider:", r1.headers.get("x-relay-provider"))

        r2 = await chat(c, q)
        print("repeat  :", r2.headers.get("x-relay-cache"))

        r3 = await chat(c, f"[{nonce}] what's the capital city of France?")
        print("paraphr :", r3.headers.get("x-relay-cache"), "| sim:", r3.headers.get("x-relay-similarity"))

        r4 = await chat(c, f"[{nonce}] how do I bake sourdough bread?")
        print("other   :", r4.headers.get("x-relay-cache"))

        # streaming still works
        body = {"model": "mock", "messages": [{"role": "user", "content": q}], "stream": True}
        lines = 0
        async with c.stream("POST", "/v1/chat/completions", json=body) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    lines += 1
        print("stream  : data lines =", lines)

        dash = await c.get("/")
        print("dashboard:", dash.status_code, dash.headers.get("content-type"))

        snap = (await c.get("/metrics.json")).json()
        print("metrics :", json.dumps({
            "counters": snap["counters"],
            "hit_rate": round(snap["hit_rate"], 3),
            "cost_saved": round(snap["cost_saved"], 6),
            "lat_miss_p95": snap["latency_ms"]["miss"]["p95"],
            "lat_hit_p95": snap["latency_ms"]["hit"]["p95"],
        }, indent=0))


if __name__ == "__main__":
    asyncio.run(main())
