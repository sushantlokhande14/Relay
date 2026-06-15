"""Quick end-to-end check of the gateway, driven in-process with httpx so it
needs no running server. Run from the repo root: python scripts/smoke.py"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from relay.app import app


async def main() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/healthz")
        print("health:", r.json())

        body = {"model": "mock", "messages": [{"role": "user", "content": "hello relay"}]}
        r = await c.post("/v1/chat/completions", json=body)
        print("non-stream status:", r.status_code)
        j = r.json()
        print("non-stream content:", j["choices"][0]["message"]["content"][:120])
        print("non-stream usage:", j["usage"])

        body["stream"] = True
        data_lines: list[str] = []
        async with c.stream("POST", "/v1/chat/completions", json=body) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data_lines.append(line[6:])

        text = ""
        for ch in data_lines:
            if ch.strip() == "[DONE]":
                continue
            delta = json.loads(ch)["choices"][0]["delta"]
            text += delta.get("content", "")
        print("stream lines:", len(data_lines), "| ends with [DONE]:", data_lines[-1].strip() == "[DONE]")
        print("stream reconstructed:", text[:120])


if __name__ == "__main__":
    asyncio.run(main())
