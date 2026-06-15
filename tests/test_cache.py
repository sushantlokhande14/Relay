"""Exact cache behaviour. Runs standalone: python tests/test_cache.py"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from relay.cache.store import Store
from relay.config import load_settings
from relay.gateway import Gateway, RequestContext
from relay.providers import build_providers
from relay.schemas import ChatCompletionRequest


def make_gateway(db_path: str) -> Gateway:
    settings = load_settings()
    providers = build_providers(settings)
    return Gateway(settings, providers, Store(db_path))


async def run(gateway: Gateway, text: str):
    req = ChatCompletionRequest(model="mock", messages=[{"role": "user", "content": text}])
    ctx = RequestContext()
    content = ""
    async for chunk in gateway.stream(req, ctx):
        content += chunk.delta
    return ctx, content


async def main():
    with tempfile.TemporaryDirectory() as d:
        gw = make_gateway(str(Path(d) / "test.db"))

        ctx1, body1 = await run(gw, "what is the capital of France?")
        assert ctx1.cache == "miss", f"first call should miss, got {ctx1.cache}"
        assert body1, "expected a response body"
        print("PASS: first identical prompt is a miss")

        ctx2, body2 = await run(gw, "what is the capital of France?")
        assert ctx2.cache == "exact", f"repeat should be an exact hit, got {ctx2.cache}"
        assert body2 == body1, "cached response should match the original"
        print("PASS: repeat of the same prompt is an exact hit with the same body")

        ctx3, _ = await run(gw, "what is the capital of Spain?")
        assert ctx3.cache == "miss", f"different prompt should miss, got {ctx3.cache}"
        print("PASS: a different prompt misses")

        # whitespace-only difference normalizes to the same key
        ctx4, _ = await run(gw, "   what is the capital of France?  ")
        assert ctx4.cache == "exact", f"normalized whitespace should hit, got {ctx4.cache}"
        print("PASS: leading/trailing whitespace normalizes to the same key")

        gw.store.close()
        print("all exact-cache tests passed")


if __name__ == "__main__":
    asyncio.run(main())
