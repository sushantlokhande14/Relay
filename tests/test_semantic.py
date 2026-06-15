"""Semantic cache and eviction-by-rebuild. Runs standalone:
    python tests/test_semantic.py
Uses the real embedder, so the model must already be downloaded (it is after
the first server start or scripts/embed_check.py)."""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from relay.cache.embed import Embedder
from relay.cache.semantic import SemanticCache
from relay.cache.store import Store
from relay.config import load_settings
from relay.gateway import Gateway, RequestContext
from relay.providers import build_providers
from relay.schemas import ChatCompletionRequest

EMBEDDER = Embedder()


def make_gateway(d: str, **cache_kwargs) -> Gateway:
    settings = load_settings()
    providers = build_providers(settings)
    store = Store(str(Path(d) / "test.db"))
    semantic = SemanticCache(store, index_path=str(Path(d) / "test.idx"), **cache_kwargs)
    return Gateway(settings, providers, store, EMBEDDER, semantic)


async def run(gw: Gateway, text: str):
    req = ChatCompletionRequest(model="mock", messages=[{"role": "user", "content": text}])
    ctx = RequestContext()
    body = ""
    async for chunk in gw.stream(req, ctx):
        body += chunk.delta
    return ctx, body


async def test_semantic_hit():
    with tempfile.TemporaryDirectory() as d:
        gw = make_gateway(d)
        ctx1, body1 = await run(gw, "what is the capital of France?")
        assert ctx1.cache == "miss"
        ctx2, body2 = await run(gw, "what's the capital city of France?")
        assert ctx2.cache == "semantic", f"expected semantic hit, got {ctx2.cache}"
        assert body2 == body1, "semantic hit should return the stored response"
        assert ctx2.similarity and ctx2.similarity >= 0.90
        print(f"PASS: paraphrase is a semantic hit (similarity={ctx2.similarity:.4f})")

        ctx3, _ = await run(gw, "how do I bake sourdough bread?")
        assert ctx3.cache == "miss", "an unrelated prompt should not hit"
        print("PASS: an unrelated prompt misses")
        gw.store.close()


async def test_eviction_rebuild():
    prompts = [
        "what is the capital of France?",
        "how do I reverse a list in python?",
        "what is the speed of light?",
        "who wrote pride and prejudice?",
        "how does photosynthesis work?",
        "what is the boiling point of water?",
    ]
    with tempfile.TemporaryDirectory() as d:
        gw = make_gateway(d, max_entries=3, rebuild_when_stale_frac=0.2, check_every=1)
        for p in prompts:
            await run(gw, p)
        counts = gw.store.counts()
        assert counts["alive"] <= 3, f"alive should stay at the cap, got {counts}"
        assert gw.semantic.rebuilds > 0, "eviction should have triggered a rebuild"
        # the index still answers after rebuilds: the last prompt is alive, so a
        # paraphrase of it should still semantic-hit
        ctx, _ = await run(gw, "what's the boiling point of water?")
        assert ctx.cache == "semantic", f"surviving entry should still hit, got {ctx.cache}"
        print(
            f"PASS: eviction holds at cap (alive={counts['alive']}), "
            f"rebuilds={gw.semantic.rebuilds}, surviving entry still hits"
        )
        gw.store.close()


async def main():
    await test_semantic_hit()
    await test_eviction_rebuild()
    print("all semantic-cache tests passed")


if __name__ == "__main__":
    asyncio.run(main())
