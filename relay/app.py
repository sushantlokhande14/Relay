from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .cache.embed import Embedder
from .cache.semantic import SemanticCache
from .cache.store import Store
from .config import load_settings
from .gateway import Gateway, RequestContext
from .metrics import Metrics
from .providers import build_providers
from .ratelimit import RateLimiter
from .schemas import (
    ChatCompletionRequest,
    chunk_payload,
    completion_payload,
    new_id,
)

settings = load_settings()
providers = build_providers(settings)
store = Store(settings.cache.db_path)
embedder = Embedder(settings.cache.embedding_model, settings.cache.dim)
semantic = SemanticCache(
    store,
    dim=settings.cache.dim,
    index_path=settings.cache.index_path,
    max_entries=settings.cache.max_entries,
    rebuild_when_stale_frac=settings.cache.rebuild_when_stale_frac,
)
gateway = Gateway(settings, providers, store, embedder, semantic)
limiter = RateLimiter()
metrics = Metrics(settings.prices)

DASHBOARD = (Path(__file__).parent / "dashboard" / "page.html").read_text(encoding="utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # The store is already durable; saving the index just avoids a rebuild on
    # the next start (it can always be rebuilt from the stored embeddings).
    semantic.save()


app = FastAPI(title="Relay", version="0.1.0", lifespan=lifespan)


def _api_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or "anon"
    return request.headers.get("x-api-key") or "anon"


@app.get("/healthz")
async def healthz() -> dict:
    return {
        "status": "ok",
        "providers": list(providers),
        "routes": list(settings.routes),
        "cache": store.counts(),
        "semantic": semantic.stats(),
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    return DASHBOARD


@app.get("/metrics.json")
async def metrics_json() -> dict:
    return metrics.snapshot()


@app.post("/metrics/reset")
async def metrics_reset() -> dict:
    # Zero the counters so a benchmark can measure a clean window after warm-up.
    metrics.reset()
    return {"status": "reset"}


@app.get("/metrics/stream")
async def metrics_stream():
    async def gen():
        while True:
            yield f"data: {json.dumps(metrics.snapshot())}\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    route_name = req.route or settings.default_route
    route_cfg = settings.routes.get(route_name)
    if route_cfg is None:
        return JSONResponse(
            {"error": {"message": f"unknown route '{route_name}'", "type": "invalid_request"}},
            status_code=400,
        )

    if route_cfg.rate_limit is not None:
        rl = route_cfg.rate_limit
        key = f"{route_name}:{_api_key(request)}"
        allowed, retry_after = limiter.check(key, rl.capacity, rl.refill_per_sec)
        if not allowed:
            metrics.record_rate_limited()
            return JSONResponse(
                {"error": {"message": "rate limit exceeded", "type": "rate_limit"}},
                status_code=429,
                headers={"Retry-After": str(max(1, int(retry_after + 0.999)))},
            )

    cid = new_id()
    ctx = RequestContext()
    start = time.perf_counter()

    if req.stream:
        async def event_stream():
            try:
                async for chunk in gateway.stream(req, ctx):
                    if chunk.delta:
                        payload = chunk_payload(cid, req.model, {"content": chunk.delta})
                        yield f"data: {json.dumps(payload)}\n\n"
                yield f"data: {json.dumps(chunk_payload(cid, req.model, {}, finish_reason='stop'))}\n\n"
                yield "data: [DONE]\n\n"
                metrics.record_request(ctx, (time.perf_counter() - start) * 1000)
            except Exception as e:
                metrics.record_error((time.perf_counter() - start) * 1000)
                err = {"error": {"message": str(e), "type": "provider_error"}}
                yield f"data: {json.dumps(err)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    content = ""
    usage = None
    try:
        async for chunk in gateway.stream(req, ctx):
            content += chunk.delta
            if chunk.usage:
                usage = chunk.usage
    except Exception as e:
        metrics.record_error((time.perf_counter() - start) * 1000)
        return JSONResponse(
            {"error": {"message": str(e), "type": "provider_error"}}, status_code=502
        )

    metrics.record_request(ctx, (time.perf_counter() - start) * 1000)
    payload = completion_payload(cid, req.model, content, usage.dict() if usage else {})
    headers = {
        "X-Relay-Cache": ctx.cache,
        "X-Relay-Provider": ctx.provider or "",
        "X-Relay-Similarity": f"{ctx.similarity:.4f}" if ctx.similarity is not None else "",
    }
    return JSONResponse(payload, headers=headers)
