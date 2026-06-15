from __future__ import annotations

import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .cache.embed import Embedder
from .cache.semantic import SemanticCache
from .cache.store import Store
from .config import load_settings
from .gateway import Gateway, RequestContext
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

app = FastAPI(title="Relay", version="0.1.0")


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
            return JSONResponse(
                {"error": {"message": "rate limit exceeded", "type": "rate_limit"}},
                status_code=429,
                headers={"Retry-After": str(max(1, int(retry_after + 0.999)))},
            )

    cid = new_id()
    ctx = RequestContext()

    if req.stream:
        async def event_stream():
            async for chunk in gateway.stream(req, ctx):
                if chunk.delta:
                    payload = chunk_payload(cid, req.model, {"content": chunk.delta})
                    yield f"data: {json.dumps(payload)}\n\n"
            yield f"data: {json.dumps(chunk_payload(cid, req.model, {}, finish_reason='stop'))}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    content = ""
    usage = None
    async for chunk in gateway.stream(req, ctx):
        content += chunk.delta
        if chunk.usage:
            usage = chunk.usage
    payload = completion_payload(cid, req.model, content, usage.dict() if usage else {})
    headers = {"X-Relay-Cache": ctx.cache, "X-Relay-Provider": ctx.provider or ""}
    return JSONResponse(payload, headers=headers)
