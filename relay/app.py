from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from .cache.store import Store
from .config import load_settings
from .gateway import Gateway, RequestContext
from .providers import build_providers
from .schemas import (
    ChatCompletionRequest,
    chunk_payload,
    completion_payload,
    new_id,
)

settings = load_settings()
providers = build_providers(settings)
store = Store(settings.cache.db_path)
gateway = Gateway(settings, providers, store)

app = FastAPI(title="Relay", version="0.1.0")


@app.get("/healthz")
async def healthz() -> dict:
    return {
        "status": "ok",
        "providers": list(providers),
        "routes": list(settings.routes),
        "cache": store.counts(),
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
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
