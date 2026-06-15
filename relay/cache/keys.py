from __future__ import annotations

import hashlib
import json

from ..schemas import ChatCompletionRequest


def normalize_request(req: ChatCompletionRequest, route: str) -> str:
    """Canonical string of the parts of a request that change the answer. Two
    requests with the same canonical form should get the same response, so this
    is what the exact cache hashes."""
    payload = {
        "route": route,
        "model": req.model,
        "messages": [{"role": m.role, "content": m.content.strip()} for m in req.messages],
        "temperature": round(float(req.temperature), 4),
        "top_p": round(float(req.top_p), 4),
        "max_tokens": req.max_tokens,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def prompt_hash(normalized: str) -> str:
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def conversation_text(req: ChatCompletionRequest) -> str:
    """Flattened conversation, used as the stored prompt text and as the input
    to the embedding model for the semantic cache."""
    return "\n".join(f"{m.role}: {m.content}" for m in req.messages)


def params_json(req: ChatCompletionRequest) -> str:
    return json.dumps(
        {
            "model": req.model,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "max_tokens": req.max_tokens,
        },
        sort_keys=True,
    )
