from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from .config import PriceConfig
from .gateway import RequestContext


def _percentiles(samples) -> dict:
    if not samples:
        return {"p50": None, "p95": None, "count": 0}
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "count": int(arr.size),
    }


class Metrics:
    """In-memory counters and latency samples. Latency is kept in bounded
    deques so percentiles reflect recent traffic without growing forever.
    Cost is modeled from a per-model price table; with no real keys these are
    reference prices, not billed amounts."""

    def __init__(self, prices: dict[str, PriceConfig], window: int = 5000):
        self.prices = prices
        self.start = time.time()
        self._lock = threading.Lock()

        self.counters = {"total": 0, "exact": 0, "semantic": 0, "miss": 0, "error": 0, "rate_limited": 0}
        self.tokens_in = 0
        self.tokens_out = 0
        self.tokens_saved_in = 0
        self.tokens_saved_out = 0
        self.cost_spent = 0.0
        self.cost_saved = 0.0

        self.lat_all = deque(maxlen=window)
        self.lat_hit = deque(maxlen=window)
        self.lat_miss = deque(maxlen=window)

    def reset(self) -> None:
        with self._lock:
            self.start = time.time()
            for k in self.counters:
                self.counters[k] = 0
            self.tokens_in = self.tokens_out = 0
            self.tokens_saved_in = self.tokens_saved_out = 0
            self.cost_spent = self.cost_saved = 0.0
            self.lat_all.clear()
            self.lat_hit.clear()
            self.lat_miss.clear()

    def _cost(self, model: str, p_in: int, p_out: int) -> float:
        price = self.prices.get(model) or self.prices.get("mock")
        if price is None:
            return 0.0
        return (p_in * price.input + p_out * price.output) / 1_000_000.0

    def record_request(self, ctx: RequestContext, latency_ms: float) -> None:
        cost = self._cost(ctx.model, ctx.prompt_tokens, ctx.completion_tokens)
        with self._lock:
            self.counters["total"] += 1
            self.counters[ctx.cache] += 1
            self.lat_all.append(latency_ms)
            if ctx.cache == "miss":
                self.tokens_in += ctx.prompt_tokens
                self.tokens_out += ctx.completion_tokens
                self.cost_spent += cost
                self.lat_miss.append(latency_ms)
            else:
                self.tokens_saved_in += ctx.prompt_tokens
                self.tokens_saved_out += ctx.completion_tokens
                self.cost_saved += cost
                self.lat_hit.append(latency_ms)

    def record_rate_limited(self) -> None:
        with self._lock:
            self.counters["total"] += 1
            self.counters["rate_limited"] += 1

    def record_error(self, latency_ms: float) -> None:
        with self._lock:
            self.counters["total"] += 1
            self.counters["error"] += 1
            self.lat_all.append(latency_ms)

    def snapshot(self) -> dict:
        with self._lock:
            c = dict(self.counters)
            served = c["exact"] + c["semantic"] + c["miss"]
            hits = c["exact"] + c["semantic"]
            tokens_saved = self.tokens_saved_in + self.tokens_saved_out
            saved_per_mtok = (
                self.cost_saved / (tokens_saved / 1_000_000.0) if tokens_saved else 0.0
            )
            return {
                "uptime_s": time.time() - self.start,
                "counters": c,
                "served": served,
                "hit_rate": hits / served if served else 0.0,
                "exact_rate": c["exact"] / served if served else 0.0,
                "semantic_rate": c["semantic"] / served if served else 0.0,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "tokens_saved": tokens_saved,
                "cost_spent": self.cost_spent,
                "cost_saved": self.cost_saved,
                "cost_saved_per_mtok": saved_per_mtok,
                "latency_ms": {
                    "all": _percentiles(self.lat_all),
                    "hit": _percentiles(self.lat_hit),
                    "miss": _percentiles(self.lat_miss),
                },
            }
