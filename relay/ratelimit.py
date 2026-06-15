from __future__ import annotations

import threading
import time


class TokenBucket:
    """Classic token bucket. tokens here are request permits, not LLM tokens.
    The bucket refills continuously at refill_per_sec up to capacity."""

    def __init__(self, capacity: float, refill_per_sec: float):
        self.capacity = capacity
        self.refill_per_sec = refill_per_sec
        self.tokens = capacity
        self.updated = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self, now: float) -> None:
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.refill_per_sec)
        self.updated = now

    def allow(self, cost: float = 1.0) -> bool:
        with self._lock:
            self._refill(time.monotonic())
            if self.tokens >= cost:
                self.tokens -= cost
                return True
            return False

    def retry_after(self, cost: float = 1.0) -> float:
        with self._lock:
            self._refill(time.monotonic())
            deficit = max(0.0, cost - self.tokens)
            return deficit / self.refill_per_sec if self.refill_per_sec > 0 else 0.0


class RateLimiter:
    """Holds one bucket per key. Keys are usually "route:apikey" so each caller
    gets its own allowance on each route."""

    def __init__(self):
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def _bucket(self, key: str, capacity: float, refill_per_sec: float) -> TokenBucket:
        with self._lock:
            b = self._buckets.get(key)
            if b is None:
                b = TokenBucket(capacity, refill_per_sec)
                self._buckets[key] = b
            return b

    def check(self, key: str, capacity: float, refill_per_sec: float, cost: float = 1.0):
        b = self._bucket(key, capacity, refill_per_sec)
        if b.allow(cost):
            return True, 0.0
        return False, b.retry_after(cost)
