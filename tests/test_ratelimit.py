"""Token bucket behaviour. Runs standalone: python tests/test_ratelimit.py"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from relay.ratelimit import RateLimiter, TokenBucket


def test_blocks_after_capacity():
    b = TokenBucket(capacity=3, refill_per_sec=0)
    assert b.allow() and b.allow() and b.allow()
    assert not b.allow()
    print("PASS: bucket allows up to capacity, then blocks")


def test_refills_over_time():
    b = TokenBucket(capacity=1, refill_per_sec=10)  # one permit every 0.1s
    assert b.allow()
    assert not b.allow()
    time.sleep(0.15)
    assert b.allow(), "bucket should have refilled within 0.15s"
    print("PASS: bucket refills over time")


def test_keys_are_isolated():
    rl = RateLimiter()
    assert rl.check("route:a", 1, 0)[0]
    assert not rl.check("route:a", 1, 0)[0]
    assert rl.check("route:b", 1, 0)[0], "a different key has its own bucket"
    print("PASS: each key gets its own bucket")


def main():
    test_blocks_after_capacity()
    test_refills_over_time()
    test_keys_are_isolated()
    print("all rate-limit tests passed")


if __name__ == "__main__":
    main()
