"""Locust load test for Relay.

Start the gateway first:
    python -m relay

Then run, for example:
    locust -f loadtest/locustfile.py --headless -u 20 -r 10 -t 60s --host http://127.0.0.1:8000

Each user posts chat requests on the bench route (no rate limit) using a mix of
exact repeats, paraphrases, and unique prompts, so the cache gets exercised the
way it would under real near-duplicate traffic. Read the hit rate and cost saved
from http://127.0.0.1:8000/metrics.json while it runs."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from locust import HttpUser, between, task

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompts  # noqa: E402

P_EXACT = float(os.environ.get("RELAY_P_EXACT", "0.45"))
P_SEMANTIC = float(os.environ.get("RELAY_P_SEMANTIC", "0.35"))


class RelayUser(HttpUser):
    wait_time = between(0.0, 0.05)

    @task
    def chat(self):
        prompt, kind = prompts.pick(P_EXACT, P_SEMANTIC)
        body = {
            "model": "mock",
            "route": "bench",
            "messages": [{"role": "user", "content": prompt}],
        }
        with self.client.post(
            "/v1/chat/completions", json=body, name=f"chat[{kind}]", catch_response=True
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"status {resp.status_code}")
