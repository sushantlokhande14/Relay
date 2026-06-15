"""Capture a dashboard screenshot for the README.

Dev tool, not part of the app. Needs Playwright, which isn't in
requirements.txt:
    pip install playwright
    python -m playwright install chromium

Start the gateway first (python -m relay), then run this. It opens the
dashboard, drives some traffic so the time-series charts fill, and saves a PNG
to docs/dashboard.png."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
URL = "http://127.0.0.1:8000/"
OUT = ROOT / "docs" / "dashboard.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1320, "height": 1180}, device_scale_factor=1.5)
    page.goto(URL, wait_until="load")

    bench = subprocess.Popen(
        [sys.executable, str(ROOT / "loadtest" / "bench.py"),
         "--n", "8000", "--concurrency", "16", "--no-warm"],
        cwd=str(ROOT),
    )
    time.sleep(35)  # let the charts accumulate a real trace while traffic flows
    page.screenshot(path=str(OUT), full_page=True)
    print("saved", OUT)

    try:
        bench.terminate()
    except Exception:
        pass
    browser.close()
