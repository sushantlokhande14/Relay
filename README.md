# Relay

Relay is a small gateway that sits in front of LLM providers and adds the things
a raw provider call doesn't give you: streaming, a cache that serves
near-duplicate prompts without paying for them again, failover between providers,
rate limiting, and a live page showing cost and latency.

The part I actually wanted to build is the semantic cache. It embeds the
incoming prompt, searches for the nearest prompt it has already answered, and if
that one is close enough it returns the stored response instead of calling a
provider. The nearest-neighbor search runs on proxima, an HNSW index I wrote in
C++. Relay is partly an excuse to run something real on top of it.

Everything runs locally. There's a mock provider so you can exercise the whole
thing with no API keys and no spend, which is also how the numbers below were
measured.

## Why

Calling a provider directly is one function call. Putting that call in front of
traffic is where the gaps show up. You pay for every request even when it's a
near-duplicate of one you answered a minute ago. One provider having a bad five
minutes takes you down with it. Nothing caps a runaway client. And you can't see
what's slow or what it costs. Relay is the layer that handles those, the same way
a service ends up behind a reverse proxy, except the concerns here are
LLM-shaped (tokens, dollars, semantic similarity).

## What's in it

- An OpenAI-compatible `POST /v1/chat/completions`, streaming (SSE) or not.
- Three providers behind one interface: OpenAI, Anthropic, and a local mock.
- An exact cache (hash of the normalized prompt and params) checked first.
- A semantic cache on proxima checked second.
- Failover: if a provider errors or stalls before the first token, try the next.
- Token-bucket rate limiting per API key and route.
- A metrics page with hit rate, p50/p95 latency, tokens, and modeled cost saved.
- A load test that produces the numbers in this README.

## How it works

A request flows through the gateway like this:

```
client ──POST /v1/chat/completions──▶ gateway
   auth + rate limit (token bucket per key/route)   ─▶ 429 if over
   normalize messages+params, sha256                ─▶ exact cache hit? serve it
   embed prompt (MiniLM, 384-d)                     ─▶ proxima nearest neighbor
       similarity ≥ threshold and same route?       ─▶ semantic hit? serve it
   miss ─▶ provider chain with failover, stream out, then store the result
```

The exact cache is just a hash lookup. Messages are trimmed and the params that
change an answer (model, temperature, top_p, max_tokens) go into the hash, so an
identical request comes straight back from SQLite without touching a provider.

The semantic cache is the interesting one. proxima is in-memory and, importantly,
has no deletes and no in-place updates. So I don't treat it as the store. SQLite
is the source of truth and holds everything real: the prompt, its embedding, the
response, tokens, and timestamps. proxima is a rebuildable index over those
embeddings, where each vector's label is the SQLite row id. To look something up
I embed the prompt, ask proxima for the nearest label, load that row, and accept
it only if the cosine similarity clears the route's threshold and the route
matches. In cosine space proxima returns distance as `1 - similarity`, so a 0.90
threshold means accepting a distance up to 0.10.

Eviction works around the no-delete limitation by rebuilding. When the cache is
over its size cap I mark the least-recently-used rows dead in SQLite. The index
keeps serving as-is until enough rows are stale, then I build a fresh index from
the survivors and swap it in. Because the embeddings live in SQLite, a rebuild
never has to call the embedding model again. Inserts and the swap are guarded by
a single lock, since proxima is a single writer; searches don't take it.

Providers all implement one async `stream()` that yields text deltas plus a final
usage chunk, so the gateway never sees a provider-specific type. The failover
chain tries them in order. The catch worth calling out: once a provider has
streamed a token to the client there's no clean way to switch, so failover only
happens before the first token. After that, a mid-stream failure just ends there.

## The numbers

I ran the load test on this machine (Windows 11, Python 3.12) against the mock
provider so the run is free and reproducible. The mock simulates latency: 120 ms
to first token, then 8 ms per token for a 40-token reply, so a miss is roughly
450 ms of "provider" time. 4000 requests, 24 concurrent. The traffic mix is 45%
exact repeats, 35% paraphrases, 20% genuinely novel prompts. Same workload run
twice, once with the cache on and once with `RELAY_CACHE_DISABLED=1`.

| | no cache | with cache |
|---|---|---|
| p50 latency (client) | 759 ms | 44 ms |
| p95 latency (client) | 770 ms | 559 ms |
| throughput | 32 req/s | 167 req/s |
| cache hit rate | 0% | 78% (54% exact, 24% semantic) |
| provider spend, 4000 reqs | $0.263 | $0.062 |

A few honest notes on reading that table. The median drops a lot because most
requests become a ~10 ms cache read instead of a ~450 ms provider call. The p95
barely moves, and that's expected: about a fifth of requests still miss and pay
full provider latency, so the 95th-percentile request is usually a miss. The
cache helps the typical request, not the worst one. Spend tracks the hit rate
almost exactly, roughly a 76% reduction, which works out to about $1.33 saved per
million tokens served at the mock's reference prices.

The dollar figures are modeled, not billed. With no real keys, cost is the
measured token counts times a published per-token price table (in `config.yaml`),
not an invoice. The token counts, latencies, and hit rate are all really measured.

On the 0.90 similarity threshold: I didn't pick it to look good. With this
corpus, 31 of 48 paraphrases land above it against their canonical question, and
none of the 4000 novel prompts collide. Lower it and the semantic hit rate goes
up but you start serving stored answers to prompts that aren't really equivalent.
That tradeoff is the whole game with a semantic cache, and 0.90 is a conservative
spot for it.

## Running it

```
pip install -r requirements.txt
python -m relay
```

proxima isn't in `requirements.txt` because it's not a PyPI package; it's my own
engine, built and installed from its own repo. Check it's importable:

```
python -c "import proxima"
```

The first start downloads the MiniLM embedding model once (a few hundred MB) and
caches it. After that it runs offline. With no keys set, the default route uses
the mock provider, so you can hit it right away:

```
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{"model":"mock","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

The dashboard is at `http://127.0.0.1:8000/` and updates live over SSE, with
hit-rate, latency, throughput, and cache-mix charts drawn in plain SVG. There's
no chart library, so it works offline like the rest of the project. To use the
real providers, put keys in a `.env` (see `.env.example`) and add them to the
route's provider chain in `config.yaml`.

To reproduce the numbers, start the server and run:

```
python loadtest/bench.py --n 4000 --concurrency 24
```

then restart with `RELAY_CACHE_DISABLED=1` and run it again. There's also a
standard Locust file at `loadtest/locustfile.py` if you'd rather drive it that
way.

## Tests

Each file under `tests/` runs on its own with plain Python, no pytest needed:

```
python tests/test_cache.py        # exact cache, normalization
python tests/test_semantic.py     # semantic hit, eviction by rebuild
python tests/test_failover.py     # failover and timeouts
python tests/test_ratelimit.py    # token bucket
```

## What's missing or rough

- Cached responses come back in one chunk, not re-streamed token by token. The
  latency win is real either way, but a client expecting a token stream gets one
  big delta on a hit.
- Embedding happens on the request thread and blocks the event loop while it
  runs. It's fast on CPU for these short prompts, but under heavy load it would
  serialize. Moving it to a thread pool is the obvious next step.
- The cache doesn't account for temperature. If you're sampling at a high
  temperature and actually want different answers each time, a cache hit will
  hand back the same one. For that case you'd disable the cache on the route.
- Rate limiting and metrics are in-process, so they're per-instance. Running more
  than one gateway would need them moved to something shared like Redis.
- Failover can't recover after the first streamed token, as described above.
- The similarity threshold is global, set per route in config. Per-prompt or
  learned thresholds would do better but aren't here.
- SQLite is a single connection guarded by a lock. Fine for one node; it would be
  the first thing to feel pressure at higher write rates.
