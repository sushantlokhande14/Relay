"""Sanity check the load-test corpus: confirm unique prompts stay below the
similarity threshold (so they really miss) and paraphrases mostly clear it."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "loadtest"))

import numpy as np
import proxima

import prompts
from relay.cache.embed import Embedder

THRESHOLD = 0.90
emb = Embedder()

# unique prompts: how many land within threshold of a previously seen unique?
uniques = [prompts.unique()[0] for _ in range(600)]
vecs = emb.embed_many(uniques).astype("float32")
idx = proxima.Index(dim=emb.dim, space="cosine", M=16, ef_construction=200, seed=1)
collisions = 0
worst = 0.0
for i, v in enumerate(vecs):
    if i:
        labels, dists = idx.search(v, k=1, ef=64)
        sim = 1.0 - float(np.atleast_2d(dists)[0][0])
        worst = max(worst, sim)
        if sim >= THRESHOLD:
            collisions += 1
    idx.add(v.reshape(1, -1), labels=np.array([i], dtype=np.int64))
print(f"unique prompts: {collisions}/600 within {THRESHOLD} of an earlier unique "
      f"(worst similarity seen: {worst:.3f})")

# paraphrases vs their canonical
hit = 0
total = 0
for t in prompts.TOPICS:
    cv = emb.embed_one(t["q"])
    for p in t["p"]:
        total += 1
        sim = float(np.dot(cv, emb.embed_one(p)) /
                    (np.linalg.norm(cv) * np.linalg.norm(emb.embed_one(p))))
        if sim >= THRESHOLD:
            hit += 1
print(f"paraphrases clearing {THRESHOLD} vs their canonical: {hit}/{total}")
