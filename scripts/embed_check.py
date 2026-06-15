"""Exploratory: look at real cosine similarities from the embedder and confirm
proxima returns them. Used to pick a sane similarity threshold. Not a test."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import proxima

from relay.cache.embed import Embedder

pairs = [
    ("what is the capital of France?", "what's the capital city of France?"),
    ("how do I reverse a list in python?", "how can I reverse a list in python"),
    ("what is the capital of France?", "how do I bake sourdough bread?"),
    ("summarize the theory of relativity", "give me a summary of relativity theory"),
]

emb = Embedder()
print("model:", emb.model_name, "dim:", emb.dim)

texts = []
for a, b in pairs:
    texts.extend([a, b])
vecs = emb.embed_many(texts)
print("embedded shape:", vecs.shape)

for i, (a, b) in enumerate(pairs):
    va, vb = vecs[2 * i], vecs[2 * i + 1]
    cos = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
    print(f"cos={cos:.4f}  |  {a!r}  vs  {b!r}")

# round-trip through proxima: index the 'a' prompts, query with the 'b' prompts
idx = proxima.Index(dim=emb.dim, space="cosine", M=16, ef_construction=200, seed=100)
a_vecs = np.vstack([vecs[2 * i] for i in range(len(pairs))]).astype("float32")
idx.add(a_vecs, labels=np.arange(len(pairs), dtype=np.int64))
print("index len:", len(idx))
for i, (a, b) in enumerate(pairs):
    labels, dists = idx.search(vecs[2 * i + 1].astype("float32"), k=1, ef=64)
    labels = np.atleast_2d(labels)
    dists = np.atleast_2d(dists)
    sim = 1.0 - float(dists[0][0])
    print(f"query {i}: nearest label={int(labels[0][0])} similarity={sim:.4f}")
