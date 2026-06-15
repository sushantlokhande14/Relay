from __future__ import annotations

import math
import threading
from pathlib import Path

import numpy as np
import proxima

from .store import Store


class SemanticCache:
    """A proxima HNSW index over the embeddings in the store. proxima can't
    delete or update in place, so eviction works by marking rows dead in the
    store and rebuilding a fresh index from the survivors. The store stays the
    source of truth; this is a rebuildable view over it.

    A single lock guards inserts and the rebuild swap (proxima is single
    writer). Searches run without the lock."""

    def __init__(
        self,
        store: Store,
        dim: int = 384,
        index_path: str = "data/cache.idx",
        max_entries: int = 5000,
        rebuild_when_stale_frac: float = 0.20,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        seed: int = 100,
        check_every: int = 64,
    ):
        self.store = store
        self.dim = dim
        self.index_path = index_path
        self.max_entries = max_entries
        self.rebuild_when_stale_frac = rebuild_when_stale_frac
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.seed = seed
        self.check_every = check_every

        self._lock = threading.Lock()
        self._index: proxima.Index | None = None
        self._count = 0
        self._adds_since_check = 0
        self.rebuilds = 0
        self._load_or_build()

    def _new_index(self) -> proxima.Index:
        return proxima.Index(
            dim=self.dim,
            space="cosine",
            M=self.m,
            ef_construction=self.ef_construction,
            seed=self.seed,
        )

    def _load_or_build(self) -> None:
        if Path(self.index_path).exists():
            try:
                self._index = proxima.load(self.index_path)
                self._count = len(self._index)
                return
            except Exception:
                pass  # fall through to a fresh build from the store
        self.rebuild()

    def rebuild(self) -> None:
        ids, mat = self.store.alive_embeddings()
        idx = self._new_index()
        if ids:
            idx.add(mat, labels=np.array(ids, dtype=np.int64))
        with self._lock:
            self._index = idx
            self._count = len(ids)
        self.rebuilds += 1

    def add(self, row_id: int, embedding: np.ndarray) -> None:
        vec = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        with self._lock:
            self._index.add(vec, labels=np.array([row_id], dtype=np.int64))
            self._count += 1
            self._adds_since_check += 1
        if self._adds_since_check >= self.check_every:
            self._adds_since_check = 0
            self._maybe_evict_and_rebuild()

    def search(self, embedding: np.ndarray, route: str, threshold: float):
        """Returns (row, similarity) for the nearest cached prompt if it clears
        the threshold and belongs to the same route, else (None, None)."""
        if self._count == 0:
            return None, None
        q = np.asarray(embedding, dtype=np.float32).reshape(-1)
        labels, dists = self._index.search(q, k=1, ef=self.ef_search)
        labels = np.atleast_2d(labels)
        dists = np.atleast_2d(dists)
        label = int(labels[0][0])
        dist = float(dists[0][0])
        if label < 0 or not math.isfinite(dist):
            return None, None
        similarity = 1.0 - dist
        if similarity < threshold:
            return None, None
        row = self.store.get_by_id(label)
        if row is None or row["route"] != route:
            return None, None
        return row, similarity

    def _maybe_evict_and_rebuild(self) -> None:
        counts = self.store.counts()
        if counts["alive"] > self.max_entries:
            victims = self.store.lru_candidates(counts["alive"] - self.max_entries)
            self.store.mark_dead(victims)
        counts = self.store.counts()
        indexable = max(1, counts["alive"] + counts["dead"])
        if counts["dead"] and counts["dead"] >= self.rebuild_when_stale_frac * indexable:
            self.rebuild()

    def save(self) -> None:
        with self._lock:
            if self._index is not None and self._count > 0:
                self._index.save(self.index_path)

    def stats(self) -> dict:
        return {"indexed": self._count, "rebuilds": self.rebuilds}
