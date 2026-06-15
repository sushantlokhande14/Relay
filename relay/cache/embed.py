from __future__ import annotations

import numpy as np


class Embedder:
    """Wraps a local fastembed model. all-MiniLM-L6-v2 gives 384-dim vectors,
    runs on CPU via onnxruntime, and needs no API key. The model downloads once
    on first use and is cached on disk after that.

    Behind a small interface so swapping in another model (or sentence-
    transformers) is a one-line change."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", dim: int = 384):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.dim = dim
        self._model = TextEmbedding(model_name=model_name)

    def embed_one(self, text: str) -> np.ndarray:
        vec = next(iter(self._model.embed([text])))
        return np.asarray(vec, dtype=np.float32)

    def embed_many(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self._model.embed(texts)), dtype=np.float32)
