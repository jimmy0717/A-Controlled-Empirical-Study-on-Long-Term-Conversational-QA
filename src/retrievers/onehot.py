"""One-hot (binary bag-of-words) retriever.

The weakest classical baseline, included for completeness because the
course explicitly lists One-hot / N-gram / TF-IDF as the classical
feature family. Each chunk and the query are represented as binary
term-presence vectors; relevance is cosine similarity. Together with
TF-IDF and BM25 this completes the IR progression
  One-hot -> N-gram (TF-IDF bigram) -> BM25 -> Dense.
"""
from __future__ import annotations

import time
from typing import List

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from ..data.loader import Sample, iter_chunks
from .base import BaseRetriever, MemoryItem


class OneHotRetriever(BaseRetriever):
    name = "onehot"

    def __init__(self, chunk_size: int = 512, ngram_range=(1, 1), max_features: int = 50000):
        self.chunk_size = chunk_size
        # binary=True turns counts into one-hot term-presence indicators
        self.vec = CountVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            lowercase=True,
            binary=True,
        )
        self.matrix = None          # sparse (n_chunks, V), binary
        self.norms: np.ndarray | None = None
        self.chunks: List[str] = []
        self.reset_stats()

    def index(self, sample: Sample) -> None:
        t0 = time.perf_counter()
        self.chunks = list(iter_chunks(sample.haystack_text, self.chunk_size))
        self.matrix = self.vec.fit_transform(self.chunks).astype(np.float32)
        # pre-compute L2 norms of each row for cosine
        sq = self.matrix.multiply(self.matrix).sum(axis=1)
        self.norms = np.sqrt(np.asarray(sq).ravel()) + 1e-8
        self.stats["wallclock_index"] += time.perf_counter() - t0

    def search(self, query: str, top_k: int) -> List[MemoryItem]:
        assert self.matrix is not None and self.norms is not None, "call index() first"
        t0 = time.perf_counter()
        q = self.vec.transform([query]).astype(np.float32)        # (1, V) binary
        qn = float(np.sqrt(q.multiply(q).sum())) + 1e-8
        dots = np.asarray((self.matrix @ q.T).todense()).ravel()  # (n_chunks,)
        scores = dots / (self.norms * qn)
        order = np.argsort(-scores)[:top_k]
        out = [
            MemoryItem(text=self.chunks[int(i)], score=float(scores[int(i)]), meta={"rank": int(r)})
            for r, i in enumerate(order)
        ]
        self.stats["wallclock_search"] += time.perf_counter() - t0
        self.stats["n_calls"] += 1
        return out
