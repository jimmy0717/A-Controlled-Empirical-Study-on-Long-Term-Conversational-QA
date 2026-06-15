"""TF-IDF retriever.

Bridges the course content (TF-IDF / N-gram, cf.\ blackboard notes) and
the BM25 baseline. We use scikit-learn's TfidfVectorizer with bigram
n-grams and L2-normalised vectors so cosine similarity is an inner
product. This is intentionally a 'classical IR' baseline.
"""
from __future__ import annotations

import time
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..data.loader import Sample, iter_chunks
from .base import BaseRetriever, MemoryItem


class TfidfRetriever(BaseRetriever):
    name = "tfidf"

    def __init__(self, chunk_size: int = 512, ngram_range=(1, 2), max_features: int = 50000):
        self.chunk_size = chunk_size
        self.vec = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            lowercase=True,
            sublinear_tf=True,
            norm="l2",
        )
        self.matrix = None
        self.chunks: List[str] = []
        self.reset_stats()

    def index(self, sample: Sample) -> None:
        t0 = time.perf_counter()
        self.chunks = list(iter_chunks(sample.haystack_text, self.chunk_size))
        self.matrix = self.vec.fit_transform(self.chunks)
        self.stats["wallclock_index"] += time.perf_counter() - t0

    def search(self, query: str, top_k: int) -> List[MemoryItem]:
        assert self.matrix is not None, "call index() before search()"
        t0 = time.perf_counter()
        q = self.vec.transform([query])             # (1, V)
        scores = (self.matrix @ q.T).toarray().ravel()  # cosine via L2-norm
        order = np.argsort(-scores)[:top_k]
        out = [
            MemoryItem(text=self.chunks[int(i)], score=float(scores[int(i)]), meta={"rank": int(r)})
            for r, i in enumerate(order)
        ]
        self.stats["wallclock_search"] += time.perf_counter() - t0
        self.stats["n_calls"] += 1
        return out
