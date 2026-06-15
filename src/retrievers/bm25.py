"""BM25 retriever (classical IR baseline)."""
from __future__ import annotations

import re
import time
from typing import List

from rank_bm25 import BM25Okapi

from ..data.loader import Sample, iter_chunks
from .base import BaseRetriever, MemoryItem

_TOK = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return _TOK.findall(text.lower())


class BM25Retriever(BaseRetriever):
    name = "bm25"

    def __init__(self, chunk_size: int = 512):
        self.chunk_size = chunk_size
        self.chunks: List[str] = []
        self.bm25: BM25Okapi | None = None
        self.reset_stats()

    def index(self, sample: Sample) -> None:
        t0 = time.perf_counter()
        self.chunks = list(iter_chunks(sample.haystack_text, self.chunk_size))
        tokenised = [_tokenize(c) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenised)
        self.stats["wallclock_index"] += time.perf_counter() - t0

    def search(self, query: str, top_k: int) -> List[MemoryItem]:
        assert self.bm25 is not None, "call index() before search()"
        t0 = time.perf_counter()
        scores = self.bm25.get_scores(_tokenize(query))
        order = scores.argsort()[::-1][:top_k]
        out = [
            MemoryItem(text=self.chunks[i], score=float(scores[i]), meta={"rank": int(r)})
            for r, i in enumerate(order)
        ]
        self.stats["wallclock_search"] += time.perf_counter() - t0
        self.stats["n_calls"] += 1
        return out
