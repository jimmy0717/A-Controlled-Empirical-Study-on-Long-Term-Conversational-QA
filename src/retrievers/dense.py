"""Dense retriever: BGE-M3 embeddings + optional cross-encoder reranker.

Design notes:
* L2-normalised embeddings → inner product == cosine similarity.
* Query and corpus encoded in batches (batched query encoding becomes
  important when the same retriever is reused across many questions).
* Optional reranker (e.g. ``BAAI/bge-reranker-base``) can be enabled
  through the YAML config; this is the *strong* dense baseline.
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from ..data.loader import Sample, iter_chunks
from .base import BaseRetriever, MemoryItem


class DenseRetriever(BaseRetriever):
    name = "dense"

    def __init__(
        self,
        encoder: str = "BAAI/bge-m3",
        device: str = "cuda",
        batch_size: int = 64,
        chunk_size: int = 512,
        reranker: Optional[str] = None,
        rerank_top_n: int = 32,
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(encoder, device=device)
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.device = device
        self.chunks: List[str] = []
        self.emb: np.ndarray | None = None

        self.reranker = None
        self.rerank_top_n = rerank_top_n
        if reranker:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(reranker, device=device)
        self.reset_stats()

    def _encode(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32, copy=False)

    def index(self, sample: Sample) -> None:
        t0 = time.perf_counter()
        self.chunks = list(iter_chunks(sample.haystack_text, self.chunk_size))
        self.emb = self._encode(self.chunks)
        self.stats["wallclock_index"] += time.perf_counter() - t0

    def search(self, query: str, top_k: int) -> List[MemoryItem]:
        assert self.emb is not None, "call index() before search()"
        t0 = time.perf_counter()
        q = self._encode([query])[0]
        scores = self.emb @ q

        # 1) shortlist with cosine
        n_short = max(top_k, self.rerank_top_n) if self.reranker else top_k
        order = np.argsort(-scores)[:n_short]
        cand = [(int(i), float(scores[int(i)])) for i in order]

        # 2) optional cross-encoder rerank
        if self.reranker is not None and len(cand) > 1:
            pairs = [(query, self.chunks[i]) for i, _ in cand]
            rr_scores = self.reranker.predict(pairs, show_progress_bar=False)
            cand = sorted(zip([i for i, _ in cand], rr_scores),
                          key=lambda x: -float(x[1]))

        cand = cand[:top_k]
        out = [
            MemoryItem(text=self.chunks[i], score=float(s), meta={"rank": r})
            for r, (i, s) in enumerate(cand)
        ]
        self.stats["wallclock_search"] += time.perf_counter() - t0
        self.stats["n_calls"] += 1
        return out
