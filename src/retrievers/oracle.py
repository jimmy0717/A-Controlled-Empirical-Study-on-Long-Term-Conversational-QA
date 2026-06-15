"""Oracle retriever: cheats by reading LongMemEval's gold ``has_answer``
flag on each turn. Returns sentences that the dataset annotators marked
as containing the answer. Provides a *retrieval-perfect* upper bound
against which BM25 / Dense / Mem0-Lite can be compared.
"""
from __future__ import annotations

import time
from typing import List

from ..data.loader import Sample
from .base import BaseRetriever, MemoryItem


class OracleRetriever(BaseRetriever):
    """Returns the gold-supporting turns directly.

    LongMemEval marks turns that contain the answer with
    ``has_answer=True``. We surface those as the only memories;
    if none is marked (a few abstention questions), we fall back to
    returning empty so the model has to reply 'I don't know'.
    """
    name = "oracle"

    def __init__(self) -> None:
        self.items: List[MemoryItem] = []
        self.reset_stats()

    def index(self, sample: Sample) -> None:
        t0 = time.perf_counter()
        items: List[MemoryItem] = []

        # Primary: turn-level has_answer flags (present on the gold turns).
        for sess in sample.sessions:
            for turn in sess.turns:
                if turn.has_answer:
                    items.append(MemoryItem(
                        text=f"[{sess.session_id} | {sess.date} | {turn.role}] {turn.content}",
                        score=1.0,
                        meta={"session_id": sess.session_id, "date": sess.date},
                    ))

        # Fallback: if no turn is flagged (some items only ship
        # answer_session_ids), surface the whole gold session(s).
        if not items and sample.answer_session_ids:
            gold = set(sample.answer_session_ids)
            for sess in sample.sessions:
                if sess.session_id in gold:
                    items.append(MemoryItem(
                        text=f"### Session {sess.session_id} ({sess.date})\n{sess.to_text()}",
                        score=1.0,
                        meta={"session_id": sess.session_id, "date": sess.date},
                    ))

        self.items = items
        self.stats["wallclock_index"] += time.perf_counter() - t0

    def search(self, query: str, top_k: int) -> List[MemoryItem]:
        t0 = time.perf_counter()
        out = self.items[:top_k] if top_k > 0 else self.items
        self.stats["wallclock_search"] += time.perf_counter() - t0
        self.stats["n_calls"] += 1
        return out
