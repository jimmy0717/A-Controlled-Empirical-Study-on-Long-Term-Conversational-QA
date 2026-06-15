"""Dense-Summarized retriever: a critical *isolation* baseline.

Mem0-Lite differs from Dense on **two** axes simultaneously:
  (a) construction: an LLM extracts memory items from raw turns;
  (b) granularity: items are roughly one-fact sentences, not 512-char chunks.

To disentangle these effects we add a third baseline that summarises
each session into K bullet sentences via the same generator LLM, then
indexes those bullets with the same dense encoder.  Comparing
Dense vs Dense-Summarized therefore isolates *granularity*; comparing
Dense-Summarized vs Mem0-Lite isolates *the four-op update agent*.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import List

import numpy as np

from ..data.loader import Sample, Session
from ..generator.llm import LLM
from ..prompting import fill_template
from .base import BaseRetriever, MemoryItem


SUMMARIZE_PROMPT = """You will be given a single conversation session between a USER and an
ASSISTANT. Summarise the *facts about the user and the world* that may
be useful in later sessions, as a JSON list of short third-person
sentences. Do NOT include greetings, acknowledgements or chit-chat.
Output strictly a JSON array of strings; no commentary.

==== SESSION ({date} | {sid}) ====
{session}

==== JSON OUTPUT ====
"""

_JSON_LIST_RE = re.compile(r"\[.*\]", re.DOTALL)


def _safe_list(s: str) -> List[str]:
    m = _JSON_LIST_RE.search(s)
    if not m:
        return []
    try:
        import json
        out = json.loads(m.group(0))
        return [x for x in out if isinstance(x, str) and x.strip()]
    except Exception:
        return []


class DenseSummarizedRetriever(BaseRetriever):
    name = "dense_summarized"

    def __init__(
        self,
        summariser: LLM,
        encoder: str = "BAAI/bge-m3",
        device: str = "cuda",
        batch_size: int = 64,
        max_facts_per_session: int = 12,
        reranker: "str | None" = None,
        rerank_top_n: int = 32,
    ):
        from sentence_transformers import SentenceTransformer

        self.llm = summariser
        self.model = SentenceTransformer(encoder, device=device)
        self.batch_size = batch_size
        self.max_facts = max_facts_per_session
        self.facts: List[str] = []
        self.emb: np.ndarray | None = None

        # Optional cross-encoder reranker, identical to DenseRetriever, so
        # that Dense vs Dense-Summarized isolates ONLY granularity /
        # construction (both share the same rerank stage) rather than
        # confounding it with the presence/absence of a reranker.
        self.reranker = None
        self.rerank_top_n = rerank_top_n
        if reranker:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(reranker, device=device)
        self.reset_stats()

    def _encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)

    def _summarise_session(self, sess: Session) -> List[str]:
        prompt = fill_template(
            SUMMARIZE_PROMPT,
            date=sess.date,
            sid=sess.session_id,
            session=sess.to_text(),
        )
        out = self.llm.chat(
            [{"role": "user", "content": prompt}],
            max_new_tokens=256,
            temperature=0.0,
        )
        facts = _safe_list(out)[: self.max_facts]
        return facts

    def index(self, sample: Sample) -> None:
        t0 = time.perf_counter()
        # All session summaries are independent -> issue them as a single batch.
        # Greedy decoding makes this identical to summarising one-by-one.
        batch = [
            [{"role": "user", "content": fill_template(
                SUMMARIZE_PROMPT,
                date=sess.date,
                sid=sess.session_id,
                session=sess.to_text(),
            )}]
            for sess in sample.sessions
        ]
        # 256 is ample for a JSON list of <=max_facts short sentences (observed
        # mean output ~181 tok); capping it halves the worst-case batched-decode
        # length, which is the dominant cost once prefill is batched.
        outs = self.llm.chat_batch(batch, max_new_tokens=256, temperature=0.0)
        all_facts: List[str] = []
        for sess, out in zip(sample.sessions, outs):
            for f in _safe_list(out)[: self.max_facts]:
                all_facts.append(f"({sess.date}) {f}")
        if not all_facts:
            # fallback: use raw last assistant turn so we never index empty
            all_facts = [sample.sessions[-1].turns[-1].content[:300]]
        self.facts = all_facts
        self.emb = self._encode(all_facts)
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

        # 2) optional cross-encoder rerank (same stage as DenseRetriever)
        if self.reranker is not None and len(cand) > 1:
            pairs = [(query, self.facts[i]) for i, _ in cand]
            rr_scores = self.reranker.predict(pairs, show_progress_bar=False)
            cand = sorted(zip([i for i, _ in cand], rr_scores),
                          key=lambda x: -float(x[1]))

        cand = cand[:top_k]
        out = [
            MemoryItem(text=self.facts[i], score=float(s), meta={"rank": r})
            for r, (i, s) in enumerate(cand)
        ]
        self.stats["wallclock_search"] += time.perf_counter() - t0
        self.stats["n_calls"] += 1
        return out
