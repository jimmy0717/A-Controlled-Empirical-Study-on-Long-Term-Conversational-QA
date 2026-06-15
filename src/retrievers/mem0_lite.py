"""Mem0-lite: a faithful but minimalist re-implementation of the
construction-side of Mem0 (Chhikara et al., 2025).

For each conversation we run a two-phase pipeline:

1. **Extraction** - per assistant/user message pair, ask an LLM to emit
   structured memory items (episodic / semantic / user_profile) as JSON.

2. **Update** - for each candidate memory, fetch the top-``k`` semantically
   similar memories already in the bank, then ask a second LLM call to commit
   exactly one of the four canonical operations
   ``ADD`` / ``UPDATE`` / ``DELETE`` / ``NOOP``.

Memories are stored as plain text along with their dense embedding and
metadata (type, timestamp, parent session, status).  Retrieval at QA time
is dense cosine over still-active memories, mirroring the production Mem0
service.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

from ..data.loader import Sample
from ..generator.llm import LLM
from ..prompting import fill_template
from .base import BaseRetriever, MemoryItem

EXTRACT_PROMPT = (
    Path(__file__).resolve().parents[2] / "prompts" / "extract.txt"
).read_text(encoding="utf-8")
UPDATE_PROMPT = (
    Path(__file__).resolve().parents[2] / "prompts" / "update.txt"
).read_text(encoding="utf-8")

# tolerant JSON extractor (LLMs often wrap output in ```json ... ```)
_JSON_LIST_RE = re.compile(r"\[.*\]", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"\{.*?\}", re.DOTALL)


@dataclass
class _Mem:
    mid: int
    text: str
    mtype: str            # episodic | semantic | user_profile
    ts: str = ""
    session_id: str = ""
    active: bool = True
    emb: np.ndarray | None = field(default=None, repr=False)


def _safe_json_list(s: str) -> list:
    m = _JSON_LIST_RE.search(s)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return []


def _safe_json_obj(s: str) -> dict:
    m = _JSON_OBJ_RE.search(s)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


class Mem0LiteRetriever(BaseRetriever):
    name = "mem0_lite"

    def __init__(
        self,
        extractor: LLM,
        updater: LLM,
        encoder: str = "BAAI/bge-m3",
        device: str = "cuda",
        similarity_threshold: float = 0.75,
        max_memories_per_pair: int = 8,
    ):
        from sentence_transformers import SentenceTransformer

        self.extractor = extractor
        self.updater = updater
        self.encoder = SentenceTransformer(encoder, device=device)
        self.threshold = similarity_threshold
        self.max_per_pair = max_memories_per_pair
        self.bank: List[_Mem] = []
        self.reset_stats()

    # ------------------------------------------------------------------
    def _embed(self, texts: List[str]) -> np.ndarray:
        return self.encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)

    def _topk_neighbors(self, query_emb: np.ndarray, k: int) -> List[_Mem]:
        active = [m for m in self.bank if m.active and m.emb is not None]
        if not active:
            return []
        E = np.stack([m.emb for m in active])
        scores = E @ query_emb
        order = np.argsort(-scores)[:k]
        return [active[int(i)] for i in order]

    # ------------------------ construction -----------------------------
    def _extract(self, summary: str, user_msg: str, asst_msg: str) -> List[dict]:
        prompt = fill_template(
            EXTRACT_PROMPT,
            summary=summary or "(empty)",
            user_msg=user_msg.strip(),
            assistant_msg=asst_msg.strip(),
        )
        out = self.extractor.chat(
            [{"role": "user", "content": prompt}],
            max_new_tokens=256,
            temperature=0.0,
        )
        cands = _safe_json_list(out)
        cleaned = []
        for c in cands[: self.max_per_pair]:
            if isinstance(c, dict) and isinstance(c.get("text"), str) and c["text"].strip():
                cleaned.append({
                    "text": c["text"].strip(),
                    "type": c.get("type", "semantic"),
                    "ts":   c.get("ts", ""),
                })
        return cleaned

    def _commit(self, candidate: dict, session_id: str) -> None:
        cand_emb = self._embed([candidate["text"]])[0]
        neighbors = self._topk_neighbors(cand_emb, k=5)
        existing_block = "\n".join(
            f"[{m.mid}] ({m.mtype}) {m.text}" for m in neighbors
        ) or "(empty)"

        decision_text = self.updater.chat(
            [{
                "role": "user",
                "content": fill_template(
                    UPDATE_PROMPT,
                    k=len(neighbors),
                    candidate=json.dumps(candidate, ensure_ascii=False),
                    existing=existing_block,
                ),
            }],
            max_new_tokens=128,
            temperature=0.0,
        )
        decision = _safe_json_obj(decision_text)
        op = (decision.get("op") or "ADD").upper()

        if op == "NOOP":
            return
        if op == "UPDATE":
            tid = decision.get("target_id")
            new_text = (decision.get("new_text") or candidate["text"]).strip()
            target = next((m for m in self.bank if m.mid == tid and m.active), None)
            if target is not None:
                target.text = new_text
                target.emb = self._embed([new_text])[0]
                return
            # fall through to ADD if target missing
        if op == "DELETE":
            tid = decision.get("target_id")
            target = next((m for m in self.bank if m.mid == tid and m.active), None)
            if target is not None:
                target.active = False
            return
        # default: ADD
        self.bank.append(
            _Mem(
                mid=len(self.bank),
                text=candidate["text"],
                mtype=candidate.get("type", "semantic"),
                ts=candidate.get("ts", ""),
                session_id=session_id,
                emb=cand_emb,
            )
        )

    # ----------------------------- API ---------------------------------
    def index(self, sample: Sample) -> None:
        """Run the full extract-and-update pipeline over one sample's
        conversation history. Memories are reset between samples so that
        each question is answered with a clean bank, mirroring how Mem0
        is benchmarked on LongMemEval."""
        t0 = time.perf_counter()
        self.bank = []

        # ---- phase 0: collect all (user, assistant) pairs with their rolling
        # summary. The rolling summary is a pure string op (last 400 chars), so
        # it does NOT depend on any LLM output and can be precomputed -> this lets
        # every extraction run in one batch.
        pairs: List = []  # (session_id, rolling_summary, user_msg, asst_msg)
        rolling_summary = ""
        for sess in sample.sessions:
            buffer: List = []
            for turn in sess.turns:
                buffer.append(turn)
                if len(buffer) >= 2 and buffer[-2].role == "user" and buffer[-1].role == "assistant":
                    user_msg, asst_msg = buffer[-2].content, buffer[-1].content
                    pairs.append((sess.session_id, rolling_summary, user_msg, asst_msg))
                    rolling_summary = (rolling_summary + " " + asst_msg)[-400:]

        # ---- phase 1: batched extraction (independent across pairs)
        extract_prompts = [
            [{"role": "user", "content": fill_template(
                EXTRACT_PROMPT,
                summary=rs or "(empty)",
                user_msg=u.strip(),
                assistant_msg=a.strip(),
            )}]
            for (_sid, rs, u, a) in pairs
        ]
        raw_extracts = (
            self.extractor.chat_batch(extract_prompts, max_new_tokens=256, temperature=0.0)
            if extract_prompts else []
        )

        # ---- phase 2: sequential commit (bank-dependent -> must stay ordered)
        for (sid, _rs, _u, _a), out in zip(pairs, raw_extracts):
            for c in _safe_json_list(out)[: self.max_per_pair]:
                if isinstance(c, dict) and isinstance(c.get("text"), str) and c["text"].strip():
                    self._commit(
                        {
                            "text": c["text"].strip(),
                            "type": c.get("type", "semantic"),
                            "ts": c.get("ts", ""),
                        },
                        session_id=sid,
                    )
        self.stats["wallclock_index"] += time.perf_counter() - t0

    def search(self, query: str, top_k: int) -> List[MemoryItem]:
        if not self.bank:
            return []
        t0 = time.perf_counter()
        q = self._embed([query])[0]
        active = [m for m in self.bank if m.active]
        E = np.stack([m.emb for m in active])
        scores = E @ q
        order = np.argsort(-scores)[:top_k]
        out = [
            MemoryItem(
                text=active[int(i)].text,
                score=float(scores[int(i)]),
                meta={
                    "type": active[int(i)].mtype,
                    "ts": active[int(i)].ts,
                    "session_id": active[int(i)].session_id,
                    "rank": int(r),
                },
            )
            for r, i in enumerate(order)
        ]
        self.stats["wallclock_search"] += time.perf_counter() - t0
        self.stats["n_calls"] += 1
        return out
