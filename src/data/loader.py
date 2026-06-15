"""LongMemEval / LOCOMO data loader.

LongMemEval ships as a single JSON list of dicts, each with fields:
    {
        "question_id": str,
        "question_type": str,        # single-session-user, multi-session, temporal-reasoning, ...
        "question_date": str,
        "question": str,
        "answer": str,
        "haystack_session_ids": [str],
        "haystack_dates":       [str],
        "haystack_sessions":    [[{role, content, has_answer}, ...], ...]
    }

We provide a thin wrapper that yields a stable :class:`Sample` dataclass and
performs deterministic sub-sampling so experiments are reproducible.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, List


@dataclass
class Turn:
    role: str           # "user" | "assistant"
    content: str
    has_answer: bool = False


@dataclass
class Session:
    session_id: str
    date: str
    turns: List[Turn] = field(default_factory=list)

    def to_text(self) -> str:
        return "\n".join(f"[{t.role}] {t.content}" for t in self.turns)


@dataclass
class Sample:
    qid: str
    question: str
    answer: str
    qtype: str
    question_date: str
    sessions: List[Session]
    answer_session_ids: List[str] = field(default_factory=list)

    @property
    def haystack_text(self) -> str:
        return "\n\n".join(
            f"### Session {s.session_id} ({s.date})\n{s.to_text()}"
            for s in self.sessions
        )


def _build_sessions(item: dict) -> List[Session]:
    out: List[Session] = []
    for sid, date, turns in zip(
        item["haystack_session_ids"],
        item["haystack_dates"],
        item["haystack_sessions"],
    ):
        out.append(
            Session(
                session_id=sid,
                date=date,
                turns=[
                    Turn(
                        role=t["role"],
                        content=t["content"],
                        has_answer=t.get("has_answer", False),
                    )
                    for t in turns
                ],
            )
        )
    return out


def load_longmemeval(
    data_dir: str | Path,
    split: str = "longmemeval_s",
    num_questions: int = -1,
    seed: int = 42,
) -> List[Sample]:
    """Load the LongMemEval split JSON shipped at ``{data_dir}/{split}.json``.

    Args:
        data_dir: directory containing the LongMemEval JSON files
        split:    one of ``longmemeval_s`` / ``longmemeval_m`` / ``longmemeval_oracle``
        num_questions: subsample size; -1 keeps the full split
        seed:     RNG seed for reproducible subsampling
    """
    path = Path(data_dir) / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/download_data.sh first."
        )
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = [
        Sample(
            qid=item["question_id"],
            question=item["question"],
            answer=item["answer"],
            qtype=item["question_type"],
            question_date=item.get("question_date", ""),
            sessions=_build_sessions(item),
            answer_session_ids=item.get("answer_session_ids", []) or [],
        )
        for item in raw
    ]

    # Prefix-stable subsampling: shuffle once with a fixed seed, then take
    # the first N. This guarantees that the n=50 subset used for the heavy
    # Mem0-Lite run is a STRICT PREFIX of the n=200 subset used by the
    # cheaper baselines, so the cross-system comparison on the shared
    # 50-question subset is well-defined.
    rng = random.Random(seed)
    rng.shuffle(samples)
    if num_questions > 0 and num_questions < len(samples):
        samples = samples[:num_questions]

    return samples


def iter_chunks(text: str, chunk_size: int = 512, overlap: int = 64) -> Iterator[str]:
    """Whitespace-friendly chunker used by BM25 / dense retrievers."""
    text = text.replace("\r\n", "\n")
    n = len(text)
    if n <= chunk_size:
        yield text
        return
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        # try to cut at a newline / whitespace boundary
        if end < n:
            cut = text.rfind("\n", start, end)
            if cut < start + chunk_size // 2:
                cut = text.rfind(" ", start, end)
            if cut > start + chunk_size // 2:
                end = cut
        yield text[start:end].strip()
        if end == n:
            break
        start = max(end - overlap, start + 1)
