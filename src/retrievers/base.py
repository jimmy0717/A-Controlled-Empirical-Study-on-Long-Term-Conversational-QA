"""Common interface for memory retrievers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class MemoryItem:
    text: str
    score: float = 0.0
    meta: dict = field(default_factory=dict)


class BaseRetriever(ABC):
    """All retrievers expose the same two methods to the experiment runner."""

    name: str = "base"

    @abstractmethod
    def index(self, sample) -> None:
        """Build any per-sample index required to answer one question."""

    @abstractmethod
    def search(self, query: str, top_k: int) -> List[MemoryItem]:
        """Return the top-``k`` memories that should be passed to the LLM."""

    # Optional: latency/cost accounting hooks
    def reset_stats(self) -> None:
        self.stats = {
            "n_calls": 0,
            "n_tokens_in": 0,
            "n_tokens_out": 0,
            "wallclock_index": 0.0,
            "wallclock_search": 0.0,
        }
