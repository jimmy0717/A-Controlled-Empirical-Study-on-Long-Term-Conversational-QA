"""Helpers shared by all retrievers.

* :func:`pack_by_token_budget` -- greedily fill a context budget with
  retrieval hits, ensuring all four retrievers are compared at matched
  *token-budget* rather than matched *top-k* (one of the design fixes
  motivated by the methodology review).

* :func:`cuda_sync_timer` -- a tiny context manager that calls
  ``torch.cuda.synchronize()`` before / after the timed block so that
  GPU latency numbers are not skewed by CUDA's asynchronous launches.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterable, List, Sequence


def estimate_tokens(text: str) -> int:
    """Rough char/4 heuristic. Avoids loading a tokenizer just to size a list."""
    return max(1, len(text) // 4)


def pack_by_token_budget(items: Sequence, budget_tokens: int = 1024):
    """Greedy packing: take items in order, stop when adding the next item
    would exceed the budget. Each ``item`` must expose ``.text``.
    Returns the packed sub-list.
    """
    out, used = [], 0
    for it in items:
        n = estimate_tokens(it.text)
        if used + n > budget_tokens and out:
            break
        out.append(it)
        used += n
    return out


@contextmanager
def cuda_sync_timer():
    """Context manager that yields a callable returning *current* elapsed
    seconds. Synchronises CUDA before start and end so GPU operations are
    fully captured. Falls back gracefully on CPU-only boxes.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:  # torch optional for CPU-only baselines
        torch = None  # type: ignore
    t0 = time.perf_counter()

    def elapsed() -> float:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    yield elapsed


def percentile(values: Iterable[float], p: float) -> float:
    """Closed-form percentile (no numpy import in retriever modules)."""
    xs = sorted(values)
    if not xs:
        return 0.0
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
