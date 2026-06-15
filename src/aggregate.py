"""Aggregate per-retriever ``metrics.json`` into a single CSV plus
bootstrap 95% confidence intervals for the accuracy column.

Two views are produced:

1. ``results/summary.csv`` -- one row per experiment.  The ``acc`` column
   is computed on each experiment's *full* prediction set (the n it was
   actually run on, e.g. 200 for the cheap baselines, 50 for Mem0-Lite).
   The ``acc_sh`` columns add accuracy restricted to the **shared subset**
   of question ids that *every* experiment answered.  Because the data
   loader sub-samples with a fixed prefix-stable shuffle, the 50-question
   Mem0-Lite run is a strict prefix of the 200-question baselines, so this
   intersection is exactly the shared comparison set referenced in the
   config comments.  Cross-system claims MUST cite ``acc_sh``, never
   ``acc`` (which mixes 200-q and 50-q numbers -- apples vs oranges).

2. ``results/pairwise_shared.csv`` -- for every ordered pair of systems,
   a paired McNemar test on the shared subset.  This is the statistically
   correct way to compare two systems evaluated on the *same* questions
   (more powerful than asking whether two independent CIs overlap).

Usage::

    python -m src.aggregate
"""
from __future__ import annotations

import json
import math
import random
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def bootstrap_ci(values: List[int], n_boot: int = 10000, alpha: float = 0.05, seed: int = 42):
    """Return (mean, lower, upper) percentile bootstrap CI for the mean of
    a 0/1 valued list."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        s = sum(values[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return sum(values) / n, lo, hi


def load_flags_by_qid(exp_dir: Path) -> Dict[str, int]:
    """Map qid -> 0/1 correctness flag for one experiment."""
    p = exp_dir / "predictions.jsonl"
    out: Dict[str, int] = {}
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "correct" in row and "qid" in row:
                out[row["qid"]] = int(bool(row["correct"]))
    return out


def mcnemar(a: List[int], b: List[int]) -> Tuple[int, int, float, str]:
    """Paired McNemar test on two aligned 0/1 sequences.

    Returns (b_only, c_only, p_value, method) where
      b_only = #(a correct, b wrong),  c_only = #(a wrong, b correct).
    Uses the exact two-sided binomial test when the number of discordant
    pairs is small (<25), otherwise the chi-square approximation with
    continuity correction.  No SciPy dependency: the chi-square(1) tail is
    ``erfc(sqrt(x/2))`` and the binomial tail is summed exactly.
    """
    b_only = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    c_only = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n_disc = b_only + c_only
    if n_disc == 0:
        return b_only, c_only, 1.0, "none"
    if n_disc < 25:
        k = min(b_only, c_only)
        tail = sum(math.comb(n_disc, i) for i in range(k + 1)) * (0.5 ** n_disc)
        return b_only, c_only, min(1.0, 2.0 * tail), "exact-binomial"
    stat = (abs(b_only - c_only) - 1.0) ** 2 / n_disc
    p = math.erfc(math.sqrt(stat / 2.0))
    return b_only, c_only, p, "chi2-cc"


def main() -> None:
    exps = []  # (name, retriever, metrics, flags_by_qid)
    for mp in sorted(Path("results").glob("*/metrics.json")):
        # Skip probe/smoke-test dirs (e.g. _probe, _smoke, _probe_mem). They
        # only hold a handful of questions, and because the shared subset is
        # the INTERSECTION of every experiment's qids, a 2-question probe
        # would otherwise crush n_sh from 50 down to 2 and silently void the
        # whole shared-subset comparison.
        if mp.parent.name.startswith("_"):
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        exps.append((m["experiment"], m.get("retriever", ""), m, load_flags_by_qid(mp.parent)))

    if not exps:
        print("No results/*/metrics.json found - run experiments first.")
        return

    # ---- shared subset = qids answered by EVERY experiment ----
    qid_sets = [set(f.keys()) for _, _, _, f in exps if f]
    shared = set.intersection(*qid_sets) if qid_sets else set()
    shared_order = sorted(shared)
    n_shared = len(shared_order)

    rows = []
    for name, retr, m, flags in exps:
        full = list(flags.values())
        mean, lo, hi = bootstrap_ci(full) if full else (m.get("accuracy", 0.0),) * 3
        sh_flags = [flags[q] for q in shared_order if q in flags]
        if sh_flags:
            s_mean, s_lo, s_hi = bootstrap_ci(sh_flags)
        else:
            s_mean = s_lo = s_hi = float("nan")
        rows.append({
            "experiment":      name,
            "retriever":       retr,
            "n":               m["n_questions"],
            "acc":             round(mean, 4),
            "acc_lo95":        round(lo, 4),
            "acc_hi95":        round(hi, 4),
            "acc_ci_pm":       round(max(hi - mean, mean - lo), 4),
            "n_sh":            n_shared,
            "acc_sh":          round(s_mean, 4) if sh_flags else "",
            "acc_sh_lo95":     round(s_lo, 4) if sh_flags else "",
            "acc_sh_hi95":     round(s_hi, 4) if sh_flags else "",
            "idx_p50_s":       round(m.get("latency", {}).get("index_p50_s", 0.0), 3),
            "idx_p95_s":       round(m.get("latency", {}).get("index_p95_s", 0.0), 3),
            "gen_p50_s":       round(m.get("latency", {}).get("generate_p50_s", 0.0), 3),
            "gen_p95_s":       round(m.get("latency", {}).get("generate_p95_s", 0.0), 3),
            "tok_in":          m.get("tokens", {}).get("generator_in", 0),
            "tok_out":         m.get("tokens", {}).get("generator_out", 0),
        })

    df = pd.DataFrame(rows).sort_values("acc", ascending=False)
    out = Path("results/summary.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nShared subset size (answered by all systems): n={n_shared}")
    print(f"Saved -> {out}")

    # ---- pairwise McNemar on the shared subset ----
    if n_shared >= 2:
        pair_rows = []
        named = [(name, flags) for name, _, _, flags in exps if flags]
        for (na, fa), (nb, fb) in combinations(named, 2):
            a = [fa[q] for q in shared_order]
            b = [fb[q] for q in shared_order]
            b_only, c_only, p, method = mcnemar(a, b)
            pair_rows.append({
                "system_a":   na,
                "system_b":   nb,
                "n_sh":       n_shared,
                "acc_a":      round(sum(a) / n_shared, 4),
                "acc_b":      round(sum(b) / n_shared, 4),
                "a>b":        b_only,   # a correct, b wrong
                "b>a":        c_only,   # b correct, a wrong
                "p_value":    round(p, 4),
                "sig_0.05":   p < 0.05,
                "method":     method,
            })
        pdf = pd.DataFrame(pair_rows)
        pout = Path("results/pairwise_shared.csv")
        pdf.to_csv(pout, index=False)
        print("\n=== Pairwise McNemar on shared subset ===")
        print(pdf.to_string(index=False))
        print(f"Saved -> {pout}")
    else:
        print("\nShared subset too small for pairwise tests (need >=2 questions).")


if __name__ == "__main__":
    main()
