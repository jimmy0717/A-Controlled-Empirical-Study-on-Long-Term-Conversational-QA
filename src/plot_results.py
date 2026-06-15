"""Visualise the LongMemEval retriever comparison.

Pure post-hoc analysis: reads the artefacts already on disk and writes PNGs.
It is completely decoupled from the experiment runner, so it can be run at
any time (including before all systems finish -- it simply plots whatever
``results/*/metrics.json`` are present).

Inputs (whatever exists):
  * ``results/summary.csv``           -- from ``python -m src.aggregate``
  * ``results/pairwise_shared.csv``   -- from ``python -m src.aggregate``
  * ``results/*/metrics.json``        -- per-system raw metrics (per-type acc)

Outputs (``results/figs/``):
  1. ``accuracy_bar.png``       accuracy on the SHARED subset (+95% bootstrap CI)
  2. ``ablation_chain.png``     weak->strong ablation ladder vs the oracle bound
  3. ``acc_vs_cost.png``        accuracy against generation latency & tokens
  4. ``per_type_heatmap.png``   accuracy per question_type x system
  5. ``mcnemar_matrix.png``     paired-significance matrix on the shared subset

All labels are in English on purpose -- AI Studio images often lack a CJK
font, and English avoids tofu boxes.  Every figure is guarded independently
so a missing input only skips that one plot.

Usage::

    python -m src.plot_results
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = Path("results")
FIGS = RESULTS / "figs"

# Canonical weak -> strong ordering; oracle is the retrieval-perfect upper
# bound so it sits at the far end.  Anything not listed is appended after.
ORDER = [
    "onehot", "tfidf", "bm25", "dense", "dense_summarized",
    "mem0_lite", "mem0_lite_14b", "full_ctx", "oracle",
]
# Short, readable display names for axis ticks.
DISPLAY = {
    "onehot": "OneHot", "tfidf": "TF-IDF", "bm25": "BM25", "dense": "Dense",
    "dense_summarized": "Dense-Sum", "mem0_lite": "Mem0-Lite",
    "mem0_lite_14b": "Mem0-14B", "full_ctx": "Full-Ctx", "oracle": "Oracle",
}


def _order_key(name: str) -> int:
    return ORDER.index(name) if name in ORDER else len(ORDER)


def _disp(name: str) -> str:
    return DISPLAY.get(name, name)


def _order_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_k"] = df["experiment"].map(_order_key)
    return df.sort_values("_k").drop(columns="_k").reset_index(drop=True)


def load_summary() -> pd.DataFrame | None:
    p = RESULTS / "summary.csv"
    if not p.exists():
        print(f"[skip] {p} not found - run `python -m src.aggregate` first.")
        return None
    df = pd.read_csv(p)
    # acc_sh columns may be empty strings for systems with no shared overlap;
    # coerce every numeric column so plotting maths never chokes on "".
    for c in ["acc", "acc_lo95", "acc_hi95", "acc_sh", "acc_sh_lo95",
              "acc_sh_hi95", "idx_p50_s", "gen_p50_s", "tok_in", "tok_out"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return _order_df(df)


def load_metrics() -> List[dict]:
    out = []
    for mp in sorted(RESULTS.glob("*/metrics.json")):
        if mp.parent.name.startswith("_"):  # skip probe/smoke-test dirs
            continue
        try:
            out.append(json.loads(mp.read_text(encoding="utf-8")))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] cannot read {mp}: {e}")
    return out


# ----------------------------------------------------------------------
def fig_accuracy_bar(df: pd.DataFrame) -> None:
    """Accuracy on the SHARED subset with asymmetric 95% bootstrap CI bars.

    We prefer ``acc_sh`` (apples-to-apples across systems); rows missing it
    fall back to ``acc`` with a hatch + footnote so the mix is never hidden.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    xs, accs, los, his, used_full = [], [], [], [], []
    for _, r in df.iterrows():
        if pd.notna(r.get("acc_sh")):
            a, lo, hi = r["acc_sh"], r.get("acc_sh_lo95"), r.get("acc_sh_hi95")
            used_full.append(False)
        else:
            a, lo, hi = r["acc"], r.get("acc_lo95"), r.get("acc_hi95")
            used_full.append(True)
        xs.append(_disp(r["experiment"]))
        accs.append(a)
        los.append(a - lo if pd.notna(lo) else 0.0)
        his.append(hi - a if pd.notna(hi) else 0.0)

    x = np.arange(len(xs))
    colors = ["#c44e52" if df.iloc[i]["experiment"] == "oracle" else "#4c72b0"
              for i in range(len(xs))]
    bars = ax.bar(x, accs, yerr=[los, his], capsize=4, color=colors,
                  edgecolor="black", linewidth=0.6)
    for i, (b, full) in enumerate(zip(bars, used_full)):
        if full:
            b.set_hatch("///")
        ax.text(b.get_x() + b.get_width() / 2, accs[i] + 0.01,
                f"{accs[i]:.2f}", ha="center", va="bottom", fontsize=9)

    n_sh = int(df["n_sh"].dropna().iloc[0]) if "n_sh" in df and df["n_sh"].notna().any() else "?"
    ax.set_xticks(x)
    ax.set_xticklabels(xs, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"LongMemEval accuracy on shared subset (n={n_sh}, 95% bootstrap CI)")
    ax.grid(axis="y", alpha=0.3)
    if any(used_full):
        ax.text(0.99, 0.97, "/// = full-set acc (no shared overlap)",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                style="italic", color="dimgray")
    fig.tight_layout()
    _save(fig, "accuracy_bar.png")


def fig_ablation_chain(df: pd.DataFrame) -> None:
    """Weak->strong ladder; oracle drawn as a dashed horizontal ceiling."""
    chain = [e for e in ORDER if e in set(df["experiment"]) and e != "oracle"]
    if len(chain) < 2:
        print("[skip] ablation_chain: need >=2 non-oracle systems.")
        return
    sub = df.set_index("experiment")

    def acc_of(e):
        v = sub.loc[e, "acc_sh"]
        return v if pd.notna(v) else sub.loc[e, "acc"]

    ys = [acc_of(e) for e in chain]
    x = np.arange(len(chain))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, ys, "-o", color="#4c72b0", linewidth=2, markersize=7)
    for xi, yi in zip(x, ys):
        ax.text(xi, yi + 0.015, f"{yi:.2f}", ha="center", fontsize=9)
    if "oracle" in sub.index:
        oacc = acc_of("oracle")
        ax.axhline(oacc, ls="--", color="#c44e52", linewidth=1.5)
        ax.text(len(chain) - 1, oacc + 0.012, f"Oracle {oacc:.2f}",
                ha="right", color="#c44e52", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([_disp(e) for e in chain], rotation=20, ha="right")
    ax.set_ylabel("Accuracy (shared subset)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Ablation ladder: memory strategy weak -> strong")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "ablation_chain.png")


def fig_acc_vs_cost(df: pd.DataFrame) -> None:
    """Two panels: accuracy vs generation latency, accuracy vs output tokens."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    acc = df["acc_sh"].where(df["acc_sh"].notna(), df["acc"])

    for ax, xcol, xlabel in [
        (axes[0], "gen_p50_s", "Generation latency p50 (s/question)"),
        (axes[1], "tok_out", "Generator output tokens (total)"),
    ]:
        if xcol not in df.columns:
            ax.set_visible(False)
            continue
        ax.scatter(df[xcol], acc, s=70, color="#4c72b0", edgecolor="black", zorder=3)
        for _, r in df.iterrows():
            yv = r["acc_sh"] if pd.notna(r.get("acc_sh")) else r["acc"]
            ax.annotate(_disp(r["experiment"]), (r[xcol], yv),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3)
    fig.suptitle("Accuracy vs. cost (upper-left = better trade-off)")
    fig.tight_layout()
    _save(fig, "acc_vs_cost.png")


def fig_per_type_heatmap(metrics: List[dict]) -> None:
    """accuracy_by_type (rows) x system (cols).  Per-type n is tiny, so this
    is a QUALITATIVE view only -- annotate that in the report."""
    if not metrics:
        print("[skip] per_type_heatmap: no metrics.json found.")
        return
    metrics = sorted(metrics, key=lambda m: _order_key(m["experiment"]))
    systems = [m["experiment"] for m in metrics]
    types: List[str] = []
    for m in metrics:
        for t in m.get("accuracy_by_type", {}):
            if t not in types:
                types.append(t)
    if not types:
        print("[skip] per_type_heatmap: no accuracy_by_type present.")
        return

    mat = np.full((len(types), len(systems)), np.nan)
    for j, m in enumerate(metrics):
        bt = m.get("accuracy_by_type", {})
        for i, t in enumerate(types):
            if t in bt:
                mat[i, j] = bt[t]

    fig, ax = plt.subplots(figsize=(1.3 * len(systems) + 3, 0.7 * len(types) + 2))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels([_disp(s) for s in systems], rotation=20, ha="right")
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels(types)
    for i in range(len(types)):
        for j in range(len(systems)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if mat[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Accuracy")
    ax.set_title("Accuracy by question type (qualitative; per-type n is small)")
    fig.tight_layout()
    _save(fig, "per_type_heatmap.png")


def fig_mcnemar(df_sum: pd.DataFrame | None) -> None:
    """Lower-triangle matrix of McNemar p-values on the shared subset; cells
    where p<0.05 are starred."""
    p = RESULTS / "pairwise_shared.csv"
    if not p.exists():
        print(f"[skip] mcnemar_matrix: {p} not found.")
        return
    pw = pd.read_csv(p)
    if pw.empty:
        print("[skip] mcnemar_matrix: pairwise file is empty.")
        return
    systems = sorted(set(pw["system_a"]) | set(pw["system_b"]), key=_order_key)
    idx = {s: i for i, s in enumerate(systems)}
    n = len(systems)
    pmat = np.full((n, n), np.nan)
    for _, r in pw.iterrows():
        i, j = idx[r["system_a"]], idx[r["system_b"]]
        pmat[i, j] = pmat[j, i] = r["p_value"]

    fig, ax = plt.subplots(figsize=(1.1 * n + 2, 1.1 * n + 2))
    masked = np.ma.masked_invalid(pmat)
    im = ax.imshow(masked, cmap="RdYlGn_r", vmin=0, vmax=0.1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels([_disp(s) for s in systems], rotation=30, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels([_disp(s) for s in systems])
    for i in range(n):
        for j in range(n):
            if not np.isnan(pmat[i, j]):
                star = "*" if pmat[i, j] < 0.05 else ""
                ax.text(j, i, f"{pmat[i, j]:.3f}{star}", ha="center",
                        va="center", fontsize=8,
                        color="white" if pmat[i, j] < 0.03 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="McNemar p-value (clipped at 0.1)")
    ax.set_title("Paired significance on shared subset (* = p<0.05)")
    fig.tight_layout()
    _save(fig, "mcnemar_matrix.png")


# ----------------------------------------------------------------------
def _save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out}")


def main() -> None:
    df = load_summary()
    metrics = load_metrics()
    if df is not None and not df.empty:
        fig_accuracy_bar(df)
        fig_ablation_chain(df)
        fig_acc_vs_cost(df)
    fig_per_type_heatmap(metrics)
    fig_mcnemar(df)
    if (df is None or df.empty) and not metrics:
        print("Nothing to plot yet - run experiments + `python -m src.aggregate`.")
    else:
        print(f"\nDone. Figures in {FIGS}/")


if __name__ == "__main__":
    main()
