"""Unified experiment runner.

Differences vs. the first cut (motivated by the methodology review):
  * **Token-budget retrieval**: matches all retrievers at the same
    *context budget* rather than a fixed top-k.
  * **CUDA-synchronised timing** with warm-up, reporting both p50 and p95.
  * **Persistent Mem0-Lite bank**: optional re-use across samples that
    belong to the same user (LongMemEval has one user per sample, so
    this only affects extension experiments).
  * Five retrievers are routed: tfidf / bm25 / dense / dense_summarized /
    mem0_lite, plus the special ``full_ctx`` and ``oracle`` modes.

Usage::

    python -m src.run --config configs/<retriever>.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import yaml
from tqdm import tqdm

from .data.loader import load_longmemeval
from .eval.judge import Judge
from .generator.llm import LLM
from .prompting import fill_template
from .retrievers.base import MemoryItem
from .retrievers.bm25 import BM25Retriever
from .retrievers.dense import DenseRetriever
from .retrievers.dense_summarized import DenseSummarizedRetriever
from .retrievers.mem0_lite import Mem0LiteRetriever
from .retrievers.onehot import OneHotRetriever
from .retrievers.oracle import OracleRetriever
from .retrievers.tfidf import TfidfRetriever
from .retrievers.utils import (
    cuda_sync_timer,
    pack_by_token_budget,
    percentile,
    seed_everything,
)


# ----------------------------------------------------------------------
def _override_backend(cfg: dict, backend: str | None) -> None:
    """Force the *local* LLM backend (generator + mem0_lite extractor/updater)
    without editing any YAML. Resolution order:
        explicit --backend arg  >  env var GEN_BACKEND  >  leave config as-is.
    The judge backend is intentionally left untouched (it talks to a remote
    OpenAI-compatible API by design)."""
    backend = backend or os.environ.get("GEN_BACKEND")
    if not backend:
        return
    if isinstance(cfg.get("generator"), dict):
        cfg["generator"]["backend"] = backend
    m = cfg.get("mem0_lite")
    if isinstance(m, dict) and "extractor_backend" in m:
        m["extractor_backend"] = backend
    print(f"[run] backend overridden -> {backend} (generator + extractor)")


# ----------------------------------------------------------------------
def _maybe_share_llm(spec_model: str, spec_backend: str, others: dict) -> LLM | None:
    """Return an existing LLM if model+backend match, else None."""
    for name, llm in others.items():
        if llm is not None and llm.model_name == spec_model and llm.backend == spec_backend:
            return llm
    return None


def build_retriever(cfg: dict, llms: dict):
    name = cfg["retriever"]["name"]
    g = cfg["generator"]
    if name == "onehot":
        return OneHotRetriever(chunk_size=cfg.get("onehot", {}).get("chunk_size", 512))
    if name == "tfidf":
        return TfidfRetriever(chunk_size=cfg.get("tfidf", {}).get("chunk_size", 512))
    if name == "bm25":
        return BM25Retriever(chunk_size=cfg.get("bm25", {}).get("chunk_size", 512))
    if name == "dense":
        d = cfg.get("dense", {})
        return DenseRetriever(
            encoder=d.get("encoder", "BAAI/bge-m3"),
            device=d.get("device", "cuda"),
            batch_size=d.get("batch_size", 64),
            chunk_size=d.get("chunk_size", 512),
            reranker=d.get("reranker"),
            rerank_top_n=d.get("rerank_top_n", 32),
        )
    if name == "dense_summarized":
        d = cfg.get("dense_summarized", {})
        return DenseSummarizedRetriever(
            summariser=llms["generator"],
            encoder=d.get("encoder", "BAAI/bge-m3"),
            device=d.get("device", "cuda"),
            batch_size=d.get("batch_size", 64),
            max_facts_per_session=d.get("max_facts_per_session", 12),
            reranker=d.get("reranker"),
            rerank_top_n=d.get("rerank_top_n", 32),
        )
    if name == "mem0_lite":
        m = cfg.get("mem0_lite", {})
        # share the generator if extractor model coincides
        ex_model = m.get("extractor_model", g["model"])
        up_model = m.get("updater_model", ex_model)
        ex_backend = m.get("extractor_backend", g.get("backend", "hf"))
        extractor = _maybe_share_llm(ex_model, ex_backend, llms) or LLM(
            model=ex_model, backend=ex_backend
        )
        updater = _maybe_share_llm(up_model, ex_backend, {**llms, "extractor": extractor}) or LLM(
            model=up_model, backend=ex_backend
        )
        return Mem0LiteRetriever(
            extractor=extractor,
            updater=updater,
            encoder=m.get("encoder", "BAAI/bge-m3"),
            device="cuda",
            similarity_threshold=m.get("similarity_threshold", 0.75),
            max_memories_per_pair=m.get("max_memories_per_pair", 8),
        )
    if name == "oracle":
        return OracleRetriever()
    if name == "full_ctx":
        return None
    raise ValueError(f"unknown retriever: {name}")


# ----------------------------------------------------------------------
def build_qa_prompt(question: str, memories, today: str) -> str:
    template = (Path(__file__).resolve().parents[1] / "prompts" / "qa.txt").read_text(
        encoding="utf-8"
    )
    if memories:
        mem_block = "\n".join(
            f"[{i+1}] {m.text}  (score={m.score:.3f})" for i, m in enumerate(memories)
        )
    else:
        mem_block = "(no memories retrieved)"
    return fill_template(template, today=today or "(unknown)", memories=mem_block, question=question)


# ----------------------------------------------------------------------
def run(cfg_path: str, backend: str | None = None) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    _override_backend(cfg, backend)
    seed_everything(cfg["experiment"].get("seed", 42))

    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- data --
    samples = load_longmemeval(
        data_dir=cfg.get("data", {}).get("data_dir", "data/longmemeval"),
        split=cfg.get("data", {}).get("split", "longmemeval_s"),
        num_questions=cfg["experiment"].get("num_questions", 100),
        seed=cfg["experiment"].get("seed", 42),
    )

    # -- generator + retriever --
    g = cfg["generator"]
    gen = LLM(
        model=g["model"],
        backend=g.get("backend", "hf"),
        max_new_tokens=g.get("max_new_tokens", 512),
        temperature=g.get("temperature", 0.0),
        top_p=g.get("top_p", 1.0),
    )
    llms = {"generator": gen}
    retriever = build_retriever(cfg, llms)

    # -- judge (separate config block) --
    j = cfg["judge"]
    judge_llm = _maybe_share_llm(j["model"], j.get("backend", "hf"), llms) or LLM(
        model=j["model"],
        backend=j.get("backend", "hf"),
        max_new_tokens=j.get("max_new_tokens", 128),
        temperature=j.get("temperature", 0.0),
    )
    judge = Judge(judge_llm)
    llms["judge"] = judge_llm

    # ---- retrieval / generation budgets ----
    top_k = cfg["retriever"].get("top_k", 50)             # shortlist size
    budget_tokens = cfg["retriever"].get("budget_tokens", 1024)  # *fair* budget

    # -- run --
    preds_path = out_dir / "predictions.jsonl"
    t_start = time.perf_counter()

    # Throw away the first sample's timing as warm-up (compilation, KV setup)
    warmup_done = False
    rows = []

    for sample in tqdm(samples, desc=f"[{cfg['experiment']['name']}] QA"):
        # ---------- retrieval -----------
        with cuda_sync_timer() as elapsed_idx:
            if retriever is not None:
                retriever.index(sample)
                hits = retriever.search(sample.question, top_k)
            else:
                hits = [MemoryItem(text=sample.haystack_text, score=1.0)]
        t_index = elapsed_idx()

        # token-budget packing (no-op for full_ctx)
        if retriever is not None and budget_tokens > 0:
            hits = pack_by_token_budget(hits, budget_tokens=budget_tokens)

        # ---------- generation -----------
        prompt = build_qa_prompt(sample.question, hits, sample.question_date)
        with cuda_sync_timer() as elapsed_gen:
            answer = gen.chat(
                [{"role": "user", "content": prompt}],
                max_new_tokens=g.get("max_new_tokens", 512),
                temperature=0.0,
            )
        t_gen = elapsed_gen()

        if not warmup_done:
            warmup_done = True
            t_index = t_gen = 0.0  # discard first-sample timings

        rows.append({
            "qid": sample.qid,
            "qtype": sample.qtype,
            "question": sample.question,
            "gold": sample.answer,
            "pred": answer,
            "n_memories": len(hits),
            "t_index": t_index,
            "t_generate": t_gen,
            "memories": [
                {"rank": i, "text": m.text, "score": m.score} for i, m in enumerate(hits)
            ],
        })

    # write predictions before judging (so a crash in judging doesn't cost us)
    with preds_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------- LLM-as-judge -----------
    correct = 0
    by_type: dict = {}
    for r in tqdm(rows, desc="judge"):
        verdict = judge.grade(r["question"], r["gold"], r["pred"])
        r["correct"] = bool(verdict["correct"])
        r["judge_reason"] = verdict["reason"]
        correct += int(r["correct"])
        by_type.setdefault(r["qtype"], [0, 0])
        by_type[r["qtype"]][0] += int(r["correct"])
        by_type[r["qtype"]][1] += 1
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - t_start

    # latency percentiles
    idx_lat = [r["t_index"]    for r in rows[1:] if r["t_index"]    > 0]
    gen_lat = [r["t_generate"] for r in rows[1:] if r["t_generate"] > 0]

    metrics = {
        "experiment": cfg["experiment"]["name"],
        "retriever":  cfg["retriever"]["name"],
        "n_questions": len(rows),
        "accuracy":   correct / max(len(rows), 1),
        "accuracy_by_type": {k: v[0] / max(v[1], 1) for k, v in by_type.items()},
        "wallclock_total_s": elapsed,
        "latency": {
            "index_p50_s":    percentile(idx_lat, 50),
            "index_p95_s":    percentile(idx_lat, 95),
            "generate_p50_s": percentile(gen_lat, 50),
            "generate_p95_s": percentile(gen_lat, 95),
        },
        "tokens": {
            "generator_in":  gen.stats.n_tokens_in,
            "generator_out": gen.stats.n_tokens_out,
            "generator_calls": gen.stats.n_calls,
            "judge_calls":   judge_llm.stats.n_calls,
        },
        "config": cfg,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in metrics.items() if k != "config"},
                     indent=2, ensure_ascii=False))


# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--backend", default=None, choices=["hf", "vllm", "openai"],
        help="override the local LLM backend (generator + extractor) without "
             "editing the YAML; default: use the config value, or env GEN_BACKEND",
    )
    args = parser.parse_args()
    run(args.config, backend=args.backend)


if __name__ == "__main__":
    main()
