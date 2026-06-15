# When Does LLM-Driven Memory Construction Help?

A controlled empirical study on long-term conversational QA. We hold
the encoder (BGE-M3), the cross-encoder reranker
(`bge-reranker-base`) and the generator (Qwen2.5-7B-Instruct) fixed,
match every retriever to a 1024-token context budget, and vary only
the construction pipeline.

The companion 6-page report is in [`paper/`](paper/); the headline
result is that under a 7B open-weight generator, a plain
Dense + reranker retriever already matches the retrieval-perfect
Oracle (54% vs 56%, McNemar p = 1.0), while LLM-driven construction
(session summaries, Mem0-style four-op update agent) makes accuracy
worse rather than better.

Course: *Natural Language Processing and Speech Technology* (final
project). Author: Yong Yang, School of Software, Beihang University.

## Repository layout

```
final_project/
├── README.md / LICENSE / .gitignore / requirements.txt
├── configs/                  one YAML per experiment
├── prompts/                  extract / update / qa / judge templates
├── scripts/
│   ├── download_data.sh      LongMemEval-S splits  -> data/longmemeval/
│   ├── download_models.sh    Qwen2.5-7B + BGE-M3 + reranker -> models/
│   └── run_all.sh            seven experiments + aggregation
├── src/
│   ├── data/loader.py
│   ├── retrievers/           onehot, tfidf, bm25, dense, dense_summarized,
│   │                         mem0_lite, oracle (full_ctx is optional)
│   ├── generator/llm.py      hf | vllm | openai backends + diskcache
│   ├── eval/judge.py
│   ├── run.py                unified entry point
│   ├── aggregate.py          summary.csv + pairwise McNemar on shared subset
│   └── plot_results.py       five PNGs into results/figs/
├── results/                  aggregate metrics (no predictions.jsonl)
└── paper/                    6-page ACL LaTeX project + figs/*.png
```

`models/`, `data/longmemeval/` and `results/*/predictions.jsonl` are
kept out of git: weights and dataset splits are obtained via the
download scripts, and per-question predictions contain LongMemEval
text that should not be redistributed.

## Setup

```bash
cd final_project
pip install -U pip
pip install -r requirements.txt
```

Tested on Python 3.10 with a single A800 80G. CPU-only runs work for
the classical baselines (one-hot, tf-idf, bm25) but the dense and
construction-heavy pipelines require a GPU.

The judge is the only component that calls a remote API:

```bash
export OPENAI_API_KEY="<your-deepseek-key>"
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
```

If you do not have a DeepSeek key, change the `judge:` block in every
`configs/*.yaml` to a local model
(e.g. `backend: "hf", model: "models/Qwen2.5-7B-Instruct"`); a
different *family* would have been preferable for self-preference
mitigation, but a different *checkpoint* of the same family is still
better than reusing the generator.

## Reproducing the results

```bash
# 1. fetch local models and data (one-time, ~18GB of weights)
bash scripts/download_models.sh
bash scripts/download_data.sh

# 2. run all seven systems end to end
bash scripts/run_all.sh
# -> results/<system>/{metrics,predictions}.json{,l}

# 3. aggregate and plot
python -m src.aggregate
python -m src.plot_results
# -> results/summary.csv
# -> results/pairwise_shared.csv
# -> results/figs/{accuracy_bar,ablation_chain,acc_vs_cost,
#                  per_type_heatmap,mcnemar_matrix}.png
```

Wall-clock on a single A800 with the HuggingFace backend (the
configuration used for the reported numbers):

| run                  | n   | time      |
|----------------------|-----|-----------|
| onehot               | 200 | ~9 min    |
| tfidf                | 200 | ~9 min    |
| bm25                 | 200 | ~9 min    |
| dense (+ reranker)   | 200 | ~15 min   |
| dense_summarized     | 200 | ~70 min   |
| oracle               | 200 | ~9 min    |
| mem0_lite (7B)       | 50  | ~3.5 h    |
| **total**            |     | **~5.5 h** |

Subsampling is *prefix-stable* (shuffle once with seed 42, then take
the first N), so the n=50 Mem0-Lite run is a strict prefix of the
n=200 baselines, and the cross-system comparison on the shared
50-question subset is well-defined.

`run_all.sh` is restartable: every LLM call is content-addressed and
cached on disk, so a re-run only spends time on the runs that have
not finished yet.

## Experiment matrix

| Retriever          | Granularity | Construction step                | Role                              |
|--------------------|-------------|----------------------------------|-----------------------------------|
| `oracle`           | turn        | gold `has_answer` flag           | retrieval-perfect upper bound     |
| `onehot`           | chunk       | binary bag-of-words              | weakest classical baseline        |
| `tfidf`            | chunk       | bigram TF-IDF                    | classical IR baseline             |
| `bm25`             | chunk       | Okapi BM25                       | strong sparse baseline            |
| `dense`            | chunk       | BGE-M3 + bge-reranker-base       | strong dense baseline             |
| `dense_summarized` | LLM fact    | session-level LLM summariser     | isolates granularity / rewriting  |
| `mem0_lite`        | LLM fact    | LLM extract + 4-op update agent  | full Mem0-style construction      |

Pairwise contrasts that the paper exploits:

* `onehot -> tfidf -> bm25 -> dense`  retrieval method, granularity fixed
* `dense vs dense_summarized`         granularity / LLM rewriting
* `dense_summarized vs mem0_lite`     contribution of the four-op update agent
* any system vs `oracle`              retrieval headroom

`configs/full_ctx.yaml` and `configs/mem0_lite_14b.yaml` are kept as
optional ablations; they are not part of the reported headline
results.

## Headline results (shared 50-question subset)

| Retriever          | Acc.  | 95% CI       | Idx p50 (s) |
|--------------------|------:|--------------|------------:|
| Oracle (bound)     | 0.56  | [0.42, 0.70] | 0.00        |
| Dense + reranker   | 0.54  | [0.40, 0.68] | 7.40        |
| BM25               | 0.44  | [0.30, 0.58] | 0.06        |
| TF-IDF             | 0.44  | [0.30, 0.58] | 0.16        |
| One-hot            | 0.38  | [0.24, 0.52] | 0.06        |
| Dense-Summarised   | 0.34  | [0.22, 0.48] | 72.6        |
| Mem0-Lite (7B)     | 0.08  | [0.02, 0.16] | 251.9       |

`results/summary.csv` and `results/pairwise_shared.csv` reproduce the
full table and the paired McNemar matrix used in the report.

## Companion survey

The paper at `paper/main.tex` cites a companion survey
(`survey_paper/main.tex` in the parent folder) which reviews Mem0,
MemOS and Zep along the construction / storage / retrieval axes.
Treat the two documents as a *survey + study* pair.

## Licences

* Code: MIT (`LICENSE`).
* Dataset: LongMemEval, MIT licence
  ([Wu *et al.*, 2024](https://arxiv.org/abs/2410.10813)). Splits are
  fetched at run time and not redistributed in this repo.
* Local models: Qwen2.5-Instruct (Apache-2.0), BGE-M3 / bge-reranker
  (MIT). Downloaded by `scripts/download_models.sh` and not
  redistributed.
* Judge: DeepSeek-V4-Flash via the DeepSeek OpenAI-compatible API.
* LaTeX templates: official ACL `acl.sty` / `acl_natbib.bst`.
