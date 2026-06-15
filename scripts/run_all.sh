#!/usr/bin/env bash
# End-to-end reproduction script for AI Studio (single-card A800 80G).
#
# Order matters: cheap retrievers first so we get a partial summary even
# if the long Mem0-Lite run is interrupted.
set -euo pipefail
cd "$(dirname "$0")/.."

# -- speed up HF downloads on AI Studio --
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER=1

# -- primary baselines (cheap → expensive ordering) --
for cfg in configs/onehot.yaml \
           configs/tfidf.yaml \
           configs/bm25.yaml \
           configs/dense.yaml \
           configs/dense_summarized.yaml \
           configs/oracle.yaml \
           configs/mem0_lite.yaml ; do
    echo "============================================================"
    echo "[run] ${cfg}"
    echo "============================================================"
    python -m src.run --config "${cfg}"
done

# -- optional ablations --
# python -m src.run --config configs/mem0_lite_14b.yaml
# python -m src.run --config configs/full_ctx.yaml

python -m src.aggregate
echo "Done. See results/summary.csv"
