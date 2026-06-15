#!/usr/bin/env bash
# ============================================================================
# CPU-instance preparation (no GPU billing).
#
# This script does ONLY network/IO work that is safe to do off-GPU:
#   1. download all local model weights -> models/
#   2. download + rename the LongMemEval splits -> data/longmemeval/
#
# It deliberately does NOT run any experiment. Every experiment produces an
# answer with the SAME generator (Qwen2.5-7B-Instruct), and 7B inference is
# not practical on CPU; running some baselines with a different (API)
# generator would introduce a confounding variable and break the
# "hold the generator fixed" design. So all experiments run on the A800
# via scripts/run_all.sh.
#
# After this finishes, snapshot models/ and data/ to shared storage (or the
# dataset/model mounts) so the A800 instance can read them directly.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> [1/2] downloading local models (~18 GB) ..."
bash scripts/download_models.sh

echo "==> [2/2] downloading LongMemEval data ..."
bash scripts/download_data.sh

echo
echo "CPU preparation done. Present:"
du -sh models/* 2>/dev/null || true
ls -lh data/longmemeval/*.json 2>/dev/null || true
echo
echo "Next: on the A800 instance, run  bash scripts/run_all.sh"
