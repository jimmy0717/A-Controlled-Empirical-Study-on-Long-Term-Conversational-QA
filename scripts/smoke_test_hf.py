#!/usr/bin/env python3
"""Minimal smoke test for the `hf` (transformers) generation path.

Loads Qwen2.5-7B-Instruct through the exact code the experiments use
(``src.generator.llm.LLM``) and runs ONE short generation. This exercises the
real chat-template + ``generate`` + decode pipeline on the installed
torch/transformers stack -- WITHOUT needing the LongMemEval dataset or the
DeepSeek judge API, so it isolates "does the local model actually load & run".

Run from the project root:

    python scripts/smoke_test_hf.py            # GPU (A800), bf16
    SMOKE_DEVICE=cpu python scripts/smoke_test_hf.py   # CPU box (slow, fp32-ish)

Env overrides:
    SMOKE_MODEL   model path        (default: models/Qwen2.5-7B-Instruct)
    SMOKE_DEVICE  cuda | cpu        (default: cuda)
    SMOKE_DTYPE   bfloat16|float16  (default: bfloat16)

Exit code 0 = the hf generation path works end-to-end.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# make `src` importable when run as a plain script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generator.llm import LLM  # noqa: E402

MODEL = os.environ.get("SMOKE_MODEL", "models/Qwen2.5-7B-Instruct")
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda")
DTYPE = os.environ.get("SMOKE_DTYPE", "bfloat16")


def main() -> int:
    print(f"[smoke] loading {MODEL!r} via hf backend on {DEVICE} ({DTYPE}) ...")
    llm = LLM(model=MODEL, backend="hf", device=DEVICE,
              max_new_tokens=32, dtype=DTYPE)

    q = "用一句话回答：中国的首都是哪里？"
    ans = llm.chat([{"role": "user", "content": q}], temperature=0.0)

    print(f"[smoke] Q: {q}")
    print(f"[smoke] A: {ans!r}")
    print(f"[smoke] tokens in/out = {llm.stats.n_tokens_in}/{llm.stats.n_tokens_out}")

    if not ans.strip():
        print("[smoke] FAIL: model returned an empty answer.")
        return 1
    print("[smoke] PASS: hf generation path works end-to-end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
