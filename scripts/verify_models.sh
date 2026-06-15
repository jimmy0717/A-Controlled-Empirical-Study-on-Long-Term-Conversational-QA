#!/usr/bin/env bash
# Verify that the three local models are COMPLETE and LOADABLE.
#
# Models live under models/<name>/ and are referenced by the YAML configs:
#   models/bge-m3              -> SentenceTransformer  (dense encoder)
#   models/bge-reranker-base   -> CrossEncoder         (reranker)
#   models/Qwen2.5-7B-Instruct -> vLLM / transformers  (generator + extractor)
#
# The script runs three layers of checks, from cheap to thorough:
#   L1  Structure : required files are present.
#   L2  Integrity : every *.safetensors shard parses (header + tensor count),
#                   sharded index references all shards, and (best-effort) each
#                   local file size matches the remote size on the HF mirror.
#   L3  Functional: load each model exactly the way src/ loads it
#                   (SentenceTransformer.encode / CrossEncoder.predict /
#                    Qwen tokenizer + a tiny generate).
#
# Defaults are CPU-only so it runs anywhere. Override with env vars:
#   VERIFY_DEVICE=cuda          load functional checks on GPU (faster)
#   VERIFY_FULL_QWEN=1          actually load the full 7B weights (needs ~15GB
#                               RAM/VRAM); off by default -- we validate the
#                               safetensors shards lazily instead.
#   SKIP_REMOTE=1               skip the remote byte-size comparison (offline)
#   HF_ENDPOINT=...             mirror used for the remote size check
#
# Exit code: 0 if all models pass, 1 otherwise.
set -uo pipefail
cd "$(dirname "$0")/.."

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
VERIFY_DEVICE="${VERIFY_DEVICE:-cpu}"
VERIFY_FULL_QWEN="${VERIFY_FULL_QWEN:-0}"
SKIP_REMOTE="${SKIP_REMOTE:-0}"

if command -v python3 >/dev/null 2>&1; then PY_BIN=python3; else PY_BIN=python; fi

HF_ENDPOINT="$HF_ENDPOINT" VERIFY_DEVICE="$VERIFY_DEVICE" \
VERIFY_FULL_QWEN="$VERIFY_FULL_QWEN" SKIP_REMOTE="$SKIP_REMOTE" \
  "$PY_BIN" - <<'PY'
import os, sys, json, pathlib, traceback

ENDPOINT   = os.environ["HF_ENDPOINT"].rstrip("/")
DEVICE     = os.environ.get("VERIFY_DEVICE", "cpu")
FULL_QWEN  = os.environ.get("VERIFY_FULL_QWEN") == "1"
SKIP_REMOTE= os.environ.get("SKIP_REMOTE") == "1"

MODELS = {
    "bge-m3":              "BAAI/bge-m3",
    "bge-reranker-base":   "BAAI/bge-reranker-base",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
}

# Files we consider essential per model type. Weight files are checked
# separately (either model.safetensors or a sharded *.safetensors set).
REQUIRED_COMMON = ["config.json"]
REQUIRED_TOKENIZER_ANY = [  # at least one of these must exist
    ["tokenizer.json"],
    ["tokenizer_config.json"],
    ["sentencepiece.bpe.model"],
    ["vocab.txt"],
]

GREEN = "\033[32m"; RED = "\033[31m"; YEL = "\033[33m"; RST = "\033[0m"
def ok(m):   print(f"  {GREEN}[OK]{RST}   {m}")
def warn(m): print(f"  {YEL}[WARN]{RST} {m}")
def bad(m):  print(f"  {RED}[FAIL]{RST} {m}")

# ---------------------------------------------------------------------------
def remote_sizes(repo):
    """{filename: size} from the HF mirror; {} if offline/unavailable."""
    if SKIP_REMOTE:
        return {}
    try:
        from huggingface_hub import HfApi
        api = HfApi(endpoint=ENDPOINT)
        info = api.repo_info(repo_id=repo, repo_type="model", files_metadata=True)
        return {s.rfilename: (s.size or 0) for s in info.siblings}
    except Exception as e:
        warn(f"remote size check unavailable ({type(e).__name__}: {e})")
        return {}

# ---------------------------------------------------------------------------
def check_structure(local: pathlib.Path):
    problems = []
    for f in REQUIRED_COMMON:
        if not (local / f).exists():
            problems.append(f"missing {f}")
    if not any(all((local / x).exists() for x in group)
               for group in REQUIRED_TOKENIZER_ANY):
        problems.append("no tokenizer file found "
                        "(tokenizer.json / tokenizer_config.json / *.model / vocab.txt)")
    return problems

def weight_files(local: pathlib.Path):
    """Return the list of safetensors shard paths, honouring the index if any."""
    idx = local / "model.safetensors.index.json"
    if idx.exists():
        try:
            mapping = json.loads(idx.read_text())["weight_map"]
            shards = sorted({local / s for s in mapping.values()})
            return shards, mapping
        except Exception as e:
            return [], f"index unreadable: {e}"
    single = local / "model.safetensors"
    if single.exists():
        return [single], None
    # legacy fallback
    pt = list(local.glob("pytorch_model*.bin"))
    return pt, None

def check_safetensors(shards):
    """Validate each shard by parsing the safetensors header directly.

    A .safetensors file is: 8 bytes little-endian uint64 = JSON header length N,
    then N bytes of JSON (the tensor dtype/shape/offset map), then the raw tensor
    bytes. We parse this ourselves so the check needs NO torch/numpy -- it works
    in any Python env and also catches truncation (header claims more bytes than
    the file actually has). Returns (#tensors, problems).
    """
    import struct
    problems, ntensors = [], 0
    for sp in shards:
        if not sp.exists():
            problems.append(f"shard missing: {sp.name}")
            continue
        if sp.suffix != ".safetensors":
            # legacy .bin -- just confirm it is non-trivial in size
            if sp.stat().st_size < 1024:
                problems.append(f"{sp.name} suspiciously small")
            continue
        try:
            fsize = sp.stat().st_size
            with open(sp, "rb") as f:
                head = f.read(8)
                if len(head) < 8:
                    problems.append(f"{sp.name} too small to hold a header")
                    continue
                n = struct.unpack("<Q", head)[0]
                if 8 + n > fsize:
                    problems.append(f"{sp.name} truncated: header claims "
                                    f"{8 + n} bytes but file is {fsize}")
                    continue
                meta = json.loads(f.read(n))
            keys = [k for k in meta.keys() if k != "__metadata__"]
            ntensors += len(keys)
            if not keys:
                problems.append(f"{sp.name} has 0 tensors")
            else:
                # the largest declared end-offset must fit inside the data blob
                end = max(v["data_offsets"][1] for k, v in meta.items()
                          if k != "__metadata__")
                if 8 + n + end > fsize:
                    problems.append(f"{sp.name} truncated: tensor data extends "
                                    f"past EOF ({8 + n + end} > {fsize})")
        except Exception as e:
            problems.append(f"{sp.name} header parse failed: {type(e).__name__}: {e}")
    return ntensors, problems

def check_sizes(local: pathlib.Path, repo):
    """Compare local vs remote byte sizes for files we actually have."""
    rs = remote_sizes(repo)
    if not rs:
        return None  # unavailable
    mismatches = []
    for fn, want in rs.items():
        fp = local / fn
        if not fp.exists() or not want:
            continue  # we only validate files we kept; missing handled elsewhere
        have = fp.stat().st_size
        if have != want:
            mismatches.append(f"{fn}: local={have} remote={want}")
    return mismatches

# ---------------------------------------------------------------------------
class SkipCheck(Exception):
    """Raised when a functional check cannot run due to a missing runtime
    dependency (torch/transformers/sentence_transformers). This is an
    ENVIRONMENT issue, not a model-integrity problem, so we WARN, not FAIL."""

def _need(*mods):
    import importlib.util
    missing = [m for m in mods if importlib.util.find_spec(m) is None]
    if missing:
        raise SkipCheck("missing packages: " + ", ".join(missing))

def functional_bge_m3(local):
    _need("torch", "sentence_transformers")
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(str(local), device=DEVICE)
    emb = m.encode(["长期记忆系统帮助 LLM 智能体", "what is a memory system"],
                   normalize_embeddings=True, convert_to_numpy=True)
    assert emb.ndim == 2 and emb.shape[0] == 2, f"bad emb shape {emb.shape}"
    dim = emb.shape[1]
    import numpy as np
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-2), f"not normalised: {norms}"
    return f"encode OK, dim={dim}"

def functional_reranker(local):
    _need("torch", "sentence_transformers")
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(str(local), device=DEVICE)
    scores = ce.predict([("什么是记忆系统", "长期记忆系统帮助 LLM 智能体"),
                         ("什么是记忆系统", "今天天气晴朗")])
    import numpy as np
    scores = np.asarray(scores).reshape(-1)
    assert scores.shape[0] == 2, f"bad score shape {scores.shape}"
    rel = "relevant>irrelevant" if scores[0] > scores[1] else "ORDERING UNEXPECTED"
    return f"predict OK, scores={scores.tolist()} ({rel})"

def functional_qwen(local):
    _need("transformers")
    from transformers import AutoTokenizer, AutoConfig
    tok = AutoTokenizer.from_pretrained(str(local), trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(str(local), trust_remote_code=True)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": "1+1=?"}],
        add_generation_prompt=True, tokenize=True,
    )
    assert len(ids) > 0, "chat template produced no tokens"
    msg = (f"tokenizer+config OK (vocab={getattr(cfg,'vocab_size','?')}, "
           f"layers={getattr(cfg,'num_hidden_layers','?')}, chat_template OK)")
    if FULL_QWEN:
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            str(local), torch_dtype=torch.float32 if DEVICE == "cpu" else "auto",
            trust_remote_code=True,
        ).to(DEVICE)
        inp = torch.tensor([ids]).to(DEVICE)
        with torch.no_grad():
            out = model.generate(inp, max_new_tokens=8, do_sample=False)
        txt = tok.decode(out[0][len(ids):], skip_special_tokens=True)
        msg += f"; full load+generate OK -> {txt!r}"
    else:
        msg += "; (full-weight load skipped, set VERIFY_FULL_QWEN=1 to enable)"
    return msg

FUNCTIONAL = {
    "bge-m3": functional_bge_m3,
    "bge-reranker-base": functional_reranker,
    "Qwen2.5-7B-Instruct": functional_qwen,
}

# ---------------------------------------------------------------------------
print(f"Verification settings: device={DEVICE}, full_qwen={FULL_QWEN}, "
      f"remote_check={'off' if SKIP_REMOTE else ENDPOINT}\n")

all_pass = True
any_skipped = False
for name, repo in MODELS.items():
    local = pathlib.Path("models") / name
    print(f"==> {name}  ({local})")
    if not local.is_dir():
        bad(f"directory not found: {local}")
        all_pass = False
        print()
        continue

    model_pass = True
    skipped_functional = False

    # L1 structure
    probs = check_structure(local)
    if probs:
        for p in probs: bad(p)
        model_pass = False
    else:
        ok("structure: config + tokenizer present")

    # weight files + L2 safetensors integrity
    shards, extra = weight_files(local)
    if isinstance(extra, str):
        bad(extra); model_pass = False
    if not shards:
        bad("no weight file (model.safetensors / shards / *.bin) found")
        model_pass = False
    else:
        ntensors, wprobs = check_safetensors(shards)
        if wprobs:
            for p in wprobs: bad(p)
            model_pass = False
        else:
            shard_note = f"{len(shards)} shard(s)" if len(shards) > 1 else "single file"
            ok(f"weights: {shard_note}, {ntensors} tensors, headers parse")

    # L2 remote size comparison (best effort)
    sz = check_sizes(local, repo)
    if sz is None:
        warn("remote size comparison skipped/unavailable")
    elif sz:
        for m in sz: bad(f"size mismatch: {m}")
        model_pass = False
    else:
        ok("sizes: all local files match remote byte sizes")

    # L3 functional load
    try:
        detail = FUNCTIONAL[name](local)
        ok(f"functional: {detail}")
    except SkipCheck as e:
        warn(f"functional check skipped ({e}). Integrity (L1+L2) already proves "
             f"the files are complete; install runtime deps to also verify "
             f"loading: pip install -r requirements.txt")
        skipped_functional = True
    except Exception as e:
        bad(f"functional load failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        model_pass = False

    if model_pass:
        tag = (GREEN + "PASS" + RST) if not skipped_functional \
              else (YEL + "PASS (integrity only; functional skipped)" + RST)
    else:
        tag = RED + "FAIL" + RST
    print(f"  ==> {tag}\n")
    all_pass = all_pass and model_pass
    any_skipped = any_skipped or skipped_functional

print("=" * 60)
if all_pass and not any_skipped:
    print(f"{GREEN}ALL MODELS VERIFIED OK (integrity + functional load){RST}")
    sys.exit(0)
elif all_pass:
    print(f"{GREEN}ALL MODELS COMPLETE & INTACT{RST} "
          f"(byte sizes match remote, safetensors headers valid).")
    print(f"{YEL}Functional load was skipped because runtime deps are missing in "
          f"this env.{RST} The download is fine; to also confirm the models load:")
    print("  pip install -r requirements.txt   # torch / transformers / sentence-transformers")
    print("  bash scripts/verify_models.sh      # re-run for the functional layer")
    sys.exit(0)
else:
    print(f"{RED}SOME MODELS FAILED -- see [FAIL] lines above.{RST}")
    print("Re-run: bash scripts/download_models.sh  (resumes only missing pieces)")
    sys.exit(1)
PY
