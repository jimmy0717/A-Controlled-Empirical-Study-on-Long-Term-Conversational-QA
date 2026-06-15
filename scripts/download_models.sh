#!/usr/bin/env bash
# Pre-download ALL models locally so experiments never depend on a live
# network connection at run time (reproducibility + AI-Studio friendliness).
#
# Models land under models/<name>/ and the YAML configs point at those
# local paths. Run this once (locally or on AI Studio).
#
# AI Studio note: the network is flaky and large weight files (e.g.
# bge-reranker-base's ~1.1 GB model.safetensors, Qwen 7B's multi-GB shards)
# repeatedly drop mid-transfer (RemoteProtocolError, stalls at the same %).
# huggingface_hub only retries the *whole* request, which is fragile. So we
# resolve the file list + exact sizes via the HF API (tiny call) and download
# each file with `curl -C -` (byte-range resume) inside OUR OWN retry loop:
# whenever the connection drops we just call curl again, which resumes from
# the bytes already on disk, until the local size matches the remote size.
# This is independent of the curl version (old curl on AI Studio does not
# support --retry-all-errors, so we never hard-depend on it). If curl is
# unavailable we fall back to hf_hub_download per file.
#
# Storage budget (bf16 / fp32 weights):
#   Qwen2.5-7B-Instruct   ~15 GB
#   Qwen2.5-14B-Instruct  ~28 GB   (only needed for the optional 14B ablation)
#   bge-m3                ~2.3 GB
#   bge-reranker-base     ~1.1 GB
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
mkdir -p models

# Set DOWNLOAD_14B=1 to also fetch the 14B extractor (optional ablation).
DOWNLOAD_14B="${DOWNLOAD_14B:-0}"

if command -v curl >/dev/null 2>&1; then HAVE_CURL=1; else HAVE_CURL=0; fi
# Detect whether this curl supports --retry-all-errors (curl >= 7.71). Old
# AI Studio curl does not, so we add the flag only when available.
CURL_ALL_ERRORS=0
if [ "$HAVE_CURL" = "1" ] && curl --help all 2>/dev/null | grep -q -- '--retry-all-errors'; then
  CURL_ALL_ERRORS=1
fi

if command -v python3 >/dev/null 2>&1; then PY_BIN=python3; else PY_BIN=python; fi

HF_ENDPOINT="$HF_ENDPOINT" HAVE_CURL="$HAVE_CURL" CURL_ALL_ERRORS="$CURL_ALL_ERRORS" \
  "$PY_BIN" - "$DOWNLOAD_14B" <<'PY'
import os, sys, pathlib, subprocess, time, fnmatch
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
ENDPOINT = os.environ["HF_ENDPOINT"].rstrip("/")
HAVE_CURL = os.environ.get("HAVE_CURL") == "1"
CURL_ALL_ERRORS = os.environ.get("CURL_ALL_ERRORS") == "1"

from huggingface_hub import HfApi, hf_hub_download
api = HfApi(endpoint=ENDPOINT)

want_14b = sys.argv[1] == "1"
repos = {
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "bge-m3":              "BAAI/bge-m3",
    "bge-reranker-base":   "BAAI/bge-reranker-base",
}
if want_14b:
    repos["Qwen2.5-14B-Instruct"] = "Qwen/Qwen2.5-14B-Instruct"

# Skip (a) duplicate/oversized weight formats we don't use, and (b) repo junk
# that is irrelevant to loading the model. Some repos (e.g. BAAI/bge-m3) ship
# a committed `imgs/` folder with README figures plus a macOS `.DS_Store`
# metadata file that the mirror returns 403 for -- none of these are needed.
IGNORE = [
    "*.pth", "*.onnx", "*.msgpack", "*.h5", "*.safetensors.index.json.bak",
    "*.DS_Store", ".DS_Store", "imgs/*", "*/imgs/*", "assets/*",
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.svg", "*.pdf",
]
def ignored(fn):
    return any(fnmatch.fnmatch(fn, pat) for pat in IGNORE)

def repo_files_with_sizes(repo):
    """{filename: size_in_bytes} via the HF API (small, retried)."""
    for attempt in range(1, 11):
        try:
            info = api.repo_info(repo_id=repo, repo_type="model", files_metadata=True)
            return {s.rfilename: (s.size or 0) for s in info.siblings}
        except Exception as e:
            print(f"    repo_info attempt {attempt}/10 -> {type(e).__name__}: {e}")
            time.sleep(min(3 * attempt, 20))
    raise RuntimeError(f"cannot read file list for {repo}")

def curl_once(repo, fn, dst):
    url = f"{ENDPOINT}/{repo}/resolve/main/{fn}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-L", "--fail", "-C", "-",
           "--retry", "20", "--retry-delay", "3", "--connect-timeout", "30"]
    if CURL_ALL_ERRORS:
        cmd.append("--retry-all-errors")
    cmd += ["-o", str(dst), url]
    return subprocess.call(cmd)

def fetch_curl(repo, fn, dst, want):
    """Our own resume loop: re-invoke curl -C - until size == want."""
    for attempt in range(1, 1000):
        have = dst.stat().st_size if dst.exists() else 0
        if want and have == want:
            return True
        if want and have > want:           # corrupt/oversized -> restart file
            dst.unlink(missing_ok=True)
            have = 0
        print(f"      [{fn}] curl attempt {attempt} (have={have} want={want})")
        rc = curl_once(repo, fn, dst)
        have = dst.stat().st_size if dst.exists() else 0
        if want and have == want:
            return True
        if rc == 0 and not want:           # unknown size -> trust curl success
            return True
        time.sleep(3)
    return False

def fetch_hf(repo, fn, local_dir):
    for attempt in range(1, 21):
        try:
            print(f"      [{fn}] hf_hub_download attempt {attempt}/20")
            return hf_hub_download(repo_id=repo, repo_type="model",
                                   filename=fn, local_dir=str(local_dir))
        except Exception as e:
            print(f"        -> {type(e).__name__}: {e}")
            time.sleep(min(5 * attempt, 30))
    return None

failures = []
for local_name, repo in repos.items():
    dst_dir = pathlib.Path("models") / local_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"==> {repo}  ->  {dst_dir}")
    sizes = repo_files_with_sizes(repo)
    files = [f for f in sizes if not ignored(f)]
    print(f"    {len(files)} files to fetch")
    for fn in files:
        dst = dst_dir / fn
        want = sizes.get(fn, 0)
        if dst.exists() and want and dst.stat().st_size == want:
            print(f"    [{fn}] already complete ({want} bytes), skip")
            continue
        ok = False
        if HAVE_CURL:
            print(f"    [{fn}] curl resume (want={want} bytes)")
            ok = fetch_curl(repo, fn, dst, want)
            if not ok:
                print(f"    [{fn}] curl loop gave up; falling back to HF client")
        if not ok:
            ok = fetch_hf(repo, fn, dst_dir) is not None
        if not ok:
            # Non-fatal: one stray file must not kill a multi-GB run. Record
            # and continue; the final check below decides if it matters.
            print(f"    [{fn}] !! could not download; SKIPPING (see summary)")
            failures.append(f"{repo}/{fn}")

if failures:
    print("\nWARNING: the following files could not be downloaded:")
    for f in failures:
        print(f"  - {f}")
    print("If they are only docs/images (e.g. imgs/, README, .DS_Store) this is"
          " harmless. If a weight/tokenizer/config file is listed, re-run the"
          " script to resume the missing pieces.")
else:
    print("\nAll models downloaded under ./models/")
PY

echo "=== models/ ==="
du -sh models/* 2>/dev/null || true
