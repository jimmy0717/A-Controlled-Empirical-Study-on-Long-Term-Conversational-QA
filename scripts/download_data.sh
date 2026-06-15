#!/usr/bin/env bash
# Download LongMemEval into data/longmemeval/.
#
# The HuggingFace repo ships the splits as extension-less files named
# `longmemeval_s` / `longmemeval_m` / `longmemeval_oracle` (their content
# is JSON). We download and rename them to `<split>.json` so the loader
# finds them.
#
# AI Studio note: the network is flaky and large downloads (longmemeval_s
# is ~278MB) repeatedly drop mid-transfer (RemoteProtocolError, stalls at
# the same %). The Python HF client only retries the whole request, which
# is fragile. So we download with `curl -C -` (byte-range resume) plus
# aggressive retries, which claws through unstable connections. If curl is
# unavailable we fall back to hf_hub_download(resume_download=True).
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p data/longmemeval data/longmemeval_raw
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SPLITS=(longmemeval_s longmemeval_oracle)

download_with_curl() {
  local name="$1"
  local url="${HF_ENDPOINT}/datasets/xiaowu0162/longmemeval/resolve/main/${name}"
  local dst="data/longmemeval_raw/${name}"
  echo "  [${name}] curl resume <- ${url}"
  # -C -            : resume from existing bytes
  # --retry 999     : retry on transient errors
  # --retry-all-errors --retry-delay 3 : also retry on connection resets
  curl -L --fail -C - \
    --retry 999 --retry-delay 3 --retry-all-errors \
    --connect-timeout 30 \
    -o "${dst}" "${url}"
}

ok=1
if command -v curl >/dev/null 2>&1; then
  for name in "${SPLITS[@]}"; do
    download_with_curl "$name" || ok=0
  done
else
  ok=0
fi

# Fallback / validation + rename via Python (python3 on macOS, python elsewhere).
if command -v python3 >/dev/null 2>&1; then PY_BIN=python3; else PY_BIN=python; fi
HF_ENDPOINT="$HF_ENDPOINT" CURL_OK="$ok" "$PY_BIN" - <<'PY'
import os, shutil, pathlib, json, time
os.environ.setdefault("HF_ENDPOINT", os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))

out = pathlib.Path("data/longmemeval"); out.mkdir(parents=True, exist_ok=True)
raw = pathlib.Path("data/longmemeval_raw"); raw.mkdir(parents=True, exist_ok=True)
SPLITS = ["longmemeval_s", "longmemeval_oracle"]
curl_ok = os.environ.get("CURL_OK") == "1"

def hf_fetch(name):
    from huggingface_hub import hf_hub_download
    for attempt in range(1, 21):
        try:
            print(f"  [{name}] hf_hub_download attempt {attempt}/20")
            return hf_hub_download(
                repo_id="xiaowu0162/longmemeval", repo_type="dataset",
                filename=name, local_dir=str(raw), resume_download=True,
            )
        except Exception as e:
            print(f"    -> {type(e).__name__}: {e}")
            time.sleep(min(5 * attempt, 30))
    raise RuntimeError(f"giving up on {name}")

for name in SPLITS:
    fp = raw / name
    # If curl didn't run or produced an unparseable/partial file, use HF client.
    need_hf = not curl_ok or not fp.exists()
    if not need_hf:
        try:
            json.load(open(fp, "r", encoding="utf-8"))
        except Exception as e:
            print(f"  [{name}] curl output not valid JSON ({e}); retrying via HF client")
            need_hf = True
    if need_hf:
        fp = pathlib.Path(hf_fetch(name))

    dst = out / f"{name}.json"
    shutil.copy(fp, dst)
    data = json.load(open(dst, "r", encoding="utf-8"))
    print(f"  {name}.json: {len(data)} questions")

print("downloaded -> data/longmemeval/")
PY

echo "=== data/longmemeval/ ==="
ls -lh data/longmemeval/*.json
