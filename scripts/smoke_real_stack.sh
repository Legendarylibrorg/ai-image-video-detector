#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
WORKDIR="${SMOKE_REAL_WORKDIR:-$(mktemp -d "${TMPDIR:-/tmp}/aid-real-smoke.XXXXXX")}"
KEEP_WORKDIR="${SMOKE_REAL_KEEP_WORKDIR:-0}"
DATA_DIR="$WORKDIR/data"
OUT_DIR="$WORKDIR/artifacts"
CACHE_DIR="$WORKDIR/hf_cache"
SOURCE_ID="${SMOKE_REAL_SOURCE_ID:-dragonintelligence/CIFAKE-image-dataset}"
TRAIN_PER_CLASS="${SMOKE_REAL_TRAIN_PER_CLASS:-8}"
VAL_PER_CLASS="${SMOKE_REAL_VAL_PER_CLASS:-2}"
TEST_PER_CLASS="${SMOKE_REAL_TEST_PER_CLASS:-2}"
IMG_SIZE="${SMOKE_REAL_IMG_SIZE:-128}"
BATCH_SIZE="${SMOKE_REAL_BATCH_SIZE:-4}"
EPOCHS="${SMOKE_REAL_EPOCHS:-1}"
source "$ROOT_DIR/scripts/lib/env.sh"

cleanup() {
  if [[ "$KEEP_WORKDIR" == "1" ]]; then
    echo "smoke_real_keep_workdir=1 path=$WORKDIR"
    return
  fi
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

load_env_file

if [[ -z "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" ]]; then
  echo "smoke_real_status=missing_hf_token"
  echo "set HF_TOKEN in the environment or .env before running smoke-real" >&2
  exit 2
fi

bash scripts/install_deps.sh >&2
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("smoke_real_status=cuda_unavailable", file=sys.stderr)
    raise SystemExit(3)
name = torch.cuda.get_device_name(0)
print(f"smoke_real_cuda_device={name}")
PY

echo "smoke_real_status=collect source=$SOURCE_ID workdir=$WORKDIR"
python scripts/build_best_dataset.py \
  --out "$DATA_DIR" \
  --train-per-class "$TRAIN_PER_CLASS" \
  --val-per-class "$VAL_PER_CLASS" \
  --test-per-class "$TEST_PER_CLASS" \
  --min-side 64 \
  --max-aspect-ratio 8 \
  --min-entropy 0 \
  --near-hamming 1 \
  --near-window 256 \
  --max-per-source-class 32 \
  --max-per-source-split-class 16 \
  --jpeg-quality 90 \
  --hardneg-fraction 0 \
  --hf-only \
  --no-default-sources \
  --no-discover-hf \
  --extra-source "$SOURCE_ID" \
  --streaming \
  --cache-dir "$CACHE_DIR" \
  --stream-buffer-size 64 \
  --max-samples-per-source 96 \
  --acceptance-warmup-samples 8 \
  --min-acceptance-rate 0 \
  --repo-base-pause-ms 0 \
  --repo-jitter-ms 0 \
  --repo-cooldown-ms 0 \
  --max-consecutive-failures 4 \
  --quiet-progress \
  --require-full-targets

echo "smoke_real_status=train data=$DATA_DIR out=$OUT_DIR"
aid-train \
  --data "$DATA_DIR" \
  --out "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --img-size "$IMG_SIZE" \
  --backbone tiny \
  --lr 3e-4 \
  --num-workers 0 \
  --no-pretrained-backbone \
  --no-compile \
  --degenerate-patience 0 \
  --save-every 1

for f in \
  "$OUT_DIR/best.safetensors" \
  "$OUT_DIR/last.pt" \
  "$OUT_DIR/config.json" \
  "$OUT_DIR/test_metrics.json"; do
  test -f "$f"
done

echo "smoke_real_status=ok workdir=$WORKDIR"
