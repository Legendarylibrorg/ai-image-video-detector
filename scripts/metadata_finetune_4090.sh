#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/core.sh"
source "$ROOT_DIR/scripts/lib/training.sh"

prepare_training_image_data

OUT_DIR="${METADATA_FINETUNE_OUT:-./artifacts_finetune_metadata}"
BASE_CKPT="${METADATA_FINETUNE_BASE_CKPT:-}"
BASE_CKPT_SEARCH_ROOT="${METADATA_FINETUNE_SEARCH_ROOT:-./artifacts_ens}"
if [[ -z "$BASE_CKPT" ]]; then
  shopt -s nullglob
  declare -a metadata_candidates=("$BASE_CKPT_SEARCH_ROOT"/m*/best.safetensors)
  shopt -u nullglob
  for candidate in "${metadata_candidates[@]}"; do
    [[ -f "$candidate" ]] || continue
    BASE_CKPT="$candidate"
    break
  done
fi

if [[ -z "$BASE_CKPT" || ! -f "$BASE_CKPT" ]]; then
  echo "metadata_finetune_fail=missing_base_checkpoint searched=$BASE_CKPT_SEARCH_ROOT/m*/best.safetensors" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

aid-train \
  --data "$PREPARED_IMAGE_DATA_DIR" \
  --out "$OUT_DIR" \
  --epochs "${METADATA_FINETUNE_EPOCHS:-6}" \
  --batch-size "${METADATA_FINETUNE_BATCH_SIZE:-32}" \
  --img-size "${METADATA_FINETUNE_IMG_SIZE:-256}" \
  --lr "${METADATA_FINETUNE_LR:-8e-5}" \
  --loss focal \
  --focal-gamma "${METADATA_FINETUNE_FOCAL_GAMMA:-2.0}" \
  --backbone "${METADATA_FINETUNE_BACKBONE:-tiny}" \
  --threshold-objective balanced \
  --patience "${METADATA_FINETUNE_PATIENCE:-3}" \
  --min-delta "${METADATA_FINETUNE_MIN_DELTA:-0.001}" \
  --degenerate-patience "${METADATA_FINETUNE_DEGENERATE_PATIENCE:-2}" \
  --init-from "$BASE_CKPT" \
  --use-metadata-features \
  "$@"

echo "metadata_finetune_out=$OUT_DIR"
echo "metadata_finetune_base=$BASE_CKPT"
