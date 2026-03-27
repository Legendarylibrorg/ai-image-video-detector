#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-./data_best}"
OUT_DIR="${2:-./artifacts_ens}"
EPOCHS="${3:-18}"
RUN_METADATA_MEMBER="${RUN_METADATA_MEMBER:-1}"
METADATA_MEMBER_OUT="${METADATA_MEMBER_OUT:-$OUT_DIR/m5_metadata}"
METADATA_MEMBER_BASE_CKPT="${METADATA_MEMBER_BASE_CKPT:-$OUT_DIR/m1/best.safetensors}"
METADATA_MEMBER_EPOCHS="${METADATA_MEMBER_EPOCHS:-$EPOCHS}"

mkdir -p "$OUT_DIR"

common_args=()
if [[ "${TRAIN_NO_PRETRAINED_BACKBONE:-0}" == "1" ]]; then
  common_args+=(--no-pretrained-backbone)
fi
if [[ "${TRAIN_NO_COMPILE:-0}" == "1" ]]; then
  common_args+=(--no-compile)
fi
if [[ -n "${TRAIN_NUM_WORKERS:-}" ]]; then
  common_args+=(--num-workers "$TRAIN_NUM_WORKERS")
fi
if [[ -n "${TRAIN_PATIENCE:-}" ]]; then
  common_args+=(--patience "$TRAIN_PATIENCE")
fi
if [[ -n "${TRAIN_MIN_DELTA:-}" ]]; then
  common_args+=(--min-delta "$TRAIN_MIN_DELTA")
fi
if [[ -n "${TRAIN_DEGENERATE_PATIENCE:-}" ]]; then
  common_args+=(--degenerate-patience "$TRAIN_DEGENERATE_PATIENCE")
fi

# Diverse runs for stronger ensemble generalization.
aid-train --data "$DATA_DIR" --epochs "$EPOCHS" --batch-size 64 --img-size 256 --lr 2e-4 --loss focal --focal-gamma 2.0 --backbone tiny --grad-accum 1 --threshold-objective balanced --out "$OUT_DIR/m1" "${common_args[@]}"
aid-train --data "$DATA_DIR" --epochs "$EPOCHS" --batch-size 40 --img-size 320 --lr 1.5e-4 --loss focal --focal-gamma 1.8 --backbone effb0 --grad-accum 1 --threshold-objective balanced --out "$OUT_DIR/m2" "${common_args[@]}"
aid-train --data "$DATA_DIR" --epochs "$EPOCHS" --batch-size 20 --img-size 384 --lr 1e-4 --loss focal --focal-gamma 2.2 --backbone effb0 --grad-accum 2 --threshold-objective balanced --out "$OUT_DIR/m3" "${common_args[@]}"
aid-train --data "$DATA_DIR" --epochs "$EPOCHS" --batch-size 12 --img-size 320 --lr 8e-5 --loss focal --focal-gamma 2.2 --backbone effb2 --grad-accum 3 --threshold-objective balanced --out "$OUT_DIR/m4" "${common_args[@]}"

if [[ "$RUN_METADATA_MEMBER" == "1" ]]; then
  if [[ ! -f "$METADATA_MEMBER_BASE_CKPT" ]]; then
    echo "metadata_member_fail=missing_base_checkpoint path=$METADATA_MEMBER_BASE_CKPT" >&2
    exit 1
  fi
  aid-train \
    --data "$DATA_DIR" \
    --epochs "$METADATA_MEMBER_EPOCHS" \
    --batch-size 48 \
    --img-size 256 \
    --lr 8e-5 \
    --loss focal \
    --focal-gamma 2.0 \
    --backbone tiny \
    --grad-accum 1 \
    --threshold-objective balanced \
    --out "$METADATA_MEMBER_OUT" \
    --init-from "$METADATA_MEMBER_BASE_CKPT" \
    --use-metadata-features \
    "${common_args[@]}"
fi

echo "ensemble_models:"
ls -1 "$OUT_DIR"/m*/best.safetensors
