#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-./data_best}"
OUT_DIR="${2:-./artifacts_ens}"
EPOCHS="${3:-18}"

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

# Diverse runs for stronger ensemble generalization.
aid-train --data "$DATA_DIR" --epochs "$EPOCHS" --batch-size 64 --img-size 256 --lr 2e-4 --loss focal --focal-gamma 2.0 --backbone tiny --grad-accum 1 --threshold-objective balanced --out "$OUT_DIR/m1" "${common_args[@]}"
aid-train --data "$DATA_DIR" --epochs "$EPOCHS" --batch-size 40 --img-size 320 --lr 1.5e-4 --loss focal --focal-gamma 1.8 --backbone effb0 --grad-accum 1 --threshold-objective balanced --out "$OUT_DIR/m2" "${common_args[@]}"
aid-train --data "$DATA_DIR" --epochs "$EPOCHS" --batch-size 20 --img-size 384 --lr 1e-4 --loss focal --focal-gamma 2.2 --backbone effb0 --grad-accum 2 --threshold-objective balanced --out "$OUT_DIR/m3" "${common_args[@]}"
aid-train --data "$DATA_DIR" --epochs "$EPOCHS" --batch-size 12 --img-size 320 --lr 8e-5 --loss focal --focal-gamma 2.2 --backbone effb2 --grad-accum 3 --threshold-objective balanced --out "$OUT_DIR/m4" "${common_args[@]}"

echo "ensemble_models:"
ls -1 "$OUT_DIR"/m*/best.safetensors
