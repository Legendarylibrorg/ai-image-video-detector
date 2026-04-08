#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-./data_best}"
OUT_ROOT="${2:-./artifacts_sweep}"
EPOCHS="${EPOCHS:-14}"
mkdir -p "$OUT_ROOT"

declare -a configs=(
  "img=256 bs=64 lr=2e-4 loss=focal gamma=2.0 bb=tiny"
  "img=320 bs=64 lr=1.5e-4 loss=focal gamma=2.0 bb=effb0"
  "img=384 bs=32 lr=1e-4 loss=focal gamma=1.8 bb=effb0"
  "img=320 bs=24 lr=8e-5 loss=focal gamma=2.2 bb=effb2"
)

idx=0
for cfg in "${configs[@]}"; do
  idx=$((idx+1))
  eval "$cfg"
  out="$OUT_ROOT/run_${idx}_img${img}_bs${bs}_lr${lr}_${loss}_${bb}"
  mkdir -p "$out"
  aid-train \
    --data "$DATA_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$bs" \
    --img-size "$img" \
    --lr "$lr" \
    --loss "$loss" \
    --focal-gamma "$gamma" \
    --backbone "$bb" \
    --threshold-objective balanced \
    --out "$out"
  echo "finished $out"
done
