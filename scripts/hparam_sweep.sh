#!/usr/bin/env bash
set -euo pipefail

# Hyperparameter sweep: fixed grid only (no eval — configs are data, not shell).
DATA_DIR="${1:-./data_best}"
OUT_ROOT="${2:-./artifacts_sweep}"
EPOCHS="${EPOCHS:-14}"
mkdir -p "$OUT_ROOT"

idx=0
while read -r img bs lr loss gamma bb; do
  [[ -z "${img:-}" ]] && continue
  idx=$((idx + 1))
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
done <<'EOF'
256 64 2e-4 focal 2.0 tiny
320 64 1.5e-4 focal 2.0 effb0
384 32 1e-4 focal 1.8 effb0
320 24 8e-5 focal 2.2 effb2
EOF
