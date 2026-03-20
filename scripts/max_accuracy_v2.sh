#!/usr/bin/env bash
set -euo pipefail

# Max-accuracy v2 pipeline:
# 1) Broad diverse collection
# 2) Image+video training
# 3) Domain threshold calibration
# 4) Optional hard-negative refresh retrain loop
#
# Usage:
#   bash scripts/max_accuracy_v2.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REFINE_LOOPS="${REFINE_LOOPS:-1}"
ENS_OUT="${ENS_OUT:-./artifacts_ens}"
DATA_DIR="${DATA_DIR:-./data_best}"
DOMAIN_CONFIG="${DOMAIN_CONFIG:-$ENS_OUT/domain_config.json}"
HARD_TOPK="${HARD_TOPK:-22000}"

bash scripts/do.sh collect-diverse
bash scripts/do.sh train-all

python scripts/fit_domain_thresholds.py \
  --data "$DATA_DIR" \
  --model "$ENS_OUT"/m1/best.safetensors "$ENS_OUT"/m2/best.safetensors "$ENS_OUT"/m3/best.safetensors "$ENS_OUT"/m4/best.safetensors \
  --ensemble-config "$ENS_OUT/ensemble_config.json" \
  --out "$DOMAIN_CONFIG" \
  --objective balanced

for ((i=1; i<=REFINE_LOOPS; i++)); do
  python scripts/mine_hard_negatives.py \
    --data "$DATA_DIR" \
    --model "$ENS_OUT"/m1/best.safetensors "$ENS_OUT"/m2/best.safetensors "$ENS_OUT"/m3/best.safetensors "$ENS_OUT"/m4/best.safetensors \
    --ensemble-config "$ENS_OUT/ensemble_config.json" \
    --out "$ENS_OUT/hard_mined_v2_loop$i" \
    --top-k "$HARD_TOPK"

  BEST_DS_HF_ONLY=1 bash scripts/do.sh collect-image
  bash scripts/do.sh train-all

  python scripts/fit_domain_thresholds.py \
    --data "$DATA_DIR" \
    --model "$ENS_OUT"/m1/best.safetensors "$ENS_OUT"/m2/best.safetensors "$ENS_OUT"/m3/best.safetensors "$ENS_OUT"/m4/best.safetensors \
    --ensemble-config "$ENS_OUT/ensemble_config.json" \
    --out "$DOMAIN_CONFIG" \
    --objective balanced
done

echo "max_accuracy_v2_complete domain_config=$DOMAIN_CONFIG"
