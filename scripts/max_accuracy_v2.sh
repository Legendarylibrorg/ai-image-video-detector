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
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
ENV_READY=0

source "$ROOT_DIR/scripts/lib/core.sh"

load_env_file

REFINE_LOOPS="${REFINE_LOOPS:-1}"
ENS_OUT="${ENS_OUT:-./artifacts_ens}"
DATA_DIR="${DATA_DIR:-./data_best}"
DOMAIN_CONFIG="${DOMAIN_CONFIG:-$ENS_OUT/domain_config.json}"
HARD_TOPK="${HARD_TOPK:-22000}"

declare -a ENSEMBLE_MODELS=()

collect_ensemble_model_paths() {
  ENSEMBLE_MODELS=()
  shopt -s nullglob
  local model_dir=""
  for model_dir in "$ENS_OUT"/m*; do
    [[ -d "$model_dir" ]] || continue
    if [[ -f "$model_dir/best.safetensors" ]]; then
      ENSEMBLE_MODELS+=("$model_dir/best.safetensors")
    fi
  done
  shopt -u nullglob
  if (( ${#ENSEMBLE_MODELS[@]} < 4 )); then
    echo "max_accuracy_v2_fail=ensemble_model_count have=${#ENSEMBLE_MODELS[@]} need=4 ens_out=$ENS_OUT" >&2
    exit 1
  fi
}

fit_domain_thresholds() {
  collect_ensemble_model_paths
  run_repo_python scripts/fit_domain_thresholds.py \
    --data "$DATA_DIR" \
    --model "${ENSEMBLE_MODELS[@]}" \
    --ensemble-config "$ENS_OUT/ensemble_config.json" \
    --out "$DOMAIN_CONFIG" \
    --objective balanced
}

mine_hard_negatives() {
  collect_ensemble_model_paths
  run_repo_python scripts/mine_hard_negatives.py \
    --data "$DATA_DIR" \
    --model "${ENSEMBLE_MODELS[@]}" \
    --ensemble-config "$ENS_OUT/ensemble_config.json" \
    --out "$ENS_OUT/hard_mined_v2_loop$1" \
    --top-k "$HARD_TOPK"
}

bash scripts/do.sh collect-diverse
bash scripts/do.sh train-all
fit_domain_thresholds

for ((i=1; i<=REFINE_LOOPS; i++)); do
  mine_hard_negatives "$i"

  BEST_DS_HF_ONLY=1 bash scripts/do.sh collect-image
  bash scripts/do.sh train-all

  fit_domain_thresholds
done

echo "max_accuracy_v2_complete domain_config=$DOMAIN_CONFIG models=${#ENSEMBLE_MODELS[@]}"
