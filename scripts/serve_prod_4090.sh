#!/usr/bin/env bash
set -euo pipefail

# Production serve script for a 4090 host.
# Usage:
#   MODEL_GLOB="./artifacts_ens/m*/best.pt" ENSEMBLE_CONFIG="./artifacts_ens/ensemble_config.json" DOMAIN_CONFIG="./artifacts_ens/domain_config.json" FUSION_CONFIG="./artifacts_ens/fusion_config.json" PORT=8000 bash scripts/serve_prod_4090.sh

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
UNKNOWN_MARGIN="${UNKNOWN_MARGIN:-0.05}"
MAX_BYTES="${MAX_BYTES:-10485760}"
RATE_LIMIT="${RATE_LIMIT:-300}"
MODEL_GLOB="${MODEL_GLOB:-./artifacts_ens/m*/best.pt}"
ENSEMBLE_CONFIG="${ENSEMBLE_CONFIG:-./artifacts_ens/ensemble_config.json}"
DOMAIN_CONFIG="${DOMAIN_CONFIG:-./artifacts_ens/domain_config.json}"
FUSION_CONFIG="${FUSION_CONFIG:-./artifacts_ens/fusion_config.json}"
TOOLS_CONFIG="${TOOLS_CONFIG:-./artifacts_ens/tools_config.json}"
TTA_VIEWS="${TTA_VIEWS:-2}"
UNCERTAIN_CAPTURE="${UNCERTAIN_CAPTURE:-0}"
UNCERTAIN_DIR="${UNCERTAIN_DIR:-./incoming_review_queue}"
UNCERTAIN_CAPTURE_MARGIN="${UNCERTAIN_CAPTURE_MARGIN:-0.05}"
UNCERTAIN_CAPTURE_RISK="${UNCERTAIN_CAPTURE_RISK:-0.85}"
IP_LOG_MODE="${IP_LOG_MODE:-masked}"
IP_LOG_SALT="${IP_LOG_SALT:-}"

source .venv/bin/activate

mapfile -t MODELS < <(ls $MODEL_GLOB)
if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "No model checkpoints found for pattern: $MODEL_GLOB"
  exit 1
fi

echo "Serving models: ${MODELS[*]}"

extra_ensemble_args=()
if [[ -f "$ENSEMBLE_CONFIG" ]]; then
  extra_ensemble_args=(--ensemble-config "$ENSEMBLE_CONFIG")
  echo "Using ensemble config: $ENSEMBLE_CONFIG"
fi

extra_domain_args=()
if [[ -f "$DOMAIN_CONFIG" ]]; then
  extra_domain_args=(--domain-config "$DOMAIN_CONFIG")
  echo "Using domain config: $DOMAIN_CONFIG"
fi

extra_fusion_args=()
if [[ -f "$FUSION_CONFIG" ]]; then
  extra_fusion_args=(--fusion-config "$FUSION_CONFIG")
  echo "Using fusion config: $FUSION_CONFIG"
fi

extra_tools_args=()
if [[ -f "$TOOLS_CONFIG" ]]; then
  extra_tools_args=(--tools-config "$TOOLS_CONFIG")
  echo "Using tools config: $TOOLS_CONFIG"
fi

# One worker is intentional for single-GPU inference stability.
aid-serve \
  --model "${MODELS[@]}" \
  "${extra_ensemble_args[@]}" \
  "${extra_domain_args[@]}" \
  "${extra_fusion_args[@]}" \
  "${extra_tools_args[@]}" \
  --tta-views "$TTA_VIEWS" \
  $([[ "$UNCERTAIN_CAPTURE" == "1" ]] && echo "--uncertain-capture") \
  --uncertain-dir "$UNCERTAIN_DIR" \
  --uncertain-capture-margin "$UNCERTAIN_CAPTURE_MARGIN" \
  --uncertain-capture-risk "$UNCERTAIN_CAPTURE_RISK" \
  --ip-log-mode "$IP_LOG_MODE" \
  --ip-log-salt "$IP_LOG_SALT" \
  --host "$HOST" \
  --port "$PORT" \
  --unknown-margin "$UNKNOWN_MARGIN" \
  --max-bytes "$MAX_BYTES" \
  --rate-limit-per-min "$RATE_LIMIT"
