#!/usr/bin/env bash
set -euo pipefail

# Production serve script for a 4090 host.
# Usage:
#   MODEL_GLOB="./artifacts_ens/m*/best.pt" PORT=8000 bash scripts/serve_prod_4090.sh

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
UNKNOWN_MARGIN="${UNKNOWN_MARGIN:-0.05}"
MAX_BYTES="${MAX_BYTES:-10485760}"
RATE_LIMIT="${RATE_LIMIT:-300}"
MODEL_GLOB="${MODEL_GLOB:-./artifacts_ens/m*/best.pt}"

source .venv/bin/activate

mapfile -t MODELS < <(ls $MODEL_GLOB)
if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "No model checkpoints found for pattern: $MODEL_GLOB"
  exit 1
fi

echo "Serving models: ${MODELS[*]}"

# One worker is intentional for single-GPU inference stability.
aid-serve \
  --model "${MODELS[@]}" \
  --host "$HOST" \
  --port "$PORT" \
  --unknown-margin "$UNKNOWN_MARGIN" \
  --max-bytes "$MAX_BYTES" \
  --rate-limit-per-min "$RATE_LIMIT"
