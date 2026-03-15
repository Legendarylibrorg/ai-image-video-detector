#!/usr/bin/env bash
set -euo pipefail

# Local retrain flow, separated from prod serve loop.
# Optional:
#   PAUSE_PROD=1 bash scripts/local_retrain_4090.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PAUSE_PROD="${PAUSE_PROD:-1}"
PIPELINE_CMD="${PIPELINE_CMD:-bash scripts/do.sh start-v2}"

if [[ "$PAUSE_PROD" == "1" ]]; then
  bash scripts/linux_service.sh pause || true
fi

cleanup() {
  if [[ "$PAUSE_PROD" == "1" ]]; then
    bash scripts/linux_service.sh resume || true
  fi
}
trap cleanup EXIT INT TERM

eval "$PIPELINE_CMD"

python scripts/benchmark_gate.py \
  --ens-out "${ENS_OUT:-./artifacts_ens}" \
  --video-out "${VIDEO_ARTIFACTS_OUT:-./video_artifacts}" \
  --min-image-auc "${GATE_MIN_IMAGE_AUC:-0.93}" \
  --min-image-f1 "${GATE_MIN_IMAGE_F1:-0.90}" \
  --min-video-acc "${GATE_MIN_VIDEO_ACC:-0.82}"
