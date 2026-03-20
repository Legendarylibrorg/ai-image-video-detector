#!/usr/bin/env bash
set -euo pipefail

# Local retrain flow for pipeline-only mode.
# Optional:
#   PIPELINE_CMD="bash scripts/do.sh start-v2" bash scripts/local_retrain_4090.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PIPELINE_CMD="${PIPELINE_CMD:-bash scripts/do.sh start-v2}"

eval "$PIPELINE_CMD"

python scripts/benchmark_gate.py \
  --ens-out "${ENS_OUT:-./artifacts_ens}" \
  --video-out "${VIDEO_ARTIFACTS_OUT:-./video_artifacts}" \
  --min-image-auc "${GATE_MIN_IMAGE_AUC:-0.93}" \
  --min-image-f1 "${GATE_MIN_IMAGE_F1:-0.90}" \
  --min-video-acc "${GATE_MIN_VIDEO_ACC:-0.82}"
