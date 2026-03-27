#!/usr/bin/env bash
set -euo pipefail

# Local retrain flow for pipeline-only mode.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
ENV_READY=0

source "$ROOT_DIR/scripts/lib/core.sh"
source "$ROOT_DIR/scripts/lib/training.sh"

bash scripts/do.sh train-existing

gate_args=(
  --ens-out "${ENS_OUT:-./artifacts_ens}"
  --video-out "${VIDEO_ARTIFACTS_OUT:-./video_artifacts}"
  --min-image-auc "${GATE_MIN_IMAGE_AUC:-0.96}"
  --min-image-f1 "${GATE_MIN_IMAGE_F1:-0.92}"
  --min-image-precision "${GATE_MIN_IMAGE_PRECISION:-0.90}"
  --min-image-recall "${GATE_MIN_IMAGE_RECALL:-0.90}"
  --max-image-ece "${GATE_MAX_IMAGE_ECE:-0.05}"
  --max-image-brier "${GATE_MAX_IMAGE_BRIER:-0.08}"
  --min-robust-worst-auc "${GATE_MIN_ROBUST_WORST_AUC:-0.90}"
  --min-robust-worst-f1 "${GATE_MIN_ROBUST_WORST_F1:-0.85}"
  --max-robust-auc-drop "${GATE_MAX_ROBUST_AUC_DROP:-0.08}"
  --min-video-acc "${GATE_MIN_VIDEO_ACC:-0.86}"
)

if [[ "${GATE_ALLOW_MISSING_VIDEO:-auto}" == "1" ]]; then
  gate_args+=(--skip-video)
elif [[ "${GATE_ALLOW_MISSING_VIDEO:-auto}" != "0" ]] && ! have_complete_video_training_data; then
  gate_args+=(--skip-video)
fi

run_repo_python scripts/benchmark_gate.py "${gate_args[@]}"
