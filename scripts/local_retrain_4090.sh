#!/usr/bin/env bash
set -euo pipefail

# Local retrain flow for pipeline-only mode.
# Optional:
#   PIPELINE_MODE=train-all bash scripts/local_retrain_4090.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

video_bucket_has_files() {
  local dir="$1"
  [[ -d "$dir" ]] || return 1
  local first_match=""
  first_match="$(find "$dir" -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.mov" -o -iname "*.avi" -o -iname "*.mkv" -o -iname "*.webm" -o -iname "*.m4v" \) -print -quit)"
  [[ -n "$first_match" ]]
}

have_complete_video_training_data() {
  local root="${VIDEO_OUT:-./video_data}"
  local split=""
  local cls=""
  for split in train val; do
    for cls in ai real; do
      if ! video_bucket_has_files "$root/$split/$cls"; then
        return 1
      fi
    done
  done
  return 0
}

PIPELINE_MODE="${PIPELINE_MODE:-}"
PIPELINE_CMD="${PIPELINE_CMD:-}"
pipeline_args=()

if [[ -n "$PIPELINE_MODE" ]]; then
  case "$PIPELINE_MODE" in
    train-existing|train|train-all)
      pipeline_args=(bash scripts/do.sh "$PIPELINE_MODE")
      ;;
    *)
      echo "unsupported_pipeline_mode=$PIPELINE_MODE allowed=train-existing|train|train-all" >&2
      exit 2
      ;;
  esac
elif [[ -n "$PIPELINE_CMD" ]]; then
  case "$PIPELINE_CMD" in
    "bash scripts/do.sh train-existing"|\
    "bash scripts/do.sh train"|\
    "bash scripts/do.sh train-all")
      read -r -a pipeline_args <<< "$PIPELINE_CMD"
      ;;
    *)
      echo "unsupported_pipeline_cmd=$PIPELINE_CMD use PIPELINE_MODE=train-existing|train|train-all" >&2
      exit 2
      ;;
  esac
else
  pipeline_args=(bash scripts/do.sh train-existing)
fi

"${pipeline_args[@]}"

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

python scripts/benchmark_gate.py "${gate_args[@]}"
