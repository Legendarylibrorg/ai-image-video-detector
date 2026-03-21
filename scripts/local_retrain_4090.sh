#!/usr/bin/env bash
set -euo pipefail

# Local retrain flow for pipeline-only mode.
# Optional:
#   PIPELINE_CMD="bash scripts/do.sh train-all" bash scripts/local_retrain_4090.sh

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

PIPELINE_CMD="${PIPELINE_CMD:-bash scripts/do.sh train-existing}"

eval "$PIPELINE_CMD"

gate_args=(
  --ens-out "${ENS_OUT:-./artifacts_ens}"
  --video-out "${VIDEO_ARTIFACTS_OUT:-./video_artifacts}"
  --min-image-auc "${GATE_MIN_IMAGE_AUC:-0.93}"
  --min-image-f1 "${GATE_MIN_IMAGE_F1:-0.90}"
  --min-video-acc "${GATE_MIN_VIDEO_ACC:-0.82}"
)

if [[ "${GATE_ALLOW_MISSING_VIDEO:-auto}" == "1" ]]; then
  gate_args+=(--skip-video)
elif [[ "${GATE_ALLOW_MISSING_VIDEO:-auto}" != "0" ]] && ! have_complete_video_training_data; then
  gate_args+=(--skip-video)
fi

python scripts/benchmark_gate.py "${gate_args[@]}"
