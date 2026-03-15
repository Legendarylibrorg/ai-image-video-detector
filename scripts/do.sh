#!/usr/bin/env bash
set -euo pipefail

# Minimal command surface for everyday use.
# Examples:
#   bash scripts/do.sh start
#   bash scripts/do.sh collect
#   bash scripts/do.sh train
#   bash scripts/do.sh serve
#   bash scripts/do.sh detect ./example.jpg

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
TRAIN_LOCK="${TRAIN_LOCK:-$ROOT_DIR/.local/training.lock}"

ensure_env() {
  if [[ ! -d .venv ]]; then
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -e . datasets
}

is_training_active() {
  [[ -f "$TRAIN_LOCK" ]]
}

with_training_lock() {
  mkdir -p "$(dirname "$TRAIN_LOCK")"
  if is_training_active; then
    echo "training lock active path=$TRAIN_LOCK"
    exit 1
  fi
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$TRAIN_LOCK"
  trap 'rm -f "$TRAIN_LOCK"' EXIT INT TERM
  "$@"
}

cmd="${1:-start}"
shift || true

case "$cmd" in
  start)
    # Full, best-quality pipeline.
    with_training_lock bash scripts/max_quality_4090.sh
    ;;

  collect)
    # Data collection only (no model training).
    if is_training_active; then
      echo "collection skipped because training is active (lock: $TRAIN_LOCK)."
      exit 0
    fi
    ensure_env
    python scripts/build_best_dataset.py \
      --out "${DATA_DIR:-./data_best}" \
      --train-per-class "${TRAIN_PER_CLASS:-80000}" \
      --val-per-class "${VAL_PER_CLASS:-20000}" \
      --test-per-class "${TEST_PER_CLASS:-20000}" \
      --discover-hf \
      --hf-discovery-limit "${BEST_DS_HF_DISCOVERY_LIMIT:-90}" \
      --hf-max-sources "${BEST_DS_HF_MAX_SOURCES:-180}" \
      --hf-cache-file "${BEST_DS_HF_CACHE_FILE:-./.local/hf_discovered_sources.txt}" \
      --streaming \
      --stream-buffer-size "${BEST_DS_STREAM_BUFFER_SIZE:-12000}" \
      --max-samples-per-source "${BEST_DS_MAX_SAMPLES_PER_SOURCE:-45000}" \
      --repo-base-pause-ms "${BEST_DS_REPO_BASE_PAUSE_MS:-1100}" \
      --repo-jitter-ms "${BEST_DS_REPO_JITTER_MS:-900}" \
      --repo-cooldown-ms "${BEST_DS_REPO_COOLDOWN_MS:-45000}" \
      --max-consecutive-failures "${BEST_DS_MAX_CONSECUTIVE_FAILURES:-2}" \
      --min-side "${BEST_DS_MIN_SIDE:-224}" \
      --max-aspect-ratio "${BEST_DS_MAX_ASPECT_RATIO:-2.5}" \
      --min-entropy "${BEST_DS_MIN_ENTROPY:-3.4}" \
      --hardneg-fraction "${BEST_DS_HARDNEG_FRACTION:-0.8}" \
      --local-source "${BEST_DS_LOCAL_SOURCES:-./data}"

    python scripts/ingest_model_outputs.py \
      --src "${MODEL_OUTPUTS_SRC:-./incoming_model_outputs}" \
      --dst "${NEW_DATA_DST:-./data_new/train}" \
      --archive "${MODEL_OUTPUTS_ARCHIVE:-./incoming_model_outputs/_processed}"

    python scripts/build_video_dataset.py \
      --out "${VIDEO_OUT:-./video_data}" \
      --train-per-class "${VIDEO_TRAIN_PER_CLASS:-1000}" \
      --val-per-class "${VIDEO_VAL_PER_CLASS:-250}" \
      --mode "${VIDEO_MODE:-snapshot}" \
      --snapshot-max-workers "${VIDEO_SNAPSHOT_MAX_WORKERS:-2}" \
      --repo-base-pause-ms "${VIDEO_REPO_BASE_PAUSE_MS:-900}" \
      --repo-jitter-ms "${VIDEO_REPO_JITTER_MS:-600}" \
      --copy-sleep-ms "${VIDEO_COPY_SLEEP_MS:-0}" \
      --sleep-ms "${VIDEO_SLEEP_MS:-120}" \
      --jitter-ms "${VIDEO_JITTER_MS:-80}" \
      --chunk-pause-ms "${VIDEO_CHUNK_PAUSE_MS:-1000}" \
      --repo-cooldown-ms "${VIDEO_REPO_COOLDOWN_MS:-3000}" \
      --retries "${VIDEO_RETRIES:-5}"
    ;;

  train)
    # Training only, assumes data already collected.
    with_training_lock env SKIP_DATA=1 RUN_VIDEO_DATA_PULL=0 bash scripts/max_quality_4090.sh
    ;;

  autocollect)
    bash scripts/continuous_collect.sh "$@"
    ;;

  serve)
    bash scripts/serve_prod_4090.sh
    ;;

  detect)
    image_path="${1:-}"
    if [[ -z "$image_path" ]]; then
      echo "usage: bash scripts/do.sh detect /path/to/image.jpg"
      exit 2
    fi
    ensure_env
    mapfile -t detect_models < <(ls ${MODEL_GLOB:-./artifacts_ens/m*/best.pt} 2>/dev/null || true)
    if [[ "${#detect_models[@]}" -eq 0 ]]; then
      detect_models=("${MODEL_PATH:-./artifacts/best.pt}")
    fi
    detect_extra=()
    if [[ -f "${ENSEMBLE_CONFIG:-./artifacts_ens/ensemble_config.json}" ]]; then
      detect_extra=(--ensemble-config "${ENSEMBLE_CONFIG:-./artifacts_ens/ensemble_config.json}")
    fi
    aid-detect --model "${detect_models[@]}" "${detect_extra[@]}" --image "$image_path" --json
    ;;

  *)
    echo "usage: bash scripts/do.sh [start|collect|train|autocollect|serve|detect <image>]"
    exit 2
    ;;
esac
