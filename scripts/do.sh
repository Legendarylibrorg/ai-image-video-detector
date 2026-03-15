#!/usr/bin/env bash
set -euo pipefail

# Minimal command surface for everyday use.
# Examples:
#   bash scripts/do.sh start
#   bash scripts/do.sh start-v2
#   bash scripts/do.sh collect
#   bash scripts/do.sh collect-diverse
#   bash scripts/do.sh collect-image
#   bash scripts/do.sh collect-video
#   bash scripts/do.sh ingest
#   bash scripts/do.sh train
#   bash scripts/do.sh train-all-types
#   bash scripts/do.sh serve
#   bash scripts/do.sh detect ./example.jpg

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
TRAIN_LOCK="${TRAIN_LOCK:-$ROOT_DIR/.local/training.lock}"
ENV_READY=0

ensure_env() {
  if [[ "$ENV_READY" == "1" ]]; then
    return
  fi
  if [[ ! -d .venv ]]; then
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -e . datasets huggingface_hub safetensors
  ENV_READY=1
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

skip_collection_if_training() {
  if is_training_active; then
    echo "collection skipped because training is active (lock: $TRAIN_LOCK)."
    return 0
  fi
  return 1
}

collect_image_data() {
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
    --hf-cache-only-if-present \
    --cache-dir "${BEST_DS_CACHE_DIR:-./.local/hf}" \
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
}

ingest_outputs() {
  ensure_env
  python scripts/ingest_model_outputs.py \
    --src "${MODEL_OUTPUTS_SRC:-./incoming_model_outputs}" \
    --dst "${NEW_DATA_DST:-./data_new/train}" \
    --archive "${MODEL_OUTPUTS_ARCHIVE:-./incoming_model_outputs/_processed}"
}

collect_diverse_image_data() {
  mapfile -t diverse_query_args < <(add_diverse_queries)
  ensure_env
  local hf_cache="${DIVERSE_HF_CACHE_FILE:-./.local/hf_diverse_sources.txt}"
  local timeout_sec="${DIVERSE_DISCOVERY_TIMEOUT_SEC:-900}"
  local -a common_args=(
    --out "${DATA_DIR:-./data_best}"
    --train-per-class "${DIVERSE_TRAIN_PER_CLASS:-100000}"
    --val-per-class "${DIVERSE_VAL_PER_CLASS:-25000}"
    --test-per-class "${DIVERSE_TEST_PER_CLASS:-25000}"
    --hf-cache-file "$hf_cache"
    --hf-cache-only-if-present
    --cache-dir "${DIVERSE_CACHE_DIR:-./.local/hf}"
    --streaming
    --stream-buffer-size "${DIVERSE_STREAM_BUFFER_SIZE:-16000}"
    --max-samples-per-source "${DIVERSE_MAX_SAMPLES_PER_SOURCE:-80000}"
    --repo-base-pause-ms "${DIVERSE_REPO_BASE_PAUSE_MS:-1400}"
    --repo-jitter-ms "${DIVERSE_REPO_JITTER_MS:-1200}"
    --repo-cooldown-ms "${DIVERSE_REPO_COOLDOWN_MS:-45000}"
    --max-consecutive-failures "${DIVERSE_MAX_CONSECUTIVE_FAILURES:-2}"
    --min-side "${DIVERSE_MIN_SIDE:-192}"
    --max-aspect-ratio "${DIVERSE_MAX_ASPECT_RATIO:-3.2}"
    --min-entropy "${DIVERSE_MIN_ENTROPY:-3.1}"
    --hardneg-fraction "${DIVERSE_HARDNEG_FRACTION:-1.0}"
    --local-source "${DIVERSE_LOCAL_SOURCES:-./data}"
    --local-source "${DIVERSE_LOCAL_NEW_SOURCES:-./data_new/train}"
  )
  local -a discover_args=(
    --discover-hf
    --hf-discovery-limit "${DIVERSE_HF_DISCOVERY_LIMIT:-140}"
    --hf-max-sources "${DIVERSE_HF_MAX_SOURCES:-320}"
  )
  local -a cache_args=(--no-discover-hf --sources-file "$hf_cache")
  local -a full_args=("${common_args[@]}")

  # Run discovery as a bounded pre-pass, then always do full build from cache/local sources.
  if [[ "${DIVERSE_SKIP_DISCOVERY:-0}" != "1" ]]; then
    if command -v timeout >/dev/null 2>&1; then
      if ! timeout "${timeout_sec}s" python scripts/build_best_dataset.py "${common_args[@]}" "${discover_args[@]}" "${diverse_query_args[@]}" --discover-only; then
        echo "collect_diverse_discovery=failed_or_timed_out fallback=cache_or_local"
      fi
    else
      if ! python scripts/build_best_dataset.py "${common_args[@]}" "${discover_args[@]}" "${diverse_query_args[@]}" --discover-only; then
        echo "collect_diverse_discovery=failed fallback=cache_or_local"
      fi
    fi
  fi

  if [[ -s "$hf_cache" ]]; then
    full_args+=("${cache_args[@]}")
  else
    full_args+=(--no-discover-hf)
    echo "collect_diverse_sources=local_only reason=hf_cache_missing"
  fi
  python scripts/build_best_dataset.py "${full_args[@]}" "${diverse_query_args[@]}"

  python scripts/audit_diversity.py \
    --data "${DATA_DIR:-./data_best}" \
    --min-unique-sources "${DIVERSE_MIN_UNIQUE_SOURCES:-20}" \
    --min-hardneg-modes "${DIVERSE_MIN_HARDNEG_MODES:-4}" \
    --max-class-imbalance "${DIVERSE_MAX_CLASS_IMBALANCE:-0.08}" || true
}

add_diverse_queries() {
  local query_csv="${DIVERSE_HF_QUERIES:-real camera photo dataset,smartphone photo dataset,dslr photo dataset,webcam image dataset,cctv frame image dataset,meme image real vs ai,captioned image real ai,screenshot dataset image,chat ui screenshot,browser screenshot image,dashboard screenshot dataset,image poster infographic,logo brand image dataset,advertisement creative image,receipt scanned document image,id card document image,invoice form document scan,anime illustration real fake,digital art illustration dataset,3d render real fake,cgi synthetic image real,game render frame dataset,watermarked social media image,recompressed image dataset,heavily edited real photo,low resolution blurry image,extreme aspect ratio image,portrait selfie real fake,group photo real fake,deepfake face swap image,diffusion generated image latest}"
  IFS=',' read -r -a _queries <<< "$query_csv"
  local args=()
  for q in "${_queries[@]}"; do
    q="$(echo "$q" | xargs)"
    [[ -z "$q" ]] && continue
    args+=(--hf-query "$q")
  done
  printf "%s\n" "${args[@]}"
}

collect_video_data() {
  ensure_env
  python scripts/build_video_dataset.py \
    --out "${VIDEO_OUT:-./video_data}" \
    --train-per-class "${VIDEO_TRAIN_PER_CLASS:-1000}" \
    --val-per-class "${VIDEO_VAL_PER_CLASS:-250}" \
    --mode "${VIDEO_MODE:-snapshot}" \
    --cache-dir "${VIDEO_CACHE_DIR:-./.local/hf}" \
    --snapshot-max-workers "${VIDEO_SNAPSHOT_MAX_WORKERS:-1}" \
    --repo-base-pause-ms "${VIDEO_REPO_BASE_PAUSE_MS:-900}" \
    --repo-jitter-ms "${VIDEO_REPO_JITTER_MS:-600}" \
    --copy-sleep-ms "${VIDEO_COPY_SLEEP_MS:-0}" \
    --sleep-ms "${VIDEO_SLEEP_MS:-120}" \
    --jitter-ms "${VIDEO_JITTER_MS:-80}" \
    --chunk-pause-ms "${VIDEO_CHUNK_PAUSE_MS:-1000}" \
    --repo-cooldown-ms "${VIDEO_REPO_COOLDOWN_MS:-3000}" \
    --retries "${VIDEO_RETRIES:-5}"
}

train_image_pipeline() {
  env SKIP_DATA=1 RUN_VIDEO_DATA_PULL=0 RUN_VIDEO_TRAIN=0 bash scripts/max_quality_4090.sh
}

train_all_pipeline() {
  env SKIP_DATA=1 RUN_VIDEO_DATA_PULL=0 bash scripts/max_quality_4090.sh
}

validate_train_artifacts() {
  local ens_dir="${ENS_OUT:-./artifacts_ens}"
  local vid_best="${VIDEO_ARTIFACTS_OUT:-./video_artifacts}/best_video.pt"
  local missing=0

  for p in "$ens_dir/m1/best.pt" "$ens_dir/m2/best.pt" "$ens_dir/m3/best.pt" "$ens_dir/m4/best.pt" "$ens_dir/test_metrics.json" "$ens_dir/prod_manifest.json"; do
    if [[ ! -f "$p" ]]; then
      echo "missing_artifact=$p"
      missing=1
    fi
  done
  if [[ ! -f "$vid_best" ]]; then
    echo "missing_artifact=$vid_best"
    missing=1
  fi

  if [[ "$missing" == "1" ]]; then
    echo "artifact_validation=failed"
    exit 1
  fi
  echo "artifact_validation=ok"
}

train_video_only() {
  ensure_env
  resume_arg=()
  if [[ "${VIDEO_TRAIN_RESUME:-1}" == "1" ]]; then
    resume_arg=(--resume "${VIDEO_ARTIFACTS_OUT:-./video_artifacts}/last_video.pt")
  fi
  aid-video-train \
    --data "${VIDEO_OUT:-./video_data}" \
    --out "${VIDEO_ARTIFACTS_OUT:-./video_artifacts}" \
    --epochs "${VIDEO_TRAIN_EPOCHS:-30}" \
    --batch-size "${VIDEO_TRAIN_BATCH_SIZE:-4}" \
    --img-size "${VIDEO_TRAIN_IMG_SIZE:-224}" \
    --frames "${VIDEO_TRAIN_FRAMES:-24}" \
    --grad-accum "${VIDEO_TRAIN_GRAD_ACCUM:-2}" \
    --lr "${VIDEO_TRAIN_LR:-1e-4}" \
    --patience "${VIDEO_TRAIN_PATIENCE:-6}" \
    --min-delta "${VIDEO_TRAIN_MIN_DELTA:-0.001}" \
    "${resume_arg[@]}"
}

show_status() {
  if is_training_active; then
    echo "training: active (lock: $TRAIN_LOCK)"
  else
    echo "training: idle"
  fi
  echo "image data: ${DATA_DIR:-./data_best}"
  echo "video data: ${VIDEO_OUT:-./video_data}"
  echo "image ensemble: ${ENS_OUT:-./artifacts_ens}"
  echo "video model: ${VIDEO_ARTIFACTS_OUT:-./video_artifacts}/best_video.pt"
}

cmd="${1:-start}"
shift || true

case "$cmd" in
  start)
    # Full, best-quality pipeline.
    with_training_lock bash scripts/max_quality_4090.sh
    ;;

  start-v2)
    # Max-accuracy v2 pipeline with domain calibration + refinement loops.
    with_training_lock bash scripts/max_accuracy_v2.sh
    ;;

  collect)
    # Full collection cycle: image pull + new-output ingest + video pull.
    if skip_collection_if_training; then
      exit 0
    fi
    collect_image_data
    ingest_outputs
    collect_video_data
    ;;

  collect-diverse)
    if skip_collection_if_training; then
      exit 0
    fi
    collect_diverse_image_data
    ingest_outputs
    VIDEO_CACHE_DIR="${VIDEO_CACHE_DIR:-./.local/hf}" VIDEO_SNAPSHOT_MAX_WORKERS="${VIDEO_SNAPSHOT_MAX_WORKERS:-1}" collect_video_data
    ;;

  collect-image)
    if skip_collection_if_training; then
      exit 0
    fi
    collect_image_data
    ;;

  collect-video)
    if skip_collection_if_training; then
      exit 0
    fi
    collect_video_data
    ;;

  ingest)
    if skip_collection_if_training; then
      exit 0
    fi
    ingest_outputs
    ;;

  train|train-image)
    # Image pipeline only, assumes data already collected.
    with_training_lock train_image_pipeline
    ;;

  train-video)
    with_training_lock train_video_only
    ;;

  train-all)
    # Image + video training, assumes data already collected.
    with_training_lock train_all_pipeline
    ;;

  train-all-types)
    # End-to-end: broad collection + image/video training + artifact checks.
    if skip_collection_if_training; then
      exit 1
    fi
    "$0" collect-diverse
    with_training_lock train_all_pipeline
    validate_train_artifacts
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
    domain_extra=()
    if [[ -f "${DOMAIN_CONFIG:-./artifacts_ens/domain_config.json}" ]]; then
      domain_extra=(--domain-config "${DOMAIN_CONFIG:-./artifacts_ens/domain_config.json}")
    fi
    tools_extra=()
    if [[ -f "${TOOLS_CONFIG:-./artifacts_ens/tools_config.json}" ]]; then
      tools_extra=(--tools-config "${TOOLS_CONFIG:-./artifacts_ens/tools_config.json}")
    fi
    aid-detect --model "${detect_models[@]}" "${detect_extra[@]}" "${domain_extra[@]}" "${tools_extra[@]}" --tta-views "${TTA_VIEWS:-2}" --image "$image_path" --json
    ;;

  status)
    show_status
    ;;

  *)
    echo "usage: bash scripts/do.sh [start|start-v2|collect|collect-diverse|collect-image|collect-video|ingest|train|train-video|train-all|train-all-types|autocollect|serve|detect <image>|status]"
    exit 2
    ;;
esac
