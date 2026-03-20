#!/usr/bin/env bash
set -euo pipefail

# Minimal command surface for everyday use.
# Examples:
#   bash scripts/do.sh start
#   bash scripts/do.sh start-v2
#   bash scripts/do.sh collect
#   bash scripts/do.sh collect-diverse
#   bash scripts/do.sh collect-fast
#   bash scripts/do.sh collect-image
#   bash scripts/do.sh collect-video
#   bash scripts/do.sh ingest
#   bash scripts/do.sh scan
#   bash scripts/do.sh train
#   bash scripts/do.sh train-all-types
#   bash scripts/do.sh deps-update
#   bash scripts/do.sh doctor
#   bash scripts/do.sh detect ./example.jpg

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
TRAIN_LOCK="${TRAIN_LOCK:-$ROOT_DIR/.local/training.lock}"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
ENV_READY=0
BEST_HF_QUERY_CSV_DEFAULT="real camera photo dataset,smartphone photo dataset,dslr photo dataset,webcam image dataset,cctv frame image dataset,meme image real vs ai,captioned image real ai,screenshot dataset image,chat ui screenshot,browser screenshot image,dashboard screenshot dataset,image poster infographic,logo brand image dataset,advertisement creative image,receipt scanned document image,id card document image,invoice form document scan,anime illustration real fake,digital art illustration dataset,3d render real fake,cgi synthetic image real,game render frame dataset,watermarked social media image,recompressed image dataset,heavily edited real photo,low resolution blurry image,extreme aspect ratio image,portrait selfie real fake,group photo real fake,deepfake face swap image,diffusion generated image latest"
DIVERSE_HF_QUERY_CSV_DEFAULT="$BEST_HF_QUERY_CSV_DEFAULT"

load_env_file() {
  if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
  fi
}

ensure_env() {
  if [[ "$ENV_READY" == "1" ]]; then
    return
  fi
  bash scripts/install_deps.sh
  # shellcheck disable=SC1091
  source .venv/bin/activate
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

run_collection_command() {
  if skip_collection_if_training; then
    return 0
  fi
  "$@"
}

print_usage() {
  echo "usage: bash scripts/do.sh [start|start-v2|collect|collect-diverse|collect-fast|collect-image|collect-video|ingest|scan [paths...]|train|train-video|train-all|train-all-types|deps-update|doctor|autocollect|detect <image>|status]"
}

run_malware_scan() {
  if [[ "${MALWARE_SCAN:-1}" != "1" ]]; then
    echo "malware_scan=disabled"
    return 0
  fi
  local -a targets=("$@")
  if [[ "${#targets[@]}" -eq 0 ]]; then
    targets=("${DATA_DIR:-./data_best}" "${NEW_DATA_DST:-./data_new/train}" "${VIDEO_OUT:-./video_data}" "${MODEL_OUTPUTS_SRC:-./incoming_model_outputs}")
  fi
  bash scripts/malware_scan.sh "${targets[@]}"
}

print_hf_query_args() {
  local query_csv="$1"
  IFS=',' read -r -a _queries <<< "$query_csv"
  local q=""
  for q in "${_queries[@]}"; do
    q="$(echo "$q" | xargs)"
    [[ -z "$q" ]] && continue
    printf "%s\n" --hf-query "$q"
  done
}

print_cli_flag() {
  printf "%s\n" "$1"
}

print_cli_flag_value() {
  printf "%s\n%s\n" "$1" "$2"
}

print_cli_flag_value_from_env() {
  local flag="$1"
  local env_name="$2"
  local default="$3"
  print_cli_flag_value "$flag" "${!env_name:-$default}"
}

run_image_dataset_builder() {
  local out="$1"
  local query_csv="$2"
  shift 2
  local -a query_args=()
  mapfile -t query_args < <(print_hf_query_args "$query_csv")
  ensure_env
  python scripts/build_best_dataset.py \
    --out "$out" \
    "$@" \
    "${query_args[@]}"
}

run_image_dataset_discovery() {
  local timeout_sec="$1"
  local out="$2"
  local query_csv="$3"
  shift 3
  local -a query_args=()
  mapfile -t query_args < <(print_hf_query_args "$query_csv")
  ensure_env
  if command -v timeout >/dev/null 2>&1; then
    timeout "${timeout_sec}s" python scripts/build_best_dataset.py \
      --out "$out" \
      "$@" \
      "${query_args[@]}" \
      --discover-only
  else
    python scripts/build_best_dataset.py \
      --out "$out" \
      "$@" \
      "${query_args[@]}" \
      --discover-only
  fi
}

run_image_dataset_build() {
  local out="$1"
  local query_csv="$2"
  shift 2
  run_image_dataset_builder "$out" "$query_csv" "$@"
  run_malware_scan "$out"
}

print_image_collection_args() {
  local profile="$1"
  local train_env=""
  local train_default=""
  local val_env=""
  local val_default=""
  local test_env=""
  local test_default=""
  local prefix=""
  local discovery_limit_default=""
  local max_sources_default=""
  local print_top_default=""
  local cache_file_default=""
  local stream_buffer_default=""
  local max_samples_default=""
  local warmup_default=""
  local min_sources_with_accepted_default=""
  local min_sources_per_class_default=""
  local base_pause_default=""
  local jitter_default=""
  local cooldown_default=""
  local min_side_default=""
  local hardneg_fraction_default=""

  case "$profile" in
    best)
      train_env="TRAIN_PER_CLASS"
      train_default="80000"
      val_env="VAL_PER_CLASS"
      val_default="20000"
      test_env="TEST_PER_CLASS"
      test_default="20000"
      prefix="BEST_DS"
      discovery_limit_default="90"
      max_sources_default="180"
      print_top_default="15"
      cache_file_default="./.local/hf_discovered_sources.txt"
      stream_buffer_default="12000"
      max_samples_default="45000"
      warmup_default="400"
      min_sources_with_accepted_default="16"
      min_sources_per_class_default="10"
      base_pause_default="1100"
      jitter_default="900"
      cooldown_default="45000"
      min_side_default="224"
      hardneg_fraction_default="0.8"
      ;;
    fast)
      train_env="FAST_TRAIN_PER_CLASS"
      train_default="4000"
      val_env="FAST_VAL_PER_CLASS"
      val_default="1000"
      test_env="FAST_TEST_PER_CLASS"
      test_default="1000"
      prefix="FAST"
      discovery_limit_default="60"
      max_sources_default="100"
      print_top_default="10"
      cache_file_default="./.local/hf_fast_sources.txt"
      stream_buffer_default="6000"
      max_samples_default="12000"
      warmup_default="250"
      min_sources_with_accepted_default="10"
      min_sources_per_class_default="6"
      base_pause_default="900"
      jitter_default="700"
      cooldown_default="30000"
      min_side_default="224"
      hardneg_fraction_default="0.8"
      ;;
    *)
      echo "unknown_image_collection_profile=$profile" >&2
      return 1
      ;;
  esac

  print_cli_flag_value_from_env --train-per-class "$train_env" "$train_default"
  print_cli_flag_value_from_env --val-per-class "$val_env" "$val_default"
  print_cli_flag_value_from_env --test-per-class "$test_env" "$test_default"
  print_cli_flag --discover-hf
  print_cli_flag_value_from_env --hf-discovery-limit "${prefix}_HF_DISCOVERY_LIMIT" "$discovery_limit_default"
  print_cli_flag_value_from_env --hf-max-sources "${prefix}_HF_MAX_SOURCES" "$max_sources_default"
  print_cli_flag_value_from_env --hf-min-downloads "${prefix}_HF_MIN_DOWNLOADS" "80"
  print_cli_flag_value_from_env --hf-min-likes "${prefix}_HF_MIN_LIKES" "2"
  print_cli_flag_value_from_env --hf-min-quality-score "${prefix}_HF_MIN_QUALITY_SCORE" "1.7"
  print_cli_flag_value_from_env --hf-print-top "${prefix}_HF_PRINT_TOP" "$print_top_default"
  print_cli_flag_value_from_env --hf-cache-file "${prefix}_HF_CACHE_FILE" "$cache_file_default"
  print_cli_flag --hf-cache-only-if-present
  print_cli_flag_value_from_env --cache-dir "${prefix}_CACHE_DIR" "./.local/hf"
  print_cli_flag --streaming
  print_cli_flag_value_from_env --stream-buffer-size "${prefix}_STREAM_BUFFER_SIZE" "$stream_buffer_default"
  print_cli_flag_value_from_env --max-samples-per-source "${prefix}_MAX_SAMPLES_PER_SOURCE" "$max_samples_default"
  print_cli_flag_value_from_env --acceptance-warmup-samples "${prefix}_ACCEPTANCE_WARMUP_SAMPLES" "$warmup_default"
  print_cli_flag_value_from_env --min-acceptance-rate "${prefix}_MIN_ACCEPTANCE_RATE" "0.01"
  print_cli_flag_value_from_env --min-hf-sources-with-accepted "${prefix}_MIN_HF_SOURCES_WITH_ACCEPTED" "$min_sources_with_accepted_default"
  print_cli_flag_value_from_env --min-hf-sources-per-class "${prefix}_MIN_HF_SOURCES_PER_CLASS" "$min_sources_per_class_default"
  print_cli_flag_value_from_env --repo-base-pause-ms "${prefix}_REPO_BASE_PAUSE_MS" "$base_pause_default"
  print_cli_flag_value_from_env --repo-jitter-ms "${prefix}_REPO_JITTER_MS" "$jitter_default"
  print_cli_flag_value_from_env --repo-cooldown-ms "${prefix}_REPO_COOLDOWN_MS" "$cooldown_default"
  print_cli_flag_value_from_env --max-consecutive-failures "${prefix}_MAX_CONSECUTIVE_FAILURES" "2"
  print_cli_flag_value_from_env --min-side "${prefix}_MIN_SIDE" "$min_side_default"
  print_cli_flag_value_from_env --max-aspect-ratio "${prefix}_MAX_ASPECT_RATIO" "2.5"
  print_cli_flag_value_from_env --min-entropy "${prefix}_MIN_ENTROPY" "3.4"
  print_cli_flag_value_from_env --hardneg-fraction "${prefix}_HARDNEG_FRACTION" "$hardneg_fraction_default"
  print_cli_flag --hf-only
  print_cli_flag --require-full-targets
}

collect_image_data() {
  local out="${DATA_DIR:-./data_best}"
  local query_csv="${BEST_DS_HF_QUERIES:-$BEST_HF_QUERY_CSV_DEFAULT}"
  local -a build_args=()
  mapfile -t build_args < <(print_image_collection_args best)
  run_image_dataset_build "$out" "$query_csv" "${build_args[@]}"
}

collect_fast_data() {
  local out="${DATA_DIR:-./data_best_fast}"
  local query_csv="${FAST_HF_QUERIES:-${BEST_DS_HF_QUERIES:-$BEST_HF_QUERY_CSV_DEFAULT}}"
  local -a build_args=()
  mapfile -t build_args < <(print_image_collection_args fast)
  run_image_dataset_build "$out" "$query_csv" "${build_args[@]}"
}

ingest_outputs() {
  ensure_env
  python scripts/ingest_model_outputs.py \
    --src "${MODEL_OUTPUTS_SRC:-./incoming_model_outputs}" \
    --dst "${NEW_DATA_DST:-./data_new/train}" \
    --archive "${MODEL_OUTPUTS_ARCHIVE:-./incoming_model_outputs/_processed}"
  run_malware_scan "${NEW_DATA_DST:-./data_new/train}" "${MODEL_OUTPUTS_SRC:-./incoming_model_outputs}"
}

print_diverse_common_args() {
  print_cli_flag_value_from_env --train-per-class "DIVERSE_TRAIN_PER_CLASS" "100000"
  print_cli_flag_value_from_env --val-per-class "DIVERSE_VAL_PER_CLASS" "25000"
  print_cli_flag_value_from_env --test-per-class "DIVERSE_TEST_PER_CLASS" "25000"
  print_cli_flag_value_from_env --hf-cache-file "DIVERSE_HF_CACHE_FILE" "./.local/hf_diverse_sources.txt"
  print_cli_flag --hf-cache-only-if-present
  print_cli_flag_value_from_env --cache-dir "DIVERSE_CACHE_DIR" "./.local/hf"
  print_cli_flag --streaming
  print_cli_flag_value_from_env --stream-buffer-size "DIVERSE_STREAM_BUFFER_SIZE" "16000"
  print_cli_flag_value_from_env --max-samples-per-source "DIVERSE_MAX_SAMPLES_PER_SOURCE" "80000"
  print_cli_flag_value_from_env --acceptance-warmup-samples "DIVERSE_ACCEPTANCE_WARMUP_SAMPLES" "450"
  print_cli_flag_value_from_env --min-acceptance-rate "DIVERSE_MIN_ACCEPTANCE_RATE" "0.012"
  print_cli_flag_value_from_env --min-hf-sources-with-accepted "DIVERSE_MIN_HF_SOURCES_WITH_ACCEPTED" "24"
  print_cli_flag_value_from_env --min-hf-sources-per-class "DIVERSE_MIN_HF_SOURCES_PER_CLASS" "14"
  print_cli_flag_value_from_env --repo-base-pause-ms "DIVERSE_REPO_BASE_PAUSE_MS" "1400"
  print_cli_flag_value_from_env --repo-jitter-ms "DIVERSE_REPO_JITTER_MS" "1200"
  print_cli_flag_value_from_env --repo-cooldown-ms "DIVERSE_REPO_COOLDOWN_MS" "45000"
  print_cli_flag_value_from_env --max-consecutive-failures "DIVERSE_MAX_CONSECUTIVE_FAILURES" "2"
  print_cli_flag_value_from_env --min-side "DIVERSE_MIN_SIDE" "192"
  print_cli_flag_value_from_env --max-aspect-ratio "DIVERSE_MAX_ASPECT_RATIO" "3.2"
  print_cli_flag_value_from_env --min-entropy "DIVERSE_MIN_ENTROPY" "3.1"
  print_cli_flag_value_from_env --hardneg-fraction "DIVERSE_HARDNEG_FRACTION" "1.0"
  print_cli_flag --hf-only
  print_cli_flag --require-full-targets
}

print_diverse_discovery_args() {
  print_cli_flag --discover-hf
  print_cli_flag_value_from_env --hf-discovery-limit "DIVERSE_HF_DISCOVERY_LIMIT" "140"
  print_cli_flag_value_from_env --hf-max-sources "DIVERSE_HF_MAX_SOURCES" "320"
  print_cli_flag_value_from_env --hf-min-downloads "DIVERSE_HF_MIN_DOWNLOADS" "100"
  print_cli_flag_value_from_env --hf-min-likes "DIVERSE_HF_MIN_LIKES" "2"
  print_cli_flag_value_from_env --hf-min-quality-score "DIVERSE_HF_MIN_QUALITY_SCORE" "1.85"
  print_cli_flag_value_from_env --hf-print-top "DIVERSE_HF_PRINT_TOP" "20"
}

print_diverse_audit_args() {
  print_cli_flag_value_from_env --min-unique-sources "DIVERSE_MIN_UNIQUE_SOURCES" "20"
  print_cli_flag_value_from_env --min-hardneg-modes "DIVERSE_MIN_HARDNEG_MODES" "4"
  print_cli_flag_value_from_env --max-class-imbalance "DIVERSE_MAX_CLASS_IMBALANCE" "0.08"
}

collect_diverse_image_data() {
  local out="${DATA_DIR:-./data_best}"
  local query_csv="${DIVERSE_HF_QUERIES:-$DIVERSE_HF_QUERY_CSV_DEFAULT}"
  local hf_cache="${DIVERSE_HF_CACHE_FILE:-./.local/hf_diverse_sources.txt}"
  local timeout_sec="${DIVERSE_DISCOVERY_TIMEOUT_SEC:-900}"
  local -a common_args=()
  local -a discover_args=()
  local -a cache_args=(--no-discover-hf --sources-file "$hf_cache")
  local -a full_args=()
  local -a audit_args=()

  mapfile -t common_args < <(print_diverse_common_args)
  full_args=("${common_args[@]}")
  mapfile -t discover_args < <(print_diverse_discovery_args)
  mapfile -t audit_args < <(print_diverse_audit_args)

  # Run discovery as a bounded pre-pass, then build from cached HF ids or live HF discovery.
  if [[ "${DIVERSE_SKIP_DISCOVERY:-0}" != "1" ]]; then
    if ! run_image_dataset_discovery "$timeout_sec" "$out" "$query_csv" "${common_args[@]}" "${discover_args[@]}"; then
      echo "collect_diverse_discovery=failed_or_timed_out fallback=cache_or_live_hf"
    fi
  fi

  if [[ -s "$hf_cache" ]]; then
    full_args+=("${cache_args[@]}")
  elif [[ "${DIVERSE_SKIP_DISCOVERY:-0}" == "1" ]]; then
    echo "collect_diverse_sources=cache_only_missing reason=hf_cache_missing"
    return 1
  else
    full_args+=("${discover_args[@]}")
    echo "collect_diverse_sources=live_discovery reason=hf_cache_missing"
  fi
  run_image_dataset_builder "$out" "$query_csv" "${full_args[@]}"
  run_malware_scan "$out"

  python scripts/audit_diversity.py \
    --data "$out" \
    "${audit_args[@]}"
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
    --retries "${VIDEO_RETRIES:-5}" \
    --min-video-bytes "${VIDEO_MIN_BYTES:-100000}" \
    --max-video-bytes "${VIDEO_MAX_BYTES:-0}"
  run_malware_scan "${VIDEO_OUT:-./video_data}"
}

collect_full_cycle() {
  collect_diverse_image_data
  ingest_outputs
  collect_video_data
}

collect_diverse_cycle() {
  collect_diverse_image_data
  ingest_outputs
  VIDEO_CACHE_DIR="${VIDEO_CACHE_DIR:-./.local/hf}" VIDEO_SNAPSHOT_MAX_WORKERS="${VIDEO_SNAPSHOT_MAX_WORKERS:-1}" collect_video_data
}

train_image_pipeline() {
  env SKIP_DATA=1 RUN_VIDEO_DATA_PULL=0 RUN_VIDEO_TRAIN=0 bash scripts/max_quality_4090.sh
}

train_all_pipeline() {
  env SKIP_DATA=1 RUN_VIDEO_DATA_PULL=0 bash scripts/max_quality_4090.sh
}

validate_train_artifacts() {
  local ens_dir="${ENS_OUT:-./artifacts_ens}"
  local vid_best_pt="${VIDEO_ARTIFACTS_OUT:-./video_artifacts}/best_video.pt"
  local vid_best_sft="${VIDEO_ARTIFACTS_OUT:-./video_artifacts}/best_video.safetensors"
  local missing=0

  for p in "$ens_dir/m1/best.safetensors" "$ens_dir/m2/best.safetensors" "$ens_dir/m3/best.safetensors" "$ens_dir/m4/best.safetensors" "$ens_dir/test_metrics.json" "$ens_dir/prod_manifest.json"; do
    if [[ ! -f "$p" ]]; then
      echo "missing_artifact=$p"
      missing=1
    fi
  done
  if [[ ! -f "$vid_best_sft" && ! -f "$vid_best_pt" ]]; then
    echo "missing_artifact=$vid_best_sft"
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
  local -a resume_arg=()
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
  echo "video model: ${VIDEO_ARTIFACTS_OUT:-./video_artifacts}/best_video.safetensors"
}

resolve_detect_models() {
  local -a models=()
  mapfile -t models < <(compgen -G "${MODEL_GLOB:-./artifacts_ens/m*/best.safetensors}" || true)
  if [[ "${#models[@]}" -eq 0 ]]; then
    mapfile -t models < <(compgen -G "./artifacts_ens/m*/best.pt" || true)
  fi
  if [[ "${#models[@]}" -eq 0 ]]; then
    models=("${MODEL_PATH:-./artifacts/best.safetensors}")
  fi
  printf "%s\n" "${models[@]}"
}

config_flag_if_exists() {
  local flag="$1"
  local path="$2"
  if [[ -f "$path" ]]; then
    printf "%s\n%s\n" "$flag" "$path"
  fi
}

run_detect() {
  local image_path="${1:-}"
  local -a detect_models=()
  local -a detect_extra=()
  local -a domain_extra=()
  local -a tools_extra=()

  if [[ -z "$image_path" ]]; then
    echo "usage: bash scripts/do.sh detect /path/to/image.jpg"
    exit 2
  fi

  ensure_env
  mapfile -t detect_models < <(resolve_detect_models)
  mapfile -t detect_extra < <(config_flag_if_exists --ensemble-config "${ENSEMBLE_CONFIG:-./artifacts_ens/ensemble_config.json}")
  mapfile -t domain_extra < <(config_flag_if_exists --domain-config "${DOMAIN_CONFIG:-./artifacts_ens/domain_config.json}")
  mapfile -t tools_extra < <(config_flag_if_exists --tools-config "${TOOLS_CONFIG:-./artifacts_ens/tools_config.json}")

  aid-detect \
    --model "${detect_models[@]}" \
    "${detect_extra[@]}" \
    "${domain_extra[@]}" \
    "${tools_extra[@]}" \
    --tta-views "${TTA_VIEWS:-2}" \
    --image "$image_path" \
    --json
}

load_env_file
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
    run_collection_command collect_full_cycle
    ;;

  collect-diverse)
    run_collection_command collect_diverse_cycle
    ;;

  collect-fast)
    run_collection_command collect_fast_data
    ;;

  collect-image)
    run_collection_command collect_image_data
    ;;

  collect-video)
    run_collection_command collect_video_data
    ;;

  ingest)
    run_collection_command ingest_outputs
    ;;

  scan)
    run_malware_scan "$@"
    ;;

  train|train-image)
    # Image pipeline only, assumes data already collected.
    with_training_lock train_image_pipeline
    ;;

  train-video)
    with_training_lock train_video_only
    ;;

  deps-update)
    bash scripts/update_deps_lock.sh
    ;;

  doctor)
    bash scripts/doctor.sh
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
    echo "serve_disabled=1 reason=pipeline_only_mode"
    exit 2
    ;;

  detect)
    run_detect "$@"
    ;;

  status)
    show_status
    ;;

  *)
    print_usage
    exit 2
    ;;
esac
