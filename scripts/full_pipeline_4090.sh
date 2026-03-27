#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
source "$ROOT_DIR/scripts/lib/env.sh"

load_env_file

DATA_DIR="${DATA_DIR:-./data_best}"
TRAIN_PER_CLASS="${TRAIN_PER_CLASS:-40000}"
VAL_PER_CLASS="${VAL_PER_CLASS:-9000}"
TEST_PER_CLASS="${TEST_PER_CLASS:-9000}"
BEST_DS_NEAR_HAMMING="${BEST_DS_NEAR_HAMMING:-2}"
BEST_DS_NEAR_WINDOW="${BEST_DS_NEAR_WINDOW:-2400}"
BEST_DS_MIN_SIDE="${BEST_DS_MIN_SIDE:-192}"
BEST_DS_MAX_ASPECT_RATIO="${BEST_DS_MAX_ASPECT_RATIO:-3.0}"
BEST_DS_MIN_ENTROPY="${BEST_DS_MIN_ENTROPY:-3.2}"
BEST_DS_MAX_UNIQUE_PER_SOURCE="${BEST_DS_MAX_UNIQUE_PER_SOURCE:-220000}"
BEST_DS_MAX_PER_SOURCE_CLASS="${BEST_DS_MAX_PER_SOURCE_CLASS:-120000}"
BEST_DS_MAX_PER_SOURCE_SPLIT_CLASS="${BEST_DS_MAX_PER_SOURCE_SPLIT_CLASS:-0}"
BEST_DS_JPEG_QUALITY="${BEST_DS_JPEG_QUALITY:-92}"
BEST_DS_HARDNEG_FRACTION="${BEST_DS_HARDNEG_FRACTION:-0.6}"
BEST_DS_DISCOVER_HF="${BEST_DS_DISCOVER_HF:-1}"
BEST_DS_HF_DISCOVERY_LIMIT="${BEST_DS_HF_DISCOVERY_LIMIT:-180}"
BEST_DS_HF_MAX_SOURCES="${BEST_DS_HF_MAX_SOURCES:-360}"
BEST_DS_HF_MIN_DOWNLOADS="${BEST_DS_HF_MIN_DOWNLOADS:-80}"
BEST_DS_HF_MIN_LIKES="${BEST_DS_HF_MIN_LIKES:-2}"
BEST_DS_HF_MIN_QUALITY_SCORE="${BEST_DS_HF_MIN_QUALITY_SCORE:-1.7}"
BEST_DS_HF_PRINT_TOP="${BEST_DS_HF_PRINT_TOP:-24}"
BEST_DS_HF_QUERY_PAUSE_MS="${BEST_DS_HF_QUERY_PAUSE_MS:-0}"
BEST_DS_HF_CACHE_FILE="${BEST_DS_HF_CACHE_FILE:-./.local/hf_discovered_sources.txt}"
BEST_DS_CACHE_DIR="${BEST_DS_CACHE_DIR:-./.local/hf}"
BEST_DS_HF_QUERIES="${BEST_DS_HF_QUERIES:-real camera photo dataset,smartphone photo dataset,dslr photo dataset,webcam image dataset,cctv frame image dataset,meme image real vs ai,captioned image real ai,screenshot dataset image,chat ui screenshot,browser screenshot image,dashboard screenshot dataset,mobile app screenshot image,website screenshot dataset,image poster infographic,logo brand image dataset,advertisement creative image,receipt scanned document image,id card document image,invoice form document scan,passport scan image,document camera capture dataset,anime illustration real fake,digital art illustration dataset,manga artwork dataset,3d render real fake,cgi synthetic image real,game render frame dataset,watermarked social media image,recompressed image dataset,heavily edited real photo,low resolution blurry image,extreme aspect ratio image,portrait selfie real fake,group photo real fake,deepfake face swap image,diffusion generated image latest,stock photo real ai,image manipulation detection,synthetic portrait dataset,screen capture ui dataset}"
BEST_DS_SOURCES_FILE="${BEST_DS_SOURCES_FILE:-}"
BEST_DS_EXTRA_SOURCES="${BEST_DS_EXTRA_SOURCES:-}"
BEST_DS_LOCAL_SOURCES="${BEST_DS_LOCAL_SOURCES:-}"
BEST_DS_HF_ONLY="${BEST_DS_HF_ONLY:-1}"
BEST_DS_NO_DEFAULT_SOURCES="${BEST_DS_NO_DEFAULT_SOURCES:-1}"
BEST_DS_STREAMING="${BEST_DS_STREAMING:-1}"
BEST_DS_STREAM_BUFFER_SIZE="${BEST_DS_STREAM_BUFFER_SIZE:-12000}"
BEST_DS_MAX_SAMPLES_PER_SOURCE="${BEST_DS_MAX_SAMPLES_PER_SOURCE:-60000}"
BEST_DS_ACCEPTANCE_WARMUP_SAMPLES="${BEST_DS_ACCEPTANCE_WARMUP_SAMPLES:-400}"
BEST_DS_MIN_ACCEPTANCE_RATE="${BEST_DS_MIN_ACCEPTANCE_RATE:-0.01}"
BEST_DS_MIN_HF_SOURCES_WITH_ACCEPTED="${BEST_DS_MIN_HF_SOURCES_WITH_ACCEPTED:-20}"
BEST_DS_MIN_HF_SOURCES_PER_CLASS="${BEST_DS_MIN_HF_SOURCES_PER_CLASS:-12}"
BEST_DS_MIN_HF_SOURCES_PER_SPLIT_CLASS="${BEST_DS_MIN_HF_SOURCES_PER_SPLIT_CLASS:-0}"
BEST_DS_REPO_BASE_PAUSE_MS="${BEST_DS_REPO_BASE_PAUSE_MS:-900}"
BEST_DS_REPO_JITTER_MS="${BEST_DS_REPO_JITTER_MS:-900}"
BEST_DS_REPO_COOLDOWN_MS="${BEST_DS_REPO_COOLDOWN_MS:-45000}"
BEST_DS_TRANSIENT_ERROR_COOLDOWN_MS="${BEST_DS_TRANSIENT_ERROR_COOLDOWN_MS:-3000}"
BEST_DS_MAX_CONSECUTIVE_FAILURES="${BEST_DS_MAX_CONSECUTIVE_FAILURES:-2}"
SWEEP_OUT="${SWEEP_OUT:-./artifacts_sweep}"
ENS_OUT="${ENS_OUT:-./artifacts_ens}"
EPOCHS="${EPOCHS:-18}"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-14}"
SKIP_DATA="${SKIP_DATA:-0}"
SKIP_SWEEP="${SKIP_SWEEP:-0}"
RUN_HARD_MINING="${RUN_HARD_MINING:-1}"
HARD_MINING_TOPK="${HARD_MINING_TOPK:-5000}"
RUN_HARD_RETRAIN="${RUN_HARD_RETRAIN:-1}"
HARD_RETRAIN_DATA_DIR="${HARD_RETRAIN_DATA_DIR:-./.local/hard_mined_training_data}"
HARD_RETRAIN_EPOCHS="${HARD_RETRAIN_EPOCHS:-$EPOCHS}"
RUN_DISTILL="${RUN_DISTILL:-1}"
DISTILL_EPOCHS="${DISTILL_EPOCHS:-10}"
RUN_ENSEMBLE_FIT="${RUN_ENSEMBLE_FIT:-1}"
ENS_CONFIG_PATH="${ENS_CONFIG_PATH:-$ENS_OUT/ensemble_config.json}"
ENS_FIT_STEPS="${ENS_FIT_STEPS:-1200}"
ENS_FIT_LR="${ENS_FIT_LR:-0.05}"
ENS_FIT_L2="${ENS_FIT_L2:-0.001}"
ENS_FIT_MAX_VAL_IMAGES="${ENS_FIT_MAX_VAL_IMAGES:-0}"
RUN_METADATA_MEMBER="${RUN_METADATA_MEMBER:-1}"
METADATA_MEMBER_OUT="${METADATA_MEMBER_OUT:-$ENS_OUT/m5_metadata}"
METADATA_MEMBER_EPOCHS="${METADATA_MEMBER_EPOCHS:-$EPOCHS}"
RUN_DOMAIN_THRESHOLDS="${RUN_DOMAIN_THRESHOLDS:-1}"
DOMAIN_CONFIG_PATH="${DOMAIN_CONFIG_PATH:-$ENS_OUT/domain_config.json}"
DOMAIN_THRESHOLD_OBJECTIVE="${DOMAIN_THRESHOLD_OBJECTIVE:-balanced}"
DOMAIN_THRESHOLD_MIN_SAMPLES="${DOMAIN_THRESHOLD_MIN_SAMPLES:-80}"
RUN_ROBUST_EVAL="${RUN_ROBUST_EVAL:-1}"
ROBUST_EVAL_OUT="${ROBUST_EVAL_OUT:-$ENS_OUT/robust_eval.json}"
ROBUST_EVAL_MAX_IMAGES="${ROBUST_EVAL_MAX_IMAGES:-1200}"
EVAL_TTA_VIEWS="${EVAL_TTA_VIEWS:-3}"
DRY_RUN="${DRY_RUN:-0}"
RUN_VIDEO_DATA_PULL="${RUN_VIDEO_DATA_PULL:-1}"
VIDEO_OUT="${VIDEO_OUT:-./video_data}"
VIDEO_CACHE_DIR="${VIDEO_CACHE_DIR:-./.local/hf}"
VIDEO_TRAIN_PER_CLASS="${VIDEO_TRAIN_PER_CLASS:-220}"
VIDEO_VAL_PER_CLASS="${VIDEO_VAL_PER_CLASS:-60}"
VIDEO_MODE="${VIDEO_MODE:-snapshot}"
VIDEO_SNAPSHOT_MAX_WORKERS="${VIDEO_SNAPSHOT_MAX_WORKERS:-1}"
VIDEO_REPO_BASE_PAUSE_MS="${VIDEO_REPO_BASE_PAUSE_MS:-2200}"
VIDEO_REPO_JITTER_MS="${VIDEO_REPO_JITTER_MS:-1800}"
VIDEO_COPY_SLEEP_MS="${VIDEO_COPY_SLEEP_MS:-15}"
VIDEO_SLEEP_MS="${VIDEO_SLEEP_MS:-120}"
VIDEO_JITTER_MS="${VIDEO_JITTER_MS:-80}"
VIDEO_CHUNK_PAUSE_MS="${VIDEO_CHUNK_PAUSE_MS:-1000}"
VIDEO_REPO_COOLDOWN_MS="${VIDEO_REPO_COOLDOWN_MS:-3000}"
VIDEO_RETRIES="${VIDEO_RETRIES:-5}"
VIDEO_MIN_BYTES="${VIDEO_MIN_BYTES:-100000}"
VIDEO_MAX_BYTES="${VIDEO_MAX_BYTES:-0}"
RUN_VIDEO_TRAIN="${RUN_VIDEO_TRAIN:-0}"
VIDEO_ARTIFACTS_OUT="${VIDEO_ARTIFACTS_OUT:-./video_artifacts}"
VIDEO_TRAIN_EPOCHS="${VIDEO_TRAIN_EPOCHS:-30}"
VIDEO_TRAIN_BATCH_SIZE="${VIDEO_TRAIN_BATCH_SIZE:-4}"
VIDEO_TRAIN_IMG_SIZE="${VIDEO_TRAIN_IMG_SIZE:-224}"
VIDEO_TRAIN_FRAMES="${VIDEO_TRAIN_FRAMES:-24}"
VIDEO_TRAIN_GRAD_ACCUM="${VIDEO_TRAIN_GRAD_ACCUM:-2}"
VIDEO_TRAIN_LR="${VIDEO_TRAIN_LR:-1e-4}"
VIDEO_TRAIN_PATIENCE="${VIDEO_TRAIN_PATIENCE:-6}"
VIDEO_TRAIN_MIN_DELTA="${VIDEO_TRAIN_MIN_DELTA:-0.001}"
VIDEO_TRAIN_RESUME="${VIDEO_TRAIN_RESUME:-1}"
PIPELINE_MIN_FREE_GB="${PIPELINE_MIN_FREE_GB:-40}"
PIPELINE_REPORT_DIR="${PIPELINE_REPORT_DIR:-./.local/reports}"
PIPELINE_COLLECTED_DATA_DIR="${PIPELINE_COLLECTED_DATA_DIR:-$DATA_DIR}"
PIPELINE_PREPARED_DATA_DIR="${PIPELINE_PREPARED_DATA_DIR:-${TRAIN_READY_DATA_DIR:-$DATA_DIR}}"
PIPELINE_DATASET_QA_OUT="${PIPELINE_DATASET_QA_OUT:-$PIPELINE_REPORT_DIR/dataset_qa_summary.json}"
PIPELINE_DATASET_PROVENANCE_OUT="${PIPELINE_DATASET_PROVENANCE_OUT:-$PIPELINE_REPORT_DIR/dataset_provenance.json}"
PIPELINE_FINAL_SUMMARY_OUT="${PIPELINE_FINAL_SUMMARY_OUT:-$ENS_OUT/final_run_summary.json}"
PIPELINE_RUN_MANIFEST_OUT="${PIPELINE_RUN_MANIFEST_OUT:-$ENS_OUT/run_manifest.json}"
PIPELINE_THRESHOLDS_OUT="${PIPELINE_THRESHOLDS_OUT:-$ENS_OUT/final_thresholds.json}"
PIPELINE_FAILURE_OUT="${PIPELINE_FAILURE_OUT:-$ENS_OUT/run_failure.json}"
PIPELINE_STAGE="bootstrap"
PIPELINE_IMAGE_TRAIN_DATA_DIR="${PIPELINE_IMAGE_TRAIN_DATA_DIR:-$DATA_DIR}"

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf "[DRY_RUN]"
    printf " %q" "$@"
    printf "\n"
  else
    "$@"
  fi
}

require_disk_free_gb() {
  local stage="$1"
  local min_gb="${2:-$PIPELINE_MIN_FREE_GB}"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] disk_guard stage=$stage required_gb=$min_gb"
    return 0
  fi
  if [[ "$min_gb" == "0" ]]; then
    echo "disk_guard=disabled stage=$stage"
    return 0
  fi
  local avail_kb=""
  avail_kb="$(df -Pk "$ROOT_DIR" | awk 'NR==2 {print $4}')"
  local avail_gb=$((avail_kb / 1024 / 1024))
  if (( avail_gb < min_gb )); then
    echo "disk_guard=failed stage=$stage avail_gb=$avail_gb required_gb=$min_gb" >&2
    return 1
  fi
  echo "disk_guard=ok stage=$stage avail_gb=$avail_gb required_gb=$min_gb"
}

write_dataset_reports() {
  run_cmd python scripts/write_pipeline_report.py dataset \
    --data "$PIPELINE_COLLECTED_DATA_DIR" \
    --prepared "$PIPELINE_PREPARED_DATA_DIR" \
    --incremental "${TRAIN_INCREMENTAL_DATA_DIR:-}" \
    --video "$VIDEO_OUT" \
    --cache-file "$BEST_DS_HF_CACHE_FILE" \
    --out "$PIPELINE_DATASET_QA_OUT" \
    --provenance-out "$PIPELINE_DATASET_PROVENANCE_OUT"
}

write_final_reports() {
  run_cmd python scripts/write_pipeline_report.py final \
    --data "$PIPELINE_COLLECTED_DATA_DIR" \
    --prepared "$PIPELINE_PREPARED_DATA_DIR" \
    --video "$VIDEO_OUT" \
    --ens-out "$ENS_OUT" \
    --ensemble-config "$ENS_CONFIG_PATH" \
    --domain-config "$DOMAIN_CONFIG_PATH" \
    --video-artifacts "$VIDEO_ARTIFACTS_OUT" \
    --dataset-qa "$PIPELINE_DATASET_QA_OUT" \
    --robust-eval "$ROBUST_EVAL_OUT" \
    --prod-manifest "$ENS_OUT/prod_manifest.json" \
    --summary-out "$PIPELINE_FINAL_SUMMARY_OUT" \
    --manifest-out "$PIPELINE_RUN_MANIFEST_OUT" \
    --thresholds-out "$PIPELINE_THRESHOLDS_OUT"
}

write_failure_report() {
  local exit_code="$1"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] write_failure_report stage=$PIPELINE_STAGE exit_code=$exit_code out=$PIPELINE_FAILURE_OUT"
    return 0
  fi
  python scripts/write_pipeline_report.py failure \
    --stage "$PIPELINE_STAGE" \
    --exit-code "$exit_code" \
    --data "$DATA_DIR" \
    --ens-out "$ENS_OUT" \
    --video "$VIDEO_OUT" \
    --video-artifacts "$VIDEO_ARTIFACTS_OUT" \
    --out "$PIPELINE_FAILURE_OUT"
}

on_exit() {
  local exit_code=$?
  if [[ "$exit_code" -ne 0 ]]; then
    write_failure_report "$exit_code" || true
  fi
}

trap on_exit EXIT

activate_repo_venv() {
  local activate_script="$VENV_DIR/bin/activate"
  if [[ -f "$activate_script" ]]; then
    # shellcheck disable=SC1090
    source "$activate_script"
    return 0
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] source $activate_script"
    return 0
  fi
  echo "missing_virtualenv_activate=$activate_script run=bash scripts/install_deps.sh" >&2
  return 1
}

reset_ensemble_outputs() {
  local path=""
  shopt -s nullglob
  local -a ensemble_dirs=("$ENS_OUT"/m*)
  shopt -u nullglob
  if (( ${#ensemble_dirs[@]} > 0 )); then
    for path in "${ensemble_dirs[@]}"; do
      if [[ -e "$path" ]]; then
        run_cmd rm -rf "$path"
      fi
    done
  fi
  for path in \
    "$ENS_OUT/distill" \
    "$ENS_OUT/test_metrics.json" \
    "$ENS_OUT/ensemble_config.json" \
    "$ENS_OUT/domain_config.json" \
    "$ENS_OUT/robust_eval.json"; do
    if [[ -e "$path" ]]; then
      run_cmd rm -rf "$path"
    fi
  done
}

ENSEMBLE_MODELS=()

collect_ensemble_model_paths() {
  local min_count="${1:-4}"
  ENSEMBLE_MODELS=()
  shopt -s nullglob
  local model_dir=""
  local candidate=""
  for model_dir in "$ENS_OUT"/m*; do
    [[ -d "$model_dir" ]] || continue
    candidate="$model_dir/best.safetensors"
    if [[ -f "$candidate" ]]; then
      ENSEMBLE_MODELS+=("$candidate")
      continue
    fi
    candidate="$model_dir/best.pt"
    if [[ -f "$candidate" ]]; then
      ENSEMBLE_MODELS+=("$candidate")
    fi
  done
  shopt -u nullglob
  if [[ "$DRY_RUN" == "1" && ${#ENSEMBLE_MODELS[@]} -lt min_count ]]; then
    local idx=1
    while (( ${#ENSEMBLE_MODELS[@]} < min_count )); do
      if (( idx <= 4 )); then
        ENSEMBLE_MODELS+=("$ENS_OUT/m${idx}/best.safetensors")
      else
        ENSEMBLE_MODELS+=("$ENS_OUT/m5_metadata/best.safetensors")
      fi
      idx=$((idx + 1))
    done
    return 0
  fi
  if (( ${#ENSEMBLE_MODELS[@]} < min_count )); then
    echo "ensemble_model_count=invalid have=${#ENSEMBLE_MODELS[@]} need=$min_count ens_out=$ENS_OUT" >&2
    return 1
  fi
}

run_ensemble_training_bundle() {
  local train_root="$1"
  local epochs="$2"

  PIPELINE_STAGE="train_ensemble"
  require_disk_free_gb "$PIPELINE_STAGE"
  reset_ensemble_outputs
  run_cmd env \
    RUN_METADATA_MEMBER="$RUN_METADATA_MEMBER" \
    METADATA_MEMBER_OUT="$METADATA_MEMBER_OUT" \
    METADATA_MEMBER_EPOCHS="$METADATA_MEMBER_EPOCHS" \
    bash scripts/train_ensemble.sh "$train_root" "$ENS_OUT" "$epochs"

  local required_model_count=4
  if [[ "$RUN_METADATA_MEMBER" == "1" ]]; then
    required_model_count=5
  fi
  collect_ensemble_model_paths "$required_model_count"

  if [[ "$RUN_ENSEMBLE_FIT" == "1" ]]; then
    PIPELINE_STAGE="fit_ensemble"
    run_cmd python scripts/fit_ensemble.py \
      --data "$train_root" \
      --model "${ENSEMBLE_MODELS[@]}" \
      --out "$ENS_CONFIG_PATH" \
      --steps "$ENS_FIT_STEPS" \
      --lr "$ENS_FIT_LR" \
      --l2 "$ENS_FIT_L2" \
      --max-val-images "$ENS_FIT_MAX_VAL_IMAGES" \
      --objective balanced
  fi

  if [[ "$RUN_DOMAIN_THRESHOLDS" == "1" ]]; then
    PIPELINE_STAGE="fit_domain_thresholds"
    declare -a domain_cmd=(
      python scripts/fit_domain_thresholds.py
      --data "$train_root"
      --model "${ENSEMBLE_MODELS[@]}"
      --out "$DOMAIN_CONFIG_PATH"
      --objective "$DOMAIN_THRESHOLD_OBJECTIVE"
      --min-samples-per-domain "$DOMAIN_THRESHOLD_MIN_SAMPLES"
    )
    if [[ -f "$ENS_CONFIG_PATH" ]]; then
      domain_cmd+=(--ensemble-config "$ENS_CONFIG_PATH")
    fi
    run_cmd "${domain_cmd[@]}"
  fi

  PIPELINE_STAGE="eval"
  declare -a eval_cmd=(
    python scripts/eval_test_ensemble.py
    --data "$train_root"
    --model "${ENSEMBLE_MODELS[@]}"
    --tta "$EVAL_TTA_VIEWS"
    --out "$ENS_OUT/test_metrics.json"
  )
  if [[ -f "$ENS_CONFIG_PATH" ]]; then
    eval_cmd+=(--ensemble-config "$ENS_CONFIG_PATH")
  fi
  run_cmd "${eval_cmd[@]}"

  if [[ "$RUN_ROBUST_EVAL" == "1" ]]; then
    PIPELINE_STAGE="robust_eval"
    declare -a robust_cmd=(
      python -m ai_image_detector.robust_eval
      --data "$train_root"
      --model "${ENSEMBLE_MODELS[@]}"
      --max-images "$ROBUST_EVAL_MAX_IMAGES"
      --out "$ROBUST_EVAL_OUT"
    )
    if [[ -f "$ENS_CONFIG_PATH" ]]; then
      robust_cmd+=(--ensemble-config "$ENS_CONFIG_PATH")
    fi
    run_cmd "${robust_cmd[@]}"
  fi
}

run_hard_mining_bundle() {
  local train_root="$1"

  [[ "$RUN_HARD_MINING" == "1" ]] || return 0

  PIPELINE_STAGE="hard_mining"
  if [[ -e "$ENS_OUT/hard_mined" ]]; then
    run_cmd rm -rf "$ENS_OUT/hard_mined"
  fi
  declare -a hard_cmd=(
    python scripts/mine_hard_negatives.py
    --data "$train_root"
    --model "${ENSEMBLE_MODELS[@]}"
  )
  if [[ -f "$ENS_CONFIG_PATH" ]]; then
    hard_cmd+=(--ensemble-config "$ENS_CONFIG_PATH")
  fi
  hard_cmd+=(--out "$ENS_OUT/hard_mined" --top-k "$HARD_MINING_TOPK")
  run_cmd "${hard_cmd[@]}"

  if [[ "$RUN_HARD_RETRAIN" != "1" ]]; then
    return 0
  fi

  PIPELINE_STAGE="hard_retrain_prepare"
  require_disk_free_gb "$PIPELINE_STAGE"
  if [[ -e "$HARD_RETRAIN_DATA_DIR" ]]; then
    run_cmd rm -rf "$HARD_RETRAIN_DATA_DIR"
  fi
  run_cmd python scripts/prepare_training_data.py \
    --base "$train_root" \
    --incremental "$ENS_OUT/hard_mined" \
    --out "$HARD_RETRAIN_DATA_DIR"

  PIPELINE_PREPARED_DATA_DIR="$HARD_RETRAIN_DATA_DIR"
  PIPELINE_IMAGE_TRAIN_DATA_DIR="$HARD_RETRAIN_DATA_DIR"
  PIPELINE_STAGE="dataset_qa"
  write_dataset_reports

  run_ensemble_training_bundle "$PIPELINE_IMAGE_TRAIN_DATA_DIR" "$HARD_RETRAIN_EPOCHS"
}

run_distill_bundle() {
  local train_root="$1"

  [[ "$RUN_DISTILL" == "1" ]] || return 0

  PIPELINE_STAGE="distill"
  if [[ -e "$ENS_OUT/distill" ]]; then
    run_cmd rm -rf "$ENS_OUT/distill"
  fi
  declare -a distill_cmd=(
    python scripts/train_distill.py
    --data "$train_root"
    --teacher "${ENSEMBLE_MODELS[@]}"
    --out "$ENS_OUT/distill"
    --student-backbone effb0
    --img-size 320
    --batch-size 64
    --epochs "$DISTILL_EPOCHS"
  )
  if [[ -f "$ENS_CONFIG_PATH" ]]; then
    distill_cmd+=(--ensemble-config "$ENS_CONFIG_PATH")
  fi
  run_cmd "${distill_cmd[@]}"
}

run_cmd bash scripts/install_deps.sh
activate_repo_venv
run_cmd mkdir -p "$PIPELINE_REPORT_DIR" "$ENS_OUT" "$VIDEO_ARTIFACTS_OUT"

if [[ "$SKIP_DATA" != "1" ]]; then
  PIPELINE_STAGE="collect_images"
  require_disk_free_gb "$PIPELINE_STAGE"
  dataset_cmd=(
    python scripts/build_best_dataset.py
    --out "$DATA_DIR"
    --train-per-class "$TRAIN_PER_CLASS"
    --val-per-class "$VAL_PER_CLASS"
    --test-per-class "$TEST_PER_CLASS"
    --near-hamming "$BEST_DS_NEAR_HAMMING"
    --near-window "$BEST_DS_NEAR_WINDOW"
    --min-side "$BEST_DS_MIN_SIDE"
    --max-aspect-ratio "$BEST_DS_MAX_ASPECT_RATIO"
    --min-entropy "$BEST_DS_MIN_ENTROPY"
    --max-unique-per-source "$BEST_DS_MAX_UNIQUE_PER_SOURCE"
    --max-per-source-class "$BEST_DS_MAX_PER_SOURCE_CLASS"
    --max-per-source-split-class "$BEST_DS_MAX_PER_SOURCE_SPLIT_CLASS"
    --jpeg-quality "$BEST_DS_JPEG_QUALITY"
    --hardneg-fraction "$BEST_DS_HARDNEG_FRACTION"
    --cache-dir "$BEST_DS_CACHE_DIR"
    --hf-cache-only-if-present
    --stream-buffer-size "$BEST_DS_STREAM_BUFFER_SIZE"
    --max-samples-per-source "$BEST_DS_MAX_SAMPLES_PER_SOURCE"
    --acceptance-warmup-samples "$BEST_DS_ACCEPTANCE_WARMUP_SAMPLES"
    --min-acceptance-rate "$BEST_DS_MIN_ACCEPTANCE_RATE"
    --min-hf-sources-with-accepted "$BEST_DS_MIN_HF_SOURCES_WITH_ACCEPTED"
    --min-hf-sources-per-class "$BEST_DS_MIN_HF_SOURCES_PER_CLASS"
    --min-hf-sources-per-split-class "$BEST_DS_MIN_HF_SOURCES_PER_SPLIT_CLASS"
    --repo-base-pause-ms "$BEST_DS_REPO_BASE_PAUSE_MS"
    --repo-jitter-ms "$BEST_DS_REPO_JITTER_MS"
    --repo-cooldown-ms "$BEST_DS_REPO_COOLDOWN_MS"
    --transient-error-cooldown-ms "$BEST_DS_TRANSIENT_ERROR_COOLDOWN_MS"
    --max-consecutive-failures "$BEST_DS_MAX_CONSECUTIVE_FAILURES"
    --require-full-targets
  )

  if [[ "$BEST_DS_STREAMING" == "1" ]]; then
    dataset_cmd+=(--streaming)
  else
    dataset_cmd+=(--no-streaming)
  fi

  if [[ "$BEST_DS_DISCOVER_HF" == "1" ]]; then
    dataset_cmd+=(
      --discover-hf
      --hf-discovery-limit "$BEST_DS_HF_DISCOVERY_LIMIT"
      --hf-max-sources "$BEST_DS_HF_MAX_SOURCES"
      --hf-min-downloads "$BEST_DS_HF_MIN_DOWNLOADS"
      --hf-min-likes "$BEST_DS_HF_MIN_LIKES"
      --hf-min-quality-score "$BEST_DS_HF_MIN_QUALITY_SCORE"
      --hf-print-top "$BEST_DS_HF_PRINT_TOP"
      --hf-query-pause-ms "$BEST_DS_HF_QUERY_PAUSE_MS"
      --hf-cache-file "$BEST_DS_HF_CACHE_FILE"
    )
  fi

  if [[ -n "$BEST_DS_HF_QUERIES" ]]; then
    IFS=',' read -r -a _queries <<< "$BEST_DS_HF_QUERIES"
    for q in "${_queries[@]}"; do
      q="$(echo "$q" | xargs)"
      [[ -z "$q" ]] && continue
      dataset_cmd+=(--hf-query "$q")
    done
  fi

  if [[ -n "$BEST_DS_SOURCES_FILE" ]]; then
    dataset_cmd+=(--sources-file "$BEST_DS_SOURCES_FILE")
  fi

  if [[ -n "$BEST_DS_EXTRA_SOURCES" ]]; then
    IFS=',' read -r -a _extra_sources <<< "$BEST_DS_EXTRA_SOURCES"
    for src in "${_extra_sources[@]}"; do
      src="$(echo "$src" | xargs)"
      [[ -z "$src" ]] && continue
      dataset_cmd+=(--extra-source "$src")
    done
  fi

  if [[ -n "$BEST_DS_LOCAL_SOURCES" && "$BEST_DS_HF_ONLY" != "1" ]]; then
    IFS=',' read -r -a _local_sources <<< "$BEST_DS_LOCAL_SOURCES"
    for src in "${_local_sources[@]}"; do
      src="$(echo "$src" | xargs)"
      [[ -z "$src" ]] && continue
      dataset_cmd+=(--local-source "$src")
    done
  fi
  if [[ "$BEST_DS_HF_ONLY" == "1" ]]; then
    dataset_cmd+=(--hf-only)
  fi
  if [[ "$BEST_DS_NO_DEFAULT_SOURCES" == "1" ]]; then
    dataset_cmd+=(--no-default-sources)
  fi

  run_cmd "${dataset_cmd[@]}"
  if [[ "${MALWARE_SCAN:-1}" == "1" ]]; then
    run_cmd bash scripts/malware_scan.sh "$DATA_DIR"
  fi
fi

if [[ "$RUN_VIDEO_DATA_PULL" == "1" ]]; then
  PIPELINE_STAGE="collect_video"
  require_disk_free_gb "$PIPELINE_STAGE"
  video_data_cmd=(
    python scripts/build_video_dataset.py
    --out "$VIDEO_OUT"
    --train-per-class "$VIDEO_TRAIN_PER_CLASS"
    --val-per-class "$VIDEO_VAL_PER_CLASS"
    --mode "$VIDEO_MODE"
    --cache-dir "$VIDEO_CACHE_DIR"
    --snapshot-max-workers "$VIDEO_SNAPSHOT_MAX_WORKERS"
    --repo-base-pause-ms "$VIDEO_REPO_BASE_PAUSE_MS"
    --repo-jitter-ms "$VIDEO_REPO_JITTER_MS"
    --copy-sleep-ms "$VIDEO_COPY_SLEEP_MS"
    --sleep-ms "$VIDEO_SLEEP_MS"
    --jitter-ms "$VIDEO_JITTER_MS"
    --chunk-pause-ms "$VIDEO_CHUNK_PAUSE_MS"
    --repo-cooldown-ms "$VIDEO_REPO_COOLDOWN_MS"
    --retries "$VIDEO_RETRIES"
    --min-video-bytes "$VIDEO_MIN_BYTES"
    --max-video-bytes "$VIDEO_MAX_BYTES"
  )
  run_cmd "${video_data_cmd[@]}"
  if [[ "${MALWARE_SCAN:-1}" == "1" ]]; then
    run_cmd bash scripts/malware_scan.sh "$VIDEO_OUT"
  fi
fi

PIPELINE_STAGE="dataset_qa"
write_dataset_reports

if [[ "$RUN_VIDEO_TRAIN" == "1" ]]; then
  PIPELINE_STAGE="train_video"
  require_disk_free_gb "$PIPELINE_STAGE"
  video_train_cmd=(
    aid-video-train
    --data "$VIDEO_OUT"
    --out "$VIDEO_ARTIFACTS_OUT"
    --epochs "$VIDEO_TRAIN_EPOCHS"
    --batch-size "$VIDEO_TRAIN_BATCH_SIZE"
    --img-size "$VIDEO_TRAIN_IMG_SIZE"
    --frames "$VIDEO_TRAIN_FRAMES"
    --grad-accum "$VIDEO_TRAIN_GRAD_ACCUM"
    --lr "$VIDEO_TRAIN_LR"
    --patience "$VIDEO_TRAIN_PATIENCE"
    --min-delta "$VIDEO_TRAIN_MIN_DELTA"
  )
  if [[ "$VIDEO_TRAIN_RESUME" == "1" ]]; then
    video_train_cmd+=(--resume "$VIDEO_ARTIFACTS_OUT/last_video.pt")
  fi
  run_cmd "${video_train_cmd[@]}"
fi

if [[ "$SKIP_SWEEP" != "1" ]]; then
  PIPELINE_STAGE="sweep"
  require_disk_free_gb "$PIPELINE_STAGE"
  run_cmd env EPOCHS="$SWEEP_EPOCHS" bash scripts/hparam_sweep.sh "$PIPELINE_IMAGE_TRAIN_DATA_DIR" "$SWEEP_OUT"
fi

run_ensemble_training_bundle "$PIPELINE_IMAGE_TRAIN_DATA_DIR" "$EPOCHS"
run_hard_mining_bundle "$PIPELINE_IMAGE_TRAIN_DATA_DIR"
run_distill_bundle "$PIPELINE_IMAGE_TRAIN_DATA_DIR"

PIPELINE_STAGE="finalize"
write_final_reports
trap - EXIT

echo "Pipeline complete."
echo "Prod manifest: $ENS_OUT/prod_manifest.json"
