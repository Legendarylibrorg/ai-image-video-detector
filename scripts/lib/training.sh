resolve_incremental_image_root() {
  if [[ -n "${TRAIN_INCREMENTAL_DATA_DIR:-}" ]]; then
    echo "${TRAIN_INCREMENTAL_DATA_DIR}"
    return
  fi
  local new_data_dst="${NEW_DATA_DST:-./data_new/train}"
  if [[ "$(basename "$new_data_dst")" == "train" ]]; then
    echo "$(dirname "$new_data_dst")"
    return
  fi
  echo "$new_data_dst"
}

bucket_has_files() {
  local dir="$1"
  shift
  [[ -d "$dir" ]] || return 1
  local -a expr=()
  local pattern=""
  for pattern in "$@"; do
    if [[ "${#expr[@]}" -gt 0 ]]; then
      expr+=(-o)
    fi
    expr+=(-iname "$pattern")
  done
  local first_match=""
  first_match="$(find "$dir" -maxdepth 1 -type f \( "${expr[@]}" \) -print -quit)"
  [[ -n "$first_match" ]]
}

image_bucket_has_files() {
  bucket_has_files "$1" "*.jpg" "*.jpeg" "*.png" "*.webp" "*.bmp" "*.tif" "*.tiff"
}

video_bucket_has_files() {
  bucket_has_files "$1" "*.mp4" "*.mov" "*.avi" "*.mkv" "*.webm" "*.m4v"
}

count_bucket_files() {
  local dir="$1"
  shift
  if [[ ! -d "$dir" ]]; then
    echo 0
    return
  fi
  local -a expr=()
  local pattern=""
  for pattern in "$@"; do
    if [[ "${#expr[@]}" -gt 0 ]]; then
      expr+=(-o)
    fi
    expr+=(-iname "$pattern")
  done
  find "$dir" -maxdepth 1 -type f \( "${expr[@]}" \) | wc -l | tr -d ' '
}

require_min_image_counts() {
  local data_root="$1"
  local train_min="${2:-0}"
  local val_min="${3:-0}"
  local test_min="${4:-0}"
  local failed=0
  local split=""
  local cls=""
  local min_required=0
  local count=0

  for split in train val test; do
    case "$split" in
      train) min_required="$train_min" ;;
      val) min_required="$val_min" ;;
      test) min_required="$test_min" ;;
    esac
    [[ "$min_required" =~ ^[0-9]+$ ]] || min_required=0
    if (( min_required <= 0 )); then
      continue
    fi
    for cls in ai real; do
      count="$(count_bucket_files "$data_root/$split/$cls" "*.jpg" "*.jpeg" "*.png" "*.webp" "*.bmp" "*.tif" "*.tiff")"
      if (( count < min_required )); then
        echo "insufficient_image_bucket=$data_root/$split/$cls have=$count need=$min_required"
        failed=1
      fi
    done
  done

  if [[ "$failed" == "1" ]]; then
    echo "image_collection_counts=invalid root=$data_root train_min=$train_min val_min=$val_min test_min=$test_min"
    return 1
  fi
  echo "image_collection_counts=ok root=$data_root train_min=$train_min val_min=$val_min test_min=$test_min"
}

require_image_training_data() {
  local data_root="$1"
  local missing=0
  local split=""
  local cls=""
  for split in train val test; do
    for cls in ai real; do
      if ! image_bucket_has_files "$data_root/$split/$cls"; then
        echo "missing_image_bucket=$data_root/$split/$cls"
        missing=1
      fi
    done
  done
  if [[ "$missing" == "1" ]]; then
    echo "image_training_data=invalid root=$data_root"
    return 1
  fi
  echo "image_training_data=ok root=$data_root"
}

require_pipeline_collection_data() {
  local data_root="${1:-${DATA_DIR:-./data_best}}"
  local train_min="${PIPELINE_MIN_TRAIN_PER_CLASS:-${TRAIN_PER_CLASS:-0}}"
  local val_min="${PIPELINE_MIN_VAL_PER_CLASS:-${VAL_PER_CLASS:-0}}"
  local test_min="${PIPELINE_MIN_TEST_PER_CLASS:-${TEST_PER_CLASS:-0}}"
  local report_path="$data_root/dataset_build_report.json"
  local have_explicit_mins=0

  if [[ -n "${PIPELINE_MIN_TRAIN_PER_CLASS:-}" || -n "${PIPELINE_MIN_VAL_PER_CLASS:-}" || -n "${PIPELINE_MIN_TEST_PER_CLASS:-}" || -n "${TRAIN_PER_CLASS:-}" || -n "${VAL_PER_CLASS:-}" || -n "${TEST_PER_CLASS:-}" ]]; then
    have_explicit_mins=1
  fi

  if [[ -f "$report_path" ]]; then
    local full_targets_ok=""
    full_targets_ok="$(
      python3 - <<'PY' "$report_path"
import json, sys
path = sys.argv[1]
try:
    data = json.loads(open(path, encoding="utf-8").read())
except Exception:
    print("")
    raise SystemExit(0)
print("1" if bool(data.get("full_targets_ok", False)) else "0")
PY
    )"
    if [[ "$full_targets_ok" != "1" ]]; then
      echo "collection_build_report=invalid path=$report_path full_targets_ok=0"
      return 1
    fi
    echo "collection_build_report=ok path=$report_path"
    if [[ "$have_explicit_mins" != "1" ]]; then
      echo "collection_min_counts=skipped reason=build_report_ok"
      return 0
    fi
  fi

  require_min_image_counts "$data_root" "$train_min" "$val_min" "$test_min"
}

have_complete_video_training_data() {
  local video_root="${1:-${VIDEO_OUT:-./video_data}}"
  local split=""
  local cls=""
  for split in train val; do
    for cls in ai real; do
      if ! video_bucket_has_files "$video_root/$split/$cls"; then
        return 1
      fi
    done
  done
  return 0
}

require_video_training_data() {
  local video_root="${1:-${VIDEO_OUT:-./video_data}}"
  if have_complete_video_training_data "$video_root"; then
    echo "video_training_data=ok root=$video_root"
    return 0
  fi
  echo "video_training_data=invalid root=$video_root"
  return 1
}

prepare_training_image_data() {
  local base_root="${DATA_DIR:-./data_best}"
  local incremental_root=""
  incremental_root="$(resolve_incremental_image_root)"
  local out_root="${TRAIN_READY_DATA_DIR:-./.local/training_data}"
  local -a cmd=(
    python scripts/prepare_training_data.py
    --base "$base_root"
    --incremental "$incremental_root"
    --out "$out_root"
  )
  if [[ "${TRAIN_DATA_COPY_ONLY:-0}" == "1" ]]; then
    cmd+=(--copy)
  fi
  ensure_env
  "${cmd[@]}"
  require_image_training_data "$out_root"
  if [[ "${TRAIN_REQUIRE_MIN_COUNTS:-0}" == "1" ]]; then
    require_min_image_counts \
      "$out_root" \
      "${PIPELINE_MIN_TRAIN_PER_CLASS:-${TRAIN_PER_CLASS:-0}}" \
      "${PIPELINE_MIN_VAL_PER_CLASS:-${VAL_PER_CLASS:-0}}" \
      "${PIPELINE_MIN_TEST_PER_CLASS:-${TEST_PER_CLASS:-0}}"
  fi
  PREPARED_IMAGE_DATA_DIR="$out_root"
}

train_image_pipeline() {
  prepare_training_image_data
  local collected_root="${DATA_DIR:-./data_best}"
  env DATA_DIR="$PREPARED_IMAGE_DATA_DIR" \
    TRAIN_READY_DATA_DIR="$PREPARED_IMAGE_DATA_DIR" \
    PIPELINE_COLLECTED_DATA_DIR="$collected_root" \
    PIPELINE_PREPARED_DATA_DIR="$PREPARED_IMAGE_DATA_DIR" \
    SKIP_DATA=1 RUN_VIDEO_DATA_PULL=0 RUN_VIDEO_TRAIN=0 bash scripts/max_quality_4090.sh
}

train_all_pipeline() {
  prepare_training_image_data
  require_video_training_data "${VIDEO_OUT:-./video_data}"
  local collected_root="${DATA_DIR:-./data_best}"
  env DATA_DIR="$PREPARED_IMAGE_DATA_DIR" \
    TRAIN_READY_DATA_DIR="$PREPARED_IMAGE_DATA_DIR" \
    PIPELINE_COLLECTED_DATA_DIR="$collected_root" \
    PIPELINE_PREPARED_DATA_DIR="$PREPARED_IMAGE_DATA_DIR" \
    SKIP_DATA=1 RUN_VIDEO_DATA_PULL=0 bash scripts/max_quality_4090.sh
}

train_existing_pipeline() {
  prepare_training_image_data
  local collected_root="${DATA_DIR:-./data_best}"
  if have_complete_video_training_data "${VIDEO_OUT:-./video_data}"; then
    echo "train_mode=image_plus_video"
    env DATA_DIR="$PREPARED_IMAGE_DATA_DIR" \
      TRAIN_READY_DATA_DIR="$PREPARED_IMAGE_DATA_DIR" \
      PIPELINE_COLLECTED_DATA_DIR="$collected_root" \
      PIPELINE_PREPARED_DATA_DIR="$PREPARED_IMAGE_DATA_DIR" \
      SKIP_DATA=1 RUN_VIDEO_DATA_PULL=0 bash scripts/max_quality_4090.sh
    return
  fi
  echo "train_mode=image_only reason=video_data_missing"
  env DATA_DIR="$PREPARED_IMAGE_DATA_DIR" \
    TRAIN_READY_DATA_DIR="$PREPARED_IMAGE_DATA_DIR" \
    PIPELINE_COLLECTED_DATA_DIR="$collected_root" \
    PIPELINE_PREPARED_DATA_DIR="$PREPARED_IMAGE_DATA_DIR" \
    SKIP_DATA=1 RUN_VIDEO_DATA_PULL=0 RUN_VIDEO_TRAIN=0 bash scripts/max_quality_4090.sh
}

run_pipeline_training_stage() {
  local train_min="${PIPELINE_MIN_TRAIN_PER_CLASS:-${DIVERSE_TRAIN_PER_CLASS:-100000}}"
  local val_min="${PIPELINE_MIN_VAL_PER_CLASS:-${DIVERSE_VAL_PER_CLASS:-25000}}"
  local test_min="${PIPELINE_MIN_TEST_PER_CLASS:-${DIVERSE_TEST_PER_CLASS:-25000}}"
  wait_for_training_to_finish "pipeline_stage=train"
  PIPELINE_MIN_TRAIN_PER_CLASS="$train_min" \
  PIPELINE_MIN_VAL_PER_CLASS="$val_min" \
  PIPELINE_MIN_TEST_PER_CLASS="$test_min" \
  require_pipeline_collection_data "${DATA_DIR:-./data_best}"
  PIPELINE_MIN_TRAIN_PER_CLASS="$train_min" \
  PIPELINE_MIN_VAL_PER_CLASS="$val_min" \
  PIPELINE_MIN_TEST_PER_CLASS="$test_min" \
  TRAIN_REQUIRE_MIN_COUNTS=1 with_training_lock train_existing_pipeline
}

run_pipeline_validation_stage() {
  validate_train_artifacts
}

run_full_pipeline() {
  run_pipeline_stage collect run_pipeline_collection_stage
  run_pipeline_stage train run_pipeline_training_stage
  run_pipeline_stage validate run_pipeline_validation_stage
}

validate_train_artifacts() {
  local ens_dir="${ENS_OUT:-./artifacts_ens}"
  local vid_best_pt="${VIDEO_ARTIFACTS_OUT:-./video_artifacts}/best_video.pt"
  local vid_best_sft="${VIDEO_ARTIFACTS_OUT:-./video_artifacts}/best_video.safetensors"
  local missing=0
  local video_required=0

  case "${VALIDATE_REQUIRE_VIDEO:-auto}" in
    1|true|yes|always)
      video_required=1
      ;;
    0|false|no|never)
      video_required=0
      ;;
    *)
      if have_complete_video_training_data "${VIDEO_OUT:-./video_data}"; then
        video_required=1
      fi
      ;;
  esac

  for p in "$ens_dir/m1/best.safetensors" "$ens_dir/m2/best.safetensors" "$ens_dir/m3/best.safetensors" "$ens_dir/m4/best.safetensors" "$ens_dir/test_metrics.json" "$ens_dir/prod_manifest.json"; do
    if [[ ! -f "$p" ]]; then
      echo "missing_artifact=$p"
      missing=1
    fi
  done
  if [[ "$video_required" == "1" && ! -f "$vid_best_sft" && ! -f "$vid_best_pt" ]]; then
    echo "missing_artifact=$vid_best_sft"
    missing=1
  elif [[ "$video_required" != "1" ]]; then
    echo "artifact_validation_video=skipped reason=video_optional"
  fi

  if [[ "$missing" == "1" ]]; then
    echo "artifact_validation=failed"
    return 1
  fi
  echo "artifact_validation=ok"
}

train_video_only() {
  require_video_training_data "${VIDEO_OUT:-./video_data}"
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
  echo "incremental image data: $(resolve_incremental_image_root)"
  echo "prepared training data: ${TRAIN_READY_DATA_DIR:-./.local/training_data}"
  echo "video data: ${VIDEO_OUT:-./video_data}"
  echo "image ensemble: ${ENS_OUT:-./artifacts_ens}"
  echo "video model: ${VIDEO_ARTIFACTS_OUT:-./video_artifacts}/best_video.safetensors"
}

show_collection_status() {
  run_repo_python -m ai_image_detector.dataset_tools collection-status \
    --data "${DATA_DIR:-./data_best}" \
    --incremental "$(resolve_incremental_image_root)" \
    --prepared "${TRAIN_READY_DATA_DIR:-./.local/training_data}" \
    --video "${VIDEO_OUT:-./video_data}" \
    "$@"
}
