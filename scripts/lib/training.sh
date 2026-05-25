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

run_dataset_layout() {
  repo_python -m ai_image_detector.dataset_layout "$@"
}

require_min_image_counts() {
  local data_root="$1"
  local train_min="${2:-0}"
  local val_min="${3:-0}"
  local test_min="${4:-0}"
  run_dataset_layout check-image-minimums \
    --root "$data_root" \
    --train-min "$train_min" \
    --val-min "$val_min" \
    --test-min "$test_min"
}

require_image_training_data() {
  local data_root="$1"
  run_dataset_layout check-image-complete --root "$data_root"
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
      run_repo_python -c 'import json,sys
path=sys.argv[1]
try:
    data=json.loads(open(path,encoding="utf-8").read())
except Exception:
    print("")
    raise SystemExit(0)
print("1" if bool(data.get("full_targets_ok",False)) else "0")' "$report_path"
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
  run_dataset_layout check-video-complete --root "$video_root" --quiet >/dev/null 2>&1
}

prepare_training_image_data() {
  local base_root="${DATA_DIR:-./data_best}"
  local incremental_root=""
  incremental_root="$(resolve_incremental_image_root)"
  local out_root="${TRAIN_READY_DATA_DIR:-./.local/training_data}"
  local -a cmd=(
    scripts/prepare_training_data.py
    --base "$base_root"
    --incremental "$incremental_root"
    --out "$out_root"
  )
  if [[ "${TRAIN_DATA_COPY_ONLY:-0}" == "1" ]]; then
    cmd+=(--copy)
  fi
  ensure_env
  run_repo_python "${cmd[@]}"
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

run_prepared_max_quality_pipeline() {
  local collected_root="$1"
  local disable_video_train="${2:-0}"
  local -a env_args=(
    DATA_DIR="$PREPARED_IMAGE_DATA_DIR"
    TRAIN_READY_DATA_DIR="$PREPARED_IMAGE_DATA_DIR"
    PIPELINE_COLLECTED_DATA_DIR="$collected_root"
    PIPELINE_PREPARED_DATA_DIR="$PREPARED_IMAGE_DATA_DIR"
    SKIP_DATA=1
    RUN_VIDEO_DATA_PULL=0
  )
  if [[ "$disable_video_train" == "1" ]]; then
    env_args+=(RUN_VIDEO_TRAIN=0)
  fi
  env_args+=(PIPELINE_PROFILE=max_quality)
  env "${env_args[@]}" bash scripts/full_pipeline_4090.sh
}

train_existing_pipeline() {
  require_pipeline_collection_data "${DATA_DIR:-./data_best}" || return 1
  prepare_training_image_data
  local collected_root="${DATA_DIR:-./data_best}"
  if have_complete_video_training_data "${VIDEO_OUT:-./video_data}"; then
    echo "train_mode=image_plus_video"
    run_prepared_max_quality_pipeline "$collected_root"
    return
  fi
  echo "train_mode=image_only reason=video_data_missing"
  run_prepared_max_quality_pipeline "$collected_root" 1
}

run_benchmark_gate() {
  local -a gate_args=(
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
}

run_retrain_pipeline() {
  bash scripts/do.sh train-existing
  run_benchmark_gate
}

run_review_queue_ingest() {
  run_repo_python scripts/review_queue_to_dataset.py \
    --queue "${REVIEW_QUEUE_DIR:-./incoming_review_queue}" \
    --dst "${NEW_DATA_DST:-./data_new/train}" || true
}

run_weekly_retrain_cycle() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) weekly_retrain_start"
  run_review_queue_ingest
  if ! run_retrain_pipeline; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) weekly_retrain_pipeline_failed"
    return 1
  fi
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) weekly_retrain_gate_passed"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) weekly_retrain_done"
}

run_full_pipeline() {
  wait_for_training_to_finish "pipeline"
  with_training_lock env PIPELINE_PROFILE=max_quality bash scripts/full_pipeline_4090.sh
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
  # Keep status/reporting commands read-only so they stay usable offline and in fresh checkouts.
  repo_python -m ai_image_detector.dataset_tools collection-status \
    --data "${DATA_DIR:-./data_best}" \
    --incremental "$(resolve_incremental_image_root)" \
    --prepared "${TRAIN_READY_DATA_DIR:-./.local/training_data}" \
    --video "${VIDEO_OUT:-./video_data}" \
    "$@"
}
