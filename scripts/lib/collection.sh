HF_CACHE_DIR_DEFAULT="${HF_CACHE_DIR_DEFAULT:-./.local/hf}"

run_image_dataset_builder() {
  local out="$1"
  local query_csv="$2"
  shift 2
  local -a query_args=()
  read_aid_csv_cli_buf --hf-query "$query_csv"
  query_args+=("${AID_CSV_CLI_BUF[@]}")
  run_repo_python scripts/build_best_dataset.py \
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
  read_aid_csv_cli_buf --hf-query "$query_csv"
  query_args+=("${AID_CSV_CLI_BUF[@]}")
  run_repo_python_with_timeout "$timeout_sec" scripts/build_best_dataset.py \
    --out "$out" \
    "$@" \
    "${query_args[@]}" \
    --discover-only
}

ingest_outputs() {
  run_malware_scan "${NEW_DATA_DST:-./data_new/train}" "${MODEL_OUTPUTS_SRC:-./incoming_model_outputs}"
  run_repo_python scripts/ingest_model_outputs.py \
    --src "${MODEL_OUTPUTS_SRC:-./incoming_model_outputs}" \
    --dst "${NEW_DATA_DST:-./data_new/train}" \
    --archive "${MODEL_OUTPUTS_ARCHIVE:-./incoming_model_outputs/_processed}"
  run_malware_scan "${NEW_DATA_DST:-./data_new/train}" "${MODEL_OUTPUTS_SRC:-./incoming_model_outputs}"
}

print_diverse_common_args() {
  print_cli_flag_value_from_env_triplets \
    --train-per-class "DIVERSE_TRAIN_PER_CLASS" "100000" \
    --val-per-class "DIVERSE_VAL_PER_CLASS" "25000" \
    --test-per-class "DIVERSE_TEST_PER_CLASS" "25000" \
    --hf-cache-file "DIVERSE_HF_CACHE_FILE" "./.local/hf_diverse_sources.txt"
  print_cli_flag --hf-cache-only-if-present
  print_cli_flag_value_from_env --cache-dir "DIVERSE_CACHE_DIR" "$HF_CACHE_DIR_DEFAULT"
  print_cli_flag --streaming
  print_cli_flag_value_from_env_triplets \
    --stream-buffer-size "DIVERSE_STREAM_BUFFER_SIZE" "24000" \
    --max-samples-per-source "DIVERSE_MAX_SAMPLES_PER_SOURCE" "22000" \
    --max-per-source-class "DIVERSE_MAX_PER_SOURCE_CLASS" "6000" \
    --max-per-source-split-class "DIVERSE_MAX_PER_SOURCE_SPLIT_CLASS" "1800" \
    --acceptance-warmup-samples "DIVERSE_ACCEPTANCE_WARMUP_SAMPLES" "160" \
    --min-acceptance-rate "DIVERSE_MIN_ACCEPTANCE_RATE" "0.008" \
    --min-hf-sources-with-accepted "DIVERSE_MIN_HF_SOURCES_WITH_ACCEPTED" "72" \
    --min-hf-sources-per-class "DIVERSE_MIN_HF_SOURCES_PER_CLASS" "32" \
    --min-hf-sources-per-split-class "DIVERSE_MIN_HF_SOURCES_PER_SPLIT_CLASS" "20" \
    --repo-base-pause-ms "DIVERSE_REPO_BASE_PAUSE_MS" "10" \
    --repo-jitter-ms "DIVERSE_REPO_JITTER_MS" "10" \
    --repo-cooldown-ms "DIVERSE_REPO_COOLDOWN_MS" "2500" \
    --transient-error-cooldown-ms "DIVERSE_TRANSIENT_ERROR_COOLDOWN_MS" "800" \
    --max-consecutive-failures "DIVERSE_MAX_CONSECUTIVE_FAILURES" "8" \
    --min-side "DIVERSE_MIN_SIDE" "160" \
    --max-aspect-ratio "DIVERSE_MAX_ASPECT_RATIO" "4.0" \
    --min-entropy "DIVERSE_MIN_ENTROPY" "3.1" \
    --hardneg-fraction "DIVERSE_HARDNEG_FRACTION" "0.5"
  if [[ "${DIVERSE_VERBOSE_PROGRESS:-0}" == "1" ]]; then
    print_cli_flag --verbose-progress
  else
    print_cli_flag --quiet-progress
  fi
  print_cli_flag --require-full-targets
}

print_diverse_discovery_args() {
  print_cli_flag --discover-hf
  print_cli_flag_value_from_env_triplets \
    --hf-discovery-limit "DIVERSE_HF_DISCOVERY_LIMIT" "480" \
    --hf-max-sources "DIVERSE_HF_MAX_SOURCES" "1200" \
    --hf-min-downloads "DIVERSE_HF_MIN_DOWNLOADS" "10" \
    --hf-min-likes "DIVERSE_HF_MIN_LIKES" "1" \
    --hf-min-quality-score "DIVERSE_HF_MIN_QUALITY_SCORE" "1.15" \
    --hf-print-top "DIVERSE_HF_PRINT_TOP" "48" \
    --hf-discovery-workers "DIVERSE_HF_DISCOVERY_WORKERS" "12" \
    --hf-query-pause-ms "DIVERSE_HF_QUERY_PAUSE_MS" "0"
}

print_diverse_audit_args() {
  local include_split_gate="${1:-0}"
  if [[ "$include_split_gate" == "1" ]]; then
    print_cli_flag_value_from_env --min-sources-per-split-class "DIVERSE_MIN_HF_SOURCES_PER_SPLIT_CLASS" "8"
  fi
  print_cli_flag_value_from_env_triplets \
    --min-unique-sources "DIVERSE_MIN_UNIQUE_SOURCES" "20" \
    --min-hardneg-modes "DIVERSE_MIN_HARDNEG_MODES" "4" \
    --max-class-imbalance "DIVERSE_MAX_CLASS_IMBALANCE" "0.08" \
    --max-source-share-per-split "DIVERSE_MAX_SOURCE_SHARE_PER_SPLIT" "0.22" \
    --max-source-share-per-split-class "DIVERSE_MAX_SOURCE_SHARE_PER_SPLIT_CLASS" "0.3"
}

collect_diverse_image_data() {
  local out="${DATA_DIR:-./data_best}"
  local query_csv="${DIVERSE_HF_QUERIES:-$BEST_HF_QUERY_CSV_DEFAULT}"
  local hf_cache="${DIVERSE_HF_CACHE_FILE:-./.local/hf_diverse_sources.txt}"
  local timeout_sec="${DIVERSE_DISCOVERY_TIMEOUT_SEC:-900}"
  local -a common_args=()
  local -a discover_args=()
  local -a cache_args=(--no-discover-hf --sources-file "$hf_cache")
  local -a full_args=()
  local -a audit_args=()
  local line=""

  while IFS= read -r line; do
    common_args+=("$line")
  done < <(print_diverse_common_args)
  full_args=("${common_args[@]}")
  while IFS= read -r line; do
    discover_args+=("$line")
  done < <(print_diverse_discovery_args)
  while IFS= read -r line; do
    audit_args+=("$line")
  done < <(print_diverse_audit_args)

  run_malware_scan "$out"

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

  run_repo_python scripts/audit_diversity.py \
    --data "$out" \
    "${audit_args[@]}"
}

print_video_collection_args() {
  print_cli_flag_value_from_env_triplets \
    --out "VIDEO_OUT" "./video_data" \
    --train-per-class "VIDEO_TRAIN_PER_CLASS" "1000" \
    --val-per-class "VIDEO_VAL_PER_CLASS" "250" \
    --mode "VIDEO_MODE" "snapshot" \
    --cache-dir "VIDEO_CACHE_DIR" "$HF_CACHE_DIR_DEFAULT" \
    --snapshot-max-workers "VIDEO_SNAPSHOT_MAX_WORKERS" "8" \
    --repo-base-pause-ms "VIDEO_REPO_BASE_PAUSE_MS" "150" \
    --repo-jitter-ms "VIDEO_REPO_JITTER_MS" "150" \
    --copy-sleep-ms "VIDEO_COPY_SLEEP_MS" "0" \
    --sleep-ms "VIDEO_SLEEP_MS" "40" \
    --jitter-ms "VIDEO_JITTER_MS" "20" \
    --chunk-pause-ms "VIDEO_CHUNK_PAUSE_MS" "250" \
    --repo-cooldown-ms "VIDEO_REPO_COOLDOWN_MS" "12000" \
    --retries "VIDEO_RETRIES" "5" \
    --min-video-bytes "VIDEO_MIN_BYTES" "200000" \
    --max-video-bytes "VIDEO_MAX_BYTES" "0"
}

collect_video_data() {
  local -a video_args=()
  local line=""
  while IFS= read -r line; do
    video_args+=("$line")
  done < <(print_video_collection_args)
  run_malware_scan "${VIDEO_OUT:-./video_data}"
  run_repo_python scripts/build_video_dataset.py "${video_args[@]}"
  run_malware_scan "${VIDEO_OUT:-./video_data}"
}

collect_full_cycle() {
  collect_diverse_image_data
  ingest_outputs
  collect_video_data
}

run_simple_collection_smoke() {
  wait_for_training_to_finish "pipeline_stage=smoke"
  ensure_env
  bash scripts/smoke_resume_eval.sh
}
