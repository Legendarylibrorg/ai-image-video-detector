read_cmd_output_into_array() {
  local array_name="$1"
  shift
  local line=""
  eval "$array_name=()"
  while IFS= read -r line; do
    eval "$array_name+=(\"\$line\")"
  done < <("$@")
}

run_image_dataset_builder() {
  local out="$1"
  local query_csv="$2"
  shift 2
  local -a query_args=()
  read_cmd_output_into_array query_args print_hf_query_args "$query_csv"
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
  read_cmd_output_into_array query_args print_hf_query_args "$query_csv"
  run_repo_python_with_timeout "$timeout_sec" scripts/build_best_dataset.py \
    --out "$out" \
    "$@" \
    "${query_args[@]}" \
    --discover-only
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
  local max_per_source_class_default=""
  local max_per_source_split_class_default=""
  local warmup_default=""
  local min_sources_with_accepted_default=""
  local min_sources_per_class_default=""
  local base_pause_default=""
  local jitter_default=""
  local cooldown_default=""
  local min_side_default=""
  local hardneg_fraction_default=""
  local min_sources_per_split_class_default=""
  local hf_only_env=""
  local no_default_sources_env=""
  local sources_file_env=""
  local extra_sources_env=""
  local local_sources_env=""
  local verbose_progress_env=""

  case "$profile" in
    best)
      train_env="TRAIN_PER_CLASS"
      train_default="80000"
      val_env="VAL_PER_CLASS"
      val_default="20000"
      test_env="TEST_PER_CLASS"
      test_default="20000"
      prefix="BEST_DS"
      discovery_limit_default="220"
      max_sources_default="320"
      print_top_default="15"
      cache_file_default="./.local/hf_discovered_sources.txt"
      stream_buffer_default="12000"
      max_samples_default="30000"
      max_per_source_class_default="12000"
      max_per_source_split_class_default="4000"
      warmup_default="400"
      min_sources_with_accepted_default="28"
      min_sources_per_class_default="16"
      base_pause_default="1100"
      jitter_default="900"
      cooldown_default="45000"
      min_side_default="224"
      hardneg_fraction_default="0.35"
      min_sources_per_split_class_default="10"
      hf_only_env="BEST_DS_HF_ONLY"
      no_default_sources_env="BEST_DS_NO_DEFAULT_SOURCES"
      sources_file_env="BEST_DS_SOURCES_FILE"
      extra_sources_env="BEST_DS_EXTRA_SOURCES"
      local_sources_env="BEST_DS_LOCAL_SOURCES"
      verbose_progress_env="BEST_DS_VERBOSE_PROGRESS"
      ;;
    fast)
      train_env="FAST_TRAIN_PER_CLASS"
      train_default="4000"
      val_env="FAST_VAL_PER_CLASS"
      val_default="1000"
      test_env="FAST_TEST_PER_CLASS"
      test_default="1000"
      prefix="FAST"
      discovery_limit_default="120"
      max_sources_default="180"
      print_top_default="10"
      cache_file_default="./.local/hf_fast_sources.txt"
      stream_buffer_default="6000"
      max_samples_default="8000"
      max_per_source_class_default="3000"
      max_per_source_split_class_default="1000"
      warmup_default="250"
      min_sources_with_accepted_default="16"
      min_sources_per_class_default="10"
      base_pause_default="900"
      jitter_default="700"
      cooldown_default="30000"
      min_side_default="224"
      hardneg_fraction_default="0.25"
      min_sources_per_split_class_default="6"
      hf_only_env="FAST_HF_ONLY"
      no_default_sources_env="FAST_NO_DEFAULT_SOURCES"
      sources_file_env="FAST_SOURCES_FILE"
      extra_sources_env="FAST_EXTRA_SOURCES"
      local_sources_env="FAST_LOCAL_SOURCES"
      verbose_progress_env="FAST_VERBOSE_PROGRESS"
      ;;
    *)
      echo "unknown_image_collection_profile=$profile" >&2
      return 1
      ;;
  esac

  print_cli_flag_value_from_env_triplets \
    --train-per-class "$train_env" "$train_default" \
    --val-per-class "$val_env" "$val_default" \
    --test-per-class "$test_env" "$test_default"
  print_cli_flag --discover-hf
  print_cli_flag_value_from_env_triplets \
    --hf-discovery-limit "${prefix}_HF_DISCOVERY_LIMIT" "$discovery_limit_default" \
    --hf-max-sources "${prefix}_HF_MAX_SOURCES" "$max_sources_default" \
    --hf-min-downloads "${prefix}_HF_MIN_DOWNLOADS" "80" \
    --hf-min-likes "${prefix}_HF_MIN_LIKES" "2" \
    --hf-min-quality-score "${prefix}_HF_MIN_QUALITY_SCORE" "1.7" \
    --hf-print-top "${prefix}_HF_PRINT_TOP" "$print_top_default" \
    --hf-cache-file "${prefix}_HF_CACHE_FILE" "$cache_file_default"
  print_cli_flag --hf-cache-only-if-present
  print_cli_flag_value_from_env --cache-dir "${prefix}_CACHE_DIR" "./.local/hf"
  print_cli_flag --streaming
  print_cli_flag_value_from_env_triplets \
    --stream-buffer-size "${prefix}_STREAM_BUFFER_SIZE" "$stream_buffer_default" \
    --max-samples-per-source "${prefix}_MAX_SAMPLES_PER_SOURCE" "$max_samples_default" \
    --max-per-source-class "${prefix}_MAX_PER_SOURCE_CLASS" "$max_per_source_class_default" \
    --max-per-source-split-class "${prefix}_MAX_PER_SOURCE_SPLIT_CLASS" "$max_per_source_split_class_default" \
    --acceptance-warmup-samples "${prefix}_ACCEPTANCE_WARMUP_SAMPLES" "$warmup_default" \
    --min-acceptance-rate "${prefix}_MIN_ACCEPTANCE_RATE" "0.01" \
    --min-hf-sources-with-accepted "${prefix}_MIN_HF_SOURCES_WITH_ACCEPTED" "$min_sources_with_accepted_default" \
    --min-hf-sources-per-class "${prefix}_MIN_HF_SOURCES_PER_CLASS" "$min_sources_per_class_default" \
    --min-hf-sources-per-split-class "${prefix}_MIN_HF_SOURCES_PER_SPLIT_CLASS" "$min_sources_per_split_class_default" \
    --repo-base-pause-ms "${prefix}_REPO_BASE_PAUSE_MS" "$base_pause_default" \
    --repo-jitter-ms "${prefix}_REPO_JITTER_MS" "$jitter_default" \
    --repo-cooldown-ms "${prefix}_REPO_COOLDOWN_MS" "$cooldown_default" \
    --max-consecutive-failures "${prefix}_MAX_CONSECUTIVE_FAILURES" "2" \
    --min-side "${prefix}_MIN_SIDE" "$min_side_default" \
    --max-aspect-ratio "${prefix}_MAX_ASPECT_RATIO" "2.5" \
    --min-entropy "${prefix}_MIN_ENTROPY" "3.4" \
    --hardneg-fraction "${prefix}_HARDNEG_FRACTION" "$hardneg_fraction_default"
  if [[ -n "${!sources_file_env:-}" ]]; then
    print_cli_flag_value --sources-file "${!sources_file_env}"
  fi
  print_cli_flag_values_from_csv --extra-source "${!extra_sources_env:-}"
  if [[ "${!hf_only_env:-1}" != "1" ]]; then
    print_cli_flag_values_from_csv --local-source "${!local_sources_env:-}"
  else
    print_cli_flag --hf-only
  fi
  if [[ "${!no_default_sources_env:-1}" == "1" ]]; then
    print_cli_flag --no-default-sources
  fi
  if [[ "${!verbose_progress_env:-0}" == "1" ]]; then
    print_cli_flag --verbose-progress
  else
    print_cli_flag --quiet-progress
  fi
  print_cli_flag --require-full-targets
}

collect_image_data() {
  local out="${DATA_DIR:-./data_best}"
  local query_csv="${BEST_DS_HF_QUERIES:-$BEST_HF_QUERY_CSV_DEFAULT}"
  local -a build_args=()
  read_cmd_output_into_array build_args print_image_collection_args best
  run_image_dataset_build "$out" "$query_csv" "${build_args[@]}"
}

collect_fast_data() {
  local out="${DATA_DIR:-./data_best_fast}"
  local query_csv="${FAST_HF_QUERIES:-${BEST_DS_HF_QUERIES:-$BEST_HF_QUERY_CSV_DEFAULT}}"
  local -a build_args=()
  read_cmd_output_into_array build_args print_image_collection_args fast
  run_image_dataset_build "$out" "$query_csv" "${build_args[@]}"
}

ingest_outputs() {
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
  print_cli_flag_value_from_env --cache-dir "DIVERSE_CACHE_DIR" "./.local/hf"
  print_cli_flag --streaming
  print_cli_flag_value_from_env_triplets \
    --stream-buffer-size "DIVERSE_STREAM_BUFFER_SIZE" "16000" \
    --max-samples-per-source "DIVERSE_MAX_SAMPLES_PER_SOURCE" "80000" \
    --max-per-source-class "DIVERSE_MAX_PER_SOURCE_CLASS" "16000" \
    --max-per-source-split-class "DIVERSE_MAX_PER_SOURCE_SPLIT_CLASS" "5500" \
    --acceptance-warmup-samples "DIVERSE_ACCEPTANCE_WARMUP_SAMPLES" "256" \
    --min-acceptance-rate "DIVERSE_MIN_ACCEPTANCE_RATE" "0.015" \
    --min-hf-sources-with-accepted "DIVERSE_MIN_HF_SOURCES_WITH_ACCEPTED" "24" \
    --min-hf-sources-per-class "DIVERSE_MIN_HF_SOURCES_PER_CLASS" "14" \
    --min-hf-sources-per-split-class "DIVERSE_MIN_HF_SOURCES_PER_SPLIT_CLASS" "8" \
    --repo-base-pause-ms "DIVERSE_REPO_BASE_PAUSE_MS" "150" \
    --repo-jitter-ms "DIVERSE_REPO_JITTER_MS" "150" \
    --repo-cooldown-ms "DIVERSE_REPO_COOLDOWN_MS" "15000" \
    --transient-error-cooldown-ms "DIVERSE_TRANSIENT_ERROR_COOLDOWN_MS" "2500" \
    --max-consecutive-failures "DIVERSE_MAX_CONSECUTIVE_FAILURES" "5" \
    --min-side "DIVERSE_MIN_SIDE" "192" \
    --max-aspect-ratio "DIVERSE_MAX_ASPECT_RATIO" "3.2" \
    --min-entropy "DIVERSE_MIN_ENTROPY" "3.1" \
    --hardneg-fraction "DIVERSE_HARDNEG_FRACTION" "0.5"
  if [[ "${DIVERSE_VERBOSE_PROGRESS:-0}" == "1" ]]; then
    print_cli_flag --verbose-progress
  else
    print_cli_flag --quiet-progress
  fi
  print_cli_flags --hf-only --require-full-targets
}

print_diverse_discovery_args() {
  print_cli_flag --discover-hf
  print_cli_flag_value_from_env_triplets \
    --hf-discovery-limit "DIVERSE_HF_DISCOVERY_LIMIT" "140" \
    --hf-max-sources "DIVERSE_HF_MAX_SOURCES" "320" \
    --hf-min-downloads "DIVERSE_HF_MIN_DOWNLOADS" "100" \
    --hf-min-likes "DIVERSE_HF_MIN_LIKES" "2" \
    --hf-min-quality-score "DIVERSE_HF_MIN_QUALITY_SCORE" "1.85" \
    --hf-print-top "DIVERSE_HF_PRINT_TOP" "20" \
    --hf-query-pause-ms "DIVERSE_HF_QUERY_PAUSE_MS" "900"
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

audit_image_dataset() {
  local out="${1:-${DATA_DIR:-./data_best}}"
  local -a audit_args=()
  read_cmd_output_into_array audit_args print_diverse_audit_args 1
  run_repo_python scripts/audit_diversity.py \
    --data "$out" \
    "${audit_args[@]}"
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

  read_cmd_output_into_array common_args print_diverse_common_args
  full_args=("${common_args[@]}")
  read_cmd_output_into_array discover_args print_diverse_discovery_args
  read_cmd_output_into_array audit_args print_diverse_audit_args

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
    --cache-dir "VIDEO_CACHE_DIR" "./.local/hf" \
    --snapshot-max-workers "VIDEO_SNAPSHOT_MAX_WORKERS" "4" \
    --repo-base-pause-ms "VIDEO_REPO_BASE_PAUSE_MS" "150" \
    --repo-jitter-ms "VIDEO_REPO_JITTER_MS" "150" \
    --copy-sleep-ms "VIDEO_COPY_SLEEP_MS" "0" \
    --sleep-ms "VIDEO_SLEEP_MS" "40" \
    --jitter-ms "VIDEO_JITTER_MS" "20" \
    --chunk-pause-ms "VIDEO_CHUNK_PAUSE_MS" "250" \
    --repo-cooldown-ms "VIDEO_REPO_COOLDOWN_MS" "12000" \
    --retries "VIDEO_RETRIES" "5" \
    --min-video-bytes "VIDEO_MIN_BYTES" "100000" \
    --max-video-bytes "VIDEO_MAX_BYTES" "0"
}

collect_video_data() {
  local -a video_args=()
  read_cmd_output_into_array video_args print_video_collection_args
  run_repo_python scripts/build_video_dataset.py "${video_args[@]}"
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

run_pipeline_collection_stage() {
  wait_for_training_to_finish "pipeline_stage=collect"
  collect_diverse_cycle
}

run_simple_collection_smoke() {
  wait_for_training_to_finish "pipeline_stage=smoke"
  collect_fast_data
}
