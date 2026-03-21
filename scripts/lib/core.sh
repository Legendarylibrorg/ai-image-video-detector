load_env_file() {
  if [[ ! -f "$ENV_FILE" ]]; then
    return
  fi
  local -a env_names=()
  local line=""
  local name=""
  local restore_script=""
  while IFS= read -r line; do
    case "$line" in
      ''|\#*) continue ;;
    esac
    if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
      name="${line%%=*}"
      env_names+=("$name")
      if eval '[[ ${'"$name"'+x} && -n "${'"$name"'}" ]]'; then
        restore_script+="$(eval "printf '%s=%q\n' '$name' \"\${$name}\"")"$'\n'
      fi
    fi
  done < "$ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  if [[ -n "$restore_script" ]]; then
    eval "$restore_script"
  fi
}

ensure_env() {
  if [[ "$ENV_READY" == "1" ]]; then
    return
  fi
  local venv_dir="${VENV_DIR:-$ROOT_DIR/.venv}"
  # Keep dependency bootstrap chatter off stdout so status commands can stay machine-readable.
  bash scripts/install_deps.sh >&2
  if [[ ! -f "$venv_dir/bin/activate" ]]; then
    echo "missing_virtualenv_activate=$venv_dir/bin/activate run=bash scripts/install_deps.sh" >&2
    return 1
  fi
  # shellcheck disable=SC1091
  source "$venv_dir/bin/activate"
  ENV_READY=1
}

is_training_active() {
  [[ -f "$TRAIN_LOCK" ]]
}

stage_file() {
  local stage="$1"
  echo "$PIPELINE_STAGE_DIR/${stage}.done"
}

stage_done() {
  local stage="$1"
  if [[ "$PIPELINE_FORCE_STAGES" == "1" ]]; then
    return 1
  fi
  [[ -f "$(stage_file "$stage")" ]]
}

mark_stage_done() {
  local stage="$1"
  mkdir -p "$PIPELINE_STAGE_DIR"
  printf "%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$(stage_file "$stage")"
}

wait_for_training_to_finish() {
  local reason="${1:-pipeline}"
  while is_training_active; do
    echo "${reason}: training lock present ($TRAIN_LOCK), sleeping ${PIPELINE_WAIT_FOR_TRAINING_SEC}s"
    sleep "$PIPELINE_WAIT_FOR_TRAINING_SEC"
  done
}

acquire_training_lock() {
  mkdir -p "$(dirname "$TRAIN_LOCK")"
  if ( set -o noclobber; printf "%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$TRAIN_LOCK" ) 2>/dev/null; then
    return 0
  fi
  echo "training lock active path=$TRAIN_LOCK"
  return 1
}

release_training_lock() {
  rm -f "$TRAIN_LOCK"
}

with_training_lock() {
  if ! acquire_training_lock; then
    return 1
  fi
  local status=0
  trap 'release_training_lock' EXIT INT TERM
  if "$@"; then
    status=0
  else
    status=$?
  fi
  release_training_lock
  trap - EXIT INT TERM
  return "$status"
}

run_with_retry() {
  local label="$1"
  shift
  local attempt=1
  while true; do
    echo "pipeline_step=${label} status=run attempt=${attempt}/${PIPELINE_MAX_ATTEMPTS}"
    if "$@"; then
      echo "pipeline_step=${label} status=done"
      return 0
    fi
    if [[ "$attempt" -ge "$PIPELINE_MAX_ATTEMPTS" ]]; then
      echo "pipeline_step=${label} status=failed attempts=$attempt"
      return 1
    fi
    echo "pipeline_step=${label} status=retry sleep_sec=${PIPELINE_RETRY_SLEEP_SEC}"
    sleep "$PIPELINE_RETRY_SLEEP_SEC"
    attempt=$((attempt + 1))
  done
}

run_stage_once() {
  local stage="$1"
  shift
  local status=0
  if stage_done "$stage"; then
    echo "pipeline_stage=${stage} status=skip_done"
    return 0
  fi
  if "$@"; then
    status=0
  else
    status=$?
  fi
  if [[ "$status" -ne 0 ]]; then
    return "$status"
  fi
  mark_stage_done "$stage"
}

run_pipeline_stage() {
  local stage="$1"
  shift
  run_with_retry "$stage" run_stage_once "$stage" "$@"
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
  echo "usage: bash scripts/do.sh [pipeline|run|smoke|smoke-real|check|start|start-v2|collect|collect-diverse|collect-fast|collect-image|collect-video|collection-status|ingest|scan [paths...]|train|train-existing|train-image|train-video|train-all|retrain|continuous|train-all-types|deps-update|doctor|status]"
}

run_doctor_check() {
  bash scripts/doctor.sh "$@"
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

print_cli_flags() {
  local flag=""
  for flag in "$@"; do
    print_cli_flag "$flag"
  done
}

print_cli_flag_value() {
  printf "%s\n%s\n" "$1" "$2"
}

print_cli_flag_value_pairs() {
  while [[ "$#" -gt 0 ]]; do
    print_cli_flag_value "$1" "$2"
    shift 2
  done
}

print_cli_flag_value_from_env() {
  local flag="$1"
  local env_name="$2"
  local default="$3"
  print_cli_flag_value "$flag" "${!env_name:-$default}"
}

print_cli_flag_value_from_env_triplets() {
  while [[ "$#" -gt 0 ]]; do
    print_cli_flag_value_from_env "$1" "$2" "$3"
    shift 3
  done
}

print_cli_flag_values_from_csv() {
  local flag="$1"
  local csv="${2:-}"
  local -a values=()
  local value=""
  [[ -z "$csv" ]] && return 0
  IFS=',' read -r -a values <<< "$csv"
  for value in "${values[@]}"; do
    value="$(echo "$value" | xargs)"
    [[ -z "$value" ]] && continue
    print_cli_flag_value "$flag" "$value"
  done
}

run_repo_python() {
  ensure_env
  python "$@"
}

run_repo_python_with_timeout() {
  local timeout_sec="$1"
  shift
  ensure_env
  if command -v timeout >/dev/null 2>&1; then
    timeout "${timeout_sec}s" python "$@"
    return
  fi
  python "$@"
}
