stage_file() {
  local stage="$1"
  echo "${SETUP_STAGE_DIR}/${stage}.done"
}

stage_done() {
  local stage="$1"
  if [[ "${SETUP_FORCE_STAGES:-0}" == "1" ]]; then
    return 1
  fi
  [[ -f "$(stage_file "$stage")" ]]
}

mark_stage_done() {
  local stage="$1"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY_RUN] mark_stage_done=$stage"
    return
  fi
  mkdir -p "$SETUP_STAGE_DIR"
  printf "%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$(stage_file "$stage")"
}

run_setup_step_with_retry() {
  local stage="$1"
  shift
  local max_attempts="${SETUP_MAX_ATTEMPTS:-4}"
  local retry_sleep_sec="${SETUP_RETRY_SLEEP_SEC:-45}"
  local attempt=1
  while true; do
    echo "setup_stage=$stage status=run attempt=$attempt/$max_attempts"
    if "$@"; then
      echo "setup_stage=$stage status=done"
      return 0
    fi
    if [[ "$attempt" -ge "$max_attempts" ]]; then
      echo "setup_stage=$stage status=failed attempts=$attempt"
      return 1
    fi
    echo "setup_stage=$stage status=retry sleep_sec=$retry_sleep_sec"
    sleep "$retry_sleep_sec"
    attempt=$((attempt + 1))
  done
}

ensure_python3() {
  if command -v python3 >/dev/null 2>&1; then
    return
  fi
  echo "setup_fail: python3_missing install_python3_and_retry=1"
  exit 1
}

prepare_local_dirs() {
  mkdir -p "$ROOT_DIR/.local" "$ROOT_DIR/.local/hf"
  echo "setup_stage=local_dirs status=done"
}
