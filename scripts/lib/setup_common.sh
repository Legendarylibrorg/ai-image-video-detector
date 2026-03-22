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

persist_env_hf_token_if_present() {
  local env_file="${ENV_FILE:-${SETUP_ENV_FILE:-}}"
  local token=""
  token="$(current_hf_token)"
  if [[ -z "$token" ]]; then
    return
  fi
  set_hf_token_vars "$token"
  save_hf_token_env "$token" "$env_file"
  echo "setup_stage=env_token status=done file=$env_file"
}

prompt_for_hf_token_if_missing() {
  local env_file="${ENV_FILE:-${SETUP_ENV_FILE:-}}"
  local token=""
  token="$(current_hf_token)"
  if [[ -n "$token" ]]; then
    return
  fi
  if [[ "${SETUP_PROMPT_FOR_HF_TOKEN:-0}" != "1" ]]; then
    echo "setup_stage=env_token status=skip_opt_out"
    return
  fi
  if [[ ! -t 0 && "${SETUP_ALLOW_STDIN_TOKEN:-0}" != "1" ]]; then
    echo "setup_stage=env_token status=skip_noninteractive"
    return
  fi

  local entered=""
  echo "Hugging Face token can be saved to .env during setup."
  if [[ -t 0 ]]; then
    printf "Enter HF_TOKEN now (input hidden, press Enter to skip): "
    read -r -s entered
    echo
  else
    printf "Enter HF_TOKEN now (press Enter to skip): "
    read -r entered || true
  fi
  if [[ -z "$entered" ]]; then
    echo "setup_stage=env_token status=skip_empty"
    return
  fi
  set_hf_token_vars "$entered"
  if [[ "${HF_SETUP_SAVE_ENV:-1}" == "1" ]]; then
    save_hf_token_env "$entered" "$env_file"
  fi
  echo "setup_stage=env_token status=done file=$env_file"
}

validate_hf_token() {
  python - <<'PY'
import os
import sys
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN", "").strip()
if not token:
    print("hf_token_status=missing")
    sys.exit(2)
try:
    me = HfApi().whoami(token=token)
    name = me.get("name") or me.get("fullname") or "unknown"
    print(f"hf_token_status=ok user={name}")
except Exception as e:
    print(f"hf_token_status=invalid reason={e}")
    sys.exit(3)
PY
}

ensure_hf_token_ready() {
  load_env_file
  local token=""
  local env_file="${ENV_FILE:-${SETUP_ENV_FILE:-}}"
  token="$(current_hf_token)"
  if [[ -n "$token" ]]; then
    set_hf_token_vars "$token"
  fi

  prompt_for_hf_token_if_missing

  if [[ -z "${HF_TOKEN:-}" ]]; then
    if [[ "${HF_SETUP_REQUIRE_TOKEN:-0}" != "1" ]]; then
      echo "hf_token_status=optional_missing"
      return 0
    fi
    echo "hf_token_status=missing_noninteractive set HF_TOKEN or add it to $env_file"
    return 1
  fi

  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    validate_hf_token
  else
    echo "[DRY_RUN] validate_hf_token"
  fi
}

print_next_step() {
  load_env_file
  if [[ "${SETUP_RUN_PIPELINE:-0}" == "1" ]]; then
    echo "setup_next=pipeline complete"
    return
  fi
  local token=""
  token="$(current_hf_token)"
  if [[ -n "$token" ]]; then
    echo "setup_next=run ./local.sh smoke, then ./local.sh run"
  else
    echo "setup_next=add HF_TOKEN in .env if needed, then run ./local.sh smoke and ./local.sh run"
  fi
}
