#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
ENV_EXAMPLE_FILE="${ENV_EXAMPLE_FILE:-$ROOT_DIR/.env.example}"
SETUP_STAGE_DIR="${SETUP_STAGE_DIR:-$ROOT_DIR/.local/stages}"
SETUP_FORCE_STAGES="${SETUP_FORCE_STAGES:-0}"
SETUP_INSTALL_SYSTEM_DEPS="${SETUP_INSTALL_SYSTEM_DEPS:-1}"
SETUP_SKIP_DOCTOR="${SETUP_SKIP_DOCTOR:-0}"
SETUP_PROMPT_FOR_HF_TOKEN="${SETUP_PROMPT_FOR_HF_TOKEN:-1}"
SETUP_ALLOW_STDIN_TOKEN="${SETUP_ALLOW_STDIN_TOKEN:-0}"
SETUP_MAX_ATTEMPTS="${SETUP_MAX_ATTEMPTS:-3}"
SETUP_RETRY_SLEEP_SEC="${SETUP_RETRY_SLEEP_SEC:-10}"
source "$ROOT_DIR/scripts/lib/env.sh"
source "$ROOT_DIR/scripts/lib/setup_common.sh"

persist_env_hf_token_if_present() {
  local token=""
  token="$(current_hf_token)"
  if [[ -z "$token" ]]; then
    return
  fi
  set_hf_token_vars "$token"
  save_hf_token_env "$token"
  echo "setup_stage=env_token status=done file=$ENV_FILE"
}

prompt_for_hf_token_if_missing() {
  local token=""
  token="$(current_hf_token)"
  if [[ -n "$token" ]]; then
    return
  fi
  if [[ "$SETUP_PROMPT_FOR_HF_TOKEN" != "1" ]]; then
    echo "setup_stage=env_token status=skip_opt_out"
    return
  fi
  if [[ ! -t 0 && "$SETUP_ALLOW_STDIN_TOKEN" != "1" ]]; then
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
  save_hf_token_env "$entered"
  echo "setup_stage=env_token status=done file=$ENV_FILE"
}

install_system_deps() {
  if [[ "$SETUP_INSTALL_SYSTEM_DEPS" != "1" ]]; then
    echo "setup_stage=system_deps status=skip_opt_out"
    return
  fi
  if [[ "$(uname -s)" != "Linux" ]] || ! command -v apt-get >/dev/null 2>&1; then
    echo "setup_stage=system_deps status=skip_unsupported_platform"
    return
  fi
  if stage_done "bootstrap_apt_deps"; then
    echo "setup_stage=system_deps status=skip_done"
    return
  fi

  echo "setup_stage=system_deps status=run"
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update
    sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip build-essential clamav clamav-daemon
    sudo freshclam || true
  else
    apt-get update
    env DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip build-essential clamav clamav-daemon
    freshclam || true
  fi
  mark_stage_done "bootstrap_apt_deps"
  echo "setup_stage=system_deps status=done"
}

install_python_deps() {
  UPGRADE_TOOLCHAIN="${UPGRADE_TOOLCHAIN:-0}" run_setup_step_with_retry python_deps bash scripts/install_deps.sh
}

run_doctor() {
  if [[ "$SETUP_SKIP_DOCTOR" == "1" ]]; then
    echo "setup_stage=doctor status=skip_opt_out"
    return
  fi
  DOCTOR_REQUIRE_TOKEN=0 run_setup_step_with_retry doctor bash scripts/doctor.sh
}

print_next_step() {
  load_env_file
  local token=""
  token="$(current_hf_token)"
  if [[ -n "$token" ]]; then
    echo "setup_next=run ./local.sh smoke, then ./local.sh run"
  else
    echo "setup_next=add HF_TOKEN in .env if needed, then run ./local.sh smoke or ./local.sh run"
  fi
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  return 0
fi

ensure_env_file
load_env_file
persist_env_hf_token_if_present
prompt_for_hf_token_if_missing
install_system_deps
ensure_python3
prepare_local_dirs
install_python_deps
run_doctor
print_next_step
echo "setup_status=ready"
