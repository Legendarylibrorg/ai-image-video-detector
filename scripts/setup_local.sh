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

load_env_file() {
  if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
  fi
}

stage_file() {
  local stage="$1"
  echo "$SETUP_STAGE_DIR/${stage}.done"
}

stage_done() {
  local stage="$1"
  if [[ "$SETUP_FORCE_STAGES" == "1" ]]; then
    return 1
  fi
  [[ -f "$(stage_file "$stage")" ]]
}

mark_stage_done() {
  local stage="$1"
  mkdir -p "$SETUP_STAGE_DIR"
  printf "%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$(stage_file "$stage")"
}

save_hf_token_env() {
  local token="$1"
  mkdir -p "$(dirname "$ENV_FILE")"
  if [[ -f "$ENV_FILE" ]]; then
    if grep -q '^HF_TOKEN=' "$ENV_FILE"; then
      sed -i.bak "s|^HF_TOKEN=.*$|HF_TOKEN='$token'|" "$ENV_FILE"
      rm -f "${ENV_FILE}.bak"
    else
      printf "\nHF_TOKEN='%s'\n" "$token" >> "$ENV_FILE"
    fi
  else
    printf "HF_TOKEN='%s'\n" "$token" > "$ENV_FILE"
  fi
}

ensure_env_file() {
  if [[ -f "$ENV_FILE" ]]; then
    echo "setup_stage=env_file status=skip_exists file=$ENV_FILE"
    return
  fi
  if [[ ! -f "$ENV_EXAMPLE_FILE" ]]; then
    echo "setup_stage=env_file status=skip_missing_example file=$ENV_EXAMPLE_FILE"
    return
  fi
  cp "$ENV_EXAMPLE_FILE" "$ENV_FILE"
  echo "setup_stage=env_file status=done file=$ENV_FILE"
}

persist_env_hf_token_if_present() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    return
  fi
  save_hf_token_env "$HF_TOKEN"
  echo "setup_stage=env_token status=done file=$ENV_FILE"
}

ensure_python3() {
  if command -v python3 >/dev/null 2>&1; then
    return
  fi
  echo "setup_fail: python3_missing install_python3_and_retry=1"
  exit 1
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
    sudo apt-get install -y python3 python3-venv python3-pip build-essential clamav clamav-daemon
    sudo freshclam || true
  else
    apt-get update
    apt-get install -y python3 python3-venv python3-pip build-essential clamav clamav-daemon
    freshclam || true
  fi
  mark_stage_done "bootstrap_apt_deps"
  echo "setup_stage=system_deps status=done"
}

prepare_local_dirs() {
  mkdir -p "$ROOT_DIR/.local" "$ROOT_DIR/.local/hf"
  echo "setup_stage=local_dirs status=done"
}

install_python_deps() {
  echo "setup_stage=python_deps status=run"
  bash scripts/install_deps.sh
  echo "setup_stage=python_deps status=done"
}

run_doctor() {
  if [[ "$SETUP_SKIP_DOCTOR" == "1" ]]; then
    echo "setup_stage=doctor status=skip_opt_out"
    return
  fi
  echo "setup_stage=doctor status=run"
  DOCTOR_REQUIRE_TOKEN=0 bash scripts/doctor.sh
  echo "setup_stage=doctor status=done"
}

print_next_step() {
  load_env_file
  local token="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
  if [[ -n "$token" ]]; then
    echo "setup_next=run ./local.sh collect-fast or ./local.sh collect"
  else
    echo "setup_next=edit .env and set HF_TOKEN before ./local.sh collect-fast or ./local.sh setup-full"
  fi
}

ensure_env_file
load_env_file
persist_env_hf_token_if_present
install_system_deps
ensure_python3
prepare_local_dirs
install_python_deps
run_doctor
print_next_step
echo "setup_status=ready"
