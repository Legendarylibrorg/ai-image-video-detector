#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SETUP_MAX_ATTEMPTS="${SETUP_MAX_ATTEMPTS:-2}"
SETUP_RETRY_SLEEP_SEC="${SETUP_RETRY_SLEEP_SEC:-5}"
DRY_RUN="${DRY_RUN:-0}"
ENV_FILE="${SETUP_ENV_FILE:-$ROOT_DIR/.env}"
ENV_EXAMPLE_FILE="${SETUP_ENV_EXAMPLE_FILE:-$ROOT_DIR/.env.example}"
SETUP_RUN_PIPELINE="${SETUP_RUN_PIPELINE:-0}"
SETUP_INSTALL_SYSTEM_DEPS="${SETUP_INSTALL_SYSTEM_DEPS:-1}"
SETUP_SKIP_DOCTOR="${SETUP_SKIP_DOCTOR:-0}"
SETUP_PROMPT_FOR_HF_TOKEN="${SETUP_PROMPT_FOR_HF_TOKEN:-0}"
SETUP_ALLOW_STDIN_TOKEN="${SETUP_ALLOW_STDIN_TOKEN:-0}"
HF_SETUP_REQUIRE_TOKEN="${HF_SETUP_REQUIRE_TOKEN:-}"
HF_SETUP_SAVE_ENV="${HF_SETUP_SAVE_ENV:-1}"
SETUP_STAGE_DIR="${SETUP_STAGE_DIR:-$ROOT_DIR/.local/stages}"
SETUP_FORCE_STAGES="${SETUP_FORCE_STAGES:-0}"
source "$ROOT_DIR/scripts/lib/env.sh"
source "$ROOT_DIR/scripts/lib/setup_common.sh"

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] $*"
  else
    eval "$*"
  fi
}

install_system_deps() {
  if [[ "$SETUP_INSTALL_SYSTEM_DEPS" != "1" ]]; then
    echo "setup_stage=apt_deps status=skip_opt_out"
    return
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "setup_stage=apt_deps status=skip_no_apt"
    return
  fi
  if stage_done "apt_deps"; then
    echo "setup_stage=apt_deps status=skip_done"
    return
  fi

  run_apt_bootstrap() {
    if command -v sudo >/dev/null 2>&1; then
      run_cmd "sudo apt-get update"
      run_cmd "sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y curl ca-certificates git python3 python3-venv python3-pip build-essential clamav clamav-daemon"
      run_cmd "sudo freshclam || true"
    else
      run_cmd "apt-get update"
      run_cmd "env DEBIAN_FRONTEND=noninteractive apt-get install -y curl ca-certificates git python3 python3-venv python3-pip build-essential clamav clamav-daemon"
      run_cmd "freshclam || true"
    fi
  }

  run_setup_step_with_retry apt_deps run_apt_bootstrap
  mark_stage_done "apt_deps"
}

install_python_deps() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "setup_stage=python_deps status=run attempt=1/$SETUP_MAX_ATTEMPTS"
    echo "[DRY_RUN] bash scripts/install_deps.sh"
    echo "setup_stage=python_deps status=done"
    return
  fi
  UPGRADE_TOOLCHAIN="${UPGRADE_TOOLCHAIN:-0}" run_setup_step_with_retry python_deps bash scripts/install_deps.sh
}

run_doctor() {
  if [[ "$SETUP_SKIP_DOCTOR" == "1" ]]; then
    echo "setup_stage=doctor status=skip_opt_out"
    return
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "setup_stage=doctor status=run attempt=1/$SETUP_MAX_ATTEMPTS"
    echo "[DRY_RUN] bash scripts/doctor.sh"
    echo "setup_stage=doctor status=done"
    return
  fi
  DOCTOR_REQUIRE_TOKEN=0 run_setup_step_with_retry doctor bash scripts/doctor.sh
}

main() {
  if [[ -z "$HF_SETUP_REQUIRE_TOKEN" ]]; then
    if [[ "$SETUP_RUN_PIPELINE" == "1" ]]; then
      HF_SETUP_REQUIRE_TOKEN="1"
    else
      HF_SETUP_REQUIRE_TOKEN="0"
    fi
  fi
  ensure_env_file
  install_system_deps
  ensure_python3
  prepare_local_dirs
  install_python_deps
  echo "setup_stage=hf_token status=run attempt=1/$SETUP_MAX_ATTEMPTS"
  ensure_hf_token_ready
  mark_stage_done "hf_token"
  echo "setup_stage=hf_token status=done"
  run_doctor

  if [[ "$SETUP_RUN_PIPELINE" != "1" ]]; then
    print_next_step
    echo "setup_status=ready"
    return 0
  fi

  if stage_done "pipeline_train_all_types"; then
    echo "setup_stage=pipeline_train_all_types status=skip_done"
    print_next_step
    echo "setup_status=complete"
    return 0
  fi

  local attempt=1
  while true; do
    echo "setup_stage=pipeline_train_all_types status=run setup_attempt=$attempt/$SETUP_MAX_ATTEMPTS"
    if run_cmd "bash scripts/do.sh train-all-types"; then
      mark_stage_done "pipeline_train_all_types"
      echo "setup_stage=pipeline_train_all_types status=done"
      print_next_step
      echo "setup_status=complete"
      return 0
    fi
    if [[ "$attempt" -ge "$SETUP_MAX_ATTEMPTS" ]]; then
      echo "setup_status=failed attempts=$attempt"
      return 1
    fi
    echo "setup_retry_in_sec=$SETUP_RETRY_SLEEP_SEC"
    sleep "$SETUP_RETRY_SLEEP_SEC"
    attempt=$((attempt + 1))
  done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
