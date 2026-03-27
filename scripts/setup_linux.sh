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
APT_PACKAGES="${APT_PACKAGES:-curl ca-certificates git unzip python3 python3-venv python3-pip build-essential clamav clamav-daemon}"
source "$ROOT_DIR/scripts/lib/env.sh"
source "$ROOT_DIR/scripts/lib/setup_common.sh"

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf "[DRY_RUN]"
    printf " %q" "$@"
    printf "\n"
  else
    "$@"
  fi
}

run_apt_bootstrap() {
  local -a apt_packages=()
  read -r -a apt_packages <<< "$APT_PACKAGES"
  if command -v sudo >/dev/null 2>&1; then
    run_cmd sudo apt-get update
    run_cmd sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y "${apt_packages[@]}"
    run_cmd sudo freshclam || true
  else
    run_cmd apt-get update
    run_cmd env DEBIAN_FRONTEND=noninteractive apt-get install -y "${apt_packages[@]}"
    run_cmd freshclam || true
  fi
}

apt_deps_marker_value() {
  printf "packages=%s\n" "$APT_PACKAGES"
}

apt_deps_stage_done() {
  if [[ "${SETUP_FORCE_STAGES:-0}" == "1" ]]; then
    return 1
  fi
  local marker_file
  marker_file="$(stage_file "apt_deps")"
  [[ -f "$marker_file" ]] || return 1
  [[ "$(cat "$marker_file")" == "$(apt_deps_marker_value)" ]]
}

mark_apt_deps_done() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY_RUN] mark_stage_done=apt_deps"
    return
  fi
  mkdir -p "$SETUP_STAGE_DIR"
  apt_deps_marker_value > "$(stage_file "apt_deps")"
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
  if apt_deps_stage_done; then
    echo "setup_stage=apt_deps status=skip_done"
    return
  fi

  run_setup_step_with_retry apt_deps run_apt_bootstrap
  mark_apt_deps_done
}

install_python_deps() {
  run_setup_command python_deps env UPGRADE_TOOLCHAIN="${UPGRADE_TOOLCHAIN:-0}" bash scripts/install_deps.sh
}

run_doctor() {
  if [[ "$SETUP_SKIP_DOCTOR" == "1" ]]; then
    echo "setup_stage=doctor status=skip_opt_out"
    return
  fi
  run_setup_command doctor env DOCTOR_REQUIRE_TOKEN=0 bash scripts/doctor.sh
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
    if run_cmd bash scripts/do.sh train-all-types; then
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
