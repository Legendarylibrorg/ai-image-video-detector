#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SETUP_MAX_ATTEMPTS="${SETUP_MAX_ATTEMPTS:-4}"
SETUP_RETRY_SLEEP_SEC="${SETUP_RETRY_SLEEP_SEC:-45}"
DRY_RUN="${DRY_RUN:-0}"
ENV_FILE="${SETUP_ENV_FILE:-$ROOT_DIR/.env}"
ENV_EXAMPLE_FILE="${SETUP_ENV_EXAMPLE_FILE:-$ROOT_DIR/.env.example}"
SETUP_RUN_PIPELINE="${SETUP_RUN_PIPELINE:-0}"
SETUP_INSTALL_SYSTEM_DEPS="${SETUP_INSTALL_SYSTEM_DEPS:-1}"
SETUP_SKIP_DOCTOR="${SETUP_SKIP_DOCTOR:-0}"
SETUP_PROMPT_FOR_HF_TOKEN="${SETUP_PROMPT_FOR_HF_TOKEN:-1}"
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
  token="$(current_hf_token)"
  if [[ -n "$token" ]]; then
    set_hf_token_vars "$token"
  fi

  if [[ -z "${HF_TOKEN:-}" ]]; then
    if [[ "$SETUP_PROMPT_FOR_HF_TOKEN" == "1" && ( -t 0 || "$SETUP_ALLOW_STDIN_TOKEN" == "1" ) ]]; then
      echo "Hugging Face token can be saved to .env during setup."
      if [[ -t 0 ]]; then
        printf "Enter HF_TOKEN now (input hidden, press Enter to skip): "
        read -r -s HF_TOKEN
        echo
      else
        printf "Enter HF_TOKEN now (press Enter to skip): "
        read -r HF_TOKEN || true
      fi
      if [[ -n "${HF_TOKEN:-}" ]]; then
        set_hf_token_vars "$HF_TOKEN"
        if [[ "$HF_SETUP_SAVE_ENV" == "1" ]]; then
          save_hf_token_env "$HF_TOKEN"
          echo "hf_token_saved=$ENV_FILE"
        fi
      fi
    fi
  fi

  if [[ -z "${HF_TOKEN:-}" ]]; then
    if [[ "$HF_SETUP_REQUIRE_TOKEN" != "1" ]]; then
      echo "hf_token_status=optional_missing"
      return 0
    fi
    echo "hf_token_status=missing_noninteractive set HF_TOKEN or add it to $ENV_FILE"
    exit 1
  fi

  if [[ "$DRY_RUN" != "1" ]]; then
    validate_hf_token
  else
    echo "[DRY_RUN] validate_hf_token"
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
      run_cmd "sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip build-essential clamav clamav-daemon"
      run_cmd "sudo freshclam || true"
    else
      run_cmd "apt-get update"
      run_cmd "env DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip build-essential clamav clamav-daemon"
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

print_next_step() {
  load_env_file
  if [[ "$SETUP_RUN_PIPELINE" == "1" ]]; then
    echo "setup_next=pipeline complete"
    return
  fi
  local token=""
  token="$(current_hf_token)"
  if [[ -n "$token" ]]; then
    echo "setup_next=run ./local.sh run"
  else
    echo "setup_next=add HF_TOKEN in .env if needed, then run ./local.sh run"
  fi
}

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
  exit 0
fi

if stage_done "pipeline_train_all_types"; then
  echo "setup_stage=pipeline_train_all_types status=skip_done"
  print_next_step
  echo "setup_status=complete"
  exit 0
fi

attempt=1
while true; do
  echo "setup_stage=pipeline_train_all_types status=run setup_attempt=$attempt/$SETUP_MAX_ATTEMPTS"
  if run_cmd "bash scripts/do.sh train-all-types"; then
    mark_stage_done "pipeline_train_all_types"
    echo "setup_stage=pipeline_train_all_types status=done"
    print_next_step
    echo "setup_status=complete"
    break
  fi
  if [[ "$attempt" -ge "$SETUP_MAX_ATTEMPTS" ]]; then
    echo "setup_status=failed attempts=$attempt"
    exit 1
  fi
  echo "setup_retry_in_sec=$SETUP_RETRY_SLEEP_SEC"
  sleep "$SETUP_RETRY_SLEEP_SEC"
  attempt=$((attempt + 1))
done
