#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MAX_ATTEMPTS="${SETUP_MAX_ATTEMPTS:-4}"
RETRY_SLEEP_SEC="${SETUP_RETRY_SLEEP_SEC:-45}"
DRY_RUN="${DRY_RUN:-0}"
ENV_FILE="${SETUP_ENV_FILE:-$ROOT_DIR/.env}"
HF_SETUP_REQUIRE_TOKEN="${HF_SETUP_REQUIRE_TOKEN:-1}"
HF_SETUP_SAVE_ENV="${HF_SETUP_SAVE_ENV:-1}"
STAGE_DIR="${SETUP_STAGE_DIR:-$ROOT_DIR/.local/stages}"
SETUP_FORCE_STAGES="${SETUP_FORCE_STAGES:-0}"

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] $*"
  else
    eval "$*"
  fi
}

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

stage_file() {
  local stage="$1"
  echo "$STAGE_DIR/${stage}.done"
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
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] mark_stage_done=$stage"
    return
  fi
  mkdir -p "$STAGE_DIR"
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
  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
  fi

  if [[ -z "${HF_TOKEN:-}" ]]; then
    if [[ "$HF_SETUP_REQUIRE_TOKEN" != "1" ]]; then
      echo "hf_token_status=optional_missing"
      return 0
    fi
    if [[ -t 0 ]]; then
      echo "Hugging Face token required for reliable collection."
      printf "Enter HF_TOKEN (input hidden): "
      read -r -s HF_TOKEN
      echo
      if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "hf_token_status=missing"
        exit 1
      fi
      export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
      if [[ "$HF_SETUP_SAVE_ENV" == "1" ]]; then
        save_hf_token_env "$HF_TOKEN"
        echo "hf_token_saved=$ENV_FILE"
      fi
    else
      echo "hf_token_status=missing_noninteractive set HF_TOKEN or add it to $ENV_FILE"
      exit 1
    fi
  fi

  if [[ "$DRY_RUN" != "1" ]]; then
    validate_hf_token
  else
    echo "[DRY_RUN] validate_hf_token"
  fi
}

if command -v apt-get >/dev/null 2>&1; then
  if stage_done "apt_deps"; then
    echo "setup_stage=apt_deps status=skip_done"
  else
    echo "setup_stage=apt_deps status=run"
    if command -v sudo >/dev/null 2>&1; then
      run_cmd "sudo apt-get update"
      run_cmd "sudo apt-get install -y python3 python3-venv python3-pip build-essential clamav clamav-daemon"
      run_cmd "sudo freshclam || true"
    else
      run_cmd "apt-get update"
      run_cmd "apt-get install -y python3 python3-venv python3-pip build-essential clamav clamav-daemon"
      run_cmd "freshclam || true"
    fi
    mark_stage_done "apt_deps"
    echo "setup_stage=apt_deps status=done"
  fi
else
  echo "setup_stage=apt_deps status=skip_no_apt"
fi

echo "setup_stage=python_deps status=run"
run_cmd "bash scripts/install_deps.sh"
mark_stage_done "python_deps"
echo "setup_stage=python_deps status=done"

echo "setup_stage=hf_token status=run"
ensure_hf_token_ready
mark_stage_done "hf_token"
echo "setup_stage=hf_token status=done"

if stage_done "pipeline_train_all_types"; then
  echo "setup_stage=pipeline_train_all_types status=skip_done"
  echo "setup_status=complete"
  exit 0
fi

attempt=1
while true; do
  echo "setup_stage=pipeline_train_all_types status=run setup_attempt=$attempt/$MAX_ATTEMPTS"
  if run_cmd "bash scripts/do.sh train-all-types"; then
    mark_stage_done "pipeline_train_all_types"
    echo "setup_stage=pipeline_train_all_types status=done"
    echo "setup_status=complete"
    break
  fi
  if [[ "$attempt" -ge "$MAX_ATTEMPTS" ]]; then
    echo "setup_status=failed attempts=$attempt"
    exit 1
  fi
  echo "setup_retry_in_sec=$RETRY_SLEEP_SEC"
  sleep "$RETRY_SLEEP_SEC"
  attempt=$((attempt + 1))
done
