#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
MIN_FREE_GB="${DOCTOR_MIN_FREE_GB:-40}"
TOKEN_CHECK_TIMEOUT_SEC="${DOCTOR_TOKEN_CHECK_TIMEOUT_SEC:-12}"
DOCTOR_REQUIRE_TOKEN="${DOCTOR_REQUIRE_TOKEN:-0}"
DOCTOR_REQUIRE_GPU="${DOCTOR_REQUIRE_GPU:-0}"
DOCTOR_REQUIRE_CLAMAV="${DOCTOR_REQUIRE_CLAMAV:-0}"

ok_count=0
warn_count=0
fail_count=0

source "$ROOT_DIR/scripts/lib/env.sh"

emit_ok() {
  ok_count=$((ok_count + 1))
  echo "doctor_ok: $*"
}

emit_warn() {
  warn_count=$((warn_count + 1))
  echo "doctor_warn: $*"
}

emit_fail() {
  fail_count=$((fail_count + 1))
  echo "doctor_fail: $*"
}

normalize_path() {
  local path="$1"
  if [[ "$path" == /* ]]; then
    printf "%s\n" "$path"
    return
  fi
  path="${path#./}"
  printf "%s/%s\n" "$ROOT_DIR" "$path"
}

check_disk_space() {
  local avail_kb
  avail_kb="$(df -Pk "$ROOT_DIR" | awk 'NR==2 {print $4}')"
  if [[ -z "$avail_kb" || ! "$avail_kb" =~ ^[0-9]+$ ]]; then
    emit_fail "unable_to_read_disk_space root=$ROOT_DIR"
    return
  fi
  local avail_gb
  avail_gb=$((avail_kb / 1024 / 1024))
  if (( avail_gb < MIN_FREE_GB )); then
    emit_fail "disk_free_gb=$avail_gb required_gb=$MIN_FREE_GB"
  else
    emit_ok "disk_free_gb=$avail_gb"
  fi
}

check_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    if [[ "$DOCTOR_REQUIRE_GPU" == "1" ]]; then
      emit_fail "nvidia_smi_missing gpu_required=1"
    else
      emit_warn "nvidia_smi_missing gpu_check_skipped=1"
    fi
    return
  fi
  local gpu_line
  gpu_line="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -n 1 || true)"
  if [[ -z "$gpu_line" ]]; then
    if [[ "$DOCTOR_REQUIRE_GPU" == "1" ]]; then
      emit_fail "gpu_query_failed gpu_required=1"
    else
      emit_warn "gpu_query_failed"
    fi
    return
  fi
  emit_ok "gpu=$gpu_line"
}

check_clamav() {
  if ! command -v clamscan >/dev/null 2>&1; then
    if [[ "$DOCTOR_REQUIRE_CLAMAV" == "1" ]]; then
      emit_fail "clamscan_missing clamav_required=1"
    else
      emit_warn "clamscan_missing"
    fi
    return
  fi
  emit_ok "clamscan_ready=1"
}

check_cache_paths() {
  local cache1="${BEST_DS_CACHE_DIR:-$ROOT_DIR/.local/hf}"
  local cache2="${VIDEO_CACHE_DIR:-$ROOT_DIR/.local/hf}"
  local cache3="$ROOT_DIR/.local"
  local -a seen=()
  local d
  for d in "$cache1" "$cache2" "$cache3"; do
    d="$(normalize_path "$d")"
    local already_seen=0
    local seen_path
    if (( ${#seen[@]} > 0 )); then
      for seen_path in "${seen[@]}"; do
        if [[ "$seen_path" == "$d" ]]; then
          already_seen=1
          break
        fi
      done
    fi
    if [[ "$already_seen" == "1" ]]; then
      continue
    fi
    seen+=("$d")
    mkdir -p "$d" 2>/dev/null || true
    if [[ ! -d "$d" ]]; then
      emit_fail "cache_dir_missing path=$d"
      continue
    fi
    if [[ ! -w "$d" ]]; then
      emit_fail "cache_dir_not_writable path=$d"
      continue
    fi
    emit_ok "cache_dir_ready path=$d"
  done
}

check_venv_and_deps() {
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    emit_warn "venv_missing path=$VENV_DIR run=./local.sh setup"
    return
  fi
  emit_ok "venv_present path=$VENV_DIR"
  if "$VENV_DIR/bin/python" - <<'PY' >/dev/null 2>&1
import ai_image_detector  # noqa: F401
import cv2  # noqa: F401
import datasets  # noqa: F401
import huggingface_hub  # noqa: F401
import PIL  # noqa: F401
import torch  # noqa: F401
PY
  then
    emit_ok "core_python_deps=ok"
  else
    emit_warn "core_python_deps=missing_or_partial run=./local.sh setup"
  fi

  local cli
  local cli_missing=0
  for cli in hf aid-train aid-video-train aid-dataset; do
    if [[ -x "$VENV_DIR/bin/$cli" ]]; then
      emit_ok "cli_ready name=$cli"
    else
      cli_missing=1
      emit_warn "cli_missing name=$cli run=./local.sh deps"
    fi
  done
  if [[ "$cli_missing" == "0" ]]; then
    emit_ok "core_cli_deps=ok"
  fi
}

check_hf_token() {
  local token="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
  if [[ -z "$token" ]]; then
    if [[ "$DOCTOR_REQUIRE_TOKEN" == "1" ]]; then
      emit_fail "hf_token_missing set HF_TOKEN in environment or .env"
    else
      emit_warn "hf_token_missing add HF_TOKEN before collection or training"
    fi
    return
  fi
  emit_ok "hf_token_present=1"
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    emit_warn "hf_token_validation_skipped reason=venv_missing"
    return
  fi

  local rc=0
  if command -v timeout >/dev/null 2>&1; then
    timeout "${TOKEN_CHECK_TIMEOUT_SEC}s" "$VENV_DIR/bin/python" - <<'PY' >/dev/null 2>&1 || rc=$?
import os
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
HfApi().whoami(token=token)
PY
  else
    "$VENV_DIR/bin/python" - <<'PY' >/dev/null 2>&1 || rc=$?
import os
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
HfApi().whoami(token=token)
PY
  fi

  if [[ "$rc" == "0" ]]; then
    emit_ok "hf_token_validation=ok"
  else
    emit_warn "hf_token_validation=failed_or_timeout"
  fi
}

load_env_file
check_disk_space
check_gpu
check_cache_paths
check_venv_and_deps
check_hf_token
check_clamav

echo "doctor_summary ok=$ok_count warn=$warn_count fail=$fail_count"
if (( fail_count > 0 )); then
  exit 2
fi
exit 0
