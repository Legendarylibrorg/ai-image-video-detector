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
DOCTOR_REQUIRE_DOCKER="${DOCTOR_REQUIRE_DOCKER:-0}"
DOCTOR_DEPS_EXTRA="${DOCTOR_DEPS_EXTRA:-${DEPS_EXTRA:-}}"
DOCTOR_DEPS_PROFILE_FILE="${DOCTOR_DEPS_PROFILE_FILE:-$VENV_DIR/.deps_profile}"

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

doctor_deps_extra() {
  resolve_deps_extra "$DOCTOR_DEPS_EXTRA" "$DOCTOR_DEPS_PROFILE_FILE"
}

doctor_extra_enabled() {
  local wanted="$1"
  deps_extra_enabled "$wanted" "$(doctor_deps_extra)"
}

doctor_setup_command() {
  local deps_extra=""
  deps_extra="$(doctor_deps_extra)"
  printf '%s./local.sh setup\n' "$(deps_extra_env_prefix "$deps_extra")"
}

doctor_deps_command() {
  local deps_extra=""
  deps_extra="$(doctor_deps_extra)"
  printf '%s./local.sh deps\n' "$(deps_extra_env_prefix "$deps_extra")"
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

check_docker_stack() {
  local compose_file="$ROOT_DIR/docker-compose.yml"
  local dockerfile_cpu="$ROOT_DIR/Dockerfile"
  local dockerfile_gpu="$ROOT_DIR/Dockerfile.gpu"

  local path=""
  for path in "$compose_file" "$dockerfile_cpu" "$dockerfile_gpu"; do
    if [[ -f "$path" ]]; then
      emit_ok "docker_path_ready path=$path"
    else
      emit_fail "docker_path_missing path=$path"
    fi
  done

  if [[ "${DOCTOR_FORCE_DOCKER_STATE:-}" == "missing" ]]; then
    if [[ "$DOCTOR_REQUIRE_DOCKER" == "1" ]]; then
      emit_fail "docker_missing docker_required=1"
    else
      emit_warn "docker_missing"
    fi
    return
  fi

  if ! command -v docker >/dev/null 2>&1; then
    if [[ "$DOCTOR_REQUIRE_DOCKER" == "1" ]]; then
      emit_fail "docker_missing docker_required=1"
    else
      emit_warn "docker_missing"
    fi
    return
  fi
  if [[ "${DOCTOR_FORCE_DOCKER_COMPOSE_STATE:-}" == "missing" ]]; then
    if [[ "$DOCTOR_REQUIRE_DOCKER" == "1" ]]; then
      emit_fail "docker_compose_missing docker_required=1"
    else
      emit_warn "docker_compose_missing"
    fi
    return
  fi
  emit_ok "docker_cli_ready=1"

  if docker compose version >/dev/null 2>&1; then
    emit_ok "docker_compose_ready=1"
  else
    if [[ "$DOCTOR_REQUIRE_DOCKER" == "1" ]]; then
      emit_fail "docker_compose_missing docker_required=1"
    else
      emit_warn "docker_compose_missing"
    fi
  fi
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

check_isolation_posture() {
  local in_container=0
  local virt_type=""
  if [[ -f "/.dockerenv" ]]; then
    in_container=1
  elif grep -Eq "(docker|containerd|kubepods|podman)" /proc/1/cgroup 2>/dev/null; then
    in_container=1
  fi

  if command -v systemd-detect-virt >/dev/null 2>&1; then
    virt_type="$(systemd-detect-virt 2>/dev/null || true)"
  fi

  local in_vm=0
  if [[ -n "$virt_type" && "$virt_type" != "docker" && "$virt_type" != "container-other" && "$virt_type" != "podman" ]]; then
    in_vm=1
  fi

  if [[ "$in_vm" == "1" && "$in_container" == "1" ]]; then
    emit_ok "isolation=vm_plus_container virt=$virt_type"
  elif [[ "$in_vm" == "1" ]]; then
    emit_warn "isolation=vm_only virt=$virt_type prefer_compose_inside_vm=1"
  elif [[ "$in_container" == "1" ]]; then
    emit_warn "isolation=container_only prefer_dedicated_vm=1"
  else
    emit_warn "isolation=host_only highest_supply_chain_risk=1"
  fi

  if [[ "$in_container" == "1" ]]; then
    if [[ -w "$ROOT_DIR/README.md" ]]; then
      emit_ok "source_checkout_writable_in_container=1"
    else
      emit_warn "source_checkout_read_only_in_container=1"
    fi
    if [[ "$VENV_DIR" == "$ROOT_DIR"* ]]; then
      emit_warn "container_venv_inside_checkout=1 prefer_isolated_venv_volume=1"
    else
      emit_ok "container_venv_isolated_from_checkout=1 path=$VENV_DIR"
    fi
  fi
}

check_venv_and_deps() {
  local deps_extra=""
  deps_extra="$(doctor_deps_extra)"
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    emit_warn "venv_missing path=$VENV_DIR profile=$deps_extra run=$(doctor_setup_command)"
    return
  fi
  emit_ok "venv_present path=$VENV_DIR"
  emit_ok "deps_profile=$deps_extra"
  if "$VENV_DIR/bin/python" "$ROOT_DIR/scripts/deps_profile.py" --extras "$deps_extra" --check-imports >/dev/null 2>&1
  then
    emit_ok "core_python_deps=ok"
  else
    emit_warn "core_python_deps=missing_or_partial profile=$deps_extra run=$(doctor_setup_command)"
  fi

  local cli
  local cli_missing=0
  local -a required_clis=()
  if doctor_extra_enabled collection; then
    required_clis+=(hf)
  fi
  if doctor_extra_enabled training; then
    required_clis+=(aid-train)
  fi
  if doctor_extra_enabled training && doctor_extra_enabled video; then
    required_clis+=(aid-video-train)
  fi
  for cli in "${required_clis[@]}"; do
    if [[ -x "$VENV_DIR/bin/$cli" ]]; then
      emit_ok "cli_ready name=$cli"
    else
      cli_missing=1
      emit_warn "cli_missing name=$cli profile=$deps_extra run=$(doctor_deps_command)"
    fi
  done
  if [[ ${#required_clis[@]} -eq 0 ]]; then
    emit_ok "core_cli_deps=not_required profile=$deps_extra"
  elif [[ "$cli_missing" == "0" ]]; then
    emit_ok "core_cli_deps=ok"
  fi
}

check_hf_token() {
  local token=""
  resolve_current_hf_token
  token="$CURRENT_HF_TOKEN"
  if [[ -z "$token" ]]; then
    if [[ "$DOCTOR_REQUIRE_TOKEN" == "1" ]]; then
      emit_fail "hf_token_missing set HF_TOKEN, add it to .env, or run hf auth login"
    else
      emit_warn "hf_token_missing add HF_TOKEN before collection or training, or run hf auth login"
    fi
    return
  fi
  set_hf_token_vars "$token"
  emit_ok "hf_token_present=1"
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    emit_warn "hf_token_validation_skipped reason=venv_missing"
    return
  fi

  local rc=0
  if command -v timeout >/dev/null 2>&1; then
    HF_TOKEN="$token" timeout "${TOKEN_CHECK_TIMEOUT_SEC}s" "$VENV_DIR/bin/python" - <<'PY' >/dev/null 2>&1 || rc=$?
import os
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN")
HfApi().whoami(token=token)
PY
  else
    HF_TOKEN="$token" "$VENV_DIR/bin/python" - <<'PY' >/dev/null 2>&1 || rc=$?
import os
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN")
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
check_isolation_posture
check_cache_paths
check_venv_and_deps
check_hf_token
check_clamav
check_docker_stack

echo "doctor_summary ok=$ok_count warn=$warn_count fail=$fail_count"
if (( fail_count > 0 )); then
  exit 2
fi
exit 0
