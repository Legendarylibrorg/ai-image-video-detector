#!/usr/bin/env bash
set -euo pipefail

# Truly one-command setup + optimized training (pipeline-only)
# Usage:
#   bash scripts/one_command_4090.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Optimized defaults (speed/quality balance)
export SKIP_SWEEP="${SKIP_SWEEP:-1}"
export EPOCHS="${EPOCHS:-12}"
export RUN_DISTILL="${RUN_DISTILL:-1}"
export RUN_HARD_MINING="${RUN_HARD_MINING:-1}"
export TRAIN_PER_CLASS="${TRAIN_PER_CLASS:-40000}"
export VAL_PER_CLASS="${VAL_PER_CLASS:-9000}"
export TEST_PER_CLASS="${TEST_PER_CLASS:-9000}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export DRY_RUN="${DRY_RUN:-0}"
export RUN_VIDEO_DATA_PULL="${RUN_VIDEO_DATA_PULL:-1}"
export VIDEO_MODE="${VIDEO_MODE:-snapshot}"
export VIDEO_SNAPSHOT_MAX_WORKERS="${VIDEO_SNAPSHOT_MAX_WORKERS:-1}"
export VIDEO_REPO_BASE_PAUSE_MS="${VIDEO_REPO_BASE_PAUSE_MS:-2200}"
export VIDEO_REPO_JITTER_MS="${VIDEO_REPO_JITTER_MS:-1800}"
export VIDEO_COPY_SLEEP_MS="${VIDEO_COPY_SLEEP_MS:-15}"
export VIDEO_SLEEP_MS="${VIDEO_SLEEP_MS:-120}"
export VIDEO_JITTER_MS="${VIDEO_JITTER_MS:-80}"
export VIDEO_CHUNK_PAUSE_MS="${VIDEO_CHUNK_PAUSE_MS:-1000}"
export VIDEO_REPO_COOLDOWN_MS="${VIDEO_REPO_COOLDOWN_MS:-3000}"
export VIDEO_RETRIES="${VIDEO_RETRIES:-5}"
export VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf "[DRY_RUN]"
    printf " %q" "$@"
    printf "\n"
  else
    "$@"
  fi
}

activate_repo_venv() {
  local activate_script="$VENV_DIR/bin/activate"
  if [[ -f "$activate_script" ]]; then
    # shellcheck disable=SC1090
    source "$activate_script"
    return 0
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] source $activate_script"
    return 0
  fi
  echo "missing_virtualenv_activate=$activate_script run=bash scripts/install_deps.sh" >&2
  return 1
}

# 1) Optional system deps for Ubuntu hosts
if command -v apt-get >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1; then
    run_cmd sudo apt-get update
    run_cmd sudo apt-get install -y curl ca-certificates git python3 python3-venv python3-pip build-essential clamav clamav-daemon
    run_cmd sudo freshclam || true
  else
    run_cmd apt-get update
    run_cmd apt-get install -y curl ca-certificates git python3 python3-venv python3-pip build-essential clamav clamav-daemon
    run_cmd freshclam || true
  fi
fi

# 2) Python environment + package deps
run_cmd bash scripts/install_deps.sh
activate_repo_venv

# 3) Optimized full training pipeline
run_cmd bash scripts/full_pipeline_4090.sh

# 4) Pipeline-only mode: no serving.
