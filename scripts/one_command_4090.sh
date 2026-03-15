#!/usr/bin/env bash
set -euo pipefail

# Truly one-command setup + optimized training + optional serve
# Usage:
#   bash scripts/one_command_4090.sh
#   AUTO_SERVE=1 bash scripts/one_command_4090.sh

# Optimized defaults (speed/quality balance)
export SKIP_SWEEP="${SKIP_SWEEP:-1}"
export EPOCHS="${EPOCHS:-12}"
export RUN_DISTILL="${RUN_DISTILL:-1}"
export RUN_HARD_MINING="${RUN_HARD_MINING:-1}"
export TRAIN_PER_CLASS="${TRAIN_PER_CLASS:-40000}"
export VAL_PER_CLASS="${VAL_PER_CLASS:-9000}"
export TEST_PER_CLASS="${TEST_PER_CLASS:-9000}"
export AUTO_SERVE="${AUTO_SERVE:-0}"
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8000}"
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

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] $*"
  else
    eval "$*"
  fi
}

# 1) Optional system deps for Ubuntu hosts
if command -v apt-get >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv python3-pip build-essential
  else
    apt-get update
    apt-get install -y python3 python3-venv python3-pip build-essential
  fi
fi

# 2) Python environment + package deps
if [[ ! -d .venv ]]; then
  run_cmd "python3 -m venv .venv"
fi
source .venv/bin/activate
run_cmd "python -m pip install --upgrade pip"
run_cmd "pip install -e . datasets huggingface_hub safetensors"

# 3) Optimized full training pipeline
run_cmd "bash scripts/full_pipeline_4090.sh"

# 4) Optional serve right after training
if [[ "$AUTO_SERVE" == "1" ]]; then
  MODEL_GLOB="${MODEL_GLOB:-./artifacts_ens/m*/best.safetensors}"
  run_cmd "HOST=\"$HOST\" PORT=\"$PORT\" MODEL_GLOB=\"$MODEL_GLOB\" bash scripts/serve_prod_4090.sh"
fi
