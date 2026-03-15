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
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e . datasets

# 3) Optimized full training pipeline
bash scripts/full_pipeline_4090.sh

# 4) Optional serve right after training
if [[ "$AUTO_SERVE" == "1" ]]; then
  MODEL_GLOB="${MODEL_GLOB:-./artifacts_ens/m*/best.pt}"
  HOST="$HOST" PORT="$PORT" MODEL_GLOB="$MODEL_GLOB" bash scripts/serve_prod_4090.sh
fi
