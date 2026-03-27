#!/usr/bin/env bash
set -euo pipefail

# Thin convenience wrapper over the canonical install + run entrypoints.
# Usage:
#   bash scripts/one_command_4090.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Quality-oriented defaults for a dedicated local GPU box.
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

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  INSTALL_ASSUME_LINUX=1 bash ./install.sh
  GPU_REQUIRED_CMDS="disabled" DOCTOR_REQUIRE_TOKEN=0 DOCTOR_REQUIRE_GPU=0 DOCTOR_REQUIRE_CLAMAV=0 bash ./local.sh run
else
  bash ./install.sh
  bash ./local.sh run
fi
