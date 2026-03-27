#!/usr/bin/env bash
set -euo pipefail

# Continuous collection + retraining loop.
# Usage:
#   bash scripts/continuous_training.sh
#   RUN_ONCE=1 bash scripts/continuous_training.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-./.local/runtime}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/continuous_training.log}"
TRAIN_LOCK="${TRAIN_LOCK:-$ROOT_DIR/.local/training.lock}"
INTERVAL_SEC="${CONTINUOUS_TRAIN_INTERVAL_SEC:-21600}"
CHECK_WHILE_TRAINING_SEC="${CHECK_WHILE_TRAINING_SEC:-600}"
PIPELINE_WAIT_FOR_TRAINING_SEC="${PIPELINE_WAIT_FOR_TRAINING_SEC:-$CHECK_WHILE_TRAINING_SEC}"
FAILURE_SLEEP_SEC="${CONTINUOUS_TRAIN_FAILURE_SLEEP_SEC:-900}"
RUN_ONCE="${RUN_ONCE:-0}"
ENV_READY=0

source "$ROOT_DIR/scripts/lib/core.sh"

exec > >(tee -a "$LOG_FILE") 2>&1

while true; do
  wait_for_training_to_finish "continuous_training"

  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) continuous_training_cycle_start"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) continuous_training_collect_start"
  if bash scripts/do.sh collect; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) continuous_training_collect_done"
  else
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) continuous_training_collect_failed sleep_sec=$FAILURE_SLEEP_SEC"
    if [[ "$RUN_ONCE" == "1" ]]; then
      exit 1
    fi
    sleep "$FAILURE_SLEEP_SEC"
    continue
  fi

  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) continuous_training_retrain_start"
  if bash scripts/weekly_retrain_v3.sh; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) continuous_training_cycle_done"
  else
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) continuous_training_retrain_failed sleep_sec=$FAILURE_SLEEP_SEC"
    if [[ "$RUN_ONCE" == "1" ]]; then
      exit 1
    fi
    sleep "$FAILURE_SLEEP_SEC"
    continue
  fi

  if [[ "$RUN_ONCE" == "1" ]]; then
    break
  fi
  sleep "$INTERVAL_SEC"
done
