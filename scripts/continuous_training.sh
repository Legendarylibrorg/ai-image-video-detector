#!/usr/bin/env bash
set -euo pipefail

# Continuous retraining loop built on the training-only pipeline flow.
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
FAILURE_SLEEP_SEC="${CONTINUOUS_TRAIN_FAILURE_SLEEP_SEC:-900}"
RUN_ONCE="${RUN_ONCE:-0}"

exec > >(tee -a "$LOG_FILE") 2>&1

is_training_active() {
  [[ -f "$TRAIN_LOCK" ]]
}

while true; do
  while is_training_active; do
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) continuous_training_wait lock=$TRAIN_LOCK sleep_sec=$CHECK_WHILE_TRAINING_SEC"
    sleep "$CHECK_WHILE_TRAINING_SEC"
  done

  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) continuous_training_cycle_start"
  if bash scripts/weekly_retrain_v3.sh; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) continuous_training_cycle_done"
  else
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) continuous_training_cycle_failed sleep_sec=$FAILURE_SLEEP_SEC"
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
