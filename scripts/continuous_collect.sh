#!/usr/bin/env bash
set -euo pipefail

# Continuous collection loop that pauses whenever training is active.
# Usage:
#   bash scripts/continuous_collect.sh
#   RUN_ONCE=1 bash scripts/continuous_collect.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TRAIN_LOCK="${TRAIN_LOCK:-$ROOT_DIR/.local/training.lock}"
INTERVAL_SEC="${INTERVAL_SEC:-21600}"           # 6h default
CHECK_WHILE_TRAINING_SEC="${CHECK_WHILE_TRAINING_SEC:-600}"
RUN_ONCE="${RUN_ONCE:-0}"

is_training_active() {
  [[ -f "$TRAIN_LOCK" ]]
}

while true; do
  if is_training_active; then
    echo "continuous_collect: training lock present ($TRAIN_LOCK), sleeping ${CHECK_WHILE_TRAINING_SEC}s"
    sleep "$CHECK_WHILE_TRAINING_SEC"
    continue
  fi

  echo "continuous_collect: starting cycle at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  bash scripts/do.sh collect
  echo "continuous_collect: finished cycle at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

  if [[ "$RUN_ONCE" == "1" ]]; then
    break
  fi
  sleep "$INTERVAL_SEC"
done
