#!/usr/bin/env bash
set -euo pipefail

# Weekly retrain with gating for training-only mode.
# 1) ingest reviewed queue labels
# 2) run the local retrain flow

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-./.local/runtime}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/weekly_retrain.log}"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) weekly_retrain_start"

python scripts/review_queue_to_dataset.py --queue "${REVIEW_QUEUE_DIR:-./incoming_review_queue}" --dst "${NEW_DATA_DST:-./data_new/train}" || true

bash scripts/local_retrain_4090.sh

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) weekly_retrain_gate_passed"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) weekly_retrain_done"
