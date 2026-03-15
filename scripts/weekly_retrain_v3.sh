#!/usr/bin/env bash
set -euo pipefail

# Weekly retrain with gating. Intended for Linux cron/systemd timers.
# 1) ingest reviewed queue labels
# 2) collect diverse data
# 3) run max-accuracy v2
# 4) run benchmark gate
# 5) if gate passes, restart service

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-./.local/runtime}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/weekly_retrain.log}"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) weekly_retrain_start"

python scripts/review_queue_to_dataset.py --queue "${REVIEW_QUEUE_DIR:-./incoming_review_queue}" --dst "${NEW_DATA_DST:-./data_new/train}" || true

bash scripts/do.sh collect-diverse
bash scripts/do.sh start-v2

python scripts/benchmark_gate.py \
  --ens-out "${ENS_OUT:-./artifacts_ens}" \
  --video-out "${VIDEO_ARTIFACTS_OUT:-./video_artifacts}" \
  --min-image-auc "${GATE_MIN_IMAGE_AUC:-0.93}" \
  --min-image-f1 "${GATE_MIN_IMAGE_F1:-0.90}" \
  --min-video-acc "${GATE_MIN_VIDEO_ACC:-0.82}"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) weekly_retrain_gate_passed"
bash scripts/linux_service.sh restart || true
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) weekly_retrain_done"
