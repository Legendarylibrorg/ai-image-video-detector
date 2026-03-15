#!/usr/bin/env bash
set -euo pipefail

# Retention cleanup helper (best-effort).
# Usage:
#   bash scripts/privacy_cleanup.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

QUEUE_DIR="${QUEUE_DIR:-./incoming_review_queue}"
RUNTIME_LOG="${RUNTIME_LOG:-./.local/runtime/runner.log}"
QUEUE_RETENTION_DAYS="${QUEUE_RETENTION_DAYS:-7}"
LOG_RETENTION_DAYS="${LOG_RETENTION_DAYS:-14}"
MODEL_OUTPUT_RETENTION_DAYS="${MODEL_OUTPUT_RETENTION_DAYS:-30}"

echo "privacy_cleanup start queue_days=$QUEUE_RETENTION_DAYS log_days=$LOG_RETENTION_DAYS"

if [[ -d "$QUEUE_DIR" ]]; then
  find "$QUEUE_DIR" -type f -mtime +"$QUEUE_RETENTION_DAYS" -print -delete || true
fi

if [[ -f "$RUNTIME_LOG" ]]; then
  find "$RUNTIME_LOG" -type f -mtime +"$LOG_RETENTION_DAYS" -print -delete || true
fi

if [[ -d "./incoming_model_outputs/_processed" ]]; then
  find "./incoming_model_outputs/_processed" -type f -mtime +"$MODEL_OUTPUT_RETENTION_DAYS" -print -delete || true
fi

echo "privacy_cleanup done"
