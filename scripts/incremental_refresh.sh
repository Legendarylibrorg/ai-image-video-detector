#!/usr/bin/env bash
set -euo pipefail

# Continual retraining with newly collected samples
# Expects fresh labeled data under: ./data_new/train/{ai,real} and ./data_new/val/{ai,real}

BASE_DATA="${BASE_DATA:-./data_best}"
NEW_DATA="${NEW_DATA:-./data_new}"
MERGED="${MERGED:-./data_refresh}"
OUT="${OUT:-./artifacts_refresh}"
EPOCHS="${EPOCHS:-8}"

mkdir -p "$MERGED/train/ai" "$MERGED/train/real" "$MERGED/val/ai" "$MERGED/val/real"

cp -n "$BASE_DATA"/train/ai/* "$MERGED/train/ai/" 2>/dev/null || true
cp -n "$BASE_DATA"/train/real/* "$MERGED/train/real/" 2>/dev/null || true
cp -n "$BASE_DATA"/val/ai/* "$MERGED/val/ai/" 2>/dev/null || true
cp -n "$BASE_DATA"/val/real/* "$MERGED/val/real/" 2>/dev/null || true

cp -n "$NEW_DATA"/train/ai/* "$MERGED/train/ai/" 2>/dev/null || true
cp -n "$NEW_DATA"/train/real/* "$MERGED/train/real/" 2>/dev/null || true
cp -n "$NEW_DATA"/val/ai/* "$MERGED/val/ai/" 2>/dev/null || true
cp -n "$NEW_DATA"/val/real/* "$MERGED/val/real/" 2>/dev/null || true

aid-train --data "$MERGED" --epochs "$EPOCHS" --batch-size 64 --img-size 320 --lr 1.5e-4 --loss focal --backbone effb0 --out "$OUT"

echo "refresh model saved in $OUT"
