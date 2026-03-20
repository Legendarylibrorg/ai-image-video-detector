#!/usr/bin/env bash
set -euo pipefail

TMP="$(mktemp -d)"
DATA="$TMP/data"
OUT="$TMP/artifacts"
mkdir -p "$DATA"/{train,val,test}/{ai,real}

python3 - <<'PY' "$DATA"
from pathlib import Path
import sys
from PIL import Image
import numpy as np

root = Path(sys.argv[1])
rng = np.random.default_rng(0)
for split, n in [("train", 4), ("val", 2), ("test", 2)]:
    for cls in ["ai", "real"]:
        d = root / split / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            arr = (rng.random((64, 64, 3)) * 255).astype("uint8")
            Image.fromarray(arr, mode="RGB").save(d / f"{cls}_{i}.jpg", quality=90)
PY

export PYTHONPYCACHEPREFIX="$TMP/pycache"

python3 -m ai_image_detector.train \
  --data "$DATA" \
  --out "$OUT" \
  --epochs 1 \
  --batch-size 2 \
  --img-size 64 \
  --no-pretrained-backbone \
  --no-compile \
  --no-amp \
  --num-workers 0 \
  --degenerate-patience 0 \
  --save-every 1

python3 -m ai_image_detector.train \
  --data "$DATA" \
  --out "$OUT" \
  --epochs 2 \
  --resume "$OUT/last.pt" \
  --batch-size 2 \
  --img-size 64 \
  --no-pretrained-backbone \
  --no-compile \
  --no-amp \
  --num-workers 0 \
  --degenerate-patience 0 \
  --save-every 1

for f in \
  "$OUT/best.pt" \
  "$OUT/last.pt" \
  "$OUT/epoch_001.pt" \
  "$OUT/config.json" \
  "$OUT/training_log.jsonl" \
  "$OUT/latest_checkpoint.txt" \
  "$OUT/test_metrics.json"; do
  test -f "$f"
done

echo "smoke_ok out=$OUT"
