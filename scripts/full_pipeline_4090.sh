#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data_best}"
TRAIN_PER_CLASS="${TRAIN_PER_CLASS:-40000}"
VAL_PER_CLASS="${VAL_PER_CLASS:-9000}"
TEST_PER_CLASS="${TEST_PER_CLASS:-9000}"
SWEEP_OUT="${SWEEP_OUT:-./artifacts_sweep}"
ENS_OUT="${ENS_OUT:-./artifacts_ens}"
EPOCHS="${EPOCHS:-18}"
SKIP_DATA="${SKIP_DATA:-0}"
SKIP_SWEEP="${SKIP_SWEEP:-0}"
RUN_HARD_MINING="${RUN_HARD_MINING:-1}"
RUN_DISTILL="${RUN_DISTILL:-1}"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e . datasets

if [[ "$SKIP_DATA" != "1" ]]; then
  python scripts/build_best_dataset.py \
    --out "$DATA_DIR" \
    --train-per-class "$TRAIN_PER_CLASS" \
    --val-per-class "$VAL_PER_CLASS" \
    --test-per-class "$TEST_PER_CLASS"
fi

if [[ "$SKIP_SWEEP" != "1" ]]; then
  EPOCHS=14 bash scripts/hparam_sweep.sh "$DATA_DIR" "$SWEEP_OUT"
fi

bash scripts/train_ensemble.sh "$DATA_DIR" "$ENS_OUT" "$EPOCHS"

if [[ "$RUN_HARD_MINING" == "1" ]]; then
  python scripts/mine_hard_negatives.py \
    --data "$DATA_DIR" \
    --model "$ENS_OUT"/m1/best.pt "$ENS_OUT"/m2/best.pt "$ENS_OUT"/m3/best.pt "$ENS_OUT"/m4/best.pt \
    --out "$ENS_OUT"/hard_mined \
    --top-k 5000
fi

python scripts/eval_test_ensemble.py \
  --data "$DATA_DIR" \
  --model "$ENS_OUT"/m1/best.pt "$ENS_OUT"/m2/best.pt "$ENS_OUT"/m3/best.pt "$ENS_OUT"/m4/best.pt \
  --out "$ENS_OUT"/test_metrics.json

if [[ "$RUN_DISTILL" == "1" ]]; then
  python scripts/train_distill.py \
    --data "$DATA_DIR" \
    --teacher "$ENS_OUT"/m1/best.pt "$ENS_OUT"/m2/best.pt "$ENS_OUT"/m3/best.pt "$ENS_OUT"/m4/best.pt \
    --out "$ENS_OUT"/distill \
    --student-backbone effb0 \
    --img-size 320 \
    --batch-size 64 \
    --epochs 10
fi

python - <<'PY'
import json
from pathlib import Path

ens = Path("./artifacts_ens")
manifest = {
    "models": [
        str((ens / "m1" / "best.pt").resolve()),
        str((ens / "m2" / "best.pt").resolve()),
        str((ens / "m3" / "best.pt").resolve()),
        str((ens / "m4" / "best.pt").resolve()),
    ],
    "test_metrics": str((ens / "test_metrics.json").resolve()),
}
out = ens / "prod_manifest.json"
out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"wrote {out}")
PY

echo "Pipeline complete."
echo "Prod manifest: $ENS_OUT/prod_manifest.json"
