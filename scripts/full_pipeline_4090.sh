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
DRY_RUN="${DRY_RUN:-0}"
RUN_VIDEO_DATA_PULL="${RUN_VIDEO_DATA_PULL:-1}"
VIDEO_OUT="${VIDEO_OUT:-./video_data}"
VIDEO_TRAIN_PER_CLASS="${VIDEO_TRAIN_PER_CLASS:-220}"
VIDEO_VAL_PER_CLASS="${VIDEO_VAL_PER_CLASS:-60}"
VIDEO_MODE="${VIDEO_MODE:-snapshot}"
VIDEO_SNAPSHOT_MAX_WORKERS="${VIDEO_SNAPSHOT_MAX_WORKERS:-1}"
VIDEO_REPO_BASE_PAUSE_MS="${VIDEO_REPO_BASE_PAUSE_MS:-2200}"
VIDEO_REPO_JITTER_MS="${VIDEO_REPO_JITTER_MS:-1800}"
VIDEO_COPY_SLEEP_MS="${VIDEO_COPY_SLEEP_MS:-15}"
VIDEO_SLEEP_MS="${VIDEO_SLEEP_MS:-120}"
VIDEO_JITTER_MS="${VIDEO_JITTER_MS:-80}"
VIDEO_CHUNK_PAUSE_MS="${VIDEO_CHUNK_PAUSE_MS:-1000}"
VIDEO_REPO_COOLDOWN_MS="${VIDEO_REPO_COOLDOWN_MS:-3000}"
VIDEO_RETRIES="${VIDEO_RETRIES:-5}"

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] $*"
  else
    eval "$*"
  fi
}

if [[ ! -d .venv ]]; then
  run_cmd "python3 -m venv .venv"
fi
source .venv/bin/activate
run_cmd "python -m pip install --upgrade pip"
run_cmd "pip install -e . datasets"

if [[ "$SKIP_DATA" != "1" ]]; then
  run_cmd "python scripts/build_best_dataset.py --out \"$DATA_DIR\" --train-per-class \"$TRAIN_PER_CLASS\" --val-per-class \"$VAL_PER_CLASS\" --test-per-class \"$TEST_PER_CLASS\""
fi

if [[ "$RUN_VIDEO_DATA_PULL" == "1" ]]; then
  run_cmd "python scripts/build_video_dataset.py --out \"$VIDEO_OUT\" --train-per-class \"$VIDEO_TRAIN_PER_CLASS\" --val-per-class \"$VIDEO_VAL_PER_CLASS\" --mode \"$VIDEO_MODE\" --snapshot-max-workers \"$VIDEO_SNAPSHOT_MAX_WORKERS\" --repo-base-pause-ms \"$VIDEO_REPO_BASE_PAUSE_MS\" --repo-jitter-ms \"$VIDEO_REPO_JITTER_MS\" --copy-sleep-ms \"$VIDEO_COPY_SLEEP_MS\" --sleep-ms \"$VIDEO_SLEEP_MS\" --jitter-ms \"$VIDEO_JITTER_MS\" --chunk-pause-ms \"$VIDEO_CHUNK_PAUSE_MS\" --repo-cooldown-ms \"$VIDEO_REPO_COOLDOWN_MS\" --retries \"$VIDEO_RETRIES\""
fi

if [[ "$SKIP_SWEEP" != "1" ]]; then
  run_cmd "EPOCHS=14 bash scripts/hparam_sweep.sh \"$DATA_DIR\" \"$SWEEP_OUT\""
fi

run_cmd "bash scripts/train_ensemble.sh \"$DATA_DIR\" \"$ENS_OUT\" \"$EPOCHS\""

if [[ "$RUN_HARD_MINING" == "1" ]]; then
  run_cmd "python scripts/mine_hard_negatives.py --data \"$DATA_DIR\" --model \"$ENS_OUT\"/m1/best.pt \"$ENS_OUT\"/m2/best.pt \"$ENS_OUT\"/m3/best.pt \"$ENS_OUT\"/m4/best.pt --out \"$ENS_OUT\"/hard_mined --top-k 5000"
fi

run_cmd "python scripts/eval_test_ensemble.py --data \"$DATA_DIR\" --model \"$ENS_OUT\"/m1/best.pt \"$ENS_OUT\"/m2/best.pt \"$ENS_OUT\"/m3/best.pt \"$ENS_OUT\"/m4/best.pt --out \"$ENS_OUT\"/test_metrics.json"

if [[ "$RUN_DISTILL" == "1" ]]; then
  run_cmd "python scripts/train_distill.py --data \"$DATA_DIR\" --teacher \"$ENS_OUT\"/m1/best.pt \"$ENS_OUT\"/m2/best.pt \"$ENS_OUT\"/m3/best.pt \"$ENS_OUT\"/m4/best.pt --out \"$ENS_OUT\"/distill --student-backbone effb0 --img-size 320 --batch-size 64 --epochs 10"
fi

if [[ "$DRY_RUN" != "1" ]]; then
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
else
  echo "[DRY_RUN] write artifacts_ens/prod_manifest.json"
fi

echo "Pipeline complete."
echo "Prod manifest: $ENS_OUT/prod_manifest.json"
