#!/usr/bin/env bash
set -euo pipefail

TMP="$(mktemp -d)"
BASE_DATA="$TMP/data_best"
NEW_DATA="$TMP/data_new"
READY_DATA="$TMP/training_ready"
VIDEO_DATA="$TMP/video_data"
REPORTS="$TMP/reports"
ENS_OUT="$TMP/artifacts_ens"
VIDEO_ARTIFACTS="$TMP/video_artifacts"
mkdir -p "$BASE_DATA"/{train,val,test}/{ai,real} "$NEW_DATA"/train/{ai,real} "$VIDEO_DATA"/{train,val}/{ai,real}

python - <<'PY' "$BASE_DATA" "$NEW_DATA"
from pathlib import Path
import sys
from PIL import Image
import numpy as np

base = Path(sys.argv[1])
new = Path(sys.argv[2])
rng = np.random.default_rng(0)
for split, n in [("train", 4), ("val", 2), ("test", 2)]:
    for cls in ["ai", "real"]:
        d = base / split / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            arr = (rng.random((64, 64, 3)) * 255).astype("uint8")
            Image.fromarray(arr, mode="RGB").save(d / f"{cls}_{i}.jpg", quality=90)
for cls in ["ai", "real"]:
    d = new / "train" / cls
    d.mkdir(parents=True, exist_ok=True)
    arr = (rng.random((64, 64, 3)) * 255).astype("uint8")
    Image.fromarray(arr, mode="RGB").save(d / f"extra_{cls}.jpg", quality=90)
PY

export PYTHONPYCACHEPREFIX="$TMP/pycache"
export TRAIN_NO_PRETRAINED_BACKBONE=1
export TRAIN_NO_COMPILE=1
export TRAIN_NUM_WORKERS=0
export PIPELINE_MIN_FREE_GB=0
export MALWARE_SCAN=0
export RUN_METADATA_MEMBER=1

ENSEMBLE_MODELS=()

collect_ensemble_model_paths() {
  ENSEMBLE_MODELS=()
  local model_dir=""
  for model_dir in "$ENS_OUT"/m*; do
    [[ -d "$model_dir" ]] || continue
    if [[ -f "$model_dir/best.safetensors" ]]; then
      ENSEMBLE_MODELS+=("$model_dir/best.safetensors")
    fi
  done
}

python scripts/prepare_training_data.py \
  --base "$BASE_DATA" \
  --incremental "$NEW_DATA" \
  --out "$READY_DATA" \
  --copy

bash scripts/train_ensemble.sh "$READY_DATA" "$ENS_OUT" 1
collect_ensemble_model_paths

python scripts/fit_ensemble.py \
  --data "$READY_DATA" \
  --model "${ENSEMBLE_MODELS[@]}" \
  --out "$ENS_OUT/ensemble_config.json" \
  --steps 10 \
  --lr 0.05 \
  --l2 0.001 \
  --batch-size 2 \
  --num-workers 0

python scripts/eval_test_ensemble.py \
  --data "$READY_DATA" \
  --model "${ENSEMBLE_MODELS[@]}" \
  --ensemble-config "$ENS_OUT/ensemble_config.json" \
  --tta 1 \
  --out "$ENS_OUT/test_metrics.json"

python scripts/fit_domain_thresholds.py \
  --data "$READY_DATA" \
  --model "${ENSEMBLE_MODELS[@]}" \
  --ensemble-config "$ENS_OUT/ensemble_config.json" \
  --out "$ENS_OUT/domain_config.json" \
  --objective balanced \
  --min-samples-per-domain 1

python -m ai_image_detector.robust_eval \
  --data "$READY_DATA" \
  --model "${ENSEMBLE_MODELS[@]}" \
  --ensemble-config "$ENS_OUT/ensemble_config.json" \
  --max-images 4 \
  --out "$ENS_OUT/robust_eval.json"

python scripts/write_pipeline_report.py dataset \
  --data "$BASE_DATA" \
  --prepared "$READY_DATA" \
  --incremental "$NEW_DATA" \
  --video "$VIDEO_DATA" \
  --cache-file "$TMP/hf_sources.txt" \
  --out "$REPORTS/dataset_qa_summary.json" \
  --provenance-out "$REPORTS/dataset_provenance.json"

python scripts/write_pipeline_report.py final \
  --data "$BASE_DATA" \
  --prepared "$READY_DATA" \
  --video "$VIDEO_DATA" \
  --ens-out "$ENS_OUT" \
  --ensemble-config "$ENS_OUT/ensemble_config.json" \
  --domain-config "$ENS_OUT/domain_config.json" \
  --video-artifacts "$VIDEO_ARTIFACTS" \
  --dataset-qa "$REPORTS/dataset_qa_summary.json" \
  --robust-eval "$ENS_OUT/robust_eval.json" \
  --prod-manifest "$ENS_OUT/prod_manifest.json" \
  --summary-out "$ENS_OUT/final_run_summary.json" \
  --manifest-out "$ENS_OUT/run_manifest.json" \
  --thresholds-out "$ENS_OUT/final_thresholds.json" \
  --release-bundle "$ENS_OUT/release"

python scripts/export_best_release.py \
  --ens-out "$ENS_OUT" \
  --video-artifacts "$VIDEO_ARTIFACTS" \
  --out "$ENS_OUT/release"

python scripts/benchmark_gate.py \
  --ens-out "$ENS_OUT" \
  --video-out "$VIDEO_ARTIFACTS" \
  --min-image-auc 0.0 \
  --min-image-f1 0.0 \
  --min-image-precision 0.0 \
  --min-image-recall 0.0 \
  --max-image-ece 1.0 \
  --max-image-brier 1.0 \
  --min-robust-worst-auc 0.0 \
  --min-robust-worst-f1 0.0 \
  --max-robust-auc-drop 1.0 \
  --skip-video

for f in \
  "$READY_DATA/training_data_report.json" \
  "$ENS_OUT/m1/best.safetensors" \
  "$ENS_OUT/m2/best.safetensors" \
  "$ENS_OUT/m3/best.safetensors" \
  "$ENS_OUT/m4/best.safetensors" \
  "$ENS_OUT/m5_metadata/best.safetensors" \
  "$ENS_OUT/m1/calibration.json" \
  "$ENS_OUT/ensemble_config.json" \
  "$ENS_OUT/domain_config.json" \
  "$ENS_OUT/robust_eval.json" \
  "$ENS_OUT/test_metrics.json" \
  "$ENS_OUT/final_run_summary.json" \
  "$ENS_OUT/final_thresholds.json" \
  "$ENS_OUT/run_manifest.json" \
  "$ENS_OUT/prod_manifest.json" \
  "$ENS_OUT/release/release_manifest.json" \
  "$REPORTS/dataset_qa_summary.json" \
  "$REPORTS/dataset_provenance.json"; do
  test -f "$f"
done

echo "smoke_ok out=$ENS_OUT"
