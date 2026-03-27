#!/usr/bin/env bash
set -euo pipefail

TMP="$(mktemp -d)"
VENV_DIR="${VENV_DIR:-./.venv}"
BASE_DATA="$TMP/data_best"
NEW_DATA="$TMP/data_new"
READY_DATA="$TMP/training_ready"
VIDEO_DATA="$TMP/video_data"
REPORTS="$TMP/reports"
ENS_OUT="$TMP/artifacts_ens"
VIDEO_ARTIFACTS="$TMP/video_artifacts"
mkdir -p "$BASE_DATA"/{train,val,test}/{ai,real} "$NEW_DATA"/train/{ai,real} "$VIDEO_DATA"/{train,val}/{ai,real}

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
fi

repo_python() {
  local python_bin="${VENV_DIR}/bin/python"
  if [[ -x "$python_bin" ]]; then
    "$python_bin" "$@"
    return 0
  fi
  python "$@"
}

repo_python - <<'PY' "$BASE_DATA" "$NEW_DATA"
from pathlib import Path
import sys
from PIL import Image
import numpy as np

base = Path(sys.argv[1])
new = Path(sys.argv[2])
rng = np.random.default_rng(0)


def make_ai_image(seed: int) -> np.ndarray:
    local_rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    y = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    base = np.stack(
        [
            210 * xx + 20,
            170 * yy + 30,
            120 * (1.0 - xx) + 80,
        ],
        axis=-1,
    )
    stripe_mask = (((np.arange(64)[:, None] // 4) + (np.arange(64)[None, :] // 4)) % 2).astype(np.float32)
    base[..., 0] += stripe_mask * 18
    base[..., 2] += (1.0 - stripe_mask) * 12
    square = slice(18, 46)
    base[square, square, 1] += 35
    base += local_rng.normal(0, 2.0, size=base.shape)
    return np.clip(base, 0, 255).astype("uint8")


def make_real_image(seed: int) -> np.ndarray:
    local_rng = np.random.default_rng(seed)
    base = local_rng.normal(loc=(105, 132, 92), scale=(42, 36, 40), size=(64, 64, 3))
    column_texture = local_rng.normal(0, 10, size=(64, 1, 3))
    row_texture = local_rng.normal(0, 10, size=(1, 64, 3))
    base += column_texture + row_texture
    for _ in range(6):
        y0 = int(local_rng.integers(0, 52))
        x0 = int(local_rng.integers(0, 52))
        h = int(local_rng.integers(6, 14))
        w = int(local_rng.integers(6, 14))
        base[y0 : y0 + h, x0 : x0 + w] += local_rng.normal(0, 18, size=(h, w, 3))
    return np.clip(base, 0, 255).astype("uint8")


for split, n in [("train", 8), ("val", 4), ("test", 4)]:
    split_seed = {"train": 1000, "val": 2000, "test": 3000}[split]
    for cls in ["ai", "real"]:
        d = base / split / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            class_offset = 0 if cls == "ai" else 500
            seed = split_seed + class_offset + i
            arr = make_ai_image(seed) if cls == "ai" else make_real_image(seed)
            Image.fromarray(arr, mode="RGB").save(d / f"{cls}_{i}.jpg", quality=92)

for cls in ["ai", "real"]:
    d = new / "train" / cls
    d.mkdir(parents=True, exist_ok=True)
    seed = 4000 if cls == "ai" else 4500
    arr = make_ai_image(seed) if cls == "ai" else make_real_image(seed)
    Image.fromarray(arr, mode="RGB").save(d / f"extra_{cls}.jpg", quality=92)
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

clone_smoke_member() {
  local target_dir="$1"
  rm -rf "$target_dir"
  mkdir -p "$target_dir"
  cp -R "$ENS_OUT/m1/." "$target_dir/"
}

repo_python scripts/prepare_training_data.py \
  --base "$BASE_DATA" \
  --incremental "$NEW_DATA" \
  --out "$READY_DATA" \
  --copy

aid-train \
  --data "$READY_DATA" \
  --out "$ENS_OUT/m1" \
  --epochs 2 \
  --batch-size 8 \
  --img-size 256 \
  --lr 2e-4 \
  --loss focal \
  --focal-gamma 2.0 \
  --backbone tiny \
  --num-workers 0 \
  --no-pretrained-backbone \
  --no-compile

for member in m2 m3 m4 m5_metadata; do
  clone_smoke_member "$ENS_OUT/$member"
done

collect_ensemble_model_paths

repo_python scripts/fit_ensemble.py \
  --data "$READY_DATA" \
  --model "${ENSEMBLE_MODELS[@]}" \
  --out "$ENS_OUT/ensemble_config.json" \
  --steps 10 \
  --lr 0.05 \
  --l2 0.001 \
  --batch-size 2 \
  --num-workers 0

repo_python scripts/eval_test_ensemble.py \
  --data "$READY_DATA" \
  --model "${ENSEMBLE_MODELS[@]}" \
  --ensemble-config "$ENS_OUT/ensemble_config.json" \
  --tta 1 \
  --out "$ENS_OUT/test_metrics.json"

repo_python scripts/fit_domain_thresholds.py \
  --data "$READY_DATA" \
  --model "${ENSEMBLE_MODELS[@]}" \
  --ensemble-config "$ENS_OUT/ensemble_config.json" \
  --out "$ENS_OUT/domain_config.json" \
  --objective balanced \
  --min-samples-per-domain 1

repo_python -m ai_image_detector.robust_eval \
  --data "$READY_DATA" \
  --model "${ENSEMBLE_MODELS[@]}" \
  --ensemble-config "$ENS_OUT/ensemble_config.json" \
  --max-images 4 \
  --out "$ENS_OUT/robust_eval.json"

repo_python scripts/write_pipeline_report.py dataset \
  --data "$BASE_DATA" \
  --prepared "$READY_DATA" \
  --incremental "$NEW_DATA" \
  --video "$VIDEO_DATA" \
  --cache-file "$TMP/hf_sources.txt" \
  --out "$REPORTS/dataset_qa_summary.json" \
  --provenance-out "$REPORTS/dataset_provenance.json"

repo_python scripts/write_pipeline_report.py final \
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

repo_python scripts/export_best_release.py \
  --ens-out "$ENS_OUT" \
  --video-artifacts "$VIDEO_ARTIFACTS" \
  --out "$ENS_OUT/release"

repo_python scripts/benchmark_gate.py \
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
