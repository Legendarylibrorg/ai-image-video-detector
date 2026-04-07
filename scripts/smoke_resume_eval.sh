#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
TMP="$(mktemp -d)"
VENV_DIR="${VENV_DIR:-./.venv}"
source "$ROOT_DIR/scripts/lib/core.sh"
source "$ROOT_DIR/scripts/lib/training.sh"
BASE_DATA="$TMP/data_best"
NEW_DATA="$TMP/data_new"
READY_DATA="$TMP/training_ready"
VIDEO_DATA="$TMP/video_data"
REPORTS="$TMP/reports"
ENS_OUT="$TMP/artifacts_ens"
VIDEO_ARTIFACTS="$TMP/video_artifacts"
mkdir -p "$BASE_DATA"/{train,val,test}/{ai,real} "$NEW_DATA"/train/{ai,real} "$VIDEO_DATA"/{train,val}/{ai,real}
export DATA_DIR="$BASE_DATA"
export TRAIN_INCREMENTAL_DATA_DIR="$NEW_DATA"
export TRAIN_READY_DATA_DIR="$READY_DATA"
export TRAIN_DATA_COPY_ONLY=1
export VIDEO_OUT="$VIDEO_DATA"
export PIPELINE_REPORT_DIR="$REPORTS"
export ENS_OUT="$ENS_OUT"
export VIDEO_ARTIFACTS_OUT="$VIDEO_ARTIFACTS"
export BEST_DS_HF_CACHE_FILE="$TMP/hf_sources.txt"
ensure_env

repo_python - <<'PY' "$BASE_DATA" "$NEW_DATA"
from pathlib import Path
import sys
from PIL import Image
import numpy as np
import piexif

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


def save_labeled_image(path: Path, arr: np.ndarray, cls: str) -> None:
    image = Image.fromarray(arr, mode="RGB")
    if cls == "ai":
        exif_dict = {
            "0th": {
                piexif.ImageIFD.Software: b"Stable Diffusion WebUI",
            },
            "Exif": {
                piexif.ExifIFD.UserComment: b"prompt: cinematic portrait; steps: 28; sampler: euler",
            },
            "GPS": {},
            "Interop": {},
            "1st": {},
            "thumbnail": None,
        }
    else:
        exif_dict = {
            "0th": {
                piexif.ImageIFD.Make: b"Canon",
                piexif.ImageIFD.Model: b"EOS R6",
                piexif.ImageIFD.DateTime: b"2024:01:02 03:04:05",
            },
            "Exif": {},
            "GPS": {},
            "Interop": {},
            "1st": {},
            "thumbnail": None,
        }
    image.save(path, format="JPEG", quality=92, exif=piexif.dump(exif_dict))


for split, n in [("train", 16), ("val", 8), ("test", 8)]:
    split_seed = {"train": 1000, "val": 2000, "test": 3000}[split]
    for cls in ["ai", "real"]:
        d = base / split / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            class_offset = 0 if cls == "ai" else 500
            seed = split_seed + class_offset + i
            arr = make_ai_image(seed) if cls == "ai" else make_real_image(seed)
            save_labeled_image(d / f"{cls}_{i}.jpg", arr, cls)

for cls in ["ai", "real"]:
    d = new / "train" / cls
    d.mkdir(parents=True, exist_ok=True)
    seed = 4000 if cls == "ai" else 4500
    arr = make_ai_image(seed) if cls == "ai" else make_real_image(seed)
    save_labeled_image(d / f"extra_{cls}.jpg", arr, cls)
PY

export PYTHONPYCACHEPREFIX="$TMP/pycache"
export TRAIN_NO_PRETRAINED_BACKBONE=1
export TRAIN_NO_COMPILE=1
export TRAIN_NUM_WORKERS=0
export PIPELINE_MIN_FREE_GB=0
export MALWARE_SCAN=0
export RUN_METADATA_MEMBER=0
export TRAIN_ENSEMBLE_PROFILE=smoke

prepare_training_image_data

TRAIN_PATIENCE="${TRAIN_PATIENCE:-2}" \
TRAIN_MIN_DELTA="${TRAIN_MIN_DELTA:-0.0}" \
TRAIN_DEGENERATE_PATIENCE="${TRAIN_DEGENERATE_PATIENCE:-2}" \
METADATA_MEMBER_EPOCHS="${METADATA_MEMBER_EPOCHS:-2}" \
SKIP_SWEEP=1 \
RUN_HARD_MINING=0 \
RUN_HARD_RETRAIN=0 \
RUN_DISTILL=0 \
RUN_METADATA_MEMBER=0 \
EPOCHS=2 \
ENS_FIT_STEPS=10 \
EVAL_TTA_VIEWS=1 \
DOMAIN_THRESHOLD_MIN_SAMPLES=1 \
ROBUST_EVAL_MAX_IMAGES=4 \
run_prepared_max_quality_pipeline "$BASE_DATA" 1

GATE_ALLOW_MISSING_VIDEO=1 \
GATE_MIN_IMAGE_AUC=0.0 \
GATE_MIN_IMAGE_F1=0.0 \
GATE_MIN_IMAGE_PRECISION=0.0 \
GATE_MIN_IMAGE_RECALL=0.0 \
GATE_MAX_IMAGE_ECE=1.0 \
GATE_MAX_IMAGE_BRIER=1.0 \
GATE_MIN_ROBUST_WORST_AUC=0.0 \
GATE_MIN_ROBUST_WORST_F1=0.0 \
GATE_MAX_ROBUST_AUC_DROP=1.0 \
run_benchmark_gate

for f in \
  "$READY_DATA/training_data_report.json" \
  "$ENS_OUT/m1/best.safetensors" \
  "$ENS_OUT/m2/best.safetensors" \
  "$ENS_OUT/m3/best.safetensors" \
  "$ENS_OUT/m4/best.safetensors" \
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
