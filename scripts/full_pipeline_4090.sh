#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data_best}"
TRAIN_PER_CLASS="${TRAIN_PER_CLASS:-40000}"
VAL_PER_CLASS="${VAL_PER_CLASS:-9000}"
TEST_PER_CLASS="${TEST_PER_CLASS:-9000}"
BEST_DS_NEAR_HAMMING="${BEST_DS_NEAR_HAMMING:-2}"
BEST_DS_NEAR_WINDOW="${BEST_DS_NEAR_WINDOW:-2400}"
BEST_DS_MIN_SIDE="${BEST_DS_MIN_SIDE:-192}"
BEST_DS_MAX_ASPECT_RATIO="${BEST_DS_MAX_ASPECT_RATIO:-3.0}"
BEST_DS_MIN_ENTROPY="${BEST_DS_MIN_ENTROPY:-3.2}"
BEST_DS_MAX_UNIQUE_PER_SOURCE="${BEST_DS_MAX_UNIQUE_PER_SOURCE:-220000}"
BEST_DS_MAX_PER_SOURCE_CLASS="${BEST_DS_MAX_PER_SOURCE_CLASS:-120000}"
BEST_DS_JPEG_QUALITY="${BEST_DS_JPEG_QUALITY:-92}"
BEST_DS_HARDNEG_FRACTION="${BEST_DS_HARDNEG_FRACTION:-0.6}"
BEST_DS_DISCOVER_HF="${BEST_DS_DISCOVER_HF:-0}"
BEST_DS_HF_DISCOVERY_LIMIT="${BEST_DS_HF_DISCOVERY_LIMIT:-120}"
BEST_DS_HF_MAX_SOURCES="${BEST_DS_HF_MAX_SOURCES:-260}"
BEST_DS_HF_CACHE_FILE="${BEST_DS_HF_CACHE_FILE:-./.local/hf_discovered_sources.txt}"
BEST_DS_CACHE_DIR="${BEST_DS_CACHE_DIR:-./.local/hf}"
BEST_DS_HF_QUERIES="${BEST_DS_HF_QUERIES:-}"
BEST_DS_SOURCES_FILE="${BEST_DS_SOURCES_FILE:-}"
BEST_DS_EXTRA_SOURCES="${BEST_DS_EXTRA_SOURCES:-}"
BEST_DS_LOCAL_SOURCES="${BEST_DS_LOCAL_SOURCES:-}"
BEST_DS_STREAMING="${BEST_DS_STREAMING:-1}"
BEST_DS_STREAM_BUFFER_SIZE="${BEST_DS_STREAM_BUFFER_SIZE:-12000}"
BEST_DS_MAX_SAMPLES_PER_SOURCE="${BEST_DS_MAX_SAMPLES_PER_SOURCE:-60000}"
BEST_DS_REPO_BASE_PAUSE_MS="${BEST_DS_REPO_BASE_PAUSE_MS:-900}"
BEST_DS_REPO_JITTER_MS="${BEST_DS_REPO_JITTER_MS:-900}"
BEST_DS_REPO_COOLDOWN_MS="${BEST_DS_REPO_COOLDOWN_MS:-45000}"
BEST_DS_MAX_CONSECUTIVE_FAILURES="${BEST_DS_MAX_CONSECUTIVE_FAILURES:-2}"
SWEEP_OUT="${SWEEP_OUT:-./artifacts_sweep}"
ENS_OUT="${ENS_OUT:-./artifacts_ens}"
EPOCHS="${EPOCHS:-18}"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-14}"
SKIP_DATA="${SKIP_DATA:-0}"
SKIP_SWEEP="${SKIP_SWEEP:-0}"
RUN_HARD_MINING="${RUN_HARD_MINING:-1}"
HARD_MINING_TOPK="${HARD_MINING_TOPK:-5000}"
RUN_DISTILL="${RUN_DISTILL:-1}"
DISTILL_EPOCHS="${DISTILL_EPOCHS:-10}"
RUN_ENSEMBLE_FIT="${RUN_ENSEMBLE_FIT:-1}"
ENS_CONFIG_PATH="${ENS_CONFIG_PATH:-$ENS_OUT/ensemble_config.json}"
ENS_FIT_STEPS="${ENS_FIT_STEPS:-1200}"
ENS_FIT_LR="${ENS_FIT_LR:-0.05}"
ENS_FIT_L2="${ENS_FIT_L2:-0.001}"
ENS_FIT_MAX_VAL_IMAGES="${ENS_FIT_MAX_VAL_IMAGES:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_VIDEO_DATA_PULL="${RUN_VIDEO_DATA_PULL:-1}"
VIDEO_OUT="${VIDEO_OUT:-./video_data}"
VIDEO_CACHE_DIR="${VIDEO_CACHE_DIR:-./.local/hf}"
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
RUN_VIDEO_TRAIN="${RUN_VIDEO_TRAIN:-0}"
VIDEO_ARTIFACTS_OUT="${VIDEO_ARTIFACTS_OUT:-./video_artifacts}"
VIDEO_TRAIN_EPOCHS="${VIDEO_TRAIN_EPOCHS:-30}"
VIDEO_TRAIN_BATCH_SIZE="${VIDEO_TRAIN_BATCH_SIZE:-4}"
VIDEO_TRAIN_IMG_SIZE="${VIDEO_TRAIN_IMG_SIZE:-224}"
VIDEO_TRAIN_FRAMES="${VIDEO_TRAIN_FRAMES:-24}"
VIDEO_TRAIN_GRAD_ACCUM="${VIDEO_TRAIN_GRAD_ACCUM:-2}"
VIDEO_TRAIN_LR="${VIDEO_TRAIN_LR:-1e-4}"
VIDEO_TRAIN_PATIENCE="${VIDEO_TRAIN_PATIENCE:-6}"
VIDEO_TRAIN_MIN_DELTA="${VIDEO_TRAIN_MIN_DELTA:-0.001}"
VIDEO_TRAIN_RESUME="${VIDEO_TRAIN_RESUME:-1}"

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
run_cmd "pip install -e . datasets huggingface_hub safetensors"

if [[ "$SKIP_DATA" != "1" ]]; then
  dataset_cmd="python scripts/build_best_dataset.py --out \"$DATA_DIR\" --train-per-class \"$TRAIN_PER_CLASS\" --val-per-class \"$VAL_PER_CLASS\" --test-per-class \"$TEST_PER_CLASS\" --near-hamming \"$BEST_DS_NEAR_HAMMING\" --near-window \"$BEST_DS_NEAR_WINDOW\" --min-side \"$BEST_DS_MIN_SIDE\" --max-aspect-ratio \"$BEST_DS_MAX_ASPECT_RATIO\" --min-entropy \"$BEST_DS_MIN_ENTROPY\" --max-unique-per-source \"$BEST_DS_MAX_UNIQUE_PER_SOURCE\" --max-per-source-class \"$BEST_DS_MAX_PER_SOURCE_CLASS\" --jpeg-quality \"$BEST_DS_JPEG_QUALITY\" --hardneg-fraction \"$BEST_DS_HARDNEG_FRACTION\" --cache-dir \"$BEST_DS_CACHE_DIR\" --hf-cache-only-if-present --stream-buffer-size \"$BEST_DS_STREAM_BUFFER_SIZE\" --max-samples-per-source \"$BEST_DS_MAX_SAMPLES_PER_SOURCE\" --repo-base-pause-ms \"$BEST_DS_REPO_BASE_PAUSE_MS\" --repo-jitter-ms \"$BEST_DS_REPO_JITTER_MS\" --repo-cooldown-ms \"$BEST_DS_REPO_COOLDOWN_MS\" --max-consecutive-failures \"$BEST_DS_MAX_CONSECUTIVE_FAILURES\""

  if [[ "$BEST_DS_STREAMING" == "1" ]]; then
    dataset_cmd="$dataset_cmd --streaming"
  else
    dataset_cmd="$dataset_cmd --no-streaming"
  fi

  if [[ "$BEST_DS_DISCOVER_HF" == "1" ]]; then
    dataset_cmd="$dataset_cmd --discover-hf --hf-discovery-limit \"$BEST_DS_HF_DISCOVERY_LIMIT\" --hf-max-sources \"$BEST_DS_HF_MAX_SOURCES\" --hf-cache-file \"$BEST_DS_HF_CACHE_FILE\""
  fi

  if [[ -n "$BEST_DS_HF_QUERIES" ]]; then
    IFS=',' read -r -a _queries <<< "$BEST_DS_HF_QUERIES"
    for q in "${_queries[@]}"; do
      q="$(echo "$q" | xargs)"
      [[ -z "$q" ]] && continue
      dataset_cmd="$dataset_cmd --hf-query \"$q\""
    done
  fi

  if [[ -n "$BEST_DS_SOURCES_FILE" ]]; then
    dataset_cmd="$dataset_cmd --sources-file \"$BEST_DS_SOURCES_FILE\""
  fi

  if [[ -n "$BEST_DS_EXTRA_SOURCES" ]]; then
    IFS=',' read -r -a _extra_sources <<< "$BEST_DS_EXTRA_SOURCES"
    for src in "${_extra_sources[@]}"; do
      src="$(echo "$src" | xargs)"
      [[ -z "$src" ]] && continue
      dataset_cmd="$dataset_cmd --extra-source \"$src\""
    done
  fi

  if [[ -n "$BEST_DS_LOCAL_SOURCES" ]]; then
    IFS=',' read -r -a _local_sources <<< "$BEST_DS_LOCAL_SOURCES"
    for src in "${_local_sources[@]}"; do
      src="$(echo "$src" | xargs)"
      [[ -z "$src" ]] && continue
      dataset_cmd="$dataset_cmd --local-source \"$src\""
    done
  fi

  run_cmd "$dataset_cmd"
fi

if [[ "$RUN_VIDEO_DATA_PULL" == "1" ]]; then
  run_cmd "python scripts/build_video_dataset.py --out \"$VIDEO_OUT\" --train-per-class \"$VIDEO_TRAIN_PER_CLASS\" --val-per-class \"$VIDEO_VAL_PER_CLASS\" --mode \"$VIDEO_MODE\" --cache-dir \"$VIDEO_CACHE_DIR\" --snapshot-max-workers \"$VIDEO_SNAPSHOT_MAX_WORKERS\" --repo-base-pause-ms \"$VIDEO_REPO_BASE_PAUSE_MS\" --repo-jitter-ms \"$VIDEO_REPO_JITTER_MS\" --copy-sleep-ms \"$VIDEO_COPY_SLEEP_MS\" --sleep-ms \"$VIDEO_SLEEP_MS\" --jitter-ms \"$VIDEO_JITTER_MS\" --chunk-pause-ms \"$VIDEO_CHUNK_PAUSE_MS\" --repo-cooldown-ms \"$VIDEO_REPO_COOLDOWN_MS\" --retries \"$VIDEO_RETRIES\""
fi

if [[ "$RUN_VIDEO_TRAIN" == "1" ]]; then
  if [[ "$VIDEO_TRAIN_RESUME" == "1" ]]; then
    run_cmd "aid-video-train --data \"$VIDEO_OUT\" --out \"$VIDEO_ARTIFACTS_OUT\" --epochs \"$VIDEO_TRAIN_EPOCHS\" --batch-size \"$VIDEO_TRAIN_BATCH_SIZE\" --img-size \"$VIDEO_TRAIN_IMG_SIZE\" --frames \"$VIDEO_TRAIN_FRAMES\" --grad-accum \"$VIDEO_TRAIN_GRAD_ACCUM\" --lr \"$VIDEO_TRAIN_LR\" --patience \"$VIDEO_TRAIN_PATIENCE\" --min-delta \"$VIDEO_TRAIN_MIN_DELTA\" --resume \"$VIDEO_ARTIFACTS_OUT/last_video.pt\""
  else
    run_cmd "aid-video-train --data \"$VIDEO_OUT\" --out \"$VIDEO_ARTIFACTS_OUT\" --epochs \"$VIDEO_TRAIN_EPOCHS\" --batch-size \"$VIDEO_TRAIN_BATCH_SIZE\" --img-size \"$VIDEO_TRAIN_IMG_SIZE\" --frames \"$VIDEO_TRAIN_FRAMES\" --grad-accum \"$VIDEO_TRAIN_GRAD_ACCUM\" --lr \"$VIDEO_TRAIN_LR\" --patience \"$VIDEO_TRAIN_PATIENCE\" --min-delta \"$VIDEO_TRAIN_MIN_DELTA\""
  fi
fi

if [[ "$SKIP_SWEEP" != "1" ]]; then
  run_cmd "EPOCHS=\"$SWEEP_EPOCHS\" bash scripts/hparam_sweep.sh \"$DATA_DIR\" \"$SWEEP_OUT\""
fi

run_cmd "bash scripts/train_ensemble.sh \"$DATA_DIR\" \"$ENS_OUT\" \"$EPOCHS\""

if [[ "$RUN_ENSEMBLE_FIT" == "1" ]]; then
  run_cmd "python scripts/fit_ensemble.py --data \"$DATA_DIR\" --model \"$ENS_OUT\"/m1/best.pt \"$ENS_OUT\"/m2/best.pt \"$ENS_OUT\"/m3/best.pt \"$ENS_OUT\"/m4/best.pt --out \"$ENS_CONFIG_PATH\" --steps \"$ENS_FIT_STEPS\" --lr \"$ENS_FIT_LR\" --l2 \"$ENS_FIT_L2\" --max-val-images \"$ENS_FIT_MAX_VAL_IMAGES\" --objective balanced"
fi

if [[ "$RUN_HARD_MINING" == "1" ]]; then
  hard_cfg=""
  if [[ -f "$ENS_CONFIG_PATH" ]]; then
    hard_cfg="--ensemble-config \"$ENS_CONFIG_PATH\""
  fi
  run_cmd "python scripts/mine_hard_negatives.py --data \"$DATA_DIR\" --model \"$ENS_OUT\"/m1/best.pt \"$ENS_OUT\"/m2/best.pt \"$ENS_OUT\"/m3/best.pt \"$ENS_OUT\"/m4/best.pt $hard_cfg --out \"$ENS_OUT\"/hard_mined --top-k \"$HARD_MINING_TOPK\""
fi

eval_cfg=""
if [[ -f "$ENS_CONFIG_PATH" ]]; then
  eval_cfg="--ensemble-config \"$ENS_CONFIG_PATH\""
fi
run_cmd "python scripts/eval_test_ensemble.py --data \"$DATA_DIR\" --model \"$ENS_OUT\"/m1/best.pt \"$ENS_OUT\"/m2/best.pt \"$ENS_OUT\"/m3/best.pt \"$ENS_OUT\"/m4/best.pt $eval_cfg --out \"$ENS_OUT\"/test_metrics.json"

if [[ "$RUN_DISTILL" == "1" ]]; then
  distill_cfg=""
  if [[ -f "$ENS_CONFIG_PATH" ]]; then
    distill_cfg="--ensemble-config \"$ENS_CONFIG_PATH\""
  fi
  run_cmd "python scripts/train_distill.py --data \"$DATA_DIR\" --teacher \"$ENS_OUT\"/m1/best.pt \"$ENS_OUT\"/m2/best.pt \"$ENS_OUT\"/m3/best.pt \"$ENS_OUT\"/m4/best.pt $distill_cfg --out \"$ENS_OUT\"/distill --student-backbone effb0 --img-size 320 --batch-size 64 --epochs \"$DISTILL_EPOCHS\""
fi

if [[ "$DRY_RUN" != "1" ]]; then
python - <<'PY'
import json
import os
from pathlib import Path

ens = Path(os.environ.get("ENS_OUT", "./artifacts_ens"))
manifest = {
    "models": [
        str((ens / "m1" / "best.pt").resolve()),
        str((ens / "m2" / "best.pt").resolve()),
        str((ens / "m3" / "best.pt").resolve()),
        str((ens / "m4" / "best.pt").resolve()),
    ],
    "test_metrics": str((ens / "test_metrics.json").resolve()),
}
ens_cfg = Path(os.environ.get("ENS_CONFIG_PATH", str(ens / "ensemble_config.json")))
if ens_cfg.exists():
    manifest["ensemble_config"] = str(ens_cfg.resolve())
domain_cfg = ens / "domain_config.json"
if domain_cfg.exists():
    manifest["domain_config"] = str(domain_cfg.resolve())
video_best = Path(os.environ.get("VIDEO_ARTIFACTS_OUT", "./video_artifacts")) / "best_video.pt"
if video_best.exists():
    manifest["video_model"] = str(video_best.resolve())
out = ens / "prod_manifest.json"
out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"wrote {out}")
PY
else
  echo "[DRY_RUN] write artifacts_ens/prod_manifest.json"
fi

echo "Pipeline complete."
echo "Prod manifest: $ENS_OUT/prod_manifest.json"
