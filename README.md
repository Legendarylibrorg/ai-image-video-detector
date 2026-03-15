# Extremely Advanced AI Image Detector

A production-oriented, forensics-first detector with:

- Multi-branch model (RGB + FFT + noise residual)
- Calibration (temperature scaling), threshold tuning, and full eval metrics
- OOD rejection (`Unknown` mode)
- Metadata anomaly and provenance signal scoring
- Ensemble inference (multiple checkpoints)
- Explainability heatmaps + patch-level ranking
- Robustness benchmarking under perturbations
- Video scoring from sampled frames
- API hardening (type/size limits, rate limiting, structured logs)
- Dataset hygiene tooling (manifest, dedupe, near-dupes, balance report)

## Quick start

Linux (Ubuntu/Debian) prerequisites:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip build-essential
```

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Dataset format

```text
data/
  train/
    real/
    ai/
  val/
    real/
    ai/
```

Tip: include `source=` and `camera=` in filenames for grouped metrics, e.g.
`source=midjourney__camera=none__img123.jpg`.

## Train (calibrated)

```bash
aid-train --data ./data --epochs 12 --img-size 256 --out ./artifacts
```

Outputs include:

- `best.pt` (model + threshold + temperature + model_id)
- `last.pt` (full training state for resume: model/EMA/optimizer/scheduler/scaler)
- `epoch_XXX.pt` (periodic full training snapshots)
- `best_metrics.json` (AUC/F1/FPR/TPR/ECE/Brier)
- `best_group_metrics.json` (source/camera grouped results)
- `calibration.json`
- `last_metrics.json` (latest validation metrics)
- `test_metrics.json` (written when `data/test/{ai,real}` exists)

Resume or control periodic checkpointing:

```bash
aid-train --data ./data --out ./artifacts --epochs 12 --resume ./artifacts/last.pt --save-every 1
```

Recommended stability flags:

```bash
aid-train --data ./data --out ./artifacts --epochs 30 --patience 5 --min-delta 0.0005 --seed 1337 --deterministic
```

Extra training artifacts:
- `config.json` (CLI args + git commit + dataset counts)
- `training_log.jsonl` (per-epoch append-only logs)
- `latest_checkpoint.txt` (points to most recent checkpoint)
- `releases/<timestamp>/...` and `latest_release.txt` (versioned export bundle)

## Predict (ensemble + unknown mode)

```bash
aid-detect --model ./artifacts/best.pt --image ./example.jpg --json
```

Ensemble with multiple models:

```bash
aid-detect --model ./m1.pt ./m2.pt ./m3.pt --image ./example.jpg --json
```

Use fitted ensemble weights/config when available:

```bash
aid-detect --model ./artifacts_ens/m1/best.pt ./artifacts_ens/m2/best.pt ./artifacts_ens/m3/best.pt ./artifacts_ens/m4/best.pt \
  --ensemble-config ./artifacts_ens/ensemble_config.json \
  --image ./example.jpg --json
```

Returns:

- `label` (`AI-generated`, `Real`, or `Unknown`)
- `prob_ai`, `threshold`, `unknown_margin`
- `metadata_score`, `metadata_flags`, `metadata_fields`
- `provenance_score`, `provenance_flags`
- `ood_score`, `ood_flags`
- `combined_risk`

## API (hardened)

```bash
aid-serve --model ./artifacts/best.pt --host 0.0.0.0 --port 8000
```

Optional hardening knobs:

```bash
aid-serve --model ./artifacts/best.pt --max-bytes 10485760 --rate-limit-per-min 60 --unknown-margin 0.08
```

Weighted ensemble serving:

```bash
aid-serve --model ./artifacts_ens/m1/best.pt ./artifacts_ens/m2/best.pt ./artifacts_ens/m3/best.pt ./artifacts_ens/m4/best.pt \
  --ensemble-config ./artifacts_ens/ensemble_config.json
```

Endpoints:

- `GET /health`
- `GET /` (web UI)
- `POST /detect` (multipart file field `image`)

## Explainability

```bash
aid-explain --model ./artifacts/best.pt --image ./example.jpg --out ./artifacts/explain --grid 8 --top-k 8
```

Creates `heatmap_overlay.jpg` and prints highest-risk patches.

## Robustness evaluation

```bash
aid-robust-eval --data ./data --model ./artifacts/best.pt --out ./artifacts/robust_eval.json
```

Evaluates clean vs perturbations (`jpeg_q60`, `jpeg_q35`, `blur`, `resize_0.6`).

## Video detection

```bash
aid-video-detect --model ./artifacts/best.pt --video ./sample.mp4 --sample-every 10 --max-frames 48
```

Temporal video model (better for deepfake/video-specific artifacts):

Video dataset format:

```text
video_data/
  train/
    real/
    ai/
  val/
    real/
    ai/
```

Train:

```bash
aid-video-train --data ./video_data --out ./video_artifacts --epochs 8 --frames 24 --img-size 224
```

Video training also writes resumable checkpoints:
- `last_video.pt` and `epoch_video_XXX.pt`
- Resume with `--resume ./video_artifacts/last_video.pt`
- Also supports `--patience`, `--min-delta`, `--seed`, `--deterministic`, and release export.

Export best artifacts into a versioned release bundle:

```bash
python scripts/export_best_release.py --out ./artifacts --model best.pt
```

4090-stable video training (VRAM-safe defaults):

```bash
aid-video-train --data ./video_data --out ./video_artifacts --epochs 20 --frames 24 --img-size 224 --batch-size 4 --grad-accum 2
```

Infer:

```bash
aid-video-detect-temporal --model ./video_artifacts/best_video.pt --video ./sample.mp4
```

Chunked/resumable video data pull (HF rate-limit friendly):

```bash
HF_TOKEN=your_token python scripts/build_video_dataset.py --out ./video_data --chunk-size 20 --sleep-ms 120 --retries 5
```

Lowest HF API call mode (default): use snapshot pulls per repo:

```bash
HF_TOKEN=your_token python scripts/build_video_dataset.py --mode snapshot --out ./video_data
```

Extra rate-limit-safe mode (slow, gentle request pattern):

```bash
HF_TOKEN=your_token python scripts/build_video_dataset.py \
  --mode snapshot \
  --snapshot-max-workers 1 \
  --repo-base-pause-ms 2200 \
  --repo-jitter-ms 1800 \
  --copy-sleep-ms 15 \
  --out ./video_data
```

Set `HF_TOKEN` safely (do not commit tokens):

Current terminal only:

```bash
export HF_TOKEN='your_token_here'
```

Persistent on zsh:

```bash
echo "export HF_TOKEN='your_token_here'" >> ~/.zshrc
source ~/.zshrc
```

## Metadata tools

Inspect EXIF metadata:

```bash
aid-metadata inspect --image ./example.jpg
```

Strip metadata:

```bash
aid-metadata strip --input ./example.jpg --output ./example_stripped.jpg
```

Modify metadata:

```bash
aid-metadata modify \
  --input ./example.jpg \
  --output ./example_modified.jpg \
  --software "CustomPipeline 1.0" \
  --artist "Lab Team" \
  --comment "manually updated metadata"
```

## Dataset hygiene tools

Build manifest:

```bash
aid-dataset manifest --data ./data --out ./artifacts/dataset_manifest.csv
```

Find exact duplicates:

```bash
aid-dataset dedupe --data ./data --dry-run
```

Find near-duplicates:

```bash
aid-dataset near-dupes --data ./data --max-images 1500 --max-hamming 6
```

Class balance report:

```bash
aid-dataset balance-report --data ./data
```

## Best-performance workflow (4090)

Build a large multi-split dataset with hard negatives:

```bash
python scripts/build_best_dataset.py --out data_best --train-per-class 30000 --val-per-class 7000 --test-per-class 7000
```

Aggressive Hugging Face expansion mode (tries many HF candidates, skips incompatible datasets automatically):

```bash
HF_TOKEN=your_token python scripts/build_best_dataset.py \
  --out data_best \
  --train-per-class 80000 --val-per-class 20000 --test-per-class 20000 \
  --discover-hf --hf-discovery-limit 90 --hf-max-sources 180 \
  --hf-cache-file ./.local/hf_discovered_sources.txt \
  --streaming --repo-base-pause-ms 1100 --repo-jitter-ms 900 --repo-cooldown-ms 45000 \
  --min-side 224 --max-aspect-ratio 2.5 --min-entropy 3.4 \
  --hardneg-fraction 0.8
```

Run hyperparameter sweep:

```bash
bash scripts/hparam_sweep.sh ./data_best ./artifacts_sweep
```

Train a 4-model ensemble:

```bash
bash scripts/train_ensemble.sh ./data_best ./artifacts_ens 12
```

Evaluate the ensemble on held-out test split:

```bash
python scripts/eval_test_ensemble.py \
  --data ./data_best \
  --model ./artifacts_ens/m1/best.pt ./artifacts_ens/m2/best.pt ./artifacts_ens/m3/best.pt ./artifacts_ens/m4/best.pt \
  --ensemble-config ./artifacts_ens/ensemble_config.json \
  --out ./artifacts_ens/test_metrics.json
```

Fit ensemble weights on validation split (recommended before test/deploy):

```bash
python scripts/fit_ensemble.py \
  --data ./data_best \
  --model ./artifacts_ens/m1/best.pt ./artifacts_ens/m2/best.pt ./artifacts_ens/m3/best.pt ./artifacts_ens/m4/best.pt \
  --out ./artifacts_ens/ensemble_config.json
```


## One-command full pipeline (4090)

Minimal usage (recommended):

```bash
bash scripts/do.sh start          # full best-quality pipeline
bash scripts/do.sh collect        # data only
bash scripts/do.sh train          # training only
bash scripts/do.sh autocollect    # continuous collection loop
bash scripts/do.sh serve          # serve API/UI
bash scripts/do.sh detect ./img.jpg
```

Linux shortcut launchers:

```bash
./start.sh                        # same as: bash scripts/do.sh start
./collect.sh                      # same as: bash scripts/do.sh collect
./train.sh                        # same as: bash scripts/do.sh train
./autocollect.sh                  # same as: bash scripts/do.sh autocollect
```

Training/collection safety:
- collection auto-skips while a training lock is active
- continuous collection waits and retries after training completes
- fresh model outputs are ingested from `./incoming_model_outputs/{ai,real}` into `./data_new/train/{ai,real}`

Run everything end-to-end:

```bash
bash scripts/full_pipeline_4090.sh
```

True one-command bootstrap (installs deps, trains optimized pipeline, optional auto-serve):

```bash
bash scripts/one_command_4090.sh
```

Maximum-quality profile (slowest, strongest defaults, includes video training):

```bash
HF_TOKEN=your_token bash scripts/max_quality_4090.sh
```

Key quality knobs: `EPOCHS`, `SWEEP_EPOCHS`, `DISTILL_EPOCHS`, `HARD_MINING_TOPK`, `VIDEO_TRAIN_EPOCHS`, `BEST_DS_DISCOVER_HF`, `BEST_DS_HF_MAX_SOURCES`, `BEST_DS_REPO_BASE_PAUSE_MS`, `BEST_DS_REPO_COOLDOWN_MS`.

Auto-start API/UI after training:

```bash
AUTO_SERVE=1 bash scripts/one_command_4090.sh
```

By default this now binds to localhost (`127.0.0.1`) for private access.
To expose intentionally, override:

```bash
HOST=0.0.0.0 AUTO_SERVE=1 bash scripts/one_command_4090.sh
```

4090 stability notes:

- AMP is enabled by default in image/video training.
- Ensemble script uses conservative per-model batch sizes and gradient accumulation.
- `PYTORCH_CUDA_ALLOC_CONF` is set in `one_command_4090.sh` to reduce fragmentation spikes.
- TF32 math + cuDNN benchmark + non-blocking GPU transfers are enabled in training for faster 4090 throughput.
- `torch.compile` is enabled by default with safe fallback if unsupported.

Useful environment overrides:

```bash
DATA_DIR=./data_best EPOCHS=14 SKIP_SWEEP=1 bash scripts/full_pipeline_4090.sh
```

This executes:

- dataset build (multi-split + hard negatives)
- video dataset pull (rate-limit-friendly, staggered snapshot mode)
- optional video temporal training (`RUN_VIDEO_TRAIN=1`)
- sweep runs
- 4-model ensemble training
- ensemble weight fitting (`artifacts_ens/ensemble_config.json`)
- held-out test evaluation
- production manifest generation (`artifacts_ens/prod_manifest.json`)
- hard-negative mining (`artifacts_ens/hard_mined`)
- teacher-student distillation (`artifacts_ens/distill/best.pt`)

Advanced toggles:

```bash
RUN_HARD_MINING=1 RUN_DISTILL=1 RUN_VIDEO_TRAIN=1 EPOCHS=20 bash scripts/full_pipeline_4090.sh
```

Disable automatic video pull if needed:

```bash
RUN_VIDEO_DATA_PULL=0 bash scripts/one_command_4090.sh
```

Backbone options in training:

- `tiny` (fast)
- `effb0` (strong default)
- `effb2` (highest capacity)

Example:

```bash
aid-train --data ./data_best --backbone effb0 --loss focal --epochs 18 --batch-size 64 --img-size 320 --out ./artifacts_custom
```

Continual refresh training:

```bash
bash scripts/incremental_refresh.sh
```

## Production hosting on a 4090

### Option A: Native process

```bash
bash scripts/serve_prod_4090.sh
```

### Option B: Docker + GPU

```bash
cd deploy
docker compose -f docker-compose.gpu.yml up -d --build
```

### Option C: systemd service

Use [ai-detector.service](/Users/devcomputer/Downloads/spam%20filter/image%20spam/deploy/ai-detector.service) on your Linux host:

```bash
sudo cp deploy/ai-detector.service /etc/systemd/system/ai-detector.service
sudo systemctl daemon-reload
sudo systemctl enable --now ai-detector
```

The frontend and API will be available on `http://<server-ip>:8000/`.
