# Local AI Image And Video Training Pipeline

This repository is for one job:
- collect Hugging Face image and video data locally
- train detectors locally
- rerun safely if a long setup stops partway through

It is not a production serving repo in the current mode.

## Open Source Notes

- License: MIT (see `LICENSE`).
- Security reporting: see `SECURITY.md`.
- Do not commit secrets (tokens, keys) or private datasets.
- Dataset/model licenses vary by source; verify each source license before commercial or production use.
- Detection outputs are probabilistic and can be wrong; review high-risk decisions with human oversight.

## Setup And Run

This repository is in pipeline-only mode:
- It collects data from Hugging Face and trains models locally.
- It does not run production serving in the current setup.
- The intended host is Linux.

If you only want the normal workflow, follow the steps below in order.

### Before you start

You need:
- Linux
- Python 3
- enough disk space for datasets and artifacts
- a Hugging Face token

Normal first run:

```bash
cd /path/to/image-spam
cp .env.example .env
./local.sh doctor
./local.sh setup
./local.sh status
```

If `doctor` says `HF_TOKEN` is missing, add it to `.env` and run `doctor` again before `setup`.

### Step 1: Open the project on Linux

```bash
cd /path/to/image-spam
```

### Step 2: Create your env file

```bash
cp .env.example .env
```

What `.env` is for:
- `HF_TOKEN` for reliable Hugging Face collection
- cache paths and safe rate-limit defaults
- diversity and quality gates
- setup and doctor defaults

### Step 3: Add your Hugging Face token

Open `.env` and set:

```bash
HF_TOKEN='your_token_here'
```

If you do not have a token yet:
- Sign in to Hugging Face
- Create an access token
- Paste it into `.env`

### Step 4: Run the preflight check

```bash
./local.sh doctor
```

What `doctor` checks:
- free disk space
- GPU visibility
- cache directories
- Python environment and core packages
- whether `HF_TOKEN` is present

If `doctor` shows a failure, fix that first before running the full pipeline.

Common fixes:
- missing `HF_TOKEN`: add it to `.env`
- missing Python or venv support: install your distro Python packages first
- GPU not visible: training can still run, but likely slower
- low disk space: free space before collection starts

### Step 5: Run the full setup

```bash
./local.sh setup
```

What `setup` does:
- installs Linux packages when `apt-get` is available
- creates or reuses `.venv`
- installs pinned Python dependencies
- validates or prompts for `HF_TOKEN`
- runs the full collection + training pipeline
- retries automatically if a stage fails
- writes resumable setup markers in `./.local/stages/*.done`

Important:
- If setup stops, you can run `./local.sh setup` again.
- Completed setup stages are skipped automatically.
- To force every setup stage to rerun, use `SETUP_FORCE_STAGES=1 ./local.sh setup`.

What to expect:
- first runs can take a while because they install deps, discover sources, collect data, and train
- reruns are usually much faster because the venv, cache, and completed stages are reused
- collection is Hugging Face-only and uses cache-first behavior to reduce rate-limit pressure

### Step 6: Check the current state

```bash
./local.sh status
```

This shows:
- whether training is active
- where image data is stored
- where video data is stored
- where trained artifacts are written

### Step 7: Run only the part you need later

Use these after the initial setup:

- `./local.sh collect`
  Runs collection only.

- `./local.sh collect-fast`
  Runs a much smaller collection job for a quick sanity check.

- `./local.sh train`
  Trains using the current collected data.

- `./local.sh start`
  Runs the best-quality pipeline path.

- `./local.sh deps-update`
  Refreshes the locked dependency set.

If you are unsure which command to use:
- first time: `./local.sh setup`
- check health: `./local.sh doctor`
- collect more data only: `./local.sh collect`
- quick pipeline check: `./local.sh collect-fast`
- retrain on existing data: `./local.sh train`

### Minimal command reference

```bash
./local.sh doctor
./local.sh setup
./local.sh status
./local.sh collect
./local.sh collect-fast
./local.sh train
./local.sh start
```

### Manual install path

Only use this if you do not want `./local.sh setup`:

```bash
python3 -m venv .venv
source .venv/bin/activate
bash scripts/install_deps.sh
./local.sh doctor
bash scripts/do.sh train-all-types
```

### Setup options

- `SETUP_MAX_ATTEMPTS`
  Default `4`.

- `SETUP_RETRY_SLEEP_SEC`
  Default `45`.

- `SETUP_FORCE_STAGES`
  Set to `1` to rerun completed setup stages.

- `SETUP_STAGE_DIR`
  Custom stage marker directory. Default `./.local/stages`.

- `HF_SETUP_REQUIRE_TOKEN`
  Set to `0` to allow setup without a token.

- `HF_SETUP_SAVE_ENV`
  Set to `0` to avoid writing the token into `.env`.

Dependency lock workflow:
- Install pinned deps with `bash scripts/install_deps.sh`
- Refresh the lock with `./local.sh deps-update`
- The install script now skips reinstallation when `requirements.lock` and `pyproject.toml` are unchanged

You can ignore the remaining sections unless you want manual control, advanced training options, or lower-level commands.

## Troubleshooting

### Setup stopped

Run:

```bash
./local.sh setup
```

Setup resumes from completed stages automatically.

### Collection seems slow

That is usually normal on first run. The pipeline is intentionally cache-first and rate-limit-aware for Hugging Face.

Try:

```bash
./local.sh status
./local.sh collect-fast
```

### You only want to retrain

Run:

```bash
./local.sh train
```

That uses the current collected dataset and does not pull new data first.

### You changed dependencies

Run:

```bash
./local.sh deps-update
bash scripts/install_deps.sh
```

## Advanced Reference

Everything below this point is optional reference material for manual workflows and lower-level commands.

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
  --domain-config ./artifacts_ens/domain_config.json \
  --tools-config ./artifacts_ens/tools_config.json \
  --tta-views 2 \
  --image ./example.jpg --json
```

Returns:

- `label` (`AI-generated`, `Real`, or `Unknown`)
- `prob_ai`, `threshold`, `unknown_margin`
- `metadata_score`, `metadata_flags`, `metadata_fields`
- `provenance_score`, `provenance_flags`
- `ood_score`, `ood_flags`
- `combined_risk`

## API (Archived; Disabled In Pipeline-Only Mode)

`aid-serve` paths are intentionally disabled in the current pipeline-only setup.

```bash
aid-serve --model ./artifacts/best.pt --host 127.0.0.1 --port 8000
```

Optional hardening knobs:

```bash
aid-serve --model ./artifacts/best.pt --max-bytes 10485760 --rate-limit-per-min 60 --unknown-margin 0.08
```

Weighted ensemble serving:

```bash
aid-serve --model ./artifacts_ens/m1/best.pt ./artifacts_ens/m2/best.pt ./artifacts_ens/m3/best.pt ./artifacts_ens/m4/best.pt \
  --ensemble-config ./artifacts_ens/ensemble_config.json \
  --domain-config ./artifacts_ens/domain_config.json \
  --tools-config ./artifacts_ens/tools_config.json \
  --tta-views 2
```

Endpoints:

- `GET /health`
- `GET /` (API info)
- `POST /detect` (multipart file field `image`)
- `POST /analyze/text` (JSON: `{ "text": "..." }`)
- `POST /analyze/conversation` (JSON: `{ "text": "..." }`)
- `POST /analyze/url` (JSON: `{ "url": "..." }`)
- `POST /analyze/pdf` (multipart file field `file`)
- `POST /analyze/audio` (multipart file field `file`)
- `POST /analyze/multimodal` (JSON: `{ "scores": { "image": 0.7, "text": 0.4, ... } }`)
- Optional: learned multimodal fusion config via `--fusion-config ./artifacts_ens/fusion_config.json`
- Optional active-learning queue capture in API serve:
  - `--uncertain-capture --uncertain-dir ./incoming_review_queue`
- Optional inference tools config:
  - `--tools-config ./artifacts_ens/tools_config.json` (risk/probability bias and rule-engine adjustments)
- Optional test-time augmentation consensus:
  - `--tta-views 2` (or `3` for stronger but slower inference)

Fit learned multimodal fusion weights from labeled outcomes:

```bash
python scripts/fit_multimodal_fusion.py \
  --csv ./artifacts_ens/fusion_train.csv \
  --label-col label \
  --out ./artifacts_ens/fusion_config.json
```

Promotion gate and weekly retrain loop:

```bash
python scripts/benchmark_gate.py --ens-out ./artifacts_ens --video-out ./video_artifacts
bash scripts/weekly_retrain_v3.sh
```

`weekly_retrain_v3.sh` now stays fully inside the pipeline-only flow and no longer restarts any serving process.

Reviewed uncertainty queue ingestion:

```bash
python scripts/review_queue_to_dataset.py --queue ./incoming_review_queue --dst ./data_new/train
```

Compliance/privacy mode (recommended baseline):
- `UNCERTAIN_CAPTURE=0`
- `IP_LOG_MODE=masked` (default; no raw IP logging)
- Optional stricter: `IP_LOG_MODE=none`
- Run periodic retention cleanup:

```bash
bash scripts/privacy_cleanup.sh
```

Useful retention envs:
- `QUEUE_RETENTION_DAYS` (default `7`)
- `LOG_RETENTION_DAYS` (default `14`)
- `MODEL_OUTPUT_RETENTION_DAYS` (default `30`)

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
bash scripts/do.sh start-v2       # max-accuracy v2 (domain calibration + refinement loops)
bash scripts/do.sh collect        # full collection cycle (image + ingest + video)
bash scripts/do.sh collect-diverse # super-diverse collection preset (recommended for robustness)
bash scripts/do.sh collect-fast   # quick small collection sanity pass
bash scripts/do.sh collect-image  # image dataset only
bash scripts/do.sh collect-video  # video dataset only
bash scripts/do.sh ingest         # ingest incoming model outputs only
bash scripts/do.sh scan           # malware scan now
bash scripts/do.sh doctor         # preflight checks before long runs
bash scripts/do.sh train          # image training pipeline only
bash scripts/do.sh train-all      # image + video training (no new data pull)
bash scripts/do.sh train-all-types # collect-diverse + full image/video training + artifact validation
bash scripts/do.sh autocollect    # continuous collection loop
bash scripts/do.sh detect ./img.jpg
bash scripts/do.sh status         # lock + artifact paths
```

Collection defaults are Hugging Face-only and diverse-first, and dataset build is fail-fast when targets or source-diversity gates are not met.
Run `bash scripts/do.sh doctor` before long jobs to verify token, disk, GPU visibility, cache paths, and Python deps.

Linux shortcut launchers:

```bash
./start.sh                        # same as: bash scripts/do.sh start
./collect.sh                      # same as: bash scripts/do.sh collect
./train.sh                        # same as: bash scripts/do.sh train (image pipeline)
./autocollect.sh                  # same as: bash scripts/do.sh autocollect
```

Training/collection safety:
- collection auto-skips while a training lock is active
- continuous collection waits and retries after training completes
- fresh model outputs are ingested from `./incoming_model_outputs/{ai,real}` into `./data_new/train/{ai,real}`
- malware scanning is automatic during collection/ingest (set `MALWARE_SCAN=0` to disable)

Diverse dataset preset:
- `bash scripts/do.sh collect-diverse` expands toward memes/captions, screenshots/UI, posters/infographics, scanned docs/receipts/IDs, anime/illustration/3D/CGI, watermark/recompression styles, and newer generator outputs.
- It now also targets: camera photos (phone/DSLR/webcam/CCTV), browser/dashboard screenshots, edited/low-quality variants, portrait/group face images, and game-render/CGI hard negatives.
- Override scale with `DIVERSE_TRAIN_PER_CLASS`, `DIVERSE_VAL_PER_CLASS`, `DIVERSE_TEST_PER_CLASS`.
- Override style query mix with `DIVERSE_HF_QUERIES` (comma-separated).
- Each run now executes `scripts/audit_diversity.py` to check unique source spread, hard-negative mode variety, and class balance.
- Audit knobs: `DIVERSE_MIN_UNIQUE_SOURCES`, `DIVERSE_MIN_HARDNEG_MODES`, `DIVERSE_MAX_CLASS_IMBALANCE`.
- HF source diversity gates are enforced by default:
  - `DIVERSE_MIN_HF_SOURCES_WITH_ACCEPTED` (default `24`)
  - `DIVERSE_MIN_HF_SOURCES_PER_CLASS` (default `14`)
- Standard collect image gate defaults:
  - `BEST_DS_MIN_HF_SOURCES_WITH_ACCEPTED` (default `16`)
  - `BEST_DS_MIN_HF_SOURCES_PER_CLASS` (default `10`)
- HF cache-first behavior is enabled by default (`--hf-cache-only-if-present`) and uses local cache dirs (`./.local/hf`) to reduce repeated Hub API/resolver calls.
- `collect-diverse` runs bounded discovery first (`--discover-only`), then builds from cache when available, otherwise falls back to live HF discovery (still HF-only).
- Discovery timeout is controlled by `DIVERSE_DISCOVERY_TIMEOUT_SEC` (default `900`).
- Set `DIVERSE_SKIP_DISCOVERY=1` to skip live Hub discovery entirely and use only cached HF source ids.

Fast sanity preset:
- `bash scripts/do.sh collect-fast` keeps the same HF-only quality gates but uses much smaller sample targets for quick validation runs.

HF 5-minute limit fast-safe profile:
- Keep `VIDEO_MODE=snapshot` (default) and `VIDEO_SNAPSHOT_MAX_WORKERS=1`.
- Reuse discovery cache files (`./.local/hf_discovered_sources.txt`, `./.local/hf_diverse_sources.txt`) across runs.
- Suggested run:

```bash
HF_TOKEN=your_token \
DIVERSE_REPO_BASE_PAUSE_MS=1400 \
DIVERSE_REPO_JITTER_MS=1200 \
DIVERSE_REPO_COOLDOWN_MS=45000 \
VIDEO_SNAPSHOT_MAX_WORKERS=1 \
bash scripts/do.sh collect-diverse
```

All-data-type training run:
- `bash scripts/do.sh train-all-types` runs broad collection and trains all trainable modalities in this repo (image + video), then verifies required artifacts exist.
- Non-trainable analyzers (text/url/pdf/audio/conversation heuristics) are deterministic modules and do not require model training.

Max-accuracy v2 run:
- `bash scripts/do.sh start-v2` runs collect-diverse -> train-all -> domain threshold fitting -> hard-negative refresh loops.
- Tune with `REFINE_LOOPS` and `HARD_TOPK` (used by `scripts/max_accuracy_v2.sh`).

Run everything end-to-end:

```bash
bash scripts/full_pipeline_4090.sh
```

True one-command bootstrap (installs deps, trains optimized pipeline):

```bash
bash scripts/one_command_4090.sh
```

This command installs all required system and Python dependencies (including Hugging Face dataset tooling and safetensors support) before training starts.

Legacy service/supervisor scripts are not part of pipeline-only mode.

Maximum-quality profile (slowest, strongest defaults, includes video training):

```bash
HF_TOKEN=your_token bash scripts/max_quality_4090.sh
```

Key quality knobs: `EPOCHS`, `SWEEP_EPOCHS`, `DISTILL_EPOCHS`, `HARD_MINING_TOPK`, `VIDEO_TRAIN_EPOCHS`, `BEST_DS_DISCOVER_HF`, `BEST_DS_HF_MAX_SOURCES`, `BEST_DS_REPO_BASE_PAUSE_MS`, `BEST_DS_REPO_COOLDOWN_MS`.

Auto-serve flags are ignored in pipeline-only mode.

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

Production serving is intentionally disabled in this repository's current pipeline-only mode.
