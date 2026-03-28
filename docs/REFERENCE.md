# Reference

This file keeps the README short and startup-focused while collecting the broader concepts in one place.

## What This Repo Does

- collects image and video data locally
- trains local detectors
- supports resumable setup and pipeline runs
- stays in training-only mode; production serving is intentionally disabled
- targets a simple local CUDA + PyTorch workflow, especially on RTX 4090-class hardware

## Repo Layout

- `local.sh`: small public entrypoint
- `install.sh`: optional installer
- `docker-compose.yml`: optional Compose workflow for isolated CPU or GPU runs
- `Dockerfile`: CPU-oriented container image definition used by Compose
- `Dockerfile.gpu`: CUDA-enabled container image definition used by the GPU Compose service
- `docs/`: user-facing documentation
- `scripts/`: internal pipeline helpers and advanced wrappers
- `src/ai_image_detector/`: Python package code
- `tests/`: regression coverage

## Public command structure

The repo is structured around the public `./local.sh` commands:

- `setup`: creates `./.venv` and verifies the repo
- `collect`: builds datasets in `./data_best` and `./video_data`
- `train`: prepares `./.local/training_data` and trains from existing data
- `retrain`: rerun training on top of the existing dataset and gate the result
- `finetune`: separate metadata-aware finetune on top of an existing checkpoint
- `run`: full collect + train pipeline
- `continuous`: repeat collection and retraining over time
- `status` and `collect-status`: inspect the current local state

Everything under `scripts/` and `src/ai_image_detector/` exists to support those public commands.

## Current pipeline shape

The current pipeline is:

1. setup a local pinned environment in `./.venv`
2. collect and curate image data into `./data_best`
3. collect video data into `./video_data`
4. ingest or preserve incremental image data under `./data_new`
5. prepare additive image training data in `./.local/training_data`
6. train image models, and optionally video models when complete video data exists
7. persist resumable state, collection manifests, and stage markers under `./.local`

This means the repo is no longer just “run one train script on one folder.” It is a local dataset-building and retraining workflow with resumability and incremental refresh support.

## Dataset and artifact basics

Typical image dataset layout:

```text
data/
  train/
    real/
    ai/
  val/
    real/
    ai/
  test/
    real/
    ai/
```

Typical video dataset layout:

```text
video_data/
  train/
    real/
    ai/
  val/
    real/
    ai/
```

Image training writes artifacts such as:
- `best.safetensors`
- `best_checkpoint.txt`
- `last.pt`
- `epoch_XXX.pt`
- `best_metrics.json`
- `test_metrics.json`
- `calibration.json`
- `best_model_summary.json`
- `config.json`
- `training_log.jsonl`

Video training writes artifacts such as:
- `best_video.safetensors`
- `last_video.pt`
- `epoch_video_XXX.pt`

Pipeline-level reports also include:
- `domain_config.json`
- `robust_eval.json`
- `final_run_summary.json`
- `final_thresholds.json`
- `run_manifest.json`
- `prod_manifest.json`
- `release/release_manifest.json`

Canonical release bundle:
- `./artifacts_ens/release/`
  Exported bundle for sharing, with the best checkpoints and the main eval/calibration sidecars in one directory.

## Pipeline tools

The packaged CLI surface is intentionally small:

```bash
aid-train
aid-video-train
```

Those commands exist to support the local pipeline scripts, not to turn this repo into a broad general-purpose app surface.
They are lightweight wrappers now: a base `pip install -e .` can expose them without pulling in the full training stack, and they will print a clear missing-extra hint if you run them without the required dependency profile.

## Dependency profiles

The package is split into capability extras:

- base install: `pip install -e .`
- full repo workflow: `pip install -e '.[pipeline]'`
- image training only: `pip install -e '.[training]'`
- Hugging Face collection only: `pip install -e '.[collection]'`
- video only: `pip install -e '.[video]'`

Normal repo usage should still go through `./local.sh deps` or `./local.sh setup`, which install the full `pipeline` profile into `./.venv`.

## Containerized path

For a more isolated runtime, the repo also includes:

```bash
docker compose run --rm pipeline ./local.sh doctor
docker compose run --rm pipeline-gpu ./local.sh doctor
docker compose run --rm pipeline-gpu ./local.sh run
```

The Compose services:
- bind-mount the repo at `/workspace`
- auto-read `HF_TOKEN` and `HUGGINGFACE_HUB_TOKEN` from the repo `.env`
- keep Hugging Face and pip caches under `./.local` and in named Docker volumes
- drop Linux capabilities and enable `no-new-privileges`
- keep the repo checkout writable and use `tmpfs` scratch space
- apply a PID limit to reduce blast radius if a process misbehaves

GPU mode requires Docker Engine, the Docker Compose plugin, and the NVIDIA Container Toolkit on the host.
This repo does not add a VM layer because that would change the normal host-GPU and local bind-mount workflow rather than simply harden the existing pipeline.

## Pipeline entrypoints

Normal users should start with:

```bash
./local.sh setup
./local.sh smoke
./local.sh run
```

For stage-by-stage Linux usage:

```bash
./local.sh collect
./local.sh collect-status
./local.sh train
./local.sh retrain
./local.sh continuous
```

For command-level control, use:

```bash
bash scripts/do.sh pipeline
bash scripts/do.sh train-existing
```

For deeper command coverage, see [COMMANDS.md](COMMANDS.md).

## Performance-oriented paths

Quality-first 4090 pipeline:

```bash
bash scripts/max_quality_4090.sh
```

Full 4090-oriented pipeline:

```bash
bash scripts/full_pipeline_4090.sh
```

Example override:

```bash
DATA_DIR=./data_best EPOCHS=14 SKIP_SWEEP=1 bash scripts/full_pipeline_4090.sh
```

## Related docs

- [STARTUP.md](STARTUP.md)
- [COMMANDS.md](COMMANDS.md)
- [../CONTRIBUTING.md](../CONTRIBUTING.md)
- [../SECURITY.md](../SECURITY.md)
