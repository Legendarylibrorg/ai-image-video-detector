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

## Public commands

Use [COMMANDS.md](COMMANDS.md) for the `./local.sh` command map and stage descriptions. Everything under `scripts/` and `src/ai_image_detector/` exists to support that surface.

## Current pipeline shape

The current pipeline is:

1. preferred path: run inside a dedicated Linux VM with Docker Compose and the isolated container venv at `/opt/aid-venv`
2. native fallback: `./local.sh setup` creates or reuses `./.venv`
3. collect and curate image data into `./data_best`
4. collect video data into `./video_data`
5. ingest or preserve incremental image data under `./data_new`
6. prepare additive image training data in `./.local/training_data`
7. train image models, and optionally video models when complete video data exists
8. persist resumable state, collection manifests, and stage markers under `./.local`

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
They are thin wrappers around the Python modules in this package. After `pip install -e .`, the declared dependencies in `pyproject.toml` should satisfy imports; if not, the CLI prints `run=pip install -e .` on stderr.

## Python dependencies

Everything needed for the default training and collection workflow is listed under `[project] dependencies` in `pyproject.toml`. Install with:

```bash
pip install -e .
```

Normal native fallback usage should still go through `./local.sh deps` or `./local.sh setup`, which install that set into `./.venv`.

## Containerized path

For the preferred more isolated runtime, the repo includes:

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

GPU mode requires Docker Engine, the Docker Compose plugin, and the NVIDIA Container Toolkit inside the dedicated Linux VM.
The intended secure model is: host -> dedicated Linux VM -> Docker Engine -> Compose containers.

## Pipeline entrypoints

Normal users should start with the Linux VM + Docker Compose path:

```bash
docker compose run --rm pipeline ./local.sh deps
docker compose run --rm pipeline-gpu ./local.sh smoke
docker compose run --rm pipeline-gpu ./local.sh run
```

For native fallback Linux usage:

```bash
./local.sh setup
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

There is a single full pipeline script: `scripts/full_pipeline_4090.sh`.

- Default (`PIPELINE_PROFILE` unset or `standard`): lighter defaults for direct runs and custom overrides.
- Quality-first (`PIPELINE_PROFILE=max_quality`): the profile used by `./local.sh run` and the training helpers in `scripts/lib/training.sh`.

```bash
PIPELINE_PROFILE=max_quality bash scripts/full_pipeline_4090.sh
```

Example override on the standard profile:

```bash
DATA_DIR=./data_best EPOCHS=14 SKIP_SWEEP=1 bash scripts/full_pipeline_4090.sh
```

## Related docs

- [STARTUP.md](STARTUP.md)
- [COMMANDS.md](COMMANDS.md)
- [../CONTRIBUTING.md](../CONTRIBUTING.md)
- [../SECURITY.md](../SECURITY.md)
