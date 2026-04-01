# Command Guide

This file lists **`./local.sh`** and related command surfaces. For VM boundaries, host requirements, native `apt-get` bootstrap, and the full secure clone-to-run script, use [STARTUP.md](STARTUP.md). On macOS or Windows, use the platform sections there instead of assuming Linux-native steps here.

The recommended path is a dedicated Linux VM, then Docker Compose. The main venv for that path is the isolated container virtualenv at `/opt/aid-venv`. Native fallback uses `./.venv` from `./local.sh setup` (also installs `huggingface_hub`, the `hf` CLI, and repo CLIs). Snippets use Linux `bash` syntax unless noted.

## Dedicated Linux VM + Docker Compose commands

From the repo root (the directory with `docker-compose.yml`, `local.sh`, and `scripts/install_deps.sh`):

```bash
git clone https://github.com/Legendarylibrorg/ai-image-video-detector.git
cd ai-image-video-detector
test -f docker-compose.yml
test -f Dockerfile
test -f Dockerfile.gpu
test -f local.sh
./local.sh docker-doctor
docker compose build
docker compose run --rm pipeline ./local.sh deps
docker compose run --rm pipeline ./local.sh doctor
printf "HF_TOKEN='your_token_here'\n" >> .env
test -f .env
docker compose run --rm pipeline-gpu ./local.sh doctor
docker compose run --rm pipeline-gpu ./local.sh smoke
docker compose run --rm pipeline-gpu ./local.sh run
docker compose run --rm pipeline-gpu ./local.sh status
```

- `pipeline`
  CPU-oriented Compose service.
- `pipeline-gpu`
  GPU-enabled Compose service for CUDA hosts.
- `./local.sh docker-doctor`
  Verifies Docker, Compose, and the repo Docker files before you try the container workflow.
- repo root in container
  `/workspace`
- container venv
  `/opt/aid-venv`
- Hugging Face cache in container
  `/workspace/.local/hf`
- general source tree in container
  writable

For the full secure startup walkthrough, use [STARTUP.md](STARTUP.md). GPU hosts need the NVIDIA Container Toolkit for `pipeline-gpu`.

## Python dependencies

Required packages are listed in `pyproject.toml` under `[project] dependencies` (training, collection, and video stacks together). Install with:

```bash
pip install -e .
```

For the native fallback workflow, use `./local.sh deps` or `./local.sh setup`; both install that set into `./.venv`.
The packaged `aid-*` commands are thin wrappers; if a dependency is missing, they suggest `pip install -e .`.

## Pipeline at a glance

The normal local workflow is the basic Linux path above.

Public command-to-path map:

- `./local.sh collect`
  Fills `./data_best`, `./video_data`, and `./.local`.
- `./local.sh train`
  Builds `./.local/training_data` from `./data_best` and `./data_new`, then trains.
- `./local.sh retrain` and `./local.sh finetune`
  Reuse the existing-data training path and gate the resulting artifacts.
- `./local.sh run`
  Does both collection and training, then writes reports to `./.local/reports` and artifacts to `./artifacts_ens` and `./video_artifacts`.
- `./local.sh continuous`
  Repeats collection and retraining over time on the same repo-local paths.

What each stage does:

- `setup`
  Creates or reuses `./.venv`, installs pinned Python deps, prepares local cache dirs, and runs a health check.
- `run`
  Executes the normal collect-plus-train pipeline.
- `collect`
  Executes the collection-only pipeline without training.
- `train`
  Trains from data already collected on disk.
- `retrain`
  Runs the retrain wrapper on top of existing collected data.
- `finetune`
  Runs the separate metadata-aware finetune wrapper on top of an existing checkpoint.
- `continuous`
  Runs the continuous collection and retraining loop.
- `status`
  Shows the current pipeline state, key artifact paths, and training lock status.
- `smoke`
  Tiny local end-to-end pipeline check before the full run.

## `./local.sh` commands

Setup details live in [STARTUP.md](STARTUP.md). This section assumes the repo is already cloned and you are already in the repo root.

Optional validation:

```bash
./local.sh smoke
./local.sh smoke-real
```

Main surface:

- `./local.sh setup`
  Bootstrap the local environment only.
- `./local.sh deps`
  Install the pinned Python dependencies into `./.venv` without running the full setup wrapper.
  This also installs the repo CLI commands and the `hf` CLI in that venv.
  This installs the full dependency set from `pyproject.toml`.
- `./local.sh doctor`
  Run the health check directly.
- `./local.sh run`
  Run the full Hugging Face collection and training pipeline.
- `./local.sh collect`
  Run the Hugging Face collection pipeline only.
- `./local.sh smoke`
  Run a tiny local end-to-end pipeline check.
- `./local.sh smoke-real`
  Run a tiny real Hugging Face collection plus real CUDA training smoke. Requires `HF_TOKEN` and a CUDA GPU.
- `./local.sh status`
  Show training lock and key data and artifact paths.
- `./local.sh collect-status`
  Show the current collection/build state and the recommended next command.
- `./local.sh train`
  Train from data that is already collected on disk without starting a new collection pass.
- `./local.sh retrain`
  Retrain on top of the existing collected dataset and run the benchmark gate.
- `./local.sh finetune`
  Run the metadata-aware finetune path and write outputs to `./artifacts_finetune_metadata`.
- `./local.sh continuous`
  Run the continuous collection and retraining loop for a long-lived machine.

Everything else in the repo is internal support for the pipeline and is intentionally not part of the normal command surface.

If you need lower-level scripts or environment controls, use [docs/REFERENCE.md](docs/REFERENCE.md).

## Sudo guidance

Use `sudo` only for Linux package-manager commands such as:

```bash
sudo apt-get update
sudo apt-get install -y curl ca-certificates git python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
```

Do not add `sudo` to the repo commands in this file. Run them as your normal user so `.venv`, `.local`, datasets, and artifacts stay writable without ownership issues.
