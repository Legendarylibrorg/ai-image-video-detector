# Command Guide

This guide collects the repo command surfaces in one place.
The repo-local Python environment is `./.venv`, created or reused by `./local.sh setup`.
That setup also installs `huggingface_hub`, the `hf` CLI, and the repo CLI commands into the same venv.
The command blocks at the top of this file are Linux commands. If you are on macOS or Windows, use the platform notes in [STARTUP.md](STARTUP.md) instead of copying the `apt-get` steps directly.

Clone path:

```bash
sudo apt-get update
sudo apt-get install -y curl ca-certificates git unzip python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
git clone https://github.com/Legendarylibrorg/ai-image-video-detector.git
cd ai-image-video-detector
python3 -m venv .venv
source .venv/bin/activate
./local.sh deps
./local.sh doctor
printf "HF_TOKEN='your_token_here'\n" >> .env
./local.sh smoke
./local.sh run
./local.sh status
```

ZIP path:

```bash
sudo apt-get update
sudo apt-get install -y curl ca-certificates git unzip python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
unzip ai-image-video-detector-main.zip
cd ai-image-video-detector-main
python3 -m venv .venv
source .venv/bin/activate
./local.sh deps
./local.sh doctor
printf "HF_TOKEN='your_token_here'\n" >> .env
./local.sh smoke
./local.sh run
./local.sh status
```

Already inside the repo root:

```bash
sudo apt-get update
sudo apt-get install -y curl ca-certificates git unzip python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
python3 -m venv .venv
source .venv/bin/activate
./local.sh deps
./local.sh doctor
printf "HF_TOKEN='your_token_here'\n" >> .env
./local.sh smoke
./local.sh run
./local.sh status
```

Run `bash ./install.sh` only from inside the repo root after cloning or unzipping.
If you unzip the repo first, `bash ./install.sh` reuses that extracted folder and does not create a nested repo inside it.
If you want the installer to fetch the repo for you, use the curl installer instead.

Shortcut installers:

```bash
curl -fsSL https://raw.githubusercontent.com/Legendarylibrorg/ai-image-video-detector/main/install.sh | bash
```

```bash
./local.sh setup
```

## Docker Compose commands

The repo also includes an optional Compose path for a more isolated runtime:

```bash
docker compose run --rm pipeline ./local.sh doctor
docker compose run --rm pipeline-gpu ./local.sh doctor
docker compose run --rm pipeline-gpu ./local.sh run
```

- `pipeline`
  CPU-oriented Compose service.
- `pipeline-gpu`
  GPU-enabled Compose service for CUDA hosts.

The Compose services mount this repo at `/workspace`, reuse the repo-local `.env`, and keep Hugging Face and pip caches in named volumes.
They also drop all Linux capabilities, enable `no-new-privileges`, use a read-only container root filesystem, and keep scratch space in `tmpfs`.
This repo does not add a VM layer because that would change the normal GPU and bind-mount workflow rather than harden it transparently.

## Dependency profiles

The package keeps the base install lightweight:

- `pip install -e .`
  Base package only.
- `pip install -e '.[pipeline]'`
  Full repo workflow dependencies.
- `pip install -e '.[training]'`
  Image-training dependencies.
- `pip install -e '.[collection]'`
  Hugging Face collection dependencies.
- `pip install -e '.[video]'`
  Video dependencies.

For the normal repo workflow, use `./local.sh deps` or `./local.sh setup`; both install the full `pipeline` profile into `./.venv`.
The packaged `aid-*` commands remain available, but they are lightweight wrappers and will print a missing-extra hint if you try to run them from a base no-deps install.

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

Recommended first: use the same basic Linux path above.

Manual fallback:

```bash
sudo apt-get update
sudo apt-get install -y curl ca-certificates git unzip python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
python3 -m venv .venv
source .venv/bin/activate
./local.sh deps
./local.sh doctor
./local.sh smoke
./local.sh run
./local.sh status
```

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
  This is the full `pipeline` profile, not the lightweight base package.
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
sudo apt-get install -y curl ca-certificates git unzip python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
```

Do not add `sudo` to the repo commands in this file. Run them as your normal user so `.venv`, `.local`, datasets, and artifacts stay writable without ownership issues.
