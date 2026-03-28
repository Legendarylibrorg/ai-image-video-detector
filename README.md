# Local AI Image And Video Training Pipeline

This repository is for one job:
- collect Hugging Face image and video data locally
- train detectors locally
- rerun safely if a long setup stops partway through

The recommended path is a dedicated Linux VM first, then Docker Compose inside that VM.
The repo also supports a native local Linux machine with CUDA/PyTorch, such as an RTX 4090 box.
The native path uses a local virtualenv at `./.venv`; `./local.sh setup` creates or reuses it and the pipeline runs from there.
Unless a section says otherwise, the shell snippets in this README use Linux `bash` command syntax.
If you are on macOS or Windows, treat the Linux-native commands below as Linux-only and use the platform notes in [docs/STARTUP.md](docs/STARTUP.md) instead.

It is not a production serving repo in the current mode.

## Secure Linux VM + Docker Compose

Use this as the default runtime path when possible.

Important boundary:
- Docker Compose is not a real VM.
- The secure model here is: host -> dedicated Linux VM -> Docker Engine -> Compose containers.
- If you want a true VM boundary on Linux, create the VM first and run Docker only inside that VM.

Minimum needed inside the dedicated Linux VM:
- `git`
- Docker Engine
- Docker Compose plugin
- NVIDIA Container Toolkit for `pipeline-gpu`

You do not need host Python, `pip`, or a host `./.venv` for this secure path.
All commands in this section must be run from the repo root, the directory that contains `docker-compose.yml`, `local.sh`, and `scripts/install_deps.sh`.

Repo-root check:

```bash
pwd
test -f docker-compose.yml
test -f Dockerfile
test -f Dockerfile.gpu
test -f local.sh
test -f scripts/install_deps.sh
./local.sh docker-doctor
```

Exact secure setup:

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

Secure path map:
- host repo root: your current working directory
- container repo root: `/workspace`
- container virtualenv: `/opt/aid-venv`
- repo env file: `./.env` on the host, auto-read by Docker Compose for `HF_TOKEN` and `HUGGINGFACE_HUB_TOKEN`
- general source tree under `/workspace`: writable
- writable data and artifact paths: `./.local`, `./data_best`, `./data_new`, `./video_data`, `./artifacts_ens`, `./artifacts_sweep`, `./artifacts_finetune_metadata`, `./video_artifacts`, `./incoming_model_outputs`, `./incoming_review_queue`

Notes:
- `pipeline` uses the CPU-only `Dockerfile`; `pipeline-gpu` uses `Dockerfile.gpu`
- dependency install happens inside the isolated container venv at `/opt/aid-venv`
- the container keeps Hugging Face and pip caches under `./.local` and in named volumes for reuse
- the Compose services keep `cap_drop: [ALL]`, `no-new-privileges`, and `tmpfs` scratch space, but the checkout and container filesystem stay writable so setup and patching are simpler
- the VM is the main isolation boundary; Compose is the second layer
- for the full step-by-step walkthrough, use [docs/STARTUP.md](docs/STARTUP.md)

Best security with GPU:
- dedicated Linux VM
- GPU passthrough
- Docker Engine
- Docker Compose plugin
- NVIDIA Container Toolkit

## Quick Start

Day-to-day secure commands after setup:

```bash
docker compose run --rm pipeline-gpu ./local.sh doctor
docker compose run --rm pipeline-gpu ./local.sh smoke
docker compose run --rm pipeline-gpu ./local.sh run
docker compose run --rm pipeline-gpu ./local.sh status
```

If you want the native Linux fallback instead:

```bash
./local.sh setup
printf "HF_TOKEN='your_token_here'\n" >> .env
./local.sh smoke
./local.sh run
./local.sh status
```

If you want to split the full flow:

```bash
./local.sh collect
./local.sh train
./local.sh retrain
```

## Dependency Profiles

The repo now keeps the base package lightweight:

- `pip install -e .`
  Installs the package without the heavy training/collection stack.
- `pip install -e '.[pipeline]'`
  Installs the full local pipeline dependency set.
- `pip install -e '.[training]'`
  Installs the image-training stack only.
- `pip install -e '.[collection]'`
  Installs the Hugging Face collection stack only.
- `pip install -e '.[video]'`
  Installs the video-specific stack only.

For native local Linux use, prefer `./local.sh deps` or `./local.sh setup`; those install the full `pipeline` profile into `./.venv`.
The packaged `aid-*` commands are lightweight wrappers and will tell you which extra to install if you run them without the required dependency profile.

## Repo Layout

Important top-level paths:

- `./.venv`
  Local virtualenv for all Python dependencies.
- `./local.sh`
  Small public command surface for setup, smoke, run, status, troubleshooting, and train-from-existing-data.
- `./install.sh`
  Optional one-line Linux installer for clone or ZIP-based use.
- `./docker-compose.yml`
  Optional Compose entrypoint for an isolated CPU or GPU container workflow.
- `./Dockerfile`
  CPU-oriented container image definition for the Compose workflow.
- `./Dockerfile.gpu`
  CUDA-enabled container image definition for the GPU Compose workflow.
- `./docs/`
  Startup, commands, and reference docs.
- `./scripts/`
  Internal pipeline helpers and advanced 4090-oriented wrappers.
- `./src/ai_image_detector/`
  Python package code for training, checkpoints, datasets, ensemble logic, and inference helpers.
- `./tests/`
  Unit and shell-surface regression coverage.
- `./data_best`
  Curated image dataset built from Hugging Face sources.
- `./data_new`
  Incremental image data waiting to be folded into training.
- `./video_data`
  Curated video dataset.
- `./.local/training_data`
  Prepared additive image training dataset.
- `./.local`
  Local caches, resumable stage markers, and collection state.

## Command Map

The public commands line up to the project structure like this:

- `./local.sh setup`
  Bootstraps `./.venv` and health-checks the repo.
- `./local.sh collect`
  Writes collected image data to `./data_best`, video data to `./video_data`, and cache/state under `./.local`.
- `./local.sh train`
  Reads `./data_best` and `./data_new`, prepares `./.local/training_data`, and trains from there.
- `./local.sh retrain`
  Reruns the train-on-existing-data path and applies the benchmark gate to the resulting artifacts.
- `./local.sh finetune`
  Runs the separate metadata-aware finetune path on top of an existing checkpoint and writes results under `./artifacts_finetune_metadata`.
- `./local.sh run`
  Runs the full collect-then-train flow and writes reports under `./.local/reports` and model artifacts under `./artifacts_ens` and `./video_artifacts`.
- `./local.sh continuous`
  Repeats the collection and retraining loop for a long-lived machine.
- `./local.sh status` and `./local.sh collect-status`
  Read the current state from the same dataset, artifact, and cache paths above.

## Open Source Notes

- License: MIT (see `LICENSE`).
- Security reporting: see `SECURITY.md`.
- Do not commit secrets (tokens, keys) or private datasets.
- Dataset and model licenses vary by source; verify each source license before commercial or production use.
- Detection outputs are probabilistic and can be wrong; review high-risk decisions with human oversight.

## Native Linux Startup

For the detailed clone path, ZIP path, and native Linux startup flow, use [docs/STARTUP.md](docs/STARTUP.md).

Shortest native Linux path after you are already in the repo root:

```bash
./local.sh setup
printf "HF_TOKEN='your_token_here'\n" >> .env
./local.sh smoke
./local.sh run
./local.sh status
```

Run `bash ./install.sh` only from inside the repo root after cloning or unzipping.
If you unzip the repo first, `bash ./install.sh` reuses that extracted folder and does not create a nested repo inside it.
If you want the installer to fetch the repo for you, use the one-line curl command instead.

Shortcuts:

```bash
curl -fsSL https://raw.githubusercontent.com/Legendarylibrorg/ai-image-video-detector/main/install.sh | bash
```

```bash
./local.sh setup
```

Important notes:
- `./local.sh setup` bootstraps `./.venv`, retries dependency install and doctor checks, and does not stop to prompt for `HF_TOKEN` by default.
- A direct package install can stay lightweight with `pip install -e .`, but the full repo workflow expects `./local.sh deps` or `pip install -e '.[pipeline]'`.
- The main operator commands are `./local.sh collect`, `./local.sh train`, `./local.sh retrain`, `./local.sh finetune`, `./local.sh continuous`, and `./local.sh collect-status`.
- `./local.sh run` is the canonical full pipeline path and writes reports under `./.local/reports` plus release artifacts under `./artifacts_ens/release`.
- `./local.sh smoke` is the tiny local end-to-end validation path. `./local.sh smoke-real` is the optional real Hugging Face + CUDA validation path.

## Docs

Use these if you need more detail:

- [docs/STARTUP.md](docs/STARTUP.md)
  Setup flow and Linux startup details.
- [docs/COMMANDS.md](docs/COMMANDS.md)
  The small public command surface.
- [docs/REFERENCE.md](docs/REFERENCE.md)
  Higher-level reference notes for datasets, training, evaluation, video, and pipeline modes.
- [CONTRIBUTING.md](CONTRIBUTING.md)
  Contribution guidance.
- [SECURITY.md](SECURITY.md)
  Security reporting guidance.
