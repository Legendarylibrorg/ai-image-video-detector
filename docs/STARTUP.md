# Startup Guide

This guide expands the startup path from the main README.

This guide assumes a Linux machine and breaks startup into system deps, repo deps, then the pipeline.
The repo uses a pinned local virtualenv at `./.venv` for its Python dependencies and runtime.

## Basic Linux commands

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

Optional shortcuts:

```bash
curl -fsSL https://raw.githubusercontent.com/Legendarylibrorg/ai-image-video-detector/main/install.sh | bash
```

```bash
./local.sh setup
```

## Dependency profiles

The package now supports a lightweight base install:

- `pip install -e .`
  Base package only, without the heavy training and collection stack.
- `pip install -e '.[pipeline]'`
  Full repo workflow dependencies.
- `pip install -e '.[training]'`
  Image-training stack.
- `pip install -e '.[collection]'`
  Hugging Face collection stack.
- `pip install -e '.[video]'`
  Video stack.

For normal use inside this repo, prefer `./local.sh deps` or `./local.sh setup`; they install the full `pipeline` profile into `./.venv`.
The packaged `aid-*` commands are thin wrappers and will print a clear “install this extra” hint if you run them without the required dependency profile.

## What the pipeline does

The repo is organized around one local pipeline:

1. setup a pinned Python environment in `./.venv`
2. run the resumable collect-plus-train pipeline
3. check status and rerun safely if needed

The main operator commands after setup are:

```bash
./local.sh smoke
./local.sh run
./local.sh status
```

Use `./local.sh run` for the normal full path. It collects first, then trains.
Use `./local.sh collect` when you want to do the Hugging Face collection step first and train later.
Use `./local.sh train` only when you already have collected data and want to skip a new collection pass.
Use `./local.sh retrain` when you want another gated training pass on top of the existing collected dataset.
Use `./local.sh finetune` when you want the separate metadata-aware finetune path.
`./local.sh smoke` is the tiny local end-to-end check before the full run.

## Where `sudo` is needed

Use `sudo` for Linux package-manager commands such as `apt-get` and `freshclam`:

```bash
sudo apt-get update
sudo apt-get install -y curl ca-certificates git unzip python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
```

Do not use `sudo` for repo commands such as:

```bash
./local.sh setup
./local.sh run
bash scripts/install_deps.sh
bash scripts/do.sh pipeline
```

Those should run as your normal user so the workspace, caches, and `.venv` stay owned by the right account.
`./local.sh setup` creates or reuses that repo-local venv, and the pipeline scripts use it instead of the system Python.
It does not stop to prompt for `HF_TOKEN` by default.

## Recommended Flow

If you only want the token for the current shell session:

```bash
export HF_TOKEN='your_token_here'
```

## Manual Linux fallback

If `./local.sh setup` does not finish cleanly, run the Linux steps one by one:

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

Fallback step summary:
- `python3 -m venv .venv`
  Creates the repo-local virtualenv directly.
- `source .venv/bin/activate`
  Activates the repo-local virtualenv in your shell.
- `./local.sh deps`
  Installs the pinned Python dependency set into `./.venv`.
  It also installs the repo CLI commands and the `hf` CLI into that venv.
  Under the hood this is the full `pipeline` profile, not the lightweight base package.
- `./local.sh doctor`
  Verifies disk space, cache dirs, venv health, core Python deps, and token state.
- `./local.sh smoke`
  Runs a smaller sanity check before the full pipeline.

`./local.sh run` uses the canonical quality pipeline wrapper:
- collects from Hugging Face before training
- reuses the shared Hugging Face cache under `./.local/hf`
- waits on active training locks instead of colliding with another run
- keeps the collection defaults tuned for authenticated Hugging Face limits and cache-first reuse

Split flow if you want more control:

```bash
./local.sh collect
./local.sh train
./local.sh retrain
```

For lower-level environment variables and internal pipeline controls, use [docs/REFERENCE.md](docs/REFERENCE.md).

## Troubleshooting

Collection seems slow on first run:

```bash
./local.sh status
./local.sh collect-status
./local.sh smoke
```

You already have collected data and only want training:

```bash
./local.sh train
./local.sh retrain
./local.sh finetune
```

- `./local.sh train` prepares `./.local/training_data` from `./data_best` plus any incremental data under `./data_new`.
- `./local.sh train` skips fresh Hugging Face collection and trains images immediately.
- `./local.sh train` includes video training only when a complete video dataset is already present.
- `./local.sh retrain` runs the retrain wrapper on top of existing data and applies the benchmark gate afterward.
- `./local.sh finetune` runs the metadata-aware finetune wrapper on top of an existing checkpoint and writes to `./artifacts_finetune_metadata`.

Continuous loop:

```bash
./local.sh continuous
```

- `./local.sh continuous` runs the continuous collection and retraining loop for long-lived boxes.

You changed dependencies:

```bash
./local.sh deps
bash scripts/install_deps.sh
```
