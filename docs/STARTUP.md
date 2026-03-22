# Startup Guide

This guide expands the startup path from the main README.

This guide assumes a Linux machine and breaks startup into system deps, repo deps, then the pipeline.
The repo uses a pinned local virtualenv at `./.venv` for its Python dependencies and runtime.

## What the pipeline does

The repo is organized around one local pipeline:

1. setup a pinned Python environment in `./.venv`
2. run the resumable collect-plus-train pipeline
3. check status and rerun safely if needed

The main operator commands are:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
./local.sh setup
./local.sh run
./local.sh status
```

Use `./local.sh run` for the normal path. Use `./local.sh smoke` only as an optional validation step before the full run.

## Where `sudo` is needed

Use `sudo` for Linux package-manager commands such as `apt-get` and `freshclam`:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip build-essential clamav clamav-daemon
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

## Recommended Flow

If you want the shortest path, use:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
./local.sh setup
./local.sh run
./local.sh status
```

Then come back to the detailed steps only if you need them.

1. Enter the repo:

```bash
cd /path/to/image-spam
```

2. Install system packages if needed:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
```

3. Install the repo-local Python environment and pinned dependencies:

```bash
./local.sh setup
```

What `setup` does:
- copies `.env.example` to `.env` if needed
- installs Linux packages when `apt-get` is available
- creates or reuses `.venv`
- installs pinned Python dependencies from `requirements.lock`
- installs `huggingface_hub` into that repo-local venv
- prepares local cache directories
- retries dependency install and doctor checks automatically
- runs `doctor` in non-strict mode so a missing `HF_TOKEN` is a warning instead of a hard failure

4. During `./local.sh setup`, paste your Hugging Face token when prompted, or add it to `.env`:

```bash
printf "HF_TOKEN='your_token_here'\n" >> .env
```

If you only want it for the current shell session:

```bash
export HF_TOKEN='your_token_here'
```

5. Run the normal resumable pipeline:

```bash
./local.sh run
```

6. Check status if you want a quick confirmation:

```bash
./local.sh status
```

Optional validation before the full run:

```bash
./local.sh smoke
```

## Manual Linux fallback

If `./local.sh setup` does not finish cleanly, run the Linux steps one by one:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
python3 -m venv .venv
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
- `./local.sh deps`
  Installs the pinned Python dependency set into `./.venv`.
- `./local.sh doctor`
  Verifies disk space, cache dirs, venv health, core Python deps, and token state.
- `./local.sh smoke`
  Runs a smaller sanity check before the full pipeline.

`./local.sh run` is resumable:
- completed stages are skipped automatically on the next run
- active training locks are waited out instead of failing immediately
- transient stage failures are retried automatically
- collection defaults are tuned for authenticated Hugging Face limits and cache-first reuse

## Manual Linux bootstrap

Only use this if you do not want `./local.sh setup`:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
python3 -m venv .venv
./local.sh deps
./local.sh doctor
```

`./local.sh deps`:
- creates or reuses `.venv`
- installs the pinned dependency set from `requirements.lock`
- installs the local package in editable mode
- installs `huggingface_hub` in that venv
- uses the CUDA PyTorch wheel index on Linux when `nvidia-smi` is available

For lower-level environment variables and internal pipeline controls, use [docs/REFERENCE.md](docs/REFERENCE.md).

## Troubleshooting

Collection seems slow on first run:

```bash
./local.sh status
./local.sh collect-status
./local.sh smoke
```

You only want to retrain:

```bash
./local.sh train
./local.sh retrain
```

- `./local.sh train` prepares `./.local/training_data` from `./data_best` plus any incremental data under `./data_new`.
- `./local.sh train` trains images immediately and includes video training only when a complete video dataset is already present.
- `./local.sh retrain` runs that no-recollect train path and then benchmark-gates the result.

You changed dependencies:

```bash
./local.sh deps-update
bash scripts/install_deps.sh
```
