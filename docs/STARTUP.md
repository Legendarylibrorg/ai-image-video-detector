# Startup Guide

This guide expands the startup path from the main README.

This guide assumes a Linux machine and keeps the setup commands explicit without burying the main path.
The repo uses a pinned local virtualenv at `./.venv` for its Python dependencies and runtime.

## What the pipeline does

The repo is now organized around a local pipeline:

1. setup a pinned Python environment in `./.venv`
2. collect image and video data locally
3. curate image data into `./data_best`
4. fold in incremental image data from `./data_new` when present
5. prepare additive training data in `./.local/training_data`
6. train local image and optional video models
7. keep setup, collection, and training resumable through files under `./.local`

The main operator commands are:

```bash
./local.sh setup
./local.sh collect
./local.sh collect-status
./local.sh train
./local.sh run
```

Use `./local.sh run` when you want the normal collect-plus-train path. Use `./local.sh collect` and `./local.sh train` when you want tighter control over each stage.

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

3. Bootstrap the repo:

```bash
./local.sh setup
```

What `setup` does:
- copies `.env.example` to `.env` if needed
- installs Linux packages when `apt-get` is available
- creates or reuses `.venv`
- refreshes `pip`, `setuptools`, and `wheel`
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

5. Run the small validation path:

```bash
./local.sh smoke
```

6. Run the normal resumable pipeline:

```bash
./local.sh run
```

`./local.sh run` is resumable:
- completed stages are skipped automatically on the next run
- active training locks are waited out instead of failing immediately
- transient stage failures are retried automatically
- collection defaults are tuned for authenticated Hugging Face limits and cache-first reuse

What `run` does in practice:
- runs image collection into `./data_best`
- ingests new labeled outputs into `./data_new` when present
- runs video collection into `./video_data`
- prepares additive image training data in `./.local/training_data`
- trains the image model
- includes video training when a complete video dataset is already present

You can inspect collection state at any time with:

```bash
./local.sh collect-status
```

That prints JSON with current dataset counts, source manifest state, and resume hints.

## One-command startup

If you want setup plus full collection and training in one command:

```bash
HF_TOKEN='your_token_here' ./local.sh setup-full
```

`setup-full`:
- installs Linux packages when `apt-get` is available
- creates or reuses `.venv`
- installs pinned Python dependencies from `requirements.lock`
- validates or prompts for `HF_TOKEN`
- runs the full collection and training pipeline
- retries automatically if a stage fails
- writes resumable setup markers in `./.local/stages/*.done`

If `setup-full` stops, run it again:

```bash
./local.sh setup-full
```

To force every stage to rerun:

```bash
SETUP_FORCE_STAGES=1 ./local.sh setup-full
```

## Manual Linux bootstrap

Only use this if you do not want `./local.sh setup`:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
python3 -m venv .venv
source .venv/bin/activate
bash scripts/install_deps.sh
./local.sh check
```

`bash scripts/install_deps.sh`:
- creates or reuses `.venv`
- installs the pinned dependency set from `requirements.lock`
- installs the local package in editable mode
- installs `huggingface_hub` in that venv
- uses the CUDA PyTorch wheel index on Linux when `nvidia-smi` is available

## Setup options

- `SETUP_INSTALL_SYSTEM_DEPS`
  Set to `0` to skip `apt-get` during `./local.sh setup`.
- `SETUP_SKIP_DOCTOR`
  Set to `1` to skip the post-setup health check during `./local.sh setup`.
- `SETUP_MAX_ATTEMPTS`
  Default `4`.
- `SETUP_RETRY_SLEEP_SEC`
  Default `45`.
- `SETUP_FORCE_STAGES`
  Set to `1` to rerun completed `setup-full` stages.
- `SETUP_STAGE_DIR`
  Custom stage marker directory. Default `./.local/stages`.
- `HF_SETUP_REQUIRE_TOKEN`
  Set to `0` to allow `./local.sh setup-full` without a token.
- `HF_SETUP_SAVE_ENV`
  Set to `0` to avoid writing the token into `.env` during `./local.sh setup-full`.

## Dependency and pipeline controls

Dependency lock workflow:
- install pinned deps with `bash scripts/install_deps.sh`
- refresh the lock with `./local.sh deps-update`
- the install script skips reinstallation when `requirements.lock` and `pyproject.toml` are unchanged

Collection and training data locations:
- `./data_best`
  Curated image dataset used as the main image training base.
- `./data_new`
  Incremental image data that will be merged into the next train path.
- `./video_data`
  Curated local video dataset.
- `./.local/training_data`
  Prepared additive image training dataset built from `./data_best` plus incremental inputs.
- `./.local`
  Cache, stage, and source-manifest state for resumable runs.

Run pipeline controls:
- `PIPELINE_MAX_ATTEMPTS`
  Default `4` per stage during `./local.sh run`.
- `PIPELINE_RETRY_SLEEP_SEC`
  Default `45` seconds between retries.
- `PIPELINE_STAGE_DIR`
  Custom resumable stage marker directory. Default `./.local/pipeline`.
- `PIPELINE_FORCE_STAGES`
  Set to `1` to rerun completed `run` stages.
- `PIPELINE_WAIT_FOR_TRAINING_SEC`
  Default `600` seconds while waiting for an active training lock to clear.

Collection speed controls:
- `DIVERSE_REPO_BASE_PAUSE_MS`, `DIVERSE_REPO_JITTER_MS`, `DIVERSE_REPO_COOLDOWN_MS`
  Control image-source pacing and rate-limit backoff for `./local.sh run` or `./local.sh collect`.
- `DIVERSE_HF_QUERY_PAUSE_MS`
  Pause between discovery queries to stay under Hugging Face page limits.
- `VIDEO_SNAPSHOT_MAX_WORKERS`
  Parallel video snapshot downloads. Default `4` for the simple pipeline path.
- `VIDEO_REPO_BASE_PAUSE_MS`, `VIDEO_REPO_JITTER_MS`, `VIDEO_REPO_COOLDOWN_MS`
  Control video collection pacing and backoff.

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
