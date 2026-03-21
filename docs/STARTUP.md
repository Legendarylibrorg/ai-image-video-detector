# Linux Startup Guide

This guide expands the startup path from the main README.

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

## Recommended startup flow

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
- installs pinned Python dependencies
- prepares local cache directories
- runs `doctor` in non-strict mode so a missing `HF_TOKEN` is a warning instead of a hard failure

4. Add your Hugging Face token to `.env`:

```bash
HF_TOKEN='your_token_here'
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

## One-command startup

If you want setup plus full collection and training in one command:

```bash
HF_TOKEN='your_token_here' ./local.sh setup-full
```

`setup-full`:
- installs Linux packages when `apt-get` is available
- creates or reuses `.venv`
- installs pinned Python dependencies
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
