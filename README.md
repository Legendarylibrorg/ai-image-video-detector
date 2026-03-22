# Local AI Image And Video Training Pipeline

This repository is for one job:
- collect Hugging Face image and video data locally
- train detectors locally
- rerun safely if a long setup stops partway through

The commands below assume a local Linux machine with CUDA/PyTorch, such as an RTX 4090 box.
The repo uses a local virtualenv at `./.venv`; `./local.sh setup` creates or reuses it and the pipeline runs from there.

It is not a production serving repo in the current mode.

## Current Pipeline

The repo now runs one simple local pipeline:

1. `./local.sh setup`
   Creates or reuses `./.venv`, installs the pinned Python dependencies, and runs a health check.
2. `./local.sh run`
   Runs the resumable collect-plus-train pipeline.
3. `./local.sh status`
   Shows the current pipeline state and key artifact paths.

Optional validation:

```bash
./local.sh smoke
./local.sh smoke-real
```

Important local directories:

- `./.venv`
  Local virtualenv for all Python dependencies.
- `./data_best`
  Curated image dataset built from Hugging Face and local inputs.
- `./data_new`
  Incremental image data waiting to be folded into training.
- `./video_data`
  Curated video dataset.
- `./.local/training_data`
  Prepared additive image training dataset.
- `./.local`
  Local caches, resumable stage markers, and collection state.

## Open Source Notes

- License: MIT (see `LICENSE`).
- Security reporting: see `SECURITY.md`.
- Do not commit secrets (tokens, keys) or private datasets.
- Dataset and model licenses vary by source; verify each source license before commercial or production use.
- Detection outputs are probabilistic and can be wrong; review high-risk decisions with human oversight.

## Startup

### Quick start

1. Enter the repo:

```bash
cd /path/to/image-spam
```

2. Install the required Linux system packages:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
```

3. Install the repo Python environment and pinned dependencies:

```bash
./local.sh setup
```

This creates or reuses a local virtualenv at `.venv`, installs the pinned Python dependency set from `requirements.lock`, installs `huggingface_hub`, and keeps the runtime inside that repo-local venv.

4. During `./local.sh setup`, paste your Hugging Face token when prompted, or add it to `.env`:

```bash
printf "HF_TOKEN='your_token_here'\n" >> .env
```

If you only want it for the current shell session instead of writing `.env`:

```bash
export HF_TOKEN='your_token_here'
```

5. Start the resumable pipeline:

```bash
./local.sh run
```

6. Check status if you want a quick confirmation:

```bash
./local.sh status
```

Important notes:
- `./local.sh setup` already tries `apt-get` automatically on supported Linux hosts and uses `sudo` when available.
- Keep `sudo` on package-manager commands only. Run `./local.sh ...` and `bash scripts/...` as your normal user.
- The repo-local virtualenv is `./.venv`. The setup and pipeline scripts create or reuse it instead of relying on a global Python install.
- `huggingface_hub` is installed into that same repo-local venv during setup.
- `./local.sh setup` retries dependency install and health checks automatically so it can finish cleanly after transient failures.
- `./local.sh run` is resumable: completed stages are skipped, training locks are waited out, and transient failures are retried.
- `./local.sh collect-status` shows the current collection/build state, recent source activity, and resume hints.

### Manual Linux fallback

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

What those fallback commands do:
- `./local.sh deps`
  Creates or reuses `./.venv` and installs the pinned Python dependencies.
- `./local.sh doctor`
  Checks disk space, cache paths, venv health, core deps, and your Hugging Face token state.

### Most people only need

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
./local.sh setup
./local.sh run
./local.sh status
```

Optional validation:

```bash
./local.sh smoke
./local.sh smoke-real
```

Everything else is internal support for the pipeline and is intentionally not part of the normal startup path.

## Docs

Use these only if you need more detail:

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
