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
2. `printf "HF_TOKEN='your_token_here'\n" >> .env`
   Adds the Hugging Face token when you want authenticated collection.
3. `./local.sh smoke`
   Runs the quick sanity check before the full run.
4. `./local.sh run`
   Runs the resumable collect-plus-train pipeline.
5. `./local.sh status`
   Shows the current pipeline state and key artifact paths.

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

### Basic Linux commands

Use this exact Linux sequence:

```bash
sudo apt-get update
sudo apt-get install -y curl ca-certificates git python3 python3-venv python3-pip build-essential clamav clamav-daemon
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

If you already have the repo checked out, start here instead:

```bash
python3 -m venv .venv
source .venv/bin/activate
./local.sh deps
./local.sh doctor
printf "HF_TOKEN='your_token_here'\n" >> .env
./local.sh smoke
./local.sh run
./local.sh status
```

Optional shortcuts:

```bash
curl -fsSL https://raw.githubusercontent.com/Legendarylibrorg/ai-image-video-detector/main/install.sh | bash
```

```bash
./local.sh setup
```

What each broken-out command does:
- `python3 -m venv .venv`
  Creates the repo-local virtualenv.
- `source .venv/bin/activate`
  Activates the repo-local virtualenv in your shell.
- `./local.sh deps`
  Installs the pinned Python dependency set from `requirements.lock`, including `huggingface_hub`.
  Also installs the repo CLI commands and the `hf` CLI into `./.venv`.
- `./local.sh doctor`
  Checks disk space, cache paths, venv health, core deps, and token state.
- `./local.sh smoke`
  Runs the smaller sanity path before the full pipeline.

Important notes:
- `./local.sh setup` already tries `apt-get` automatically on supported Linux hosts and uses `sudo` when available.
- `./local.sh setup` does not stop to prompt for `HF_TOKEN` by default. Add the token to `.env` after setup, then run `./local.sh smoke` or `./local.sh run`.
- Keep `sudo` on package-manager commands only. Run `./local.sh ...` and `bash scripts/...` as your normal user.
- The repo-local virtualenv is `./.venv`. The setup and pipeline scripts create or reuse it instead of relying on a global Python install.
- `huggingface_hub`, the `hf` CLI, and the repo CLI commands are installed into that same repo-local venv during setup.
- `./local.sh setup` retries dependency install and health checks automatically so it can finish cleanly after transient failures.
- `./local.sh run` is resumable: completed stages are skipped, training locks are waited out, and transient failures are retried.
- `./local.sh collect-status` shows the current collection/build state, recent source activity, and resume hints.

`./local.sh smoke-real` is the optional real Hugging Face + CUDA validation path. Everything else is internal support for the pipeline and is intentionally not part of the normal startup path.

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
