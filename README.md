# Local AI Image And Video Training Pipeline

This repository is for one job:
- collect Hugging Face image and video data locally
- train detectors locally
- rerun safely if a long setup stops partway through

The commands below assume a local Linux machine with CUDA/PyTorch, such as an RTX 4090 box.
The repo uses a local virtualenv at `./.venv`; `./local.sh setup` creates or reuses it and the pipeline runs from there.

It is not a production serving repo in the current mode.

## Quick Start

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

Most people only need these commands:

```bash
./local.sh setup
printf "HF_TOKEN='your_token_here'\n" >> .env
./local.sh smoke
./local.sh run
./local.sh status
```

## Repo Layout

Important top-level paths:

- `./.venv`
  Local virtualenv for all Python dependencies.
- `./local.sh`
  Small public command surface for setup, smoke, run, status, troubleshooting, and train-from-existing-data.
- `./install.sh`
  Optional one-line Linux installer for clone or ZIP-based use.
- `./docs/`
  Startup, commands, and reference docs.
- `./scripts/`
  Internal pipeline helpers and advanced 4090-oriented wrappers.
- `./src/ai_image_detector/`
  Python package code for training, checkpoints, datasets, ensemble logic, and inference helpers.
- `./tests/`
  Unit and shell-surface regression coverage.
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

Use this exact Linux sequence if you want the manual Linux path:

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

If you downloaded the GitHub ZIP instead of cloning, the extracted folder is usually named `ai-image-video-detector-main`:

```bash
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

If you are already inside that extracted repo root, `bash ./install.sh` also works. If you already have the repo checked out, start here instead:

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

Shortcuts:

```bash
curl -fsSL https://raw.githubusercontent.com/Legendarylibrorg/ai-image-video-detector/main/install.sh | bash
```

```bash
./local.sh setup
```

Important notes:
- `./local.sh setup` already tries `apt-get` automatically on supported Linux hosts and uses `sudo` when available.
- `./local.sh setup` does not stop to prompt for `HF_TOKEN` by default. Add the token to `.env` after setup, then run `./local.sh smoke` or `./local.sh run`.
- Keep `sudo` on package-manager commands only. Run `./local.sh ...` and `bash scripts/...` as your normal user.
- The repo-local virtualenv is `./.venv`. The setup and pipeline scripts create or reuse it instead of relying on a global Python install.
- `huggingface_hub`, the `hf` CLI, and the repo CLI commands are installed into that same repo-local venv during setup.
- `./local.sh setup` retries dependency install and health checks automatically so it can finish cleanly after transient failures.
- `./local.sh run` is resumable: completed stages are skipped, training locks are waited out, and transient failures are retried.
- Hugging Face dataset and hub cache reuse the shared repo-local cache under `./.local/hf`, and discovery reuses cached source lists before doing live discovery calls.
- `./local.sh run` prefers high-signal HF sources, cuts weak sources earlier, and keeps repo/query pauses tuned for faster collection without hammering rate limits.
- `./local.sh collect-status` shows the current collection/build state, recent source activity, and resume hints.

`./local.sh smoke-real` is the optional real Hugging Face + CUDA validation path. Everything else is internal support for the pipeline and is intentionally not part of the normal startup path.

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
