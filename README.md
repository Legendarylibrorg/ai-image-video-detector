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
   Runs a tiny local end-to-end sanity check before the full run.
4. `./local.sh run`
   Runs the canonical collect-plus-train pipeline.
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

If you want to split the full flow:

```bash
./local.sh collect
./local.sh train
./local.sh retrain
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

## Command Map

The public commands line up to the project structure like this:

- `./local.sh setup`
  Bootstraps `./.venv` and health-checks the repo.
- `./local.sh collect`
  Writes collected image data to `./data_best`, video data to `./video_data`, and cache/state under `./.local`.
- `./local.sh train`
  Reads `./data_best` and `./data_new`, prepares `./.local/training_data`, and trains from there.
- `./local.sh retrain` or `./local.sh finetune`
  Reruns the train-on-existing-data path and applies the benchmark gate to the resulting artifacts.
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

## Startup

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
If you want the installer to fetch the repo for you, use the one-line curl command instead.

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
- `./local.sh collect` is the collection-only path when you want Hugging Face image/video data without training yet.
- `./local.sh run` is the canonical full pipeline path: it collects from Hugging Face first, then trains from that collected dataset.
- `./local.sh train` is the train-only path when you already have data on disk and do not want a new collection pass.
- `./local.sh retrain` and `./local.sh finetune` rerun training on top of existing collected data with the benchmark gate applied.
- `./local.sh continuous` runs the continuous collection/retraining loop.
- Hugging Face dataset and hub cache reuse the shared repo-local cache under `./.local/hf`, and discovery reuses cached source lists before doing live discovery calls.
- `./local.sh run` prefers high-signal HF sources, cuts weak sources earlier, and keeps repo/query pauses tuned for faster collection without hammering rate limits.
- `./local.sh collect-status` shows the current collection/build state, recent source activity, and resume hints.
- `./local.sh run` now also writes simple machine-readable reports:
  `./.local/reports/dataset_qa_summary.json`, `./.local/reports/dataset_provenance.json`, `./artifacts_ens/final_run_summary.json`, `./artifacts_ens/final_thresholds.json`, `./artifacts_ens/run_manifest.json`, `./artifacts_ens/prod_manifest.json`, `./artifacts_ens/domain_config.json`, and `./artifacts_ens/robust_eval.json`.

`./local.sh smoke` is the tiny local end-to-end validation path. `./local.sh smoke-real` is the optional real Hugging Face + CUDA validation path.

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
