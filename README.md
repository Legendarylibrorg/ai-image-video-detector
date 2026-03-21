# Local AI Image And Video Training Pipeline

This repository is for one job:
- collect Hugging Face image and video data locally
- train detectors locally
- rerun safely if a long setup stops partway through

The commands below assume a local Linux machine with CUDA/PyTorch, such as an RTX 4090 box.
The repo uses a local virtualenv at `./.venv`; `./local.sh setup` creates or reuses it and the pipeline runs from there.

It is not a production serving repo in the current mode.

## Current Pipeline

The repo now runs a local-first pipeline with a pinned `.venv`, Hugging Face dataset ingestion, resumable collection, additive training-data preparation, and local PyTorch training.

The normal flow is:

1. `./local.sh setup`
   Installs system dependencies when available, creates or reuses `.venv`, installs the pinned Python dependency set from `requirements.lock`, prepares local cache directories, and runs a health check.
2. `./local.sh collect` or `./local.sh run`
   Collects image and video data locally. Image collection uses `datasets` and `huggingface_hub`, normalizes labels, filters low-quality or duplicate samples, and writes a curated dataset under `./data_best`. Video collection writes into `./video_data`.
3. Incremental inputs are merged when present
   Fresh labeled samples or ingested outputs under `./data_new` are folded into the next train path instead of forcing a full rebuild.
4. `./local.sh train` or the train stage in `./local.sh run`
   Prepares additive training data in `./.local/training_data` from `./data_best` plus any incremental inputs, then trains the image model. Video training is included when a complete video dataset is already present.
5. Status and resume support stay local
   Setup stages, pipeline stages, and per-source collection state are written under `./.local` so long runs can be resumed safely.

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

2. Install the required system packages:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
```

3. Bootstrap the repo as your normal user:

```bash
./local.sh setup
```

This creates or reuses a local virtualenv at `.venv` and installs the pinned Python dependency set from `requirements.lock`.

4. During `./local.sh setup`, paste your Hugging Face token when prompted, or add it to `.env`:

```bash
printf "HF_TOKEN='your_token_here'\n" >> .env
```

If you only want it for the current shell session instead of writing `.env`:

```bash
export HF_TOKEN='your_token_here'
```

5. Run a quick sanity check:

```bash
./local.sh smoke
```

If you want a tiny real Hugging Face collection plus real CUDA training smoke on a 4090-class box:

```bash
./local.sh smoke-real
```

6. Start the resumable pipeline:

```bash
./local.sh run
```

Important notes:
- `./local.sh setup` already tries `apt-get` automatically on supported Linux hosts and uses `sudo` when available.
- Keep `sudo` on package-manager commands only. Run `./local.sh ...` and `bash scripts/...` as your normal user.
- The repo-local virtualenv is `./.venv`. The setup and pipeline scripts create or reuse it instead of relying on a global Python install.
- `./local.sh run` is resumable: completed stages are skipped, training locks are waited out, and transient failures are retried.
- `./local.sh collect-status` shows the current collection/build state, recent source activity, and resume hints.

### Most people only need

```bash
./local.sh setup
./local.sh smoke
./local.sh run
./local.sh status
```

If you want the smallest real end-to-end validation on a tokenized CUDA box:

```bash
./local.sh smoke-real
```

Everything else is advanced/internal:

```bash
./local.sh collect
./local.sh collect-status
./local.sh train
./local.sh retrain
./local.sh continuous
./local.sh check
./local.sh setup-full
```

### One-command startup

If you want setup plus the full collect-and-train flow in one command:

```bash
HF_TOKEN='your_token_here' ./local.sh setup-full
```

### Manual bootstrap

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
- installs pinned dependencies from `requirements.lock`
- installs the repo itself in editable mode with `pip install -e .`
- uses the CUDA PyTorch wheel index on Linux when `nvidia-smi` is present

### Startup troubleshooting

If setup stopped:

```bash
./local.sh setup-full
```

If collection seems slow:

```bash
./local.sh status
./local.sh collect-status
./local.sh smoke
```

If you only want to retrain:

```bash
./local.sh train
./local.sh retrain
```

If you changed dependencies:

```bash
./local.sh deps-update
bash scripts/install_deps.sh
```

`./local.sh deps-update` refreshes `requirements.lock`. `bash scripts/install_deps.sh` then applies that lock into `.venv`.

## Docs

Use these only if you need more detail:

- [docs/STARTUP.md](docs/STARTUP.md)
  Setup flow, manual bootstrap, setup options, and troubleshooting.
- [docs/COMMANDS.md](docs/COMMANDS.md)
  Main commands first, with advanced/internal entrypoints after that.
- [docs/REFERENCE.md](docs/REFERENCE.md)
  Higher-level reference notes for datasets, training, evaluation, video, and pipeline modes.
- [CONTRIBUTING.md](CONTRIBUTING.md)
  Contribution guidance.
- [SECURITY.md](SECURITY.md)
  Security reporting guidance.
