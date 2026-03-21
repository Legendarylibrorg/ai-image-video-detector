# Command Guide

This guide collects the repo command surfaces in one place.
The repo-local Python environment is `./.venv`, created or reused by `./local.sh setup`.
That setup also installs `huggingface_hub` into the same venv.

The default path below assumes Linux:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
./local.sh setup
./local.sh run
```

## Pipeline at a glance

The normal local workflow is:

```bash
./local.sh setup
./local.sh smoke
./local.sh run
./local.sh status
```

What each stage does:

- `setup`
  Creates or reuses `./.venv`, installs pinned Python deps, prepares local cache dirs, and runs a health check.
- `smoke`
  Runs a much smaller collection job for a quick sanity check before the full pipeline.
- `run`
  Executes the normal resumable collect-plus-train pipeline.
- `status`
  Shows the current pipeline state, key artifact paths, and training lock status.

## `./local.sh` commands

Recommended first:

```bash
./local.sh setup
./local.sh smoke
./local.sh smoke-real
./local.sh run
./local.sh status
```

Main surface:

- `./local.sh setup`
  Bootstrap the local environment only.
- `./local.sh run`
  Run the full collection and training pipeline with retries and resumable stages.
- `./local.sh smoke`
  Run a much smaller collection job for a quick sanity check.
- `./local.sh smoke-real`
  Run a tiny real Hugging Face collection plus real CUDA training smoke. Requires `HF_TOKEN` and a CUDA GPU.
- `./local.sh status`
  Show training lock and key data and artifact paths.

Advanced/internal commands:

- `./local.sh check`
  Run the preflight check directly.
- `./local.sh setup-full`
  End-to-end setup, collection, and training.
- `./local.sh collect`
  Run collection only.
- `./local.sh collect-status`
  Print JSON status for the current image/video collection state, source manifest, and resume hints.
- `./local.sh train`
  Prepare additive training data from `./data_best` plus `./data_new` and train without recollecting.
- `./local.sh retrain`
  Reuse collected data, fold in fresh incremental samples, and run the local retrain flow plus benchmark gate.
- `./local.sh continuous`
  Run the continuous training loop.
- `./local.sh start`
  Run the best-quality pipeline path.
- `./local.sh scan [paths...]`
  Run the malware scan directly.
- `./local.sh deps-update`
  Refresh the locked dependency set.

## Raw `scripts/do.sh` commands

```bash
bash scripts/do.sh pipeline
bash scripts/do.sh run
bash scripts/do.sh smoke
bash scripts/do.sh smoke-real
bash scripts/do.sh check
bash scripts/do.sh doctor
bash scripts/do.sh start
bash scripts/do.sh start-v2
bash scripts/do.sh collect
bash scripts/do.sh collect-diverse
bash scripts/do.sh collect-fast
bash scripts/do.sh collect-image
bash scripts/do.sh collect-video
bash scripts/do.sh collection-status
bash scripts/do.sh ingest
bash scripts/do.sh scan [paths...]
bash scripts/do.sh train
bash scripts/do.sh train-existing
bash scripts/do.sh train-image
bash scripts/do.sh train-video
bash scripts/do.sh train-all
bash scripts/do.sh retrain
bash scripts/do.sh continuous
bash scripts/do.sh train-all-types
bash scripts/do.sh deps-update
bash scripts/do.sh status
```

## Collection status

```bash
bash scripts/do.sh collection-status
aid-dataset collection-status --data ./data_best
```

## Collection progress controls

Image collection defaults to quiet Hugging Face `datasets` progress output so source-level logs stay readable.

Set one of these env vars to `1` if you want verbose `map`/`filter` progress bars during collection:

```bash
BEST_DS_VERBOSE_PROGRESS=1 bash scripts/do.sh collect
DIVERSE_VERBOSE_PROGRESS=1 bash scripts/do.sh collect-diverse
FAST_VERBOSE_PROGRESS=1 bash scripts/do.sh collect-fast
```

## Top-level wrappers

These are convenience launchers:

```bash
./start.sh
./run.sh
./collect.sh
./train.sh
./retrain.sh
./autocollect.sh
./continuous.sh
```

## Lower-level setup and pipeline scripts

```bash
bash scripts/install_deps.sh
bash scripts/setup_local.sh
bash scripts/setup_linux.sh
bash scripts/full_pipeline_4090.sh
bash scripts/max_quality_4090.sh
bash scripts/max_accuracy_v2.sh
```

## Sudo guidance

Use `sudo` only for Linux package-manager commands such as:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
```

Do not add `sudo` to the repo commands in this file. Run them as your normal user so `.venv`, `.local`, datasets, and artifacts stay writable without ownership issues.
