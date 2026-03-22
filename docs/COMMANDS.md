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
./local.sh status
```

## Pipeline at a glance

The normal local workflow is:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
./local.sh setup
./local.sh run
./local.sh status
```

What each stage does:

- `setup`
  Creates or reuses `./.venv`, installs pinned Python deps, prepares local cache dirs, and runs a health check.
- `run`
  Executes the normal resumable collect-plus-train pipeline.
- `status`
  Shows the current pipeline state, key artifact paths, and training lock status.
- `smoke`
  Optional smaller collection job for a quick sanity check before the full pipeline.

## `./local.sh` commands

Recommended first:

```bash
./local.sh setup
./local.sh run
./local.sh status
```

Manual fallback:

```bash
python3 -m venv .venv
./local.sh deps
./local.sh doctor
./local.sh smoke
./local.sh run
./local.sh status
```

Optional validation:

```bash
./local.sh smoke
./local.sh smoke-real
```

Main surface:

- `./local.sh setup`
  Bootstrap the local environment only.
- `./local.sh deps`
  Install the pinned Python dependencies into `./.venv` without running the full setup wrapper.
- `./local.sh doctor`
  Run the health check directly.
- `./local.sh run`
  Run the full collection and training pipeline with retries and resumable stages.
- `./local.sh smoke`
  Run a much smaller collection job for a quick sanity check.
- `./local.sh smoke-real`
  Run a tiny real Hugging Face collection plus real CUDA training smoke. Requires `HF_TOKEN` and a CUDA GPU.
- `./local.sh status`
  Show training lock and key data and artifact paths.

Everything else in the repo is internal support for the pipeline and is intentionally not part of the normal command surface.

If you need lower-level scripts or environment controls, use [docs/REFERENCE.md](docs/REFERENCE.md).

## Sudo guidance

Use `sudo` only for Linux package-manager commands such as:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
```

Do not add `sudo` to the repo commands in this file. Run them as your normal user so `.venv`, `.local`, datasets, and artifacts stay writable without ownership issues.
