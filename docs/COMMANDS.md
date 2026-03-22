# Command Guide

This guide collects the repo command surfaces in one place.
The repo-local Python environment is `./.venv`, created or reused by `./local.sh setup`.
That setup also installs `huggingface_hub`, the `hf` CLI, and the repo CLI commands into the same venv.

The basic Linux command path is:

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

Shortcut installers:

```bash
curl -fsSL https://raw.githubusercontent.com/Legendarylibrorg/ai-image-video-detector/main/install.sh | bash
```

```bash
./local.sh setup
```

## Pipeline at a glance

The normal local workflow is the basic Linux path above.

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

Recommended first: use the same basic Linux path above.

Manual fallback:

```bash
sudo apt-get update
sudo apt-get install -y curl ca-certificates git python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
python3 -m venv .venv
source .venv/bin/activate
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
  This also installs the repo CLI commands and the `hf` CLI in that venv.
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
sudo apt-get install -y curl ca-certificates git python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
```

Do not add `sudo` to the repo commands in this file. Run them as your normal user so `.venv`, `.local`, datasets, and artifacts stay writable without ownership issues.
