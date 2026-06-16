# Startup Guide

This guide expands the startup path from the main README for the **local AI image & video research pipeline** (clone → environment → first `./local.sh` runs).

**Short copy-paste paths:** see [Quick start](../README.md#quick-start) in the README for numbered steps on **Linux**, **macOS**, and **Windows**.

This guide is Linux-VM-first for security detail.
The main venv for that path is the isolated container virtualenv at `/opt/aid-venv`.
The repo also supports a native Linux fallback that uses a pinned local virtualenv at `./.venv` for its Python dependencies and runtime.
Unless a section says otherwise, the shell snippets in this document use Linux `bash` command syntax.

If you are on macOS or Windows, start with the README **Quick start**; do not copy Linux-native `apt-get` commands into macOS or PowerShell.

## Dedicated Linux VM + Docker Compose startup

Use this as the default startup path when possible:

Important boundary:
- Docker Compose does not create a real VM inside Docker.
- The secure model here is: host -> dedicated Linux VM -> Docker Engine -> Compose containers.
- On Linux, if you want a real VM boundary, you must create the VM first and then run Docker inside that VM.
- On macOS and Windows, Docker Desktop may already run containers inside a lightweight Linux VM or WSL2-backed microVM-style layer. That is useful isolation for CPU workflows, but it is not the same as the dedicated GPU Linux VM path in this guide.

Minimum needed inside the dedicated Linux VM:
- `git`
- Docker Engine
- Docker Compose plugin
- NVIDIA Container Toolkit for `pipeline-gpu`

You do not need host Python, `pip`, or a host `./.venv` for this secure path.
Run every command in this section from the repo root, the directory that contains `docker-compose.yml`, `local.sh`, and `scripts/install_deps.sh`.

Linux VM setup checklist:
1. Create a dedicated Linux VM for this repo.
2. If you need GPU training, enable GPU passthrough for that VM.
3. Install Docker Engine inside the VM.
4. Install the Docker Compose plugin inside the VM.
5. Install NVIDIA Container Toolkit inside the VM if `pipeline-gpu` will be used.
6. Clone this repo inside that VM.
7. Run the Compose workflow only from inside that VM.

Repo-root check:

```bash
pwd
test -f docker-compose.yml
test -f Dockerfile
test -f Dockerfile.gpu
test -f local.sh
test -f scripts/install_deps.sh
./local.sh docker-doctor
```

What `./local.sh docker-doctor` checks:
- `docker-compose.yml` exists
- `Dockerfile` exists
- `Dockerfile.gpu` exists
- `docker` is installed
- `docker compose` is available

Detailed secure Docker config flow:
1. Build the Compose images.
2. Install repo dependencies inside the isolated container venv at `/opt/aid-venv`.
3. Run the normal doctor check in the CPU container first.
4. Add `HF_TOKEN` to `./.env` in the repo root.
5. Run the normal doctor check in the GPU container.
6. Run the smoke pipeline in the GPU container.
7. Run the full pipeline in the GPU container.
8. Use `status` from the GPU container to inspect outputs.

Exact secure startup:

```bash
git clone https://github.com/Legendarylibrorg/ai-image-video-detector.git
cd ai-image-video-detector
test -f docker-compose.yml
test -f Dockerfile
test -f Dockerfile.gpu
test -f local.sh
./local.sh docker-doctor
docker compose build
docker compose run --rm pipeline ./local.sh deps
docker compose run --rm pipeline ./local.sh doctor
printf "HF_TOKEN='your_token_here'\n" >> .env
test -f .env
docker compose run --rm pipeline-gpu ./local.sh doctor
docker compose run --rm pipeline-gpu ./local.sh smoke
docker compose run --rm pipeline-gpu ./local.sh run
docker compose run --rm pipeline-gpu ./local.sh status
```

What each step does:
1. `docker compose build`
   Builds the container images.
2. `./local.sh docker-doctor`
   Verifies the Docker CLI, Compose plugin, and repo Docker files before container work starts.
3. `docker compose run --rm pipeline ./local.sh deps`
   Installs the pipeline dependencies in the isolated container venv at `/opt/aid-venv`.
4. `docker compose run --rm pipeline ./local.sh doctor`
   Verifies the secure container path before GPU work.
5. `printf "HF_TOKEN='your_token_here'\n" >> .env`
   Stores a Hugging Face `read` token in the repo root for Compose to load.
6. `docker compose run --rm pipeline-gpu ./local.sh doctor`
   Verifies GPU access inside the secure path.
7. `docker compose run --rm pipeline-gpu ./local.sh smoke`
   Runs the smallest end-to-end pipeline check.
8. `docker compose run --rm pipeline-gpu ./local.sh run`
   Runs the full secure pipeline.
9. `docker compose run --rm pipeline-gpu ./local.sh status`
   Prints the current runtime and artifact state.

Path map:
- host repo root: your current working directory
- container repo root: `/workspace`
- container venv: `/opt/aid-venv`
- repo env file: `./.env` on the host, auto-read by Docker Compose for `HF_TOKEN`
- general source tree: writable inside the container
- writable host/container path pairs:
  - `./.local` <-> `/workspace/.local` for general repo-local state
  - `./data_best` <-> `/workspace/data_best`
  - `./data_new` <-> `/workspace/data_new`
  - `./video_data` <-> `/workspace/video_data`
  - `./artifacts_ens` <-> `/workspace/artifacts_ens`
  - `./artifacts_sweep` <-> `/workspace/artifacts_sweep`
  - `./artifacts_finetune_metadata` <-> `/workspace/artifacts_finetune_metadata`
  - `./video_artifacts` <-> `/workspace/video_artifacts`
  - `./incoming_model_outputs` <-> `/workspace/incoming_model_outputs`
  - `./incoming_review_queue` <-> `/workspace/incoming_review_queue`
- named volume/container path pairs:
  - `aid_hf_cache` <-> `/workspace/.local/hf` for the active Hugging Face cache
  - `aid_pip_cache` <-> `/workspace/.local/pip` for the active pip cache

Notes:
- the Compose services mount the source checkout at `/workspace` so normal editing, setup, and patching still work in-container
- datasets and artifacts still live in the checkout you started from; the active Hugging Face and pip caches live in the named volumes mounted at `/workspace/.local/hf` and `/workspace/.local/pip`
- `pipeline` uses the CPU-only `Dockerfile`, while `pipeline-gpu` uses `Dockerfile.gpu` with the CUDA runtime
- the container entrypoint creates or reuses an isolated venv volume at `/opt/aid-venv` and runs `bash scripts/install_deps.sh`
- the Compose services drop all Linux capabilities, enable `no-new-privileges`, and keep scratch space in `tmpfs`
- the VM is the main isolation boundary; Compose is the second layer inside it
- Docker Desktop's lightweight VM/microVM-style boundary is appropriate for local CPU checks, but production-like GPU training should still use the dedicated Linux VM + Compose path

Security model:
- the dedicated Linux VM is the main defense against malicious packages reaching your normal host
- Docker reduces exposure further inside that VM, but it does not guarantee safety from malicious packages
- dependency installers and imported packages still execute code, only inside the container
- because selected repo data and artifact directories stay writable, malicious code could still change those writable paths inside the VM
- the general source checkout is writable in Compose so the normal repo workflow stays simple; the VM remains the main isolation boundary
- keep the repo in a dedicated VM workspace and avoid mounting unrelated secrets into the container

Best security with GPU:
- dedicated Linux VM
- GPU passthrough
- Docker Engine
- Docker Compose plugin
- NVIDIA Container Toolkit

## Native Linux fallback

Linux is the supported native host path.

`./local.sh` and `install.sh` are tracked as executable (`100755`); if your filesystem strips execute bits, run `bash ./local.sh …` instead.

**Bootstrap note:** `./local.sh setup` runs `doctor` with a **relaxed disk check** (`SETUP_DOCTOR_MIN_FREE_GB`, default `0` during setup) so first-time setup succeeds on smaller disks. A manual `./local.sh doctor` still defaults to **40GB** free space for full training; the full pipeline also enforces its own disk guard.

### Obtain the source

Use **one** of the following.

#### 1. Git clone (recommended)

```bash
git clone https://github.com/Legendarylibrorg/ai-image-video-detector.git
cd ai-image-video-detector
```

#### 2. Source tarball with `curl` and `tar`

For environments where `git` is unavailable, fetch an archive from GitHub and extract it with standard Linux tools. Replace `refs/heads/main` with `refs/tags/<tag>` when pinning a release.

```bash
curl -fsSL -o ai-image-video-detector.tar.gz \
  https://github.com/Legendarylibrorg/ai-image-video-detector/archive/refs/heads/main.tar.gz
tar -xzf ai-image-video-detector.tar.gz
mv ai-image-video-detector-main ai-image-video-detector
cd ai-image-video-detector
```

### System packages (`sudo`)

```bash
sudo apt-get update
sudo apt-get install -y curl ca-certificates git python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
```

### Bootstrap the repo (no `sudo`)

From the repository root (after clone or tarball extract):

```bash
./local.sh setup
printf "HF_TOKEN='your_token_here'\n" >> .env
./local.sh smoke
./local.sh run
./local.sh status
```

Use a Hugging Face `read` token unless you need write access. On native Linux, `hf auth login` is also supported; this repo keeps showing the `./.env` flow because it is easiest to automate and matches Docker Compose.

### Combined: clone + packages + bootstrap

```bash
sudo apt-get update
sudo apt-get install -y curl ca-certificates git python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
git clone https://github.com/Legendarylibrorg/ai-image-video-detector.git
cd ai-image-video-detector
./local.sh setup
printf "HF_TOKEN='your_token_here'\n" >> .env
./local.sh smoke
./local.sh run
./local.sh status
```

### Already inside the repository

```bash
sudo apt-get update
sudo apt-get install -y curl ca-certificates git python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
./local.sh setup
printf "HF_TOKEN='your_token_here'\n" >> .env
./local.sh smoke
./local.sh run
./local.sh status
```

Run `bash ./install.sh` only from inside this repository after you have the source tree (clone or tarball). If you extracted a tarball, `install.sh` reuses that directory and does not create a nested checkout.

One-line installer (clones with `git` when needed):

```bash
curl -fsSL https://raw.githubusercontent.com/Legendarylibrorg/ai-image-video-detector/main/install.sh | bash
```

For a **pinned installer revision** (commit or tag) and fork/mirror notes, see **Safer bootstrap** in [README.md](../README.md) and [SECURITY.md](../SECURITY.md).

`./local.sh setup` is the short native bootstrap once `local.sh` is present.

## macOS startup

macOS is **not** a substitute for a Linux + NVIDIA box for full CUDA training. Use macOS for:

1. **Docker Desktop + Compose (recommended)** — same repo commands as Linux, CPU-only `pipeline` service (no `pipeline-gpu` unless you have a supported GPU stack).
2. **Native Python (optional)** — clone the repo, create `./.venv`, run **unit tests** or small experiments; large training and HF collection still belong in Linux or Docker.

Numbered steps: [README Quick start — Docker Compose](../README.md#docker-compose-linux-and-macos).

### macOS: Docker Desktop workflow

Prerequisites: [Docker Desktop](https://www.docker.com/products/docker-desktop/) for Mac, `git`, and a clone of this repo.

From the repo root:

1. `./local.sh docker-doctor`
2. `docker compose build`
3. `docker compose run --rm pipeline ./local.sh deps`
4. `docker compose run --rm pipeline ./local.sh doctor`
5. `printf "HF_TOKEN='your_token_here'\n" >> .env`
6. `docker compose run --rm pipeline ./local.sh smoke`
7. `docker compose run --rm pipeline ./local.sh run`
8. `docker compose run --rm pipeline ./local.sh status`

Use `pipeline` only on Mac (CPU). Do **not** expect `pipeline-gpu` to provide CUDA on Apple Silicon the same way as an NVIDIA Linux VM.

### macOS: native checkout for development / tests

Use this when you want to edit code and run the test suite without building GPU images.

```bash
cd ai-image-video-detector
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[pipeline]'
python3 -m unittest discover -s tests -p 'test_*.py'
```

Notes:
- Use `python3` / `pip` from Homebrew or python.org if `python3` is not on your PATH.
- PyTorch wheels for Mac may use MPS or CPU; behavior can differ from Linux+CUDA training.
- The Linux-native `apt-get` and ClamAV steps in this document do not apply on macOS.
- For anything matching production training, prefer **Linux VM + Compose** or **Docker `pipeline`** on Mac.

## Windows startup

Windows is not a supported native PowerShell or Command Prompt path for this repo.

Numbered steps: [README Quick start — Windows](../README.md#windows).

### Option A — WSL2 Ubuntu (recommended)

1. Install WSL2 with Ubuntu ([Microsoft WSL install guide](https://learn.microsoft.com/en-us/windows/wsl/install)).
2. Open the **Ubuntu** terminal.
3. Follow the [Linux native](../README.md#linux-native-ubuntudebian--nvidia-gpu) steps in the README (from system packages through `./local.sh status`).
4. For NVIDIA GPU in WSL, install [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) before the full run.

### Option B — Docker Desktop

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (enable WSL2 backend when prompted).
2. Open **WSL2 Ubuntu** or **Git Bash** — not PowerShell for `./local.sh`.
3. Clone the repo and follow [Docker Compose](../README.md#docker-compose-linux-and-macos) steps in the README.

Notes:
- do not run Linux `apt-get` or `./local.sh` from PowerShell or Command Prompt unless you are inside WSL
- WSL2 Ubuntu is the closest match to the documented Linux native path
- Compose avoids mixing repo Python deps into the Windows host

## Python dependencies

**Authoritative versions** are in the repo root **`requirements.lock`** and **`requirements.lock.json`** (SHA256 per artifact). **Docs do not list package versions**—always read those files (or the [README](../README.md) “Python dependencies” section for the full policy).

- **Python:** `requires-python >=3.11` in **`pyproject.toml`** (floor only).
- **Minimums:** **`[project.optional-dependencies]`** extras (`pipeline`, `training`, `collection`, `video`, `inference`).
- **Install matching the lock:** `./local.sh deps` or `./local.sh setup` (see **`scripts/install_deps.sh`**); profiles via `DEPS_EXTRA=…`.
- **Refresh pins:** `bash scripts/update_deps_lock.sh` then `python3 scripts/update_deps_lock.py verify --require-current`; commit both lock files.
- **CI alignment:** the local quality gate and **Dependency Updates** use **`.github/ci-python-version.txt`** (via **`.github/actions/setup-aid-python`** for the workflow); **`scripts/update_deps_lock.py`** uses **`MANIFEST_MAX_WHEEL_CP`** so **`requirements.lock.json`** wheel tags stay in step with that interpreter (see [REFERENCE.md](REFERENCE.md) and [CI_LOCAL.md](CI_LOCAL.md)).

Manual install (may differ from the lock until you re-run `./local.sh deps`):

```bash
pip install -e '.[pipeline]'
```

## What the pipeline does

The repo is organized around one local pipeline:

1. setup a pinned Python environment in `./.venv`
2. run the resumable collect-plus-train pipeline
3. check status and rerun safely if needed

The main operator commands after setup are:

```bash
./local.sh smoke
./local.sh run
./local.sh status
```

Use `./local.sh run` for the normal full path. It collects first, then trains.
Use `./local.sh collect` when you want to do the Hugging Face collection step first and train later.
Use `./local.sh train` only when you already have collected data and want to skip a new collection pass.
Use `./local.sh retrain` when you want another gated training pass on top of the existing collected dataset.
Use `./local.sh finetune` when you want the separate metadata-aware finetune path.
`./local.sh smoke` is the tiny local end-to-end check before the full run.

## Where `sudo` is needed

Use `sudo` for Linux package-manager commands such as `apt-get` and `freshclam`:

```bash
sudo apt-get update
sudo apt-get install -y curl ca-certificates git python3 python3-venv python3-pip build-essential clamav clamav-daemon
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
It does not stop to prompt for `HF_TOKEN` by default.

## Manual Linux fallback

If `./local.sh setup` does not finish cleanly, use the **Already inside the repository** path above and then follow this fallback step summary:

Fallback step summary:
- `python3 -m venv .venv`
  Creates the repo-local virtualenv directly.
- `source .venv/bin/activate`
  Activates the repo-local virtualenv in your shell.
- `./local.sh deps`
  Installs the pinned Python dependency set into `./.venv`.
  It also installs the repo CLI commands and the `hf` CLI into that venv.
  By default this installs the full `pipeline` profile from `pyproject.toml`; set `DEPS_EXTRA=...` for a narrower profile.
- `./local.sh doctor`
  Verifies disk space, cache dirs, venv health, core Python deps, and token state.
- `./local.sh smoke`
  Runs a smaller sanity check before the full pipeline.

`./local.sh run` uses the canonical quality pipeline wrapper:
- collects from Hugging Face before training
- reuses the shared Hugging Face cache mounted at `/workspace/.local/hf` in Compose or the repo-local cache paths on native Linux
- waits on active training locks instead of colliding with another run
- keeps the collection defaults tuned for authenticated Hugging Face limits and cache-first reuse

Split flow if you want more control:

```bash
./local.sh collect
./local.sh train
./local.sh retrain
```

For lower-level environment variables and internal pipeline controls, use [docs/REFERENCE.md](docs/REFERENCE.md).

## Troubleshooting

Collection seems slow on first run:

```bash
./local.sh status
./local.sh collect-status
./local.sh smoke
```

You already have collected data and only want training:

```bash
./local.sh train
./local.sh retrain
./local.sh finetune
```

- `./local.sh train` prepares `./.local/training_data` from `./data_best` plus any incremental data under `./data_new`.
- `./local.sh train` skips fresh Hugging Face collection and trains images immediately.
- `./local.sh train` includes video training only when a complete video dataset is already present.
- `./local.sh retrain` runs the retrain wrapper on top of existing data and applies the benchmark gate afterward.
- `./local.sh finetune` runs the metadata-aware finetune wrapper on top of an existing checkpoint and writes to `./artifacts_finetune_metadata`.

Continuous loop:

```bash
./local.sh continuous
```

- `./local.sh continuous` runs the continuous collection and retraining loop for long-lived boxes.

You changed dependencies:

```bash
./local.sh deps
```
