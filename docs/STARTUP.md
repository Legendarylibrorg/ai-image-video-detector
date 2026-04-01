# Startup Guide

This guide expands the startup path from the main README.

This guide is Linux-VM-first.
The main venv for that path is the isolated container virtualenv at `/opt/aid-venv`.
The repo also supports a native Linux fallback that uses a pinned local virtualenv at `./.venv` for its Python dependencies and runtime.
Unless a section says otherwise, the shell snippets in this document use Linux `bash` command syntax.

If you are on macOS or Windows, do not copy the Linux-native `apt-get` commands below into your shell; use the Docker or platform sections in this document instead.

## Dedicated Linux VM + Docker Compose startup

Use this as the default startup path when possible:

Important boundary:
- Docker Compose does not create a real VM inside Docker.
- The secure model here is: host -> dedicated Linux VM -> Docker Engine -> Compose containers.
- On Linux, if you want a real VM boundary, you must create the VM first and then run Docker inside that VM.

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
   Stores the Hugging Face token in the repo root for Compose to load.
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
- repo env file: `./.env` on the host, auto-read by Docker Compose for `HF_TOKEN` and `HUGGINGFACE_HUB_TOKEN`
- general source tree: writable inside the container
- writable host/container path pairs:
  - `./.local` <-> `/workspace/.local`
  - `./data_best` <-> `/workspace/data_best`
  - `./data_new` <-> `/workspace/data_new`
  - `./video_data` <-> `/workspace/video_data`
  - `./artifacts_ens` <-> `/workspace/artifacts_ens`
  - `./artifacts_sweep` <-> `/workspace/artifacts_sweep`
  - `./artifacts_finetune_metadata` <-> `/workspace/artifacts_finetune_metadata`
  - `./video_artifacts` <-> `/workspace/video_artifacts`
  - `./incoming_model_outputs` <-> `/workspace/incoming_model_outputs`
  - `./incoming_review_queue` <-> `/workspace/incoming_review_queue`

Notes:
- the Compose services mount the source checkout at `/workspace` so normal editing, setup, and patching still work in-container
- datasets, artifacts, and caches still live in the checkout you started from
- `pipeline` uses the CPU-only `Dockerfile`, while `pipeline-gpu` uses `Dockerfile.gpu` with the CUDA runtime
- the container entrypoint creates or reuses an isolated venv volume at `/opt/aid-venv` and runs `bash scripts/install_deps.sh`
- the Compose services drop all Linux capabilities, enable `no-new-privileges`, and keep scratch space in `tmpfs`
- the VM is the main isolation boundary; Compose is the second layer inside it

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

`./local.sh setup` is the short native bootstrap once `local.sh` is present.

## macOS startup

macOS is **not** a substitute for a Linux + NVIDIA box for full CUDA training. Use macOS for:

1. **Docker Desktop + Compose (recommended)** — same repo commands as Linux, CPU-only `pipeline` service (no `pipeline-gpu` unless you have a supported GPU stack).
2. **Native Python (optional)** — clone the repo, create `./.venv`, run **unit tests** or small experiments; large training and HF collection still belong in Linux or Docker.

### macOS: Docker Desktop workflow

Prerequisites: [Docker Desktop](https://www.docker.com/products/docker-desktop/) for Mac, `git`, and a clone of this repo.

From the repo root (same as Linux):

```bash
cd ai-image-video-detector
./local.sh docker-doctor
docker compose build
docker compose run --rm pipeline ./local.sh deps
docker compose run --rm pipeline ./local.sh doctor
printf "HF_TOKEN='your_token_here'\n" >> .env
docker compose run --rm pipeline ./local.sh smoke
```

Use `pipeline` only on Mac (CPU). Do **not** expect `pipeline-gpu` to provide CUDA on Apple Silicon the same way as an NVIDIA Linux VM.

### macOS: native checkout for development / tests

Use this when you want to edit code and run the test suite without building GPU images.

```bash
cd ai-image-video-detector
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
export PYTHONPATH="$(pwd)/src"
python -m unittest discover -s tests -v
```

Notes:
- Use `python3` / `pip` from Homebrew or python.org if `python3` is not on your PATH.
- PyTorch wheels for Mac may use MPS or CPU; behavior can differ from Linux+CUDA training.
- The Linux-native `apt-get` and ClamAV steps in this document do not apply on macOS.
- For anything matching production training, prefer **Linux VM + Compose** or **Docker `pipeline`** on Mac.

## Windows startup

Windows is not a supported native PowerShell or Command Prompt path for this repo.

Use one of these options instead:

1. WSL2 Ubuntu, then follow the Linux commands in this document from inside WSL.
2. Docker Desktop plus Compose:

```bash
docker compose run --rm pipeline ./local.sh doctor
docker compose run --rm pipeline ./local.sh collect
```

Notes:
- do not run the Linux `apt-get` commands from PowerShell or Command Prompt
- if you want the closest match to the documented Linux path, use WSL2 Ubuntu
- Compose is the cleaner isolation path when you want to avoid mixing repo deps into the Windows host

## Python dependencies

The project declares its runtime stack in `pyproject.toml` (`torch`, `datasets`, `opencv-python-headless`, and related packages). Install everything with:

```bash
pip install -e .
```

For normal use inside this repo, prefer `./local.sh deps` or `./local.sh setup`; they install the same set into `./.venv`.
The packaged `aid-*` commands are thin wrappers; if imports fail, they print `run=pip install -e .` on stderr.

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
  Under the hood this installs the full dependency set from `pyproject.toml` into `./.venv`.
- `./local.sh doctor`
  Verifies disk space, cache dirs, venv health, core Python deps, and token state.
- `./local.sh smoke`
  Runs a smaller sanity check before the full pipeline.

`./local.sh run` uses the canonical quality pipeline wrapper:
- collects from Hugging Face before training
- reuses the shared Hugging Face cache under `./.local/hf`
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
