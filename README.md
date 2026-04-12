# Local AI Image & Video Research Pipeline

**Summary:** Local AI image & video research pipeline — data, training, reports (not serving).

**Research pipeline:** a reproducible, local-first workflow to **curate data**, **train image and video detectors**, and **iterate on models** (collection → preparation → training → reports), without shipping a hosted inference product.

This repository is for one job:
- collect Hugging Face image and video data locally
- train detectors locally
- rerun safely if a long setup stops partway through

The recommended path is a dedicated Linux VM first, then Docker Compose inside that VM.
The main venv story for that path is the isolated container virtualenv at `/opt/aid-venv`.
The repo also supports a native local Linux machine with CUDA/PyTorch, such as an RTX 4090 box.
The native fallback uses a local virtualenv at `./.venv`; `./local.sh setup` creates or reuses it and the pipeline runs from there.
Unless a section says otherwise, the shell snippets in this README use Linux `bash` command syntax.
If you are on macOS or Windows, treat the Linux-native commands below as Linux-only and use the platform notes in [docs/STARTUP.md](docs/STARTUP.md) instead.

## Documentation map

| Doc | Audience |
|-----|----------|
| [docs/STARTUP.md](docs/STARTUP.md) | Full walkthrough: Linux VM + Docker, native Linux (`apt`), **macOS** (Docker Desktop + optional Python dev), Windows / WSL2 |
| [docs/COMMANDS.md](docs/COMMANDS.md) | **`./local.sh`** subcommands, **`scripts/do.sh`** stages, Compose one-liners (canonical command map) |
| [docs/REFERENCE.md](docs/REFERENCE.md) | Research pipeline diagram, **`scripts/*.py`** roles, repo layout, artifacts, **`AID_*`**, `aid-train` flags |
| [AGENTS.md](AGENTS.md) | Short orientation for contributors and **coding agents** (architecture, commands, security pointer) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contributor workflow, checks, and PR expectations |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | Community behavior expectations |

**Training Python layout:** `src/ai_image_detector/train.py` is the tiny **`python -m ai_image_detector.train`** entry; `train_main.py` holds the CLI argument parser and training loop; `train_support.py` has loss, EMA, and metric helpers; `train_run_artifacts.py` writes run config and dataset manifest; `train_post.py` runs optional holdout **test/** eval and release export. Pipeline drivers stay under **`scripts/`** (see [docs/REFERENCE.md](docs/REFERENCE.md)).

## Linux First Start

Choose one source path, then run the same repo-local bootstrap:

1. Git clone (recommended)

```bash
git clone https://github.com/Legendarylibrorg/ai-image-video-detector.git
cd ai-image-video-detector
```

2. `curl` + `tar` source archive

```bash
curl -fsSL -o ai-image-video-detector.tar.gz \
  https://github.com/Legendarylibrorg/ai-image-video-detector/archive/refs/heads/main.tar.gz
tar -xzf ai-image-video-detector.tar.gz
mv ai-image-video-detector-main ai-image-video-detector
cd ai-image-video-detector
```

3. One-line installer

```bash
curl -fsSL https://raw.githubusercontent.com/Legendarylibrorg/ai-image-video-detector/main/install.sh | bash
```

Safer bootstrap (pin the installer to a **known commit or tag** instead of tracking `main`):

```bash
export INSTALL_REV="main"   # replace with a release tag or full commit SHA
curl -fsSL "https://raw.githubusercontent.com/Legendarylibrorg/ai-image-video-detector/${INSTALL_REV}/install.sh" | bash
```

Official `install.sh` clones this repo’s default URL only. For a **GitHub fork**, set **`INSTALL_ALLOW_CUSTOM_REPO=1`** and **`INSTALL_ALLOW_NON_OFFICIAL_GITHUB_REPO=1`**. For mirrors on other hosts, add them to **`INSTALL_REPO_HOST_ALLOWLIST`** (comma-separated) or, only if you accept the risk, **`INSTALL_ALLOW_ANY_HTTPS_HOST=1`**. See [SECURITY.md](SECURITY.md).

After the source tree is present:

```bash
./local.sh setup
printf "HF_TOKEN='your_token_here'\n" >> .env
./local.sh smoke
./local.sh run
./local.sh status
```

Use a Hugging Face `read` token unless you truly need write access. On native Linux, `hf auth login` also works; `./.env` is the simplest path for this repo and for Docker Compose.

### Verify wiring (end-to-end)

From the **repo root** after `./local.sh setup` (native) or an equivalent container shell at `/workspace`:

```bash
./local.sh help
bash scripts/do.sh
python3 -m unittest discover -s tests -p 'test_*.py'
```

- **`./local.sh`**: **most** subcommands call **`bash scripts/do.sh`** (`run`→`pipeline`, `smoke`, `train`→`train-existing`, …). **Exceptions** (bootstrap, no `do.sh` hop): **`setup`**, **`deps`**, **`docker-doctor`**. If `help` prints, the operator surface is wired.
- **`bash scripts/do.sh`** with no arguments prints usage (exit code 2); that confirms `scripts/lib/core.sh` / `env.sh` load.
- **`unittest discover`** exercises Python wiring (`src/`, `scripts/` imports, checkpoints, shell contract tests). CI runs the same command against **`requirements.lock`** (see `.github/workflows/tests.yml`).

Optional **full smoke** (synthetic data, tiny training; requires a venv with **PyTorch** from **`./local.sh deps`**, default **`./.venv`**):

```bash
AID_E2E_SMOKE=1 ./.venv/bin/python -m unittest tests.test_e2e_smoke -v
# Ephemeral or alternate venv (same variable as install_deps.sh / smoke scripts):
# VENV_DIR=/path/to/venv AID_E2E_SMOKE=1 python -m unittest tests.test_e2e_smoke -v
```

That runs **`scripts/smoke_resume_eval.sh`** end-to-end. A scheduled / manual GitHub job does the same after a lock install (`.github/workflows/e2e-smoke.yml`).

## Secure Linux VM + Docker Compose

Use this as the default runtime path when possible.

Important boundary:
- Docker Compose is not a real VM.
- The secure model here is: host -> dedicated Linux VM -> Docker Engine -> Compose containers.
- If you want a true VM boundary on Linux, create the VM first and run Docker only inside that VM.

Minimum needed inside the dedicated Linux VM:
- `git`
- Docker Engine
- Docker Compose plugin
- NVIDIA Container Toolkit for `pipeline-gpu`

You do not need host Python, `pip`, or a host `./.venv` for this secure path.
All commands in this section must be run from the repo root, the directory that contains `docker-compose.yml`, `local.sh`, and `scripts/install_deps.sh`.

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

Full clone-to-`docker compose` walkthrough (including `docker compose build`, CPU deps/doctor, `.env`, and GPU steps) lives in [docs/STARTUP.md](docs/STARTUP.md) under **Exact secure startup**.

Secure path map:
- host repo root: your current working directory
- container repo root: `/workspace`
- container virtualenv: `/opt/aid-venv`
- repo env file: `./.env` on the host, auto-read by Docker Compose for `HF_TOKEN`
- general source tree under `/workspace`: writable
- writable data and artifact paths: `./.local`, `./data_best`, `./data_new`, `./video_data`, `./artifacts_ens`, `./artifacts_sweep`, `./artifacts_finetune_metadata`, `./video_artifacts`, `./incoming_model_outputs`, `./incoming_review_queue`

Notes:
- `pipeline` uses the CPU-only `Dockerfile`; `pipeline-gpu` uses `Dockerfile.gpu`
- dependency install happens inside the isolated container venv at `/opt/aid-venv`
- the container keeps Hugging Face and pip caches under `./.local` and in named volumes for reuse
- the Compose services keep `cap_drop: [ALL]`, `no-new-privileges`, and `tmpfs` scratch space, but the checkout and container filesystem stay writable so setup and patching are simpler
- the VM is the main isolation boundary; Compose is the second layer
- for the full step-by-step walkthrough, use [docs/STARTUP.md](docs/STARTUP.md)

Best security with GPU:
- dedicated Linux VM
- GPU passthrough
- Docker Engine
- Docker Compose plugin
- NVIDIA Container Toolkit

## Quick Start

Day-to-day secure commands after setup:

```bash
docker compose run --rm pipeline-gpu ./local.sh doctor
docker compose run --rm pipeline-gpu ./local.sh smoke
docker compose run --rm pipeline-gpu ./local.sh run
docker compose run --rm pipeline-gpu ./local.sh status
```

If you want the native Linux fallback instead:

```bash
./local.sh setup
printf "HF_TOKEN='your_token_here'\n" >> .env
./local.sh smoke
./local.sh run
./local.sh status
```

Setup runs a lenient disk check so clone-to-smoke works on typical dev disks; `./local.sh doctor` alone still defaults to **40GB** free for full training. One-liner install: `curl -fsSL https://raw.githubusercontent.com/Legendarylibrorg/ai-image-video-detector/main/install.sh | bash` (see [docs/STARTUP.md](docs/STARTUP.md)).

If you want to split the full flow:

```bash
./local.sh collect
./local.sh train
./local.sh retrain
```

## Python dependencies

**Where versions actually live:** committed **`requirements.lock`** (exact pins) and **`requirements.lock.json`** (each artifact’s **PyPI SHA256**). **Markdown docs do not pin package versions**—if you need a number, open those files in the repo (or your checkout after `git pull`).

**What `pyproject.toml` is for:** `requires-python` (**≥3.11**, a floor—3.12+ is fine) and **`[project.optional-dependencies]`** (`>=` **minimums** per extra: `pipeline`, `training`, `collection`, `video`, `inference`). That file declares compatibility; it does **not** replace the lock for `./local.sh deps`.

**Default install (matches the lock):** `./local.sh deps` or `./local.sh setup` → **`scripts/install_deps.sh`** installs the selected profile from **`requirements.lock`** (default profile is **pipeline**), then editable-installs the package with **`--no-deps`** so the venv matches the lock. Narrower installs: `DEPS_EXTRA=collection ./local.sh deps`, etc. The `aid-*` wrappers land in `./.venv/bin`; if imports fail, stderr points you at `./local.sh deps`.

**Refreshing dependencies:** `bash scripts/update_deps_lock.sh` picks **newest stable** PyPI releases for the pipeline profile (see **`scripts/update_deps_lock.py`**; **torch** / **torchvision** series map; **`MANIFEST_MAX_WHEEL_CP`** matches CI Python). Then `python3 scripts/update_deps_lock.py verify --require-current` and commit **both** lock files. **Security Checks** re-verifies digests; **Dependency Updates** runs **daily** and can open a refresh PR. Details: [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md).

**Manual fallback (resolver, not the lock):**

```bash
pip install -e '.[pipeline]'
```

Uses **`pyproject.toml`** minimums only; versions can differ from **`requirements.lock`** until you run `./local.sh deps` or regenerate the lock.

## Repo Layout

Important top-level paths:

- `./local.sh`
  Small public command surface for setup, smoke, run, status, troubleshooting, and train-from-existing-data.
- `./install.sh`
  Optional one-line Linux installer (`git clone` or reuse an existing tree from tarball extract).
- `./docker-compose.yml`
  Optional Compose entrypoint for an isolated CPU or GPU container workflow.
- `./Dockerfile`
  CPU-oriented container image definition for the Compose workflow.
- `./Dockerfile.gpu`
  CUDA-enabled container image definition for the GPU Compose workflow.
- `./docs/`
  Startup, commands, and reference docs.
- `./scripts/`
  Internal pipeline helpers and advanced 4090-oriented wrappers.
- `./src/ai_image_detector/`
  Python package code for training, checkpoints, datasets, ensemble logic, and inference helpers.
- `./tests/`
  Unit and shell-surface regression coverage.
- `./data_best`
  Curated image dataset built from Hugging Face sources.
- `./data_new`
  Incremental image data waiting to be folded into training.
- `./video_data`
  Curated video dataset.
- `./.local/training_data`
  Prepared additive image training dataset.
- `./.local`
  Local caches, resumable stage markers, and collection state.

## Command map

`./local.sh` is the operator surface; most commands forward to **`scripts/do.sh`**. **Bootstrap-only:** `setup`, `deps`, `docker-doctor` (see [docs/REFERENCE.md](docs/REFERENCE.md) *Architecture at a glance*).

| `./local.sh` | Typical `do.sh` / effect | Notes |
|--------------|-------------------------|--------|
| `run` | `pipeline` | Full collect → prepare → train → reports |
| `smoke` / `smoke-real` | `smoke` / `smoke-real` | Fast synthetic vs optional real HF path |
| `collect` | `collect` | Refresh `data_best`, `video_data`, `.local` |
| `train` | `train-existing` | Prepare `.local/training_data`, train |
| `retrain` / `finetune` / `continuous` | same | Retrain gate, metadata finetune script, or loop |

Full flags, Compose one-liners, and stage semantics: **[docs/COMMANDS.md](docs/COMMANDS.md)** (canonical detail—avoid duplicating it here).

## Open Source Notes

- License: MIT (see `LICENSE`).
- Contributing: see [CONTRIBUTING.md](CONTRIBUTING.md) for setup, checks, and PR expectations.
- Community standards: see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
- Security reporting: see `SECURITY.md`.
- Do not commit secrets (tokens, keys) or private datasets.
- Dataset and model licenses vary by source; verify each source license before commercial or production use.
- Detection outputs are probabilistic and can be wrong; review high-risk decisions with human oversight.

## Native Linux Startup

For obtain-the-source options (`git clone` vs `curl` + `tar`), system packages, and native Linux startup, use [docs/STARTUP.md](docs/STARTUP.md). The short Linux-first commands also live above under **Linux First Start**.

Shortest native Linux fallback after you are already in the repo root:

```bash
./local.sh setup
printf "HF_TOKEN='your_token_here'\n" >> .env
./local.sh smoke
./local.sh run
./local.sh status
```

Run `bash ./install.sh` only from inside the repo root after a `git clone` or after extracting a **`.tar.gz`** source archive (`curl` + `tar`; see [docs/STARTUP.md](docs/STARTUP.md)). The installer reuses that directory and does not create a nested checkout. To fetch and bootstrap in one step, use the curl installer below.

Fastest installer:

```bash
curl -fsSL https://raw.githubusercontent.com/Legendarylibrorg/ai-image-video-detector/main/install.sh | bash
```

Important notes: `./local.sh setup` bootstraps `./.venv` and does not prompt for `HF_TOKEN` by default; prefer **`./local.sh deps`** or **`setup`** over bare **`pip install -e '.[pipeline]'`** when you want **lock** alignment (see **Python dependencies** above). Command details: **[docs/COMMANDS.md](docs/COMMANDS.md)**.

## Docs

Use these if you need more detail:

- [docs/STARTUP.md](docs/STARTUP.md) — setup flow and Linux startup details.
- [docs/COMMANDS.md](docs/COMMANDS.md) — canonical **`./local.sh`** / **`do.sh`** / Compose command map.
- [docs/REFERENCE.md](docs/REFERENCE.md) — architecture, datasets, training, video, pipeline modes, **`AID_*`**.
- [SECURITY.md](SECURITY.md) — security reporting guidance.
