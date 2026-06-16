# Local AI Image & Video Research Pipeline-(Not to use against ai artists or artists, simply a local pipeline)

**Summary:** Local AI image & video research pipeline — data, training, reports (not serving).

**Research pipeline:** a reproducible, local-first workflow to **curate data**, **train image and video detectors**, and **iterate on models** (collection → preparation → training → reports), without shipping a hosted inference product.

This repository is for one job:
- collect Hugging Face image and video data locally
- train detectors locally
- rerun safely if a long setup stops partway through

All paths below need a Hugging Face **read** token in `./.env` unless you use `hf auth login` on native Linux. Full walkthroughs, tarball install, and security notes: [docs/STARTUP.md](docs/STARTUP.md).

## Quick start

Pick your platform. Run every command from the **repo root** (the directory with `local.sh` and `docker-compose.yml`).

### Linux native (Ubuntu/Debian + NVIDIA GPU)

Best when Linux is your main training host (for example an RTX 4090 box). Creates a repo-local venv at `./.venv`.

1. **Install system packages**

```bash
sudo apt-get update
sudo apt-get install -y curl ca-certificates git python3 python3-venv python3-pip build-essential clamav clamav-daemon
sudo freshclam || true
```

2. **Clone the repository**

```bash
git clone https://github.com/Legendarylibrorg/ai-image-video-detector.git
cd ai-image-video-detector
```

3. **Bootstrap Python and run checks**

```bash
./local.sh setup
```

4. **Add your Hugging Face token**

```bash
printf "HF_TOKEN='your_token_here'\n" >> .env
```

5. **Smoke test (small end-to-end check)**

```bash
./local.sh smoke
```

6. **Full pipeline (collect → train → reports)**

```bash
./local.sh run
```

7. **Check status**

```bash
./local.sh status
```

**Linux + Docker (isolated VM or shared machine):** use the [Docker Compose steps](#docker-compose-linux-and-macos) below inside a dedicated Linux VM when possible. GPU training uses the `pipeline-gpu` service; see [docs/STARTUP.md](docs/STARTUP.md).

### Docker Compose (Linux and macOS)

Use this on **macOS** (CPU only) or on **Linux** when you want container isolation. Venv inside the container: `/opt/aid-venv`. Repo mounted at `/workspace`.

**VM boundary note:** Docker Compose is not itself a VM. On Linux, run it inside a dedicated VM when you need a hard host boundary. On macOS and Windows, Docker Desktop typically runs Linux containers inside a lightweight VM or WSL2-backed microVM-style layer, but the repo still treats `pipeline` as the CPU container path and keeps CUDA training on Linux/WSL2.

1. **Prerequisites**
   - **Linux:** Docker Engine + Compose plugin (+ NVIDIA Container Toolkit for `pipeline-gpu`)
   - **macOS:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) and `git`

2. **Clone the repository**

```bash
git clone https://github.com/Legendarylibrorg/ai-image-video-detector.git
cd ai-image-video-detector
```

3. **Verify Docker setup**

```bash
./local.sh docker-doctor
```

4. **Build images and install dependencies**

```bash
docker compose build
docker compose run --rm pipeline ./local.sh deps
docker compose run --rm pipeline ./local.sh doctor
```

5. **Add your Hugging Face token**

```bash
printf "HF_TOKEN='your_token_here'\n" >> .env
```

6. **Smoke test**

```bash
# macOS and CPU-only Linux:
docker compose run --rm pipeline ./local.sh smoke

# Linux with NVIDIA GPU (inside a GPU-enabled VM):
docker compose run --rm pipeline-gpu ./local.sh doctor
docker compose run --rm pipeline-gpu ./local.sh smoke
```

7. **Full pipeline and status**

```bash
# CPU (macOS / pipeline):
docker compose run --rm pipeline ./local.sh run
docker compose run --rm pipeline ./local.sh status

# GPU (Linux pipeline-gpu):
docker compose run --rm pipeline-gpu ./local.sh run
docker compose run --rm pipeline-gpu ./local.sh status
```

On **macOS**, use `pipeline` only — do not expect CUDA from `pipeline-gpu` the same way as on an NVIDIA Linux host.

### Windows

There is no native PowerShell path. Use **WSL2 Ubuntu** (closest to Linux native) or **Docker Desktop**.

#### Option A — WSL2 Ubuntu (recommended for GPU training on Windows)

1. **Install WSL2** with Ubuntu (Microsoft docs: [Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install)).
2. **Open an Ubuntu terminal** and follow the [Linux native](#linux-native-ubuntudebian--nvidia-gpu) steps above from step 1.
3. For NVIDIA GPU inside WSL, install the [Windows NVIDIA driver + WSL CUDA support](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) before `./local.sh run`.

#### Option B — Docker Desktop on Windows

1. **Install** [Docker Desktop](https://www.docker.com/products/docker-desktop/) and enable WSL2 backend if prompted.
2. **Open a WSL2 or Git Bash terminal**, clone the repo, then follow [Docker Compose steps](#docker-compose-linux-and-macos) from step 2.
3. Use the **`pipeline`** service (CPU). Full GPU training still belongs on Linux or WSL2 with CUDA.

Do **not** run `apt-get` or `./local.sh` from PowerShell or Command Prompt unless you are already inside WSL.

## Documentation map

| Doc | Audience |
|-----|----------|
| [docs/STARTUP.md](docs/STARTUP.md) | Full walkthrough: Linux VM + Docker, native Linux (`apt`), **macOS** (Docker Desktop + optional Python dev), Windows / WSL2 |
| [docs/COMMANDS.md](docs/COMMANDS.md) | **`./local.sh`** subcommands, **`scripts/do.sh`** stages, Compose one-liners (canonical command map) |
| [docs/REFERENCE.md](docs/REFERENCE.md) | Research pipeline diagram, **`scripts/*.py`** roles, repo layout, artifacts, **`AID_*`**, `aid-train` flags |
| [AGENTS.md](AGENTS.md) | Short orientation for contributors and **coding agents** (architecture, commands, security pointer) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contributor workflow, checks, and PR expectations |
| [Local quality gate](docs/CI_LOCAL.md) | `make ci-fast` / `make ci` — test, security, E2E smoke |

**Training Python layout:** `src/ai_image_detector/train.py` is the tiny **`python -m ai_image_detector.train`** entry; `train_main.py` holds the CLI argument parser and training loop; `train_support.py` has loss, EMA, and metric helpers; `train_run_artifacts.py` writes run config and dataset manifest; `train_post.py` runs optional holdout **test/** eval and release export. Pipeline drivers stay under **`scripts/`** (see [docs/REFERENCE.md](docs/REFERENCE.md)).

## Verify wiring

After `./local.sh setup` (native Linux) or `docker compose run … ./local.sh deps` (Docker), from the repo root or container `/workspace`:

```bash
./local.sh help
bash scripts/do.sh
python3 -m unittest discover -s tests -p 'test_*.py'
```

- **`./local.sh`**: **most** subcommands call **`bash scripts/do.sh`** (`run`→`pipeline`, `smoke`, `train`→`train-existing`, …). **Exceptions** (bootstrap, no `do.sh` hop): **`setup`**, **`deps`**, **`docker-doctor`**. If `help` prints, the operator surface is wired.
- **`bash scripts/do.sh`** with no arguments prints usage (exit code 2); that confirms `scripts/lib/core.sh` / `env.sh` load.
- **`unittest discover`** exercises Python wiring (`src/`, `scripts/` imports, checkpoints, shell contract tests). The local quality gate runs the same command against **`requirements.lock`** (see [docs/CI_LOCAL.md](docs/CI_LOCAL.md)).

Optional **full smoke** (synthetic data, tiny training; requires a venv with **PyTorch** from **`./local.sh deps`**, default **`./.venv`**):

```bash
AID_E2E_SMOKE=1 ./.venv/bin/python -m unittest tests.test_e2e_smoke -v
# Ephemeral or alternate venv (same variable as install_deps.sh / smoke scripts):
# VENV_DIR=/path/to/venv AID_E2E_SMOKE=1 python -m unittest tests.test_e2e_smoke -v
```

That runs **`scripts/smoke_resume_eval.sh`** end-to-end. Run via **`make ci`** or **`python3 scripts/run_ci_local.py --job e2e-smoke`** before release merges.

## Secure Linux VM + Docker Compose

Prefer a **dedicated Linux VM**, then Docker Compose on that VM—not on the bare metal laptop you use for email. The [Quick start](#quick-start) section above has the copy-paste steps; this section is the rationale.

Compose is **not** a VM; the VM is the main isolation boundary. Inside the VM you want **Docker Engine**, the **Compose plugin**, and **NVIDIA Container Toolkit** for **`pipeline-gpu`**. You do **not** need host Python or a host **`./.venv`** for the container path.

Paths: host repo root → container **`/workspace`**; deps and CLIs → **`/opt/aid-venv`**; **`./.env`** on the host supplies **`HF_TOKEN`** to Compose. Writable data/artifact roots include **`./.local`**, **`./data_best`**, **`./data_new`**, **`./video_data`**, **`./artifacts_ens`**, and the other dirs under **Repo layout** below. **`pipeline`** is CPU-only; **`pipeline-gpu`** uses **`Dockerfile.gpu`**.

More detail (tarball install, `install.sh`, path map): [docs/STARTUP.md](docs/STARTUP.md) and [docs/COMMANDS.md](docs/COMMANDS.md).

`./local.sh setup` uses a lenient disk check for clone-to-smoke; **`./local.sh doctor`** alone still defaults to **40GB** free for full training.

## Python dependencies

**Where versions actually live:** committed **`requirements.lock`** (exact pins) and **`requirements.lock.json`** (each artifact’s **PyPI SHA256**). **Markdown docs do not pin package versions**—if you need a number, open those files in the repo (or your checkout after `git pull`).

**What `pyproject.toml` is for:** `requires-python` (**≥3.11**, a floor—3.12+ is fine) and **`[project.optional-dependencies]`** (`>=` **minimums** per extra: `pipeline`, `training`, `collection`, `video`, `inference`). That file declares compatibility; it does **not** replace the lock for `./local.sh deps`.

**Default install (matches the lock):** `./local.sh deps` or `./local.sh setup` → **`scripts/install_deps.sh`** installs the selected profile from **`requirements.lock`** (default profile is **pipeline**), then editable-installs the package with **`--no-deps`** so the venv matches the lock. Narrower installs: `DEPS_EXTRA=collection ./local.sh deps`, etc. The `aid-*` wrappers land in `./.venv/bin`; if imports fail, stderr points you at `./local.sh deps`.

**Refreshing dependencies:** `bash scripts/update_deps_lock.sh` picks **newest stable** PyPI releases for the pipeline profile (see **`scripts/update_deps_lock.py`**; **torch** / **torchvision** series map; **`MANIFEST_MAX_WHEEL_CP`** matches CI Python). Then `python3 scripts/update_deps_lock.py verify --require-current` and commit **both** lock files. Run **`make ci-fast`** to re-verify digests locally. Details: [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), [docs/CI_LOCAL.md](docs/CI_LOCAL.md).

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
- Security reporting: see `SECURITY.md`.
- Do not commit secrets (tokens, keys) or private datasets.
- Dataset and model licenses vary by source; verify each source license before commercial or production use.
- Detection outputs are probabilistic and can be wrong; review high-risk decisions with human oversight.
