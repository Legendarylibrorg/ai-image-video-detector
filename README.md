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

## Get the code and run

Long-form options (**`git clone`**, **`curl` + `tar`**, **`install.sh`**, safer **`INSTALL_REV`**, fork/mirror flags) live in [docs/STARTUP.md](docs/STARTUP.md). Copy-paste **Compose** and stage-by-stage **`./local.sh`** blocks live in [docs/COMMANDS.md](docs/COMMANDS.md)—this README does not repeat them.

Typical **native** flow from the repo root after you have a source tree:

```bash
./local.sh setup
printf "HF_TOKEN='your_token_here'\n" >> .env
./local.sh smoke
./local.sh run
./local.sh status
```

Use a Hugging Face **read** token unless you need write access. On native Linux, `hf auth login` also works; `./.env` is the simplest path for this repo and for Docker Compose.

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

Prefer a **dedicated Linux VM**, then Docker Compose on that VM—not on the bare metal laptop you use for email. Compose is **not** a VM; the VM is the main isolation boundary. Inside the VM you want **Docker Engine**, the **Compose plugin**, and **NVIDIA Container Toolkit** for **`pipeline-gpu`**. You do **not** need host Python or a host **`./.venv`** for the container path.

Paths: host repo root → container **`/workspace`**; deps and CLIs → **`/opt/aid-venv`**; **`./.env`** on the host supplies **`HF_TOKEN`** to Compose. Writable data/artifact roots include **`./.local`**, **`./data_best`**, **`./data_new`**, **`./video_data`**, **`./artifacts_ens`**, and the other dirs under **Repo layout** below. **`pipeline`** is CPU-only; **`pipeline-gpu`** uses **`Dockerfile.gpu`**.

**Exact** clone → **`docker compose build`** → CPU/GPU doctor/smoke/run commands: [docs/STARTUP.md](docs/STARTUP.md) (**Exact secure startup**) and [docs/COMMANDS.md](docs/COMMANDS.md) (**Dedicated Linux VM + Docker Compose commands**).

`./local.sh setup` uses a lenient disk check for clone-to-smoke; **`./local.sh doctor`** alone still defaults to **40GB** free for full training.

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
