# Agent and contributor orientation

Short guide for humans and **coding agents** working in this repository. Deep detail lives in [docs/REFERENCE.md](docs/REFERENCE.md).

## Read first

1. **[docs/REFERENCE.md](docs/REFERENCE.md)** — **Architecture at a glance** (operator → `do.sh` → drivers → library), `src/` vs `scripts/`, env vars, pipeline diagram.
2. **[docs/COMMANDS.md](docs/COMMANDS.md)** — Canonical **`./local.sh`** / **`scripts/do.sh`** command map.
3. **[README.md](README.md)** — Summary, first-time flow, **Verify wiring** commands, Compose notes.

## Rules of thumb

- Use **`./local.sh …`** or **`bash scripts/do.sh …`** for pipeline work so **`PYTHONPATH`** (`./src` + `./scripts`) and the venv match **`run_repo_python`**. Avoid bare **`python3 scripts/some_driver.py`** from an arbitrary cwd (see REFERENCE, *Python modules*).
- **Bootstrap** commands **`./local.sh setup`**, **`deps`**, and **`docker-doctor`** do **not** go through **`do.sh`**; they must work before a full pipeline session exists.
- Prefer **`./local.sh deps`** (or **`setup`**) over **`pip install -e '.[pipeline]'`** when you need the same pins as **`requirements.lock`** / CI.
- Image training code: **`train.py`** (CLI) → **`train_main.py`** (loop) + **`train_support.py`** + **`train_run_artifacts.py`** + **`train_post.py`**.
- Before a PR: **`python3 -m unittest discover -s tests -p 'test_*.py'`** and **`ruff check src/ai_image_detector tests`** (same as CI; see [CONTRIBUTING.md](CONTRIBUTING.md) for install and shell checks).

## Security

Do not open public issues for vulnerabilities. Follow **[SECURITY.md](SECURITY.md)**. Collection, checkpoints, and path rules are bounded by **`AID_*`** env vars documented there.
