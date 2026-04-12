# Contributing

Thanks for contributing.

## Before You Start

- Open an issue or draft PR when the change is large, user-facing, or changes the training/data flow.
- Do not open public issues for security problems. Use the process in [SECURITY.md](SECURITY.md).
- Keep changes focused. Small PRs are much easier to review and safer to merge.

## Development Setup

Native Linux:

```bash
./local.sh setup
```

Narrower local installs are also supported:

```bash
DEPS_EXTRA=collection ./local.sh deps
DEPS_EXTRA=training ./local.sh deps
```

## Expectations For Changes

- Preserve the existing `ai` vs `real` training/data flow unless the change explicitly updates that contract.
- Avoid committing datasets, model artifacts, caches, or secrets.
- Keep docs in sync when command behavior, paths, or bootstrap flow changes.
- Add or update tests for behavior changes, especially around shell wrappers and install paths.

## Dependencies and lock files

- **Exact runtime versions** live in **`requirements.lock`** and **`requirements.lock.json`** at the repo root. **Do not treat README or `docs/` prose as a version list.**
- **`requirements.lock.json`** pins **one** PyPI artifact per package. CI verifies that exact file’s SHA256. Another OS may still install the same **version** from a different wheel; regenerate on **Linux x86_64** when you need the JSON to match CI (see [SECURITY.md](SECURITY.md) supply-chain section).
- If you change **`pyproject.toml`** optional dependencies or `requires-python`, run **`bash scripts/update_deps_lock.sh`**, then **`python3 scripts/update_deps_lock.py verify --require-current`**, and commit **both** lock files in the same PR. When you bump CI’s Python in **`.github/ci-python-version.txt`**, also bump **`MANIFEST_MAX_WHEEL_CP`** in **`scripts/update_deps_lock.py`** to the same minor, then refresh the lock again.
- **`pip install -e '.[pipeline]'`** alone follows **`pyproject.toml`** minimums only; **`./local.sh deps`** matches the lock.

## GitHub automation (layout)

This repository is **Python + shell + GitHub Actions** only (no application JavaScript/TypeScript, no npm lockfile). CI is scoped to that stack.

**CI Python version:** single source **`.github/ci-python-version.txt`** (one line, e.g. `3.14`). **Security Checks** and **Dependency Updates** install it via **`.github/actions/setup-aid-python`**. When you bump CI Python, edit that file and **`MANIFEST_MAX_WHEEL_CP`** in **`scripts/update_deps_lock.py`** to the same minor, then refresh locks (see below).

| What | File | When |
|------|------|------|
| **Security Checks** | `.github/workflows/security.yml` | Every **PR** and **push to `main`**, **weekly** `cron` (Monday 14:00 UTC), **manual** (`workflow_dispatch`). Verifies lock digests, runs secret scan + `pip-audit`. |
| **Code scanning (Python)** | `.github/workflows/codeql.yml` | **PRs** and **pushes to `main`**, **weekly** `cron` (Monday 06:30 UTC), **manual**. CodeQL **Python only**, `build-mode: none` (no JS/TS job). |
| **Dependency Updates** | `.github/workflows/deps-update.yml` | **Daily** `cron` (13:00 UTC) + **manual** only. Refreshes `requirements.lock` / `requirements.lock.json` and opens a PR if they change. |
| **Dependabot** | `.github/dependabot.yml` | **Once per day** grouped PRs for **pip** (`/` manifests) and **github-actions** only. |

### CodeQL default setup vs this workflow

If **GitHub → Settings → Code security → Code scanning** shows **Default setup** *and* you see extra jobs (e.g. **Analyze (javascript-typescript)**) or duplicate CodeQL runs, **disable Default setup** or **Edit** it to **Python only** and rely on `.github/workflows/codeql.yml`. One configuration avoids noise and failed no-op language jobs.

## Checks To Run

Run the full test suite before opening a PR:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

Useful targeted checks:

```bash
bash -n local.sh install.sh scripts/install_deps.sh scripts/doctor.sh scripts/lib/apt_packages_validate.sh scripts/lib/core.sh scripts/lib/training.sh scripts/verify_secrets_scan.sh
python3 -m py_compile src/ai_image_detector/cli.py src/ai_image_detector/checkpoint_io.py scripts/lib/install_validate.py
python3 scripts/update_deps_lock.py verify --require-current
bash scripts/verify_secrets_scan.sh
```

`scripts/verify_secrets_scan.sh` matches **CI** / **`.pre-commit-config.yaml`**: ephemeral venv, **`detect-secrets==1.5.0`**, scan with **`.secrets.baseline`** (no PEP 668 changes to your system Python). Also run **`git grep -E 'sk-|ghp_|AKIA[0-9A-Z]{16}'`** before pushing.

If you use pre-commit, install it and run:

```bash
pre-commit run --all-files
```

## Pull Requests

- Branch from `main`.
- Explain what changed and why.
- Call out behavior changes, dependency changes, and docs updates.
- Mention any skipped checks or environment-specific limitations.

The PR template in [`.github/pull_request_template.md`](.github/pull_request_template.md) is the default checklist.
