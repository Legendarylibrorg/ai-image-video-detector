# Local quality gate

This repository does **not** run GitHub Actions CI workflows on push/PR for tests, security, E2E smoke, or CodeQL. Run the same checks **locally** before opening or updating a pull request.

**Entrypoints:**

```bash
python3 scripts/run_ci_local.py          # all jobs (includes heavy e2e-smoke)
python3 scripts/run_ci_local.py --fast   # test + security (daily PR loop)
python3 scripts/run_ci_local.py --list   # show jobs and recommended matrix
make ci
make ci-fast
```

**CI Python pin:** `.github/ci-python-version.txt` (must match `MANIFEST_MAX_WHEEL_CP` in `scripts/update_deps_lock.py`). Use that interpreter when refreshing locks on Linux x86_64.

---

## Jobs

| Job | Steps (matches former GitHub workflow) |
| --- | --- |
| **test** | `bash scripts/install_deps.sh` → `ruff check src/ai_image_detector tests` → `python -m unittest discover` |
| **security** | `update_deps_lock.py verify --require-current` → `detect-secrets-hook` → `pip-audit -r requirements.lock` |
| **e2e-smoke** | lock install → `AID_E2E_SMOKE=1` `unittest tests.test_e2e_smoke` (runs `scripts/smoke_resume_eval.sh`) |

### Test detail (former `tests.yml`)

1. **Install:** `bash scripts/install_deps.sh` (pipeline profile from `requirements.lock`)
2. **Lint:** `ruff==0.15.14` on `src/ai_image_detector` and `tests`
3. **Tests:** `python -m unittest discover -s tests -p 'test_*.py' -v`

### Security detail (former `security.yml`)

1. **Verify:** `python3 scripts/update_deps_lock.py verify --require-current`
2. **Secrets:** `detect-secrets-hook` with `.secrets.baseline` (excludes `requirements.lock.json`)
3. **Audit:** `python3 -m pip_audit -r requirements.lock`

Also available: `bash scripts/verify_secrets_scan.sh` and `pre-commit run --all-files`.

### E2E smoke (former `e2e-smoke.yml`)

Heavy end-to-end training smoke. Skipped by `--fast`; run before release merges or when changing training/shell paths:

```bash
python3 scripts/run_ci_local.py --job e2e-smoke
# or
AID_E2E_SMOKE=1 ./.venv/bin/python -m unittest tests.test_e2e_smoke -v
```

---

## Recommended matrix

Run **`python3 scripts/run_ci_local.py --fast`** on each row before merging:

| OS | Python |
| --- | --- |
| Linux | from `.github/ci-python-version.txt` |
| macOS | same pin (dev convenience; lock wheels target Linux x86_64) |

---

## Lighter checks

Quick iteration without full lock install:

```bash
ruff check src/ai_image_detector tests
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/update_deps_lock.py verify --require-current
bash scripts/verify_secrets_scan.sh
```

---

## Environment

`run_ci_local.py` sets:

| Variable | Value |
| --- | --- |
| `PIP_DISABLE_PIP_VERSION_CHECK` | `1` |
| `PYTHONUTF8` | `1` |
| `PYTHONIOENCODING` | `utf-8` |

Optional: **`PYTHON_BIN`** to force the interpreter for security-tool installs.

See also [CONTRIBUTING.md](../CONTRIBUTING.md) and [SECURITY.md](../SECURITY.md).
