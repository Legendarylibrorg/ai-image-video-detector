#!/usr/bin/env python3
"""Run the ai-image-video-detector quality gate locally (see docs/CI_LOCAL.md).

Jobs (former .github/workflows/*.yml):

  1. test         — install_deps.sh, Ruff, unittest discover
  2. security     — lock digest verify, detect-secrets, pip-audit
  3. e2e-smoke    — locked install + AID_E2E_SMOKE unittest (heavy)

Usage:
  python3 scripts/run_ci_local.py              # all jobs
  python3 scripts/run_ci_local.py --list       # show jobs and recommended matrix
  python3 scripts/run_ci_local.py --job test
  python3 scripts/run_ci_local.py --fast       # test + security (skip e2e-smoke)
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

CI_DOC = "docs/CI_LOCAL.md"
CI_PYTHON_FILE = ".github/ci-python-version.txt"
RUFF_VERSION = "0.15.14"

CI_ENV: dict[str, str] = {
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
}

ALL_JOBS = ("test", "security", "e2e-smoke")
FAST_JOBS = ("test", "security")

SECRET_SCAN_SHELL = r"""
set -euo pipefail
if [ -f .secrets.baseline ]; then
  git ls-files -z | xargs -0 detect-secrets-hook --baseline .secrets.baseline \
    --exclude-files '^requirements\.lock\.json$'
else
  detect-secrets scan . > .secrets.baseline
  git diff --exit-code .secrets.baseline
fi
"""


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ci_python_version(root: Path) -> str:
    path = root / CI_PYTHON_FILE
    if not path.is_file():
        return "unknown"
    token = "".join(path.read_text(encoding="utf-8").split())
    return token or "unknown"


def venv_python(root: Path) -> Path:
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def _ci_env(*, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.update(CI_ENV)
    if extra:
        env.update(extra)
    return env


def _banner(title: str) -> None:
    width = max(len(title) + 4, 72)
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _step(title: str, cmd: Sequence[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(f"\n--- {title} ---")
    print("$", " ".join(cmd))
    subprocess.run(list(cmd), cwd=str(cwd), env=env, check=True)


def _shell_step(title: str, script: str, *, cwd: Path, env: dict[str, str]) -> None:
    print(f"\n--- {title} ---")
    print(script.strip())
    subprocess.run(["bash", "-c", script], cwd=str(cwd), env=env, check=True)


def job_test(root: Path, env: dict[str, str]) -> None:
    py = venv_python(root)
    ci_py = ci_python_version(root)
    _banner(f"Job: test (CI Python {ci_py}, host {platform.system()})")

    _step(
        "Install dependencies from lock",
        ["bash", "scripts/install_deps.sh"],
        cwd=root,
        env=env,
    )
    if not py.is_file():
        raise SystemExit(f"missing venv python at {py}; install_deps.sh should create .venv")

    _step(
        "Install Ruff",
        [str(py), "-m", "pip", "install", "--progress-bar", "off", f"ruff=={RUFF_VERSION}"],
        cwd=root,
        env=env,
    )
    ruff = py.parent / ("ruff.exe" if os.name == "nt" else "ruff")
    _step(
        "Ruff lint",
        [str(ruff), "check", "src/ai_image_detector", "tests"],
        cwd=root,
        env=env,
    )
    _step(
        "Unittest discover",
        [str(py), "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"],
        cwd=root,
        env=env,
    )


def job_security(root: Path, env: dict[str, str]) -> None:
    python3 = os.environ.get("PYTHON_BIN") or shutil.which("python3") or sys.executable
    _banner("Job: security (lock + secrets + pip-audit)")

    _step(
        "Install security tools",
        [
            python3,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "detect-secrets>=1.5.0",
            "pip-audit>=2.10.0",
        ],
        cwd=root,
        env=env,
    )
    _step(
        "Verify lock digests vs PyPI",
        [python3, "scripts/update_deps_lock.py", "verify", "--require-current"],
        cwd=root,
        env=env,
    )
    _shell_step("Secret scan", SECRET_SCAN_SHELL, cwd=root, env=env)
    lock = root / "requirements.lock"
    if lock.is_file():
        _step(
            "Audit locked dependencies",
            [python3, "-m", "pip_audit", "-r", str(lock)],
            cwd=root,
            env=env,
        )
    else:
        _step("Audit dependencies", [python3, "-m", "pip_audit"], cwd=root, env=env)


def job_e2e_smoke(root: Path, env: dict[str, str]) -> None:
    py = venv_python(root)
    _banner("Job: e2e-smoke (AID_E2E_SMOKE)")

    _step(
        "Install dependencies from lock",
        ["bash", "scripts/install_deps.sh"],
        cwd=root,
        env=env,
    )
    if not py.is_file():
        raise SystemExit(f"missing venv python at {py}; install_deps.sh should create .venv")

    smoke_env = _ci_env(extra={"AID_E2E_SMOKE": "1"})
    _step(
        "Run smoke_resume_eval (AID_E2E_SMOKE)",
        [str(py), "-m", "unittest", "tests.test_e2e_smoke", "-v"],
        cwd=root,
        env=smoke_env,
    )


def _print_plan(*, jobs: Sequence[str]) -> None:
    root = repo_root()
    host = platform.system()
    ci_py = ci_python_version(root)
    print(f"Quality gate: {CI_DOC}")
    print(f"Local host: {host}  |  CI Python pin: {ci_py} ({CI_PYTHON_FILE})")
    print("\nRecommended matrix (run --fast before release merges):")
    print(f"  - Linux, Python {ci_py}")
    print(f"  - macOS, Python {ci_py}")
    print("\nAll local jobs:")
    for job in ALL_JOBS:
        print(f"  - {job}")
    print("\nLocal run plan:")
    for job in jobs:
        print(f"  - {job}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local quality gate (docs/CI_LOCAL.md).")
    parser.add_argument(
        "--job",
        choices=ALL_JOBS,
        action="append",
        help="Run one job (repeatable). Default: all jobs.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run test + security only (skip e2e-smoke).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print recommended matrix vs local plan and exit.",
    )
    args = parser.parse_args(argv)

    root = repo_root()

    if args.fast:
        jobs = list(FAST_JOBS)
    elif args.job:
        jobs = list(dict.fromkeys(args.job))
    else:
        jobs = list(ALL_JOBS)

    if args.list:
        _print_plan(jobs=jobs)
        return 0

    _print_plan(jobs=jobs)
    env = _ci_env()
    success = False

    try:
        for job in jobs:
            if job == "test":
                job_test(root, env)
            elif job == "security":
                job_security(root, env)
            elif job == "e2e-smoke":
                job_e2e_smoke(root, env)
            else:
                raise SystemExit(f"unknown job: {job}")
        success = True
    except subprocess.CalledProcessError as exc:
        print(f"\nLocal quality gate failed (exit {exc.returncode}).", file=sys.stderr)
        return exc.returncode

    if not success:
        return 1

    print("\nOK: run_ci_local.py finished successfully (all selected jobs passed).")
    if len(jobs) == len(ALL_JOBS):
        print(f"See {CI_DOC} for the recommended cross-platform matrix before large merges.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
