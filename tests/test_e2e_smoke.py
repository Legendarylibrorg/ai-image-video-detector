"""End-to-end smoke: full training/eval/report path in a temp directory.

Runs `scripts/smoke_resume_eval.sh` (synthetic data, tiny epochs, no malware scan).
Disabled by default so `python -m unittest discover` stays fast locally.

Enable explicitly (default venv is ``./.venv``, same as ``scripts/install_deps.sh``):

  AID_E2E_SMOKE=1 ./.venv/bin/python -m unittest tests.test_e2e_smoke

Use a different virtualenv (absolute path, or path relative to repo root) without
touching ``./.venv``:

  VENV_DIR=/path/to/venv AID_E2E_SMOKE=1 python -m unittest tests.test_e2e_smoke

GitHub Actions runs the fast suite on every PR (``.github/workflows/tests.yml``) and can run this
case weekly or manually via ``.github/workflows/e2e-smoke.yml``. Locally, opt in with
``AID_E2E_SMOKE=1`` before release.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from tests._support import ROOT


def _resolved_venv_dir() -> Path:
    """Virtualenv directory for smoke (``VENV_DIR`` or repo ``./.venv``)."""
    raw = os.environ.get("VENV_DIR", "").strip()
    if not raw:
        return (ROOT / ".venv").resolve()
    candidate = Path(raw)
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


@unittest.skipUnless(
    os.environ.get("AID_E2E_SMOKE", "").strip() in {"1", "true", "yes"},
    "set AID_E2E_SMOKE=1 to run the full smoke_resume_eval pipeline (several minutes)",
)
class E2ESmokeTests(unittest.TestCase):
    def test_smoke_resume_eval_script_exits_zero(self) -> None:
        venv_dir = _resolved_venv_dir()
        venv_py = venv_dir / "bin" / "python"
        if not venv_py.is_file():
            self.skipTest(
                "missing venv python at "
                f"{venv_py}; set VENV_DIR or run "
                "`VENV_DIR=... bash scripts/install_deps.sh` / `./local.sh deps`"
            )

        env = {**os.environ, "VENV_DIR": str(venv_dir)}
        env["PATH"] = str(venv_dir / "bin") + os.pathsep + env.get("PATH", "")
        proc = subprocess.run(
            ["bash", "scripts/smoke_resume_eval.sh"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if proc.returncode != 0:
            sys.stderr.write("--- stdout ---\n" + (proc.stdout or ""))
            sys.stderr.write("--- stderr ---\n" + (proc.stderr or ""))
        self.assertEqual(proc.returncode, 0, "smoke_resume_eval.sh failed")
        self.assertIn("smoke_ok", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
