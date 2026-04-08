"""End-to-end smoke: full training/eval/report path in a temp directory.

Runs `scripts/smoke_resume_eval.sh` (synthetic data, tiny epochs, no malware scan).
Disabled by default so `python -m unittest discover` stays fast locally.

Enable explicitly:
  AID_E2E_SMOKE=1 .venv/bin/python -m unittest tests.test_e2e_smoke

CI sets `AID_E2E_SMOKE=1` in `.github/workflows/smoke.yml`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from _support import ROOT


@unittest.skipUnless(
    os.environ.get("AID_E2E_SMOKE", "").strip() in {"1", "true", "yes"},
    "set AID_E2E_SMOKE=1 to run the full smoke_resume_eval pipeline (several minutes)",
)
class E2ESmokeTests(unittest.TestCase):
    def test_smoke_resume_eval_script_exits_zero(self) -> None:
        venv_py = ROOT / ".venv" / "bin" / "python"
        if not venv_py.is_file():
            self.skipTest("missing .venv; run bash scripts/install_deps.sh first")

        # Same entrypoint as CI: repo_python via core.sh expects .venv
        proc = subprocess.run(
            ["bash", "scripts/smoke_resume_eval.sh"],
            cwd=ROOT,
            env={**os.environ, "PATH": str(ROOT / ".venv" / "bin") + os.pathsep + os.environ.get("PATH", "")},
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
