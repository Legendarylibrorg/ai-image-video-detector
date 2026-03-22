from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class InstallShTests(unittest.TestCase):
    def test_install_script_supports_repo_bootstrap_without_zip(self) -> None:
        text = (ROOT / "install.sh").read_text(encoding="utf-8")
        self.assertIn("git clone --depth 1", text)
        self.assertIn("python3 -m venv .venv", text)
        self.assertIn("source .venv/bin/activate", text)
        self.assertIn("./local.sh deps", text)
        self.assertIn("./local.sh doctor", text)
        self.assertIn("install_status=ready", text)

    def test_install_script_dry_run_works_inside_repo(self) -> None:
        proc = subprocess.run(
            ["bash", "./install.sh"],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "1",
                "INSTALL_SYSTEM_DEPS": "0",
                "INSTALL_ASSUME_LINUX": "1",
            },
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("install_stage=repo status=using_current", proc.stdout)
        self.assertIn("[DRY_RUN] cd", proc.stdout)
        self.assertIn("install_stage=venv status=", proc.stdout)
        self.assertIn("source .venv/bin/activate", proc.stdout)
        self.assertIn("./local.sh deps", proc.stdout)
        self.assertIn("./local.sh doctor", proc.stdout)
        self.assertIn("install_status=ready", proc.stdout)


if __name__ == "__main__":
    unittest.main()
