from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LocalShTests(unittest.TestCase):
    def test_help_shows_simple_workflow(self) -> None:
        proc = subprocess.run(
            ["bash", "./local.sh", "help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        out = proc.stdout
        self.assertIn("linux setup", out.lower())
        self.assertIn("sudo apt-get update", out)
        self.assertIn("./local.sh setup", out)
        self.assertIn("./local.sh deps", out)
        self.assertIn("./local.sh doctor", out)
        self.assertIn("./local.sh smoke", out)
        self.assertIn("./local.sh smoke-real", out)
        self.assertIn("./local.sh run", out)
        self.assertIn("./local.sh status", out)
        self.assertIn("repo-local venv", out.lower())
        self.assertNotIn("advanced aliases still work", out.lower())
        self.assertNotIn("detect <image>", out)

    def test_setup_uses_linux_setup_path_in_dry_run(self) -> None:
        proc = subprocess.run(
            ["bash", "./local.sh", "setup"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "DRY_RUN": "1",
                "SETUP_INSTALL_SYSTEM_DEPS": "0",
                "SETUP_PROMPT_FOR_HF_TOKEN": "0",
                "HF_SETUP_REQUIRE_TOKEN": "0",
            },
        )

        out = proc.stdout
        self.assertIn("setup_stage=python_deps status=run", out)
        self.assertIn("[DRY_RUN] bash scripts/install_deps.sh", out)
        self.assertIn("[DRY_RUN] bash scripts/doctor.sh", out)
        self.assertIn("setup_status=ready", out)

    def test_deps_command_runs_install_script(self) -> None:
        proc = subprocess.run(
            ["bash", "./local.sh", "deps"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("deps_status=up_to_date", proc.stdout)

    def test_collect_status_stdout_is_valid_json(self) -> None:
        proc = subprocess.run(
            ["bash", "./local.sh", "collect-status"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        payload = json.loads(proc.stdout)
        self.assertIn("data_root", payload)
        self.assertTrue(proc.stdout.lstrip().startswith("{"))


if __name__ == "__main__":
    unittest.main()
