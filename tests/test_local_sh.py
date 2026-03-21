from __future__ import annotations

import json
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
        self.assertIn("linux quick start", out.lower())
        self.assertIn("sudo apt-get update", out)
        self.assertIn("./local.sh setup", out)
        self.assertIn("./local.sh smoke", out)
        self.assertIn("./local.sh smoke-real", out)
        self.assertIn("./local.sh run", out)
        self.assertIn("./local.sh status", out)
        self.assertIn("repo-local venv", out.lower())
        self.assertIn("advanced aliases still work", out.lower())
        self.assertNotIn("detect <image>", out)

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
