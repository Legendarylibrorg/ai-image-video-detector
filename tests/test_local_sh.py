from __future__ import annotations

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
        self.assertIn("./local.sh setup", out)
        self.assertIn("./local.sh smoke", out)
        self.assertIn("./local.sh run", out)
        self.assertIn("./local.sh check", out)


if __name__ == "__main__":
    unittest.main()
