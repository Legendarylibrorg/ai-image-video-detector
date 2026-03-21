from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class InstallDepsSurfaceTests(unittest.TestCase):
    def test_install_deps_verifies_huggingface_python_and_cli(self) -> None:
        text = (ROOT / "scripts" / "install_deps.sh").read_text(encoding="utf-8")
        self.assertIn("import huggingface_hub", text)
        self.assertIn("command -v hf", text)
        self.assertIn("deps_fail=huggingface_cli_missing", text)


if __name__ == "__main__":
    unittest.main()
