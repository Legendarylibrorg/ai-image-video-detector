from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DocsSurfaceTests(unittest.TestCase):
    def test_readme_points_to_split_startup_docs(self) -> None:
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("[docs/STARTUP.md](docs/STARTUP.md)", text)
        self.assertIn("[docs/COMMANDS.md](docs/COMMANDS.md)", text)
        self.assertIn("[docs/REFERENCE.md](docs/REFERENCE.md)", text)
        self.assertIn("local Linux machine", text)
        self.assertIn("local virtualenv at `./.venv`", text)
        self.assertIn("sudo apt-get update", text)
        self.assertIn("./local.sh setup", text)
        self.assertIn("./local.sh run", text)
        self.assertNotIn("## Advanced Reference", text)

    def test_startup_doc_marks_sudo_only_for_system_commands(self) -> None:
        text = (ROOT / "docs" / "STARTUP.md").read_text(encoding="utf-8")
        self.assertIn("sudo apt-get update", text)
        self.assertIn(
            "sudo apt-get install -y python3 python3-venv python3-pip build-essential clamav clamav-daemon",
            text,
        )
        self.assertIn("Do not use `sudo` for repo commands", text)
        self.assertIn("pinned local virtualenv at `./.venv`", text)

    def test_commands_doc_starts_with_linux_quick_start(self) -> None:
        text = (ROOT / "docs" / "COMMANDS.md").read_text(encoding="utf-8")
        self.assertIn("The default path below assumes Linux", text)
        self.assertIn("repo-local Python environment is `./.venv`", text)
        self.assertIn("sudo apt-get update", text)
        self.assertIn("./local.sh setup", text)
        self.assertIn("./local.sh run", text)


if __name__ == "__main__":
    unittest.main()
