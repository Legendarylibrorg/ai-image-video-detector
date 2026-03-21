from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SmokeWorkflowTests(unittest.TestCase):
    def test_smoke_workflow_uses_project_bootstrap(self) -> None:
        text = (ROOT / ".github" / "workflows" / "smoke.yml").read_text(encoding="utf-8")
        self.assertIn("bash scripts/install_deps.sh", text)
        self.assertIn(".venv/bin/python -m unittest discover -s tests -p 'test_*.py'", text)
        self.assertIn("source .venv/bin/activate && bash scripts/smoke_resume_eval.sh", text)
        self.assertNotIn("python -m pip install -e .", text)


if __name__ == "__main__":
    unittest.main()
