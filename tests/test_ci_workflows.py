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

    def test_smoke_script_exercises_end_to_end_pipeline_outputs(self) -> None:
        text = (ROOT / "scripts" / "smoke_resume_eval.sh").read_text(encoding="utf-8")
        self.assertIn("scripts/prepare_training_data.py", text)
        self.assertIn("bash scripts/train_ensemble.sh", text)
        self.assertIn("scripts/fit_ensemble.py", text)
        self.assertIn("scripts/fit_domain_thresholds.py", text)
        self.assertIn("scripts/eval_test_ensemble.py", text)
        self.assertIn("ai_image_detector.robust_eval", text)
        self.assertIn("scripts/write_pipeline_report.py dataset", text)
        self.assertIn("scripts/write_pipeline_report.py final", text)
        self.assertIn("scripts/benchmark_gate.py", text)
        self.assertIn("final_run_summary.json", text)
        self.assertIn("run_manifest.json", text)
        self.assertIn("prod_manifest.json", text)
        self.assertIn("robust_eval.json", text)


if __name__ == "__main__":
    unittest.main()
