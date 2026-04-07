from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SmokeWorkflowTests(unittest.TestCase):
    def test_smoke_workflow_uses_project_bootstrap(self) -> None:
        text = (ROOT / ".github" / "workflows" / "smoke.yml").read_text(encoding="utf-8")
        self.assertIn("bash scripts/install_deps.sh", text)
        self.assertIn(".venv/bin/python -m unittest discover -s tests -p 'test_*.py'", text)
        self.assertIn("AID_E2E_SMOKE", text)
        self.assertNotIn("source .venv/bin/activate && bash scripts/smoke_resume_eval.sh", text)
        self.assertNotIn("python -m pip install -e .", text)

    def test_smoke_script_exercises_end_to_end_pipeline_outputs(self) -> None:
        text = (ROOT / "scripts" / "smoke_resume_eval.sh").read_text(encoding="utf-8")
        self.assertIn("scripts/prepare_training_data.py", text)
        self.assertIn("scripts/train_ensemble.sh", text)
        self.assertIn('export TRAIN_ENSEMBLE_PROFILE=smoke', text)
        self.assertIn('export RUN_METADATA_MEMBER=0', text)
        self.assertIn("scripts/fit_ensemble.py", text)
        self.assertIn("scripts/fit_domain_thresholds.py", text)
        self.assertIn("scripts/eval_test_ensemble.py", text)
        self.assertIn("ai_image_detector.robust_eval", text)
        self.assertIn("scripts/write_pipeline_report.py dataset", text)
        self.assertIn("scripts/write_pipeline_report.py final", text)
        self.assertIn("scripts/export_best_release.py", text)
        self.assertIn("scripts/benchmark_gate.py", text)
        self.assertIn("final_run_summary.json", text)
        self.assertIn("run_manifest.json", text)
        self.assertIn("prod_manifest.json", text)
        self.assertIn("release_manifest.json", text)
        self.assertIn("robust_eval.json", text)
        self.assertIn("collect_ensemble_model_paths", text)
        self.assertNotIn('"$ENS_OUT/m5_metadata/best.safetensors"', text)
        self.assertNotIn("clone_smoke_member", text)


class SecurityWorkflowTests(unittest.TestCase):
    def test_security_workflow_has_manual_and_scheduled_entrypoints(self) -> None:
        text = (ROOT / ".github" / "workflows" / "security.yml").read_text(encoding="utf-8")
        self.assertIn("schedule:", text)
        self.assertIn('cron: "0 14 * * 1"', text)
        self.assertIn("workflow_dispatch:", text)
        self.assertIn("timeout-minutes: 15", text)

    def test_security_workflow_fails_on_vulnerability_audit(self) -> None:
        text = (ROOT / ".github" / "workflows" / "security.yml").read_text(encoding="utf-8")
        self.assertIn('cache: "pip"', text)
        self.assertIn("cache-dependency-path: requirements.lock", text)
        self.assertIn("pip-audit -r requirements.lock", text)
        self.assertIn("pip-audit", text)
        self.assertNotIn("|| true", text)


if __name__ == "__main__":
    unittest.main()
