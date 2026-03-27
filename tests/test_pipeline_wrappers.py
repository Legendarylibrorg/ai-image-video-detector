from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PipelineWrapperTests(unittest.TestCase):
    def test_pipeline_scripts_use_repo_python_helper_for_internal_python_calls(self) -> None:
        full_text = (ROOT / "scripts" / "full_pipeline_4090.sh").read_text(encoding="utf-8")
        smoke_text = (ROOT / "scripts" / "smoke_resume_eval.sh").read_text(encoding="utf-8")
        self.assertIn("repo_python() {", full_text)
        self.assertIn("run_cmd repo_python scripts/fit_ensemble.py", full_text)
        self.assertIn("repo_python scripts/eval_test_ensemble.py", full_text)
        self.assertIn("repo_python -m ai_image_detector.robust_eval", full_text)
        self.assertIn("repo_python() {", smoke_text)
        self.assertIn("repo_python scripts/benchmark_gate.py", smoke_text)
        self.assertIn("repo_python -m ai_image_detector.robust_eval", smoke_text)

    def test_full_pipeline_dry_run_allows_missing_venv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_venv = Path(tmpdir) / "missing-venv"
            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "MALWARE_SCAN": "0",
                    "VENV_DIR": str(missing_venv),
                }
            )
            proc = subprocess.run(
                ["bash", "scripts/full_pipeline_4090.sh"],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn(f"[DRY_RUN] source {missing_venv / 'bin' / 'activate'}", proc.stdout)
        self.assertIn("Pipeline complete.", proc.stdout)

if __name__ == "__main__":
    unittest.main()
