from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]

# Minimal full-pipeline env so dry-run completes in low-disk / CI sandboxes.
_PIPELINE_DRY_RUN_LIGHT_ENV: dict[str, str] = {
    "DRY_RUN": "1",
    "MALWARE_SCAN": "0",
    "PIPELINE_MIN_FREE_GB": "0",
    "SKIP_DATA": "1",
    "RUN_VIDEO_DATA_PULL": "0",
    "SKIP_SWEEP": "1",
    "RUN_HARD_MINING": "0",
    "RUN_DISTILL": "0",
    "RUN_ENSEMBLE_FIT": "0",
    "RUN_DOMAIN_THRESHOLDS": "0",
    "RUN_ROBUST_EVAL": "0",
    "RUN_METADATA_MEMBER": "0",
}


class PipelineWrapperTests(unittest.TestCase):
    def test_pipeline_scripts_reuse_shared_repo_python_helper(self) -> None:
        full_text = (ROOT / "scripts" / "full_pipeline_4090.sh").read_text(encoding="utf-8")
        smoke_text = (ROOT / "scripts" / "smoke_resume_eval.sh").read_text(encoding="utf-8")
        core_text = (ROOT / "scripts" / "lib" / "core.sh").read_text(encoding="utf-8")
        self.assertIn('source "$ROOT_DIR/scripts/lib/hf_default_queries.inc.sh"', core_text)
        self.assertIn("repo_python() {", core_text)
        self.assertIn("ensure_env() {", core_text)
        self.assertIn('source "$ROOT_DIR/scripts/lib/core.sh"', full_text)
        self.assertIn("read_aid_csv_cli_buf --hf-query", full_text)
        self.assertIn("run_cmd repo_python scripts/fit_ensemble.py", full_text)
        self.assertIn("repo_python scripts/eval_test_ensemble.py", full_text)
        self.assertIn("repo_python -m ai_image_detector.robust_eval", full_text)
        self.assertNotIn("trim_ws() {", full_text)
        self.assertNotIn("repo_python() {", full_text)
        self.assertNotIn("activate_repo_venv()", full_text)
        self.assertIn("\nensure_env\n", full_text)
        self.assertIn('source "$ROOT_DIR/scripts/lib/core.sh"', smoke_text)
        self.assertIn('source "$ROOT_DIR/scripts/lib/training.sh"', smoke_text)
        self.assertIn("\nensure_env\n", smoke_text)
        self.assertIn("prepare_training_image_data", smoke_text)
        self.assertIn("run_prepared_max_quality_pipeline", smoke_text)
        self.assertIn("run_benchmark_gate", smoke_text)
        self.assertNotIn("scripts/fit_ensemble.py", smoke_text)
        self.assertNotIn("repo_python() {", smoke_text)

    def test_core_trims_csv_without_xargs(self) -> None:
        core_text = (ROOT / "scripts" / "lib" / "core.sh").read_text(encoding="utf-8")
        self.assertIn("trim_ws() {", core_text)
        self.assertIn('value="$(trim_ws "$value")"', core_text)
        self.assertIn("read_aid_csv_cli_buf() {", core_text)
        self.assertIn("print_cli_flag_values_from_csv() {", core_text)
        self.assertNotIn('echo "$q" | xargs', core_text)
        self.assertNotIn('echo "$value" | xargs', core_text)

    def test_hf_query_default_is_single_sourced(self) -> None:
        full_text = (ROOT / "scripts" / "full_pipeline_4090.sh").read_text(encoding="utf-8")
        do_text = (ROOT / "scripts" / "do.sh").read_text(encoding="utf-8")
        inc_text = (ROOT / "scripts" / "lib" / "hf_default_queries.inc.sh").read_text(encoding="utf-8")
        marker = "synthetic portrait dataset"
        self.assertEqual(inc_text.count(marker), 1)
        self.assertEqual(do_text.count(marker), 0)
        self.assertEqual(full_text.count(marker), 0)
        self.assertIn(
            'BEST_DS_HF_QUERIES="${BEST_DS_HF_QUERIES:-$BEST_HF_QUERY_CSV_DEFAULT}"',
            full_text,
        )
        self.assertIn(": \"${BEST_HF_QUERY_CSV_DEFAULT:=", inc_text)
        self.assertNotIn("BEST_HF_QUERY_CSV_DEFAULT=", do_text)

    def test_full_pipeline_dry_run_allows_missing_venv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_venv = Path(tmpdir) / "missing-venv"
            env = os.environ.copy()
            env.update(_PIPELINE_DRY_RUN_LIGHT_ENV)
            env["VENV_DIR"] = str(missing_venv)
            proc = subprocess.run(
                ["bash", "scripts/full_pipeline_4090.sh"],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("[DRY_RUN] bash scripts/install_deps.sh", proc.stdout)
        self.assertIn(f"[DRY_RUN] source {missing_venv / 'bin' / 'activate'}", proc.stdout)
        self.assertIn("Pipeline complete.", proc.stdout)

    def test_local_collect_status_executes_without_bootstrapping_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_root = tmp / "data_best"
            prepared_root = tmp / "prepared"
            video_root = tmp / "video_data"
            (data_root / "train" / "ai").mkdir(parents=True, exist_ok=True)

            proc = subprocess.run(
                ["bash", "./local.sh", "collect-status"],
                cwd=ROOT,
                env={
                    **os.environ,
                    "DATA_DIR": str(data_root),
                    "TRAIN_READY_DATA_DIR": str(prepared_root),
                    "VIDEO_OUT": str(video_root),
                    "VENV_DIR": str(tmp / "missing-venv"),
                },
                check=True,
                capture_output=True,
                text=True,
            )

        payload = json.loads(proc.stdout)
        self.assertEqual(payload["data_root"], str(data_root.resolve()))
        self.assertEqual(payload["prepared_training_root"]["path"], str(prepared_root.resolve()))
        self.assertEqual(payload["video_root"]["path"], str(video_root.resolve()))
        self.assertNotIn("install_deps.sh", proc.stdout + proc.stderr)

if __name__ == "__main__":
    unittest.main()
