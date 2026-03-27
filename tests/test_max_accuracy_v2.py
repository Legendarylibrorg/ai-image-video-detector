from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MaxAccuracyV2Tests(unittest.TestCase):
    def test_refine_loop_keeps_collection_hf_only(self) -> None:
        script = (ROOT / "scripts" / "max_accuracy_v2.sh").read_text()
        self.assertIn("BEST_DS_HF_ONLY=1 bash scripts/do.sh collect-image", script)
        self.assertNotIn("BEST_DS_LOCAL_SOURCES=", script)

    def test_wrapper_discovers_current_ensemble_members_dynamically(self) -> None:
        script = (ROOT / "scripts" / "max_accuracy_v2.sh").read_text()
        self.assertIn("collect_ensemble_model_paths()", script)
        self.assertIn('for model_dir in "$ENS_OUT"/m*; do', script)
        self.assertIn('--model "${ENSEMBLE_MODELS[@]}"', script)
        self.assertNotIn('"$ENS_OUT"/m4/best.safetensors', script)

    def test_wrapper_uses_repo_python_helpers(self) -> None:
        script = (ROOT / "scripts" / "max_accuracy_v2.sh").read_text()
        self.assertIn('source "$ROOT_DIR/scripts/lib/core.sh"', script)
        self.assertIn("load_env_file", script)
        self.assertIn("run_repo_python scripts/fit_domain_thresholds.py", script)
        self.assertIn("run_repo_python scripts/mine_hard_negatives.py", script)
        self.assertNotRegex(script, r"(?m)^\\s*python scripts/fit_domain_thresholds\\.py\\b")
        self.assertNotRegex(script, r"(?m)^\\s*python scripts/mine_hard_negatives\\.py\\b")


if __name__ == "__main__":
    unittest.main()
