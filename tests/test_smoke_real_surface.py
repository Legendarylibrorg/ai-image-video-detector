from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SmokeRealSurfaceTests(unittest.TestCase):
    def test_smoke_real_script_exercises_real_hf_and_cuda_paths(self) -> None:
        text = (ROOT / "scripts" / "smoke_real_stack.sh").read_text(encoding="utf-8")
        self.assertIn("scripts/build_best_dataset.py", text)
        self.assertIn("aid-train", text)
        self.assertIn("HF_TOKEN", text)
        self.assertIn("torch.cuda.is_available()", text)
        self.assertIn("dragonintelligence/CIFAKE-image-dataset", text)


if __name__ == "__main__":
    unittest.main()
