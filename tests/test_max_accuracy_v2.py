from __future__ import annotations

import unittest

from _support import ROOT


class MaxAccuracyV2Tests(unittest.TestCase):
    def test_refine_loop_keeps_collection_hf_only(self) -> None:
        script = (ROOT / "scripts" / "max_accuracy_v2.sh").read_text()
        self.assertIn("BEST_DS_HF_ONLY=1 bash scripts/do.sh collect-image", script)
        self.assertNotIn("BEST_DS_LOCAL_SOURCES=", script)


if __name__ == "__main__":
    unittest.main()
