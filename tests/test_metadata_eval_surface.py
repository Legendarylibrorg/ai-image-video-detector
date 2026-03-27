from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from _support import ROOT, write_rgb_image  # noqa: F401
from ai_image_detector.ensemble import metadata_features_from_paths


class MetadataEvalSurfaceTests(unittest.TestCase):
    def test_metadata_helper_builds_batch_tensor_from_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            img1 = Path(tmp) / "one.png"
            img2 = Path(tmp) / "two.jpg"
            write_rgb_image(img1)
            write_rgb_image(img2)

            features = metadata_features_from_paths([str(img1), str(img2)], device=torch.device("cpu"))

        self.assertEqual(features.shape[0], 2)
        self.assertGreater(features.shape[1], 0)

    def test_eval_and_refinement_scripts_support_metadata_aware_models(self) -> None:
        fit_text = (ROOT / "scripts" / "fit_ensemble.py").read_text(encoding="utf-8")
        eval_text = (ROOT / "scripts" / "eval_test_ensemble.py").read_text(encoding="utf-8")
        domain_text = (ROOT / "scripts" / "fit_domain_thresholds.py").read_text(encoding="utf-8")
        robust_text = (ROOT / "src" / "ai_image_detector" / "robust_eval.py").read_text(encoding="utf-8")
        hard_text = (ROOT / "scripts" / "mine_hard_negatives.py").read_text(encoding="utf-8")
        distill_text = (ROOT / "scripts" / "train_distill.py").read_text(encoding="utf-8")

        self.assertIn("loaded.uses_metadata_features", fit_text)
        self.assertIn("MetadataImageFolder", fit_text)
        self.assertIn("metadata_features=metadata_features", fit_text)

        self.assertIn("loaded.uses_metadata_features", eval_text)
        self.assertIn("MetadataImageFolder", eval_text)
        self.assertIn("metadata_features=metadata_features", eval_text)

        self.assertIn("metadata_features_from_paths", domain_text)
        self.assertIn("loaded.uses_metadata_features", domain_text)

        self.assertIn("metadata_features_from_paths", robust_text)
        self.assertIn("loaded.uses_metadata_features", robust_text)

        self.assertIn("metadata_features_from_paths", hard_text)
        self.assertIn("loaded.uses_metadata_features", hard_text)

        self.assertIn("loaded.uses_metadata_features", distill_text)
        self.assertIn("MetadataImageFolder", distill_text)
        self.assertIn("teacher(x, metadata_features=metadata_features)", distill_text)


if __name__ == "__main__":
    unittest.main()
