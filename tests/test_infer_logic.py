from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from PIL import Image

from _support import ROOT, write_rgb_image  # noqa: F401
from ai_image_detector.decision import combined_risk, decide_label
from ai_image_detector.metadata import analyze_metadata


class InferLogicTests(unittest.TestCase):
    def test_combined_risk_keeps_model_probability_primary(self) -> None:
        mostly_model = combined_risk(0.80, 0.0, 0.0, 0.0)
        mostly_side_signals = combined_risk(0.20, 1.0, 1.0, 1.0)

        self.assertGreater(mostly_model, mostly_side_signals)
        self.assertAlmostEqual(combined_risk(1.0, 0.0, 0.0, 0.0), 0.84, places=6)

    def test_borderline_predictions_without_ood_are_not_unknown(self) -> None:
        self.assertEqual(
            decide_label(
                0.53,
                threshold=0.50,
                unknown_margin=0.04,
                ood_score=0.10,
                borderline_ood_score=0.45,
                hard_ood_score=0.80,
                ai_unknown_margin=0.03,
                real_unknown_margin=0.05,
            ),
            "AI-generated",
        )
        self.assertEqual(
            decide_label(
                0.47,
                threshold=0.50,
                unknown_margin=0.04,
                ood_score=0.10,
                borderline_ood_score=0.45,
                hard_ood_score=0.80,
                ai_unknown_margin=0.03,
                real_unknown_margin=0.05,
            ),
            "Real",
        )

    def test_borderline_predictions_require_ood_to_be_unknown(self) -> None:
        self.assertEqual(
            decide_label(
                0.52,
                threshold=0.50,
                unknown_margin=0.04,
                ood_score=0.50,
                borderline_ood_score=0.45,
                hard_ood_score=0.80,
                ai_unknown_margin=0.03,
                real_unknown_margin=0.05,
            ),
            "Unknown",
        )
        self.assertEqual(
            decide_label(
                0.67,
                threshold=0.50,
                unknown_margin=0.04,
                ood_score=0.85,
                borderline_ood_score=0.45,
                hard_ood_score=0.80,
                ai_unknown_margin=0.03,
                real_unknown_margin=0.05,
            ),
            "Unknown",
        )

    def test_web_export_formats_get_lighter_missing_metadata_penalties(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            png_path = tmp_path / "sample.png"
            jpg_path = tmp_path / "sample.jpg"
            Image.new("RGB", (64, 64), (64, 128, 192)).save(png_path)
            write_rgb_image(jpg_path)

            png_analysis = analyze_metadata(str(png_path))
            jpg_analysis = analyze_metadata(str(jpg_path))

        self.assertLess(png_analysis["metadata_score"], jpg_analysis["metadata_score"])
        self.assertLess(png_analysis["metadata_score"], 0.20)
        self.assertIn("missing_exif", png_analysis["metadata_flags"])


if __name__ == "__main__":
    unittest.main()
