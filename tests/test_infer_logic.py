from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw

from _support import ROOT, write_rgb_image  # noqa: F401
from ai_image_detector.decision import combined_risk, decide_label
from ai_image_detector.metadata import METADATA_FEATURE_NAMES, analyze_metadata, extract_metadata_features, metadata_feature_dim


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

    def test_auxiliary_feature_vector_includes_provenance_and_text_signals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "overlay.png"
            image = Image.new("RGB", (192, 96), (248, 248, 248))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 191, 20), fill=(16, 16, 16))
            draw.text((12, 28), "SALE NOW", fill=(0, 0, 0))
            draw.text((12, 56), "LIMITED OFFER", fill=(0, 0, 0))
            image.save(img_path)

            features = extract_metadata_features(str(img_path))

        self.assertEqual(len(features), metadata_feature_dim())
        self.assertIn("provenance_score", METADATA_FEATURE_NAMES)
        self.assertIn("text_score", METADATA_FEATURE_NAMES)
        self.assertIn("has_text_overlay_signal", METADATA_FEATURE_NAMES)
        text_score_idx = METADATA_FEATURE_NAMES.index("text_score")
        text_overlay_idx = METADATA_FEATURE_NAMES.index("has_text_overlay_signal")
        self.assertGreater(features[text_score_idx], 0.0)
        self.assertEqual(features[text_overlay_idx], 1.0)


if __name__ == "__main__":
    unittest.main()
