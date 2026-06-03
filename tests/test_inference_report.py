from __future__ import annotations

import unittest

from ai_image_detector.inference_report import ConfigReport, DecisionOptions, ModelReport, build_inference_report


class InferenceReportTests(unittest.TestCase):
    def test_build_inference_report_applies_risk_tools_and_formats_model_context(self) -> None:
        report = build_inference_report(
            prob_ai=0.49,
            threshold=0.50,
            metadata={
                "metadata_score": 0.2,
                "metadata_flags": ["edited_with_software_tag"],
                "metadata_fields": {"Software": "editor"},
            },
            provenance={"provenance_score": 0.1, "provenance_flags": ["has_c2pa_marker"]},
            text={"text_score": 0.0, "text_flags": [], "text_regions": 0},
            ood={"ood_score": 0.1, "ood_flags": ["oversharpened_or_noisy"]},
            domain="photo",
            decision=DecisionOptions(
                unknown_margin=0.04,
                unknown_margin_ai=0.03,
                unknown_margin_real=0.05,
                borderline_ood_threshold=0.45,
                hard_ood_threshold=0.80,
                tta_views=0,
            ),
            model=ModelReport(model_ids=["m1", "m2"], weights=[0.25, 0.75], temperature=1.3, ensemble_config="ens.json"),
            config=ConfigReport(domain_config="domains.json", tools_config="tools.json"),
            tools_cfg={"prob_bias": 0.03},
        )

        self.assertEqual(report["label"], "AI-generated")
        self.assertAlmostEqual(report["prob_ai"], 0.52)
        self.assertGreater(report["combined_risk"], 0.49)
        self.assertEqual(report["model_count"], 2)
        self.assertEqual(report["ensemble_weights"], [0.25, 0.75])
        self.assertEqual(report["ensemble_config"], "ens.json")
        self.assertEqual(report["domain_config"], "domains.json")
        self.assertEqual(report["tools_config"], "tools.json")
        self.assertEqual(report["tta_views"], 1)
        self.assertIn("rule_meta_plus_ood", report["tool_adjustments"])
        self.assertIn("cfg_prob_bias", report["tool_adjustments"])

    def test_build_inference_report_preserves_unknown_decision_near_threshold_with_ood(self) -> None:
        report = build_inference_report(
            prob_ai=0.52,
            threshold=0.50,
            metadata={"metadata_score": 0.0, "metadata_flags": [], "metadata_fields": {}},
            provenance={"provenance_score": 0.0, "provenance_flags": []},
            text={"text_score": 0.0, "text_flags": []},
            ood={"ood_score": 0.50, "ood_flags": ["very_low_resolution"]},
            domain="photo",
            decision=DecisionOptions(
                unknown_margin=0.04,
                unknown_margin_ai=0.03,
                unknown_margin_real=0.05,
                borderline_ood_threshold=0.45,
                hard_ood_threshold=0.80,
                tta_views=3,
            ),
            model=ModelReport(model_ids=["m1"], weights=[1.0], temperature=1.0),
            config=ConfigReport(),
            tools_cfg={},
        )

        self.assertEqual(report["label"], "Unknown")
        self.assertIsNone(report["ensemble_config"])
        self.assertIsNone(report["domain_config"])
        self.assertEqual(report["text_regions"], 0)
        self.assertEqual(report["tta_views"], 3)


if __name__ == "__main__":
    unittest.main()
