from __future__ import annotations

import unittest

from _support import ROOT  # noqa: F401
from ai_image_detector.risk_tools import apply_risk_tools


class RiskToolsTests(unittest.TestCase):
    def test_metadata_plus_ood_rule_uses_current_metadata_flag_name(self) -> None:
        out = apply_risk_tools(
            prob_ai=0.4,
            combined_risk=0.5,
            metadata_flags=["edited_with_software_tag"],
            ood_flags=["oversharpened_or_noisy"],
            text_flags=[],
            cfg={},
        )

        self.assertIn("rule_meta_plus_ood", out["tool_adjustments"])
        self.assertGreater(out["combined_risk"], 0.5)


if __name__ == "__main__":
    unittest.main()
