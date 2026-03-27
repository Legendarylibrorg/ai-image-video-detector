from __future__ import annotations

import unittest

from ai_image_detector.metrics import find_best_threshold


class MetricsThresholdTests(unittest.TestCase):
    def test_threshold_search_prefers_operable_thresholds(self) -> None:
        probs = [0.06, 0.18, 0.41, 0.63, 0.81]
        labels = [0, 0, 1, 1, 1]

        threshold, score, metrics = find_best_threshold(probs, labels, objective="balanced")

        self.assertGreater(score, 0.5)
        self.assertTrue(metrics["operable"])
        self.assertEqual(metrics["search_status"], "operable")
        self.assertGreater(metrics["tp"], 0.0)
        self.assertGreater(metrics["tn"], 0.0)
        self.assertNotEqual(threshold, 0.05)

    def test_threshold_search_falls_back_when_no_operable_cutoff_exists(self) -> None:
        probs = [0.50, 0.50, 0.50, 0.50]
        labels = [0, 0, 1, 1]

        threshold, score, metrics = find_best_threshold(probs, labels, objective="balanced")

        self.assertEqual(threshold, 0.5)
        self.assertEqual(score, 0.5)
        self.assertFalse(metrics["operable"])
        self.assertEqual(metrics["search_status"], "fallback_no_operable_threshold")

    def test_threshold_search_handles_low_probability_but_well_ranked_outputs(self) -> None:
        probs = [0.01, 0.02, 0.03, 0.04]
        labels = [0, 0, 1, 1]

        threshold, score, metrics = find_best_threshold(probs, labels, objective="balanced")

        self.assertGreater(score, 0.5)
        self.assertTrue(metrics["operable"])
        self.assertEqual(metrics["search_status"], "operable")
        self.assertGreater(threshold, 0.02)
        self.assertLessEqual(threshold, 0.03)


if __name__ == "__main__":
    unittest.main()
