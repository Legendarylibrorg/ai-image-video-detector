from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _support import write_rgb_image
import audit_diversity


class AuditDiversityTests(unittest.TestCase):
    def test_max_source_share_zero_for_empty_counts(self) -> None:
        self.assertEqual(audit_diversity._max_source_share({}), 0.0)

    def test_max_source_share_detects_dominant_source(self) -> None:
        self.assertAlmostEqual(audit_diversity._max_source_share({"a": 9, "b": 1}), 0.9)

    def test_count_sources_ignores_hardneg_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            split_dir = Path(tmp)
            write_rgb_image(split_dir / "source=repo_a__train_ai_0000001.jpg")
            write_rgb_image(split_dir / "source=repo_a__train_ai_0000002.jpg")
            write_rgb_image(split_dir / "source=repo_b__train_ai_0000003.jpg")
            write_rgb_image(split_dir / "hardneg=blur__repo_a__hn0000001.jpg")

            counts = audit_diversity._count_sources(split_dir)

            self.assertEqual(counts, {"repo_a": 2, "repo_b": 1})


if __name__ == "__main__":
    unittest.main()
