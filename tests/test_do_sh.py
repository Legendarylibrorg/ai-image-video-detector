from __future__ import annotations

import subprocess
import unittest

from _support import ROOT


class DoShTests(unittest.TestCase):
    def run_bash(self, script: str) -> str:
        proc = subprocess.run(
            ["bash", "-lc", script],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout

    def test_best_profile_includes_local_sources_when_hf_only_disabled(self) -> None:
        out = self.run_bash(
            "source scripts/do.sh; "
            "BEST_DS_HF_ONLY=0; "
            "BEST_DS_LOCAL_SOURCES='/tmp/a,/tmp/b'; "
            "print_image_collection_args best"
        )

        self.assertIn("--local-source\n/tmp/a\n", out)
        self.assertIn("--local-source\n/tmp/b\n", out)
        self.assertNotIn("--hf-only\n", out)

    def test_best_profile_defaults_to_hf_only(self) -> None:
        out = self.run_bash("source scripts/do.sh; print_image_collection_args best")
        self.assertIn("--hf-only\n", out)

    def test_best_profile_emits_split_source_diversity_gate(self) -> None:
        out = self.run_bash("source scripts/do.sh; print_image_collection_args best")
        self.assertIn("--min-hf-sources-per-split-class\n10\n", out)
        self.assertIn("--max-per-source-class\n12000\n", out)
        self.assertIn("--max-per-source-split-class\n4000\n", out)
        self.assertIn("--hardneg-fraction\n0.35\n", out)

    def test_fast_profile_defaults_to_hf_only_and_split_source_gate(self) -> None:
        out = self.run_bash("source scripts/do.sh; print_image_collection_args fast")
        self.assertIn("--hf-only\n", out)
        self.assertIn("--min-hf-sources-per-split-class\n6\n", out)
        self.assertIn("--max-per-source-class\n3000\n", out)
        self.assertIn("--max-per-source-split-class\n1000\n", out)
        self.assertIn("--hardneg-fraction\n0.25\n", out)


if __name__ == "__main__":
    unittest.main()
