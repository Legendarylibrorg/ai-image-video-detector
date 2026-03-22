from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class InstallDepsSurfaceTests(unittest.TestCase):
    def test_install_deps_verifies_huggingface_python_and_cli(self) -> None:
        text = (ROOT / "scripts" / "install_deps.sh").read_text(encoding="utf-8")
        self.assertIn("import ai_image_detector", text)
        self.assertIn("import huggingface_hub", text)
        self.assertIn("import cv2", text)
        self.assertIn("command -v hf", text)
        self.assertIn("command -v aid-train", text)
        self.assertIn("command -v aid-video-train", text)
        self.assertIn("command -v aid-dataset", text)
        self.assertIn("deps_fail=huggingface_cli_missing", text)
        self.assertIn("deps_fail=repo_cli_missing", text)
        self.assertIn("PIP_DISABLE_PIP_VERSION_CHECK=1", text)
        self.assertIn("python -m pip --progress-bar off", text)
        self.assertIn('"$UPGRADE_TOOLCHAIN" != "1"', text)

    def test_install_deps_fast_path_skips_work_when_current_by_default(self) -> None:
        proc = subprocess.run(
            ["bash", "scripts/install_deps.sh"],
            cwd=ROOT,
            env={**os.environ, "UPGRADE_TOOLCHAIN": "0"},
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("deps_status=up_to_date", proc.stdout)
        self.assertNotIn("warning_toolchain_upgrade_failed", proc.stdout)


if __name__ == "__main__":
    unittest.main()
