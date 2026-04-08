from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest

from _support import ROOT


class TrainingSurfaceTests(unittest.TestCase):
    def test_api_module_is_removed(self) -> None:
        self.assertFalse((ROOT / "src" / "ai_image_detector" / "api.py").exists())
        self.assertFalse((ROOT / "src" / "ai_image_detector" / "multimodal.py").exists())

    def test_cli_wrapper_import_stays_lightweight(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import ai_image_detector.cli as c; "
                    "print(int('torch' in sys.modules)); "
                    "print(int('cv2' in sys.modules)); "
                    "print(hasattr(c, 'train_main')); "
                    "print(hasattr(c, 'video_train_main')); "
                    "print(hasattr(c, 'dataset_main'))"
                ),
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            check=True,
            capture_output=True,
            text=True,
        )
        lines = proc.stdout.strip().splitlines()
        self.assertEqual(lines[0], "0")
        self.assertEqual(lines[1], "0")
        self.assertEqual(lines[2], "True")
        self.assertEqual(lines[3], "True")
        self.assertEqual(lines[4], "False")

    def test_key_training_shell_scripts_are_bash_valid(self) -> None:
        for rel_path in [
            "scripts/continuous_training.sh",
            "scripts/full_pipeline_4090.sh",
            "scripts/lib/training.sh",
            "scripts/metadata_finetune_4090.sh",
            "scripts/train_ensemble.sh",
        ]:
            with self.subTest(script=rel_path):
                subprocess.run(["bash", "-n", rel_path], cwd=ROOT, check=True)

    def test_key_training_python_modules_compile(self) -> None:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "py_compile",
                "scripts/train_distill.py",
                "src/ai_image_detector/train.py",
                "src/ai_image_detector/video_temporal.py",
                "src/ai_image_detector/checkpoint_io.py",
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
