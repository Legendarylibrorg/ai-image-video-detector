from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class UtilsSurfaceTests(unittest.TestCase):
    def test_utils_import_does_not_eagerly_load_heavy_modules(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import ai_image_detector.utils as u; "
                    "print(int('torch' in sys.modules)); "
                    "print(int('numpy' in sys.modules)); "
                    "print(int('PIL' in sys.modules)); "
                    "print(hasattr(u, 'read_json_dict'))"
                ),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        lines = proc.stdout.strip().splitlines()
        self.assertEqual(lines[0], "0")
        self.assertEqual(lines[1], "0")
        self.assertEqual(lines[2], "0")
        self.assertEqual(lines[3], "True")


if __name__ == "__main__":
    unittest.main()
