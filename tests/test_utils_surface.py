from __future__ import annotations

import json
import subprocess
import sys
import tempfile
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

    def test_write_json_atomic_writes_valid_json(self) -> None:
        from ai_image_detector.utils.jsonio import write_json_atomic

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "out.json"
            write_json_atomic(path, {"a": 1, "b": [2, 3]}, indent=2)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["a"], 1)
            self.assertEqual(data["b"], [2, 3])


if __name__ == "__main__":
    unittest.main()
