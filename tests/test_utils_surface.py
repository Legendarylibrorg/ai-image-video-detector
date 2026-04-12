from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests._support import ROOT, source_tree_env


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
            env=source_tree_env(),
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

    def test_read_json_dict_missing_returns_empty(self) -> None:
        from ai_image_detector.utils.jsonio import read_json_dict

        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.json"
            self.assertEqual(read_json_dict(missing), {})

    def test_read_json_dict_valid_object_round_trips(self) -> None:
        from ai_image_detector.utils.jsonio import read_json_dict, write_json_atomic

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            write_json_atomic(path, {"k": 1}, indent=2)
            self.assertEqual(read_json_dict(path), {"k": 1})

    def test_read_json_dict_invalid_json_raises(self) -> None:
        from ai_image_detector.utils.jsonio import read_json_dict

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                read_json_dict(path)
            self.assertIn("invalid_json_config", str(ctx.exception))

    def test_read_json_dict_non_object_json_raises(self) -> None:
        from ai_image_detector.utils.jsonio import read_json_dict

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "arr.json"
            path.write_text("[1, 2]", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                read_json_dict(path)
            self.assertIn("json_config_must_be_object", str(ctx.exception))

    def test_read_nonempty_lines_respects_line_cap(self) -> None:
        from ai_image_detector.io_limits import MAX_NONEMPTY_LINES_COUNT
        from ai_image_detector.utils.jsonio import read_nonempty_lines

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "many.txt"
            path.write_text("\n".join(f"x{i}" for i in range(MAX_NONEMPTY_LINES_COUNT + 1)), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                read_nonempty_lines(path)
            self.assertIn("nonempty_lines_too_many", str(ctx.exception))

    def test_read_nonempty_lines_rejects_symlink_leaf(self) -> None:
        from ai_image_detector.utils.jsonio import read_nonempty_lines

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a.txt"
            target.write_text("ok\n", encoding="utf-8")
            link = Path(tmp) / "b.txt"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("symlinks not supported")
            with self.assertRaises(ValueError):
                read_nonempty_lines(link)


if __name__ == "__main__":
    unittest.main()
