from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from _support import write_rgb_image
from ai_image_detector.io_limits import (
    MAX_JSON_CONFIG_BYTES,
    open_image_rgb,
    path_must_be_under,
    prepare_video_path,
    read_json_file_limited,
    validate_domain_config,
    validate_ensemble_config,
    validate_tools_config,
)


class IoLimitsTests(unittest.TestCase):
    def test_path_must_reject_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            safe = root / "train" / "real"
            safe.mkdir(parents=True)
            outside = Path(tmp) / "secret.txt"
            outside.write_text("x", encoding="utf-8")
            link = safe / "evil.jpg"
            try:
                link.symlink_to(outside)
            except OSError:
                self.skipTest("symlinks not supported")
            with self.assertRaises(ValueError):
                path_must_be_under(link, root)

    def test_read_json_file_limited_rejects_huge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "big.json"
            p.write_bytes(b"{" + b"0" * (MAX_JSON_CONFIG_BYTES + 2) + b"}")
            with self.assertRaises(ValueError):
                read_json_file_limited(p)

    def test_validate_ensemble_rejects_bad_temperature(self) -> None:
        with self.assertRaises(ValueError):
            validate_ensemble_config({"weights": [0.5, 0.5], "temperature": 0.0})

    def test_validate_domain_rejects_bad_threshold(self) -> None:
        with self.assertRaises(ValueError):
            validate_domain_config({"thresholds": {"photo": 1.5}})

    def test_validate_tools_rejects_large_bias(self) -> None:
        with self.assertRaises(ValueError):
            validate_tools_config({"risk_bias": 9.0})

    def test_prepare_video_path_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a.bin"
            target.write_bytes(b"x")
            link = Path(tmp) / "v.mp4"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("symlinks not supported")
            with self.assertRaises(ValueError):
                prepare_video_path(link)

    def test_open_image_rgb_writes_small_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "jail"
            root.mkdir()
            img_path = root / "a.png"
            write_rgb_image(img_path, color=(1, 2, 3), size=(8, 8))
            out = open_image_rgb(img_path, root=root.resolve())
            self.assertEqual(out.size, (8, 8))
            out.close()


if __name__ == "__main__":
    unittest.main()
