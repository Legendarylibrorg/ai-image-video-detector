from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
import tempfile
import unittest

from _support import ROOT, SRC

from ai_image_detector import dataset_layout


def write_stub(path: Path, payload: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


class DatasetLayoutTests(unittest.TestCase):
    def test_image_counts_supports_train_root_alias_and_shared_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_stub(root / "ai" / "one.png")
            write_stub(root / "real" / "two.tiff")

            counts = dataset_layout.image_counts(root, allow_train_root_alias=True)

            self.assertEqual(counts["train"]["ai"], 1)
            self.assertEqual(counts["train"]["real"], 1)
            self.assertEqual(counts["val"]["ai"], 0)

    def test_dataset_layout_cli_reports_image_shortfalls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_stub(root / "train" / "ai" / "one.jpg")
            write_stub(root / "train" / "real" / "two.jpg")
            write_stub(root / "val" / "ai" / "three.jpg")

            proc = subprocess.run(
                [
                    "python3",
                    "-m",
                    "ai_image_detector.dataset_layout",
                    "check-image-minimums",
                    "--root",
                    str(root),
                    "--train-min",
                    "1",
                    "--val-min",
                    "1",
                    "--test-min",
                    "1",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(SRC)},
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(proc.returncode, 1)
            self.assertIn("insufficient_image_bucket=", proc.stdout)
            self.assertIn("image_collection_counts=invalid", proc.stdout)

    def test_dataset_layout_cli_counts_videos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_stub(root / "train" / "ai" / "one.mp4")
            write_stub(root / "train" / "real" / "two.mov")
            write_stub(root / "val" / "ai" / "three.webm")
            write_stub(root / "val" / "real" / "four.mkv")

            proc = subprocess.run(
                [
                    "python3",
                    "-m",
                    "ai_image_detector.dataset_layout",
                    "counts",
                    "--root",
                    str(root),
                    "--kind",
                    "video",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(SRC)},
                check=True,
                capture_output=True,
                text=True,
            )

            counts = json.loads(proc.stdout)
            self.assertEqual(counts["train"]["ai"], 1)
            self.assertEqual(counts["val"]["real"], 1)


if __name__ == "__main__":
    unittest.main()
