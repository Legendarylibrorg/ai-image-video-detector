from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest


import prepare_training_data


def write_image_stub(path: Path, payload: bytes = b"image-bytes") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


class PrepareTrainingDataTests(unittest.TestCase):
    def test_prepare_training_data_merges_base_and_incremental_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            incremental = root / "data_new"
            out = root / "training"

            write_image_stub(base / "train" / "ai" / "base_ai.jpg", b"base-ai")
            write_image_stub(base / "train" / "real" / "base_real.jpg", b"base-real")
            write_image_stub(base / "val" / "ai" / "val_ai.jpg", b"val-ai")
            write_image_stub(base / "val" / "real" / "val_real.jpg", b"val-real")
            write_image_stub(base / "test" / "ai" / "test_ai.jpg", b"test-ai")
            write_image_stub(base / "test" / "real" / "test_real.jpg", b"test-real")

            write_image_stub(incremental / "train" / "ai" / "inc_ai.jpg", b"inc-ai")
            write_image_stub(incremental / "train" / "real" / "inc_train_real.jpg", b"inc-real")

            result = subprocess.run(
                [
                    "python3",
                    str(Path(prepare_training_data.__file__)),
                    "--base",
                    str(base),
                    "--incremental",
                    str(incremental),
                    "--out",
                    str(out),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            self.assertTrue((out / "train" / "ai" / "base_ai.jpg").exists())
            self.assertTrue((out / "train" / "ai" / "inc_ai.jpg").exists())
            self.assertTrue((out / "train" / "real" / "base_real.jpg").exists())
            self.assertTrue((out / "train" / "real" / "inc_train_real.jpg").exists())
            self.assertTrue((out / "val" / "ai" / "val_ai.jpg").exists())
            self.assertTrue((out / "test" / "real" / "test_real.jpg").exists())

    def test_prepare_training_data_accepts_train_root_incremental_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            incremental_train_root = root / "data_new" / "train"
            out = root / "training"

            for split in ("train", "val", "test"):
                write_image_stub(base / split / "ai" / f"{split}_ai.jpg", f"{split}-ai".encode("utf-8"))
                write_image_stub(base / split / "real" / f"{split}_real.jpg", f"{split}-real".encode("utf-8"))

            write_image_stub(incremental_train_root / "ai" / "new_train_ai.jpg", b"new-train-ai")

            result = subprocess.run(
                [
                    "python3",
                    str(Path(prepare_training_data.__file__)),
                    "--base",
                    str(base),
                    "--incremental",
                    str(incremental_train_root),
                    "--out",
                    str(out),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue((out / "train" / "ai" / "new_train_ai.jpg").exists())

    def test_prepare_training_data_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            out = root / "training"

            for split in ("train", "val", "test"):
                write_image_stub(base / split / "ai" / f"{split}_ai.jpg", f"{split}-ai".encode("utf-8"))
                write_image_stub(base / split / "real" / f"{split}_real.jpg", f"{split}-real".encode("utf-8"))

            script = str(Path(prepare_training_data.__file__))
            for _ in range(2):
                result = subprocess.run(
                    [
                        "python3",
                        script,
                        "--base",
                        str(base),
                        "--incremental",
                        str(root / "missing_incremental"),
                        "--out",
                        str(out),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, msg=result.stderr)

            from ai_image_detector.dataset_layout import image_counts

            counts = image_counts(out, allow_train_root_alias=True, include_symlinks=False)
            self.assertEqual(counts["train"]["ai"], 1)
            self.assertEqual(counts["train"]["real"], 1)
            self.assertEqual(counts["val"]["ai"], 1)
            self.assertEqual(counts["test"]["real"], 1)

    def test_prepare_training_data_fails_when_required_buckets_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            out = root / "training"

            write_image_stub(base / "train" / "ai" / "only_train_ai.jpg", b"only-train-ai")

            result = subprocess.run(
                [
                    "python3",
                    str(Path(prepare_training_data.__file__)),
                    "--base",
                    str(base),
                    "--incremental",
                    str(root / "missing_incremental"),
                    "--out",
                    str(out),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("missing_buckets", result.stderr)


if __name__ == "__main__":
    unittest.main()
