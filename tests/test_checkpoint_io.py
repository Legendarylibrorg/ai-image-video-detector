from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from ai_image_detector.checkpoint_io import materialize_checkpoint_file


class CheckpointIoTests(unittest.TestCase):
    def test_materialize_checkpoint_file_copies_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "m.bin"
            src.write_bytes(b"hello-checkpoint")
            staged = materialize_checkpoint_file(src, max_bytes=1024)
            try:
                self.assertTrue(staged.exists())
                self.assertEqual(staged.read_bytes(), b"hello-checkpoint")
            finally:
                staged.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
