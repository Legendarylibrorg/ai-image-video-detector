from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests._support import ROOT, source_tree_env, write_rgb_image


class ReviewQueueToDatasetTests(unittest.TestCase):
    def test_ingest_rejects_paths_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            workspace = Path(parent) / "ws"
            workspace.mkdir()
            outsider = Path(parent) / "outside"
            outsider.mkdir()
            env = source_tree_env({"AID_WORKSPACE_ROOT": str(workspace)})
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/review_queue_to_dataset.py",
                    "--queue",
                    str(outsider),
                    "--dst",
                    str(workspace / "data_new" / "train"),
                    "--archive",
                    str(workspace / "archive"),
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("collection_path_escapes_workspace", proc.stderr + proc.stdout)

    def test_ingest_rejects_symlink_image_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            queue = workspace / "incoming_review_queue" / "real"
            queue.mkdir(parents=True)
            target = workspace / "target.jpg"
            write_rgb_image(target, size=(8, 8))
            link = queue / "evil.jpg"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("symlinks not supported")
            env = source_tree_env({"AID_WORKSPACE_ROOT": str(workspace)})
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/review_queue_to_dataset.py",
                    "--queue",
                    str(workspace / "incoming_review_queue"),
                    "--dst",
                    str(workspace / "data_new" / "train"),
                    "--archive",
                    str(workspace / "incoming_review_queue" / "_processed"),
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertFalse((workspace / "data_new" / "train" / "real" / "evil.jpg").exists())
            self.assertEqual(proc.stdout.strip().splitlines()[-1], f"review_queue_ingested dst={workspace / 'data_new' / 'train'} count=0")

    def test_ingest_moves_valid_jpg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            queue = workspace / "incoming_review_queue" / "ai"
            queue.mkdir(parents=True)
            src = queue / "sample.jpg"
            write_rgb_image(src, size=(8, 8))
            env = source_tree_env({"AID_WORKSPACE_ROOT": str(workspace)})
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/review_queue_to_dataset.py",
                    "--queue",
                    str(workspace / "incoming_review_queue"),
                    "--dst",
                    str(workspace / "data_new" / "train"),
                    "--archive",
                    str(workspace / "incoming_review_queue" / "_processed"),
                ],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            dst = workspace / "data_new" / "train" / "ai" / "sample.jpg"
            self.assertTrue(dst.exists())
            self.assertFalse(src.exists())
            self.assertIn("count=1", proc.stdout)


if __name__ == "__main__":
    unittest.main()
