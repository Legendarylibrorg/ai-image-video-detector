from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _run_review_queue(
    *extra_args: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "review_queue_to_dataset.py"), *extra_args],
        cwd=ROOT,
        env=merged,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_jpg(path: Path, *, color: str = "red") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=color).save(path, format="JPEG")


class ReviewQueueToDatasetTests(unittest.TestCase):
    def test_rejects_path_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            parent_p = Path(parent).resolve()
            workspace = parent_p / "workspace"
            workspace.mkdir()
            outsider = parent_p / "queue"
            outsider.mkdir()
            (outsider / "ai").mkdir()
            _write_jpg(outsider / "ai" / "x.jpg")
            proc = _run_review_queue(
                "--queue",
                str(outsider),
                "--dst",
                str(workspace / "dst"),
                env={"AID_WORKSPACE_ROOT": str(workspace), "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'scripts'}"},
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("collection_path_escapes_workspace", proc.stderr + proc.stdout)

    def test_ingests_valid_jpg_and_archives_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = Path(tmp).resolve()
            queue = w / "incoming_review_queue"
            dst = w / "data_new" / "train"
            archive = w / "incoming_review_queue" / "_processed"
            img = queue / "real" / "sample.jpg"
            sidecar = queue / "real" / "sample.json"
            _write_jpg(img)
            sidecar.write_text('{"reviewer":"test"}\n', encoding="utf-8")
            proc = _run_review_queue(
                "--queue",
                str(queue),
                "--dst",
                str(dst),
                "--archive",
                str(archive),
                env={"AID_WORKSPACE_ROOT": str(w), "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'scripts'}"},
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue((dst / "real" / "sample.jpg").is_file())
            self.assertFalse(img.exists())
            self.assertTrue((archive / "real" / "sample.json").is_file())
            self.assertIn("review_queue_ingested", proc.stdout)

    def test_skips_symlink_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = Path(tmp).resolve()
            queue = w / "incoming_review_queue"
            dst = w / "data_new" / "train"
            real = w / "real.jpg"
            _write_jpg(real)
            (queue / "ai").mkdir(parents=True)
            link = queue / "ai" / "linked.jpg"
            try:
                link.symlink_to(real)
            except OSError:
                self.skipTest("symlinks not supported")
            proc = _run_review_queue(
                "--queue",
                str(queue),
                "--dst",
                str(dst),
                "--archive",
                str(queue / "_processed"),
                env={"AID_WORKSPACE_ROOT": str(w), "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'scripts'}"},
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertFalse((dst / "ai" / "linked.jpg").exists())
            self.assertIn("ingested=0", proc.stdout)


if __name__ == "__main__":
    unittest.main()
