from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

from _support import ROOT, SRC

from ai_image_detector import dataset_tools


def write_stub(path: Path, payload: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


class DatasetToolsTests(unittest.TestCase):
    def test_walk_images_includes_tiff_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tiff_path = root / "train" / "ai" / "sample.tiff"
            write_stub(tiff_path)

            walked = list(dataset_tools._walk_images(root))

            self.assertEqual(walked, [tiff_path])

    def test_collection_status_reports_manifest_resume_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data_best"
            write_stub(root / "train" / "ai" / "a.jpg")
            write_stub(root / "train" / "real" / "b.jpg")
            write_stub(root / "val" / "ai" / "c.jpg")
            write_stub(root / "val" / "real" / "d.jpg")
            write_stub(root / "test" / "ai" / "e.jpg")
            write_stub(root / "test" / "real" / "f.jpg")
            (root / "dataset_state.json").write_text(
                json.dumps({"source_candidates": ["repo/a", "repo/b"], "full_targets_ok": False}),
                encoding="utf-8",
            )
            (root / "dataset_source_manifest.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "source": "repo/a",
                                "status": "completed",
                                "accepted_total": 12,
                                "processed_total": 30,
                                "finished_utc": "2026-03-21T00:00:00+00:00",
                                "skip_future_runs": True,
                            }
                        ),
                        json.dumps(
                            {
                                "source": "repo/b",
                                "status": "load_failed",
                                "accepted_total": 0,
                                "processed_total": 0,
                                "finished_utc": "2026-03-21T00:10:00+00:00",
                                "skip_future_runs": False,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            buf = io.StringIO()
            with redirect_stdout(buf):
                dataset_tools.cmd_collection_status(str(root))
            out = json.loads(buf.getvalue())

            self.assertEqual(out["current_counts"]["train"]["ai"], 1)
            self.assertEqual(out["manifest"]["unique_sources"], 2)
            self.assertEqual(out["manifest"]["skipped_future_runs"], 1)
            self.assertEqual(out["resume"]["remaining_candidates_estimate"], 1)
            self.assertEqual(out["resume"]["recommended_command"], "./local.sh run")

    def test_collection_status_recommends_train_when_targets_are_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data_best"
            write_stub(root / "train" / "ai" / "a.jpg")
            write_stub(root / "train" / "real" / "b.jpg")
            write_stub(root / "val" / "ai" / "c.jpg")
            write_stub(root / "val" / "real" / "d.jpg")
            write_stub(root / "test" / "ai" / "e.jpg")
            write_stub(root / "test" / "real" / "f.jpg")
            (root / "dataset_state.json").write_text(
                json.dumps({"source_candidates": ["repo/a"], "full_targets_ok": True}),
                encoding="utf-8",
            )

            buf = io.StringIO()
            with redirect_stdout(buf):
                dataset_tools.cmd_collection_status(str(root))
            out = json.loads(buf.getvalue())

            self.assertFalse(out["resume"]["resume_needed"])
            self.assertEqual(out["resume"]["recommended_command"], "./local.sh train")

    def test_collection_status_counts_train_root_incremental_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data_best"
            incremental = Path(tmp) / "data_new" / "train"
            write_stub(root / "train" / "ai" / "a.jpg")
            write_stub(root / "train" / "real" / "b.jpg")
            write_stub(root / "val" / "ai" / "c.jpg")
            write_stub(root / "val" / "real" / "d.jpg")
            write_stub(root / "test" / "ai" / "e.jpg")
            write_stub(root / "test" / "real" / "f.jpg")
            write_stub(incremental / "ai" / "g.jpg")
            write_stub(incremental / "real" / "h.jpg")

            buf = io.StringIO()
            with redirect_stdout(buf):
                dataset_tools.cmd_collection_status(str(root), incremental_root=str(incremental))
            out = json.loads(buf.getvalue())

            self.assertEqual(out["incremental_root"]["counts"]["train"]["ai"], 1)
            self.assertEqual(out["incremental_root"]["counts"]["train"]["real"], 1)


if __name__ == "__main__":
    unittest.main()
