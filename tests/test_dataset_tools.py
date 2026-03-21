from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_image_detector import dataset_tools


def write_stub(path: Path, payload: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


class DatasetToolsTests(unittest.TestCase):
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
            self.assertEqual(out["resume"]["recommended_command"], "./local.sh collect")


if __name__ == "__main__":
    unittest.main()
