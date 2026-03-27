from __future__ import annotations

import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stdout

from _support import ROOT  # noqa: F401
import build_video_dataset


class BuildVideoDatasetTests(unittest.TestCase):
    def test_per_file_download_retry_uses_cache_dir(self) -> None:
        with mock.patch.object(build_video_dataset, "hf_hub_download", return_value="/tmp/video.mp4") as download:
            result = build_video_dataset._download_with_retry(
                "org/repo",
                "video.mp4",
                "tok",
                "/tmp/hf-cache",
                retries=1,
                sleep_ms=0,
            )

        self.assertEqual(result, "/tmp/video.mp4")
        download.assert_called_once_with("org/repo", "video.mp4", token="tok", cache_dir="/tmp/hf-cache")

    def test_snapshot_mode_skips_existing_duplicate_video_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            snapshot_root = tmp_path / "snapshot"
            fake_src = snapshot_root / "dataset" / "fake" / "a.mp4"
            real_src = snapshot_root / "dataset" / "real" / "b.mp4"
            fake_src.parent.mkdir(parents=True, exist_ok=True)
            real_src.parent.mkdir(parents=True, exist_ok=True)
            fake_src.write_bytes(b"fake-video-bytes")
            real_src.write_bytes(b"real-video-bytes")

            out = tmp_path / "out"
            existing_fake = out / "train" / "ai" / "existing_a.mp4"
            existing_real = out / "train" / "real" / "existing_b.mp4"
            existing_fake.parent.mkdir(parents=True, exist_ok=True)
            existing_real.parent.mkdir(parents=True, exist_ok=True)
            existing_fake.write_bytes(b"fake-video-bytes")
            existing_real.write_bytes(b"real-video-bytes")

            argv = [
                "prog",
                "--out",
                str(out),
                "--train-per-class",
                "2",
                "--val-per-class",
                "0",
                "--repo-base-pause-ms",
                "0",
                "--repo-jitter-ms",
                "0",
                "--copy-sleep-ms",
                "0",
                "--min-video-bytes",
                "1",
            ]
            sources = [{"repo": "org/repo", "real_prefixes": ["dataset/real/"], "fake_prefixes": ["dataset/fake/"]}]

            with mock.patch("sys.argv", argv), \
                mock.patch.object(build_video_dataset, "SOURCES", sources), \
                mock.patch.object(build_video_dataset, "snapshot_download", return_value=str(snapshot_root)), \
                mock.patch.object(build_video_dataset.time, "sleep", return_value=None):
                with redirect_stdout(io.StringIO()):
                    build_video_dataset.main()

            self.assertEqual(len(list((out / "train" / "ai").glob("*"))), 1)
            self.assertEqual(len(list((out / "train" / "real").glob("*"))), 1)

    def test_count_existing_ignores_non_video_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "train" / "ai").mkdir(parents=True)
            (out / "train" / "ai" / "note.txt").write_text("not a video", encoding="utf-8")
            (out / "train" / "ai" / "clip.mp4").write_bytes(b"video")

            counts = build_video_dataset.count_existing(out)

        self.assertEqual(counts["train"]["ai"], 1)


if __name__ == "__main__":
    unittest.main()
