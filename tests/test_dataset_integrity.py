from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from ai_image_detector.dataset_integrity import (
    assert_no_train_val_hash_overlap,
    build_manifest_records,
    preflight_dataset_tree,
    write_dataset_manifest,
)


class DatasetIntegrityTests(unittest.TestCase):
    def test_preflight_rejects_symlinked_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "train" / "cls" / "a.png"
            real.parent.mkdir(parents=True)
            real.write_bytes(b"\x89PNG\r\n\x1a\n")
            link = root / "train" / "cls" / "b.png"
            os.symlink(real, link)

            with self.assertRaisesRegex(ValueError, "dataset_image_symlink"):
                preflight_dataset_tree(root)

    def test_preflight_skipped_when_env_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "train" / "cls" / "a.png"
            real.parent.mkdir(parents=True)
            real.write_bytes(b"\x89PNG\r\n\x1a\n")
            link = root / "train" / "cls" / "b.png"
            os.symlink(real, link)

            old = os.environ.get("AID_SKIP_DATA_PREFLIGHT")
            try:
                os.environ["AID_SKIP_DATA_PREFLIGHT"] = "1"
                preflight_dataset_tree(root)
            finally:
                if old is None:
                    os.environ.pop("AID_SKIP_DATA_PREFLIGHT", None)
                else:
                    os.environ["AID_SKIP_DATA_PREFLIGHT"] = old

    def test_overlap_raises(self) -> None:
        train = [{"sha256": "abc", "rel": "t/x.png"}]
        val = [{"sha256": "abc", "rel": "v/y.png"}]
        with self.assertRaisesRegex(ValueError, "train_val_content_overlap"):
            assert_no_train_val_hash_overlap(train, val)

    def test_manifest_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p1 = root / "train" / "c" / "1.png"
            p1.parent.mkdir(parents=True)
            p1.write_bytes(b"a")
            p2 = root / "val" / "c" / "2.png"
            p2.parent.mkdir(parents=True)
            p2.write_bytes(b"b")
            classes = ["c"]
            train_samples = [(str(p1), 0)]
            val_samples = [(str(p2), 0)]
            tr = build_manifest_records(train_samples, classes, root, hash_files=True)
            vr = build_manifest_records(val_samples, classes, root, hash_files=True)
            assert_no_train_val_hash_overlap(tr, vr)
            out = Path(tmp) / "manifest.json"
            write_dataset_manifest(out, data_root=root, train_records=tr, val_records=vr)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["schema"], "ai-image-detector-dataset-manifest-v1")
            self.assertEqual(data["train_count"], 1)
            self.assertEqual(data["val_count"], 1)
            self.assertIn("sha256", data["train"][0])

