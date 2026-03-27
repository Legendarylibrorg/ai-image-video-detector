from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from _support import ROOT  # noqa: F401
from ai_image_detector import checkpoints


class CheckpointsTests(unittest.TestCase):
    def test_load_checkpoint_blocks_pickle_files_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            path.write_bytes(b"not-a-real-checkpoint")

            with self.assertRaisesRegex(RuntimeError, "pickle_checkpoint_blocked"):
                checkpoints.load_checkpoint(path, map_location="cpu")

    def test_load_checkpoint_allows_pickle_files_with_explicit_opt_in(self) -> None:
        fake_ckpt = {"state_dict": {}, "threshold": 0.5}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            path.write_bytes(b"not-a-real-checkpoint")

            with mock.patch.dict(os.environ, {"ALLOW_UNSAFE_PICKLE_CHECKPOINTS": "1"}, clear=False):
                with mock.patch.object(checkpoints.torch, "load", return_value=fake_ckpt) as mock_load:
                    loaded = checkpoints.load_checkpoint(path, map_location="cpu")

        self.assertEqual(loaded, fake_ckpt)
        mock_load.assert_called_once_with(path, map_location="cpu")

    def test_load_checkpoint_uses_safetensors_without_pickle_opt_in(self) -> None:
        fake_ckpt = {"state_dict": {"x": object()}, "_checkpoint_format": "safetensors"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            path.write_bytes(b"placeholder")

            with mock.patch.object(checkpoints, "load_safetensors_checkpoint", return_value=fake_ckpt) as mock_load:
                loaded = checkpoints.load_checkpoint(path, map_location="cpu")

        self.assertEqual(loaded, fake_ckpt)
        mock_load.assert_called_once_with(path, map_location="cpu")

    def test_train_module_does_not_call_torch_load_for_best_checkpoint_eval(self) -> None:
        train_text = (ROOT / "src" / "ai_image_detector" / "train.py").read_text(encoding="utf-8")
        self.assertIn("best = load_checkpoint(best_path, map_location=device)", train_text)
        self.assertNotIn('torch.load(out / "best.pt"', train_text)

    def test_distill_script_uses_shared_checkpoint_loader_for_resume(self) -> None:
        distill_text = (ROOT / "scripts" / "train_distill.py").read_text(encoding="utf-8")
        self.assertIn("from ai_image_detector.checkpoints import load_checkpoint", distill_text)
        self.assertIn("ckpt = load_checkpoint(resume_path, map_location=device)", distill_text)
        self.assertNotIn("ckpt = torch.load(resume_path, map_location=device)", distill_text)
