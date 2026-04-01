from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from _support import ROOT  # noqa: F401
from ai_image_detector import checkpoints


class CheckpointsTests(unittest.TestCase):
    def test_load_checkpoint_rejects_pickle_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            path.write_bytes(b"not-a-real-checkpoint")

            with self.assertRaisesRegex(RuntimeError, "unsupported_checkpoint_format"):
                checkpoints.load_checkpoint(path, map_location="cpu")

    def test_load_checkpoint_uses_safetensors_without_pickle_opt_in(self) -> None:
        fake_ckpt = {"state_dict": {"x": object()}, "_checkpoint_format": "safetensors"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            path.write_bytes(b"placeholder")

            with mock.patch.object(checkpoints, "load_safetensors_checkpoint", return_value=fake_ckpt) as mock_load:
                loaded = checkpoints.load_checkpoint(path, map_location="cpu")

        self.assertEqual(loaded, fake_ckpt)
        mock_load.assert_called_once_with(path, map_location="cpu")

    def test_resolve_checkpoint_path_prefers_safetensors_when_both_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pt_path = Path(tmp) / "model.pt"
            sft_path = Path(tmp) / "model.safetensors"
            pt_path.write_bytes(b"pt")
            sft_path.write_bytes(b"sft")

            resolved = checkpoints.resolve_checkpoint_path(pt_path)

        self.assertEqual(resolved, sft_path)

    def test_save_training_checkpoint_updates_latest_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.pt"

            checkpoints.save_training_checkpoint(path, {"epoch": 3})

            self.assertTrue(path.exists())
            self.assertEqual((Path(tmp) / "latest_checkpoint.txt").read_text(encoding="utf-8"), "last.pt")

    def test_train_module_does_not_call_torch_load_for_best_checkpoint_eval(self) -> None:
        train_text = (ROOT / "src" / "ai_image_detector" / "train.py").read_text(encoding="utf-8")
        self.assertIn("best = load_checkpoint(best_path, map_location=device)", train_text)
        self.assertIn("save_training_checkpoint(", train_text)
        self.assertIn("ckpt = load_training_checkpoint(resume_path, map_location=device)", train_text)
        self.assertNotIn("torch.load(", train_text)

    def test_distill_script_uses_shared_checkpoint_loader_for_resume(self) -> None:
        distill_text = (ROOT / "scripts" / "train_distill.py").read_text(encoding="utf-8")
        self.assertIn("ckpt = load_training_checkpoint(resume_path, map_location=device)", distill_text)
        self.assertIn("save_training_checkpoint(", distill_text)

    def test_distill_script_writes_safetensors_and_summary_artifacts(self) -> None:
        distill_text = (ROOT / "scripts" / "train_distill.py").read_text(encoding="utf-8")
        self.assertIn("save_safetensors_checkpoint", distill_text)
        self.assertIn('out / "best.safetensors"', distill_text)
        self.assertIn('out / "best_checkpoint.txt"', distill_text)
        self.assertIn('out / "best_model_summary.json"', distill_text)

    def test_load_training_checkpoint_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.pt"
            payload = {
                "epoch": 1,
                "best_auc": 0.5,
                "state_dict": {"w": torch.zeros(2, 2)},
            }
            checkpoints.save_training_checkpoint(path, payload)
            loaded = checkpoints.load_training_checkpoint(path, map_location="cpu")
        self.assertEqual(loaded["epoch"], 1)
        self.assertEqual(loaded["best_auc"], 0.5)
        self.assertEqual(tuple(loaded["state_dict"]["w"].shape), (2, 2))

    def test_args_dict_for_checkpoint_omits_sensitive_keys(self) -> None:
        ns = argparse.Namespace(
            data="./data",
            hf_token="supersecret",
            api_key="x",
            oauth_access_token="t",
            tokenizer_name="gpt2",
            normal_flag=True,
        )
        d = checkpoints.args_dict_for_checkpoint(ns)
        self.assertEqual(d["data"], "./data")
        self.assertTrue(d["normal_flag"])
        self.assertEqual(d["tokenizer_name"], "gpt2")
        self.assertNotIn("hf_token", d)
        self.assertNotIn("api_key", d)
        self.assertNotIn("oauth_access_token", d)
