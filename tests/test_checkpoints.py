from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests._support import ROOT  # noqa: F401
import script_support
from ai_image_detector import checkpoints, io_limits

try:
    import torch
    from ai_image_detector import runtime as aid_runtime
except ModuleNotFoundError:  # pragma: no cover - optional dependency path
    torch = None  # type: ignore[assignment]
    aid_runtime = None  # type: ignore[assignment]


class CheckpointsTests(unittest.TestCase):
    def test_maybe_compile_model_returns_original_when_disabled(self) -> None:
        if torch is None or aid_runtime is None:
            self.skipTest("requires torch")

        model = torch.nn.Linear(2, 1)

        compiled = aid_runtime.maybe_compile_model(model, enabled=False)

        self.assertIs(compiled, model)

    def test_maybe_compile_model_falls_back_on_compile_error(self) -> None:
        if torch is None or aid_runtime is None:
            self.skipTest("requires torch")

        model = torch.nn.Linear(2, 1)
        with mock.patch.object(aid_runtime.torch, "compile", side_effect=RuntimeError("boom")):
            compiled = aid_runtime.maybe_compile_model(model, enabled=True)

        self.assertIs(compiled, model)

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

    def test_load_checkpoint_rejects_symlink_safetensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "weights.safetensors"
            real.write_bytes(b"x")
            link = Path(tmp) / "via_link.safetensors"
            try:
                link.symlink_to(real)
            except OSError:
                self.skipTest("symlinks not supported")
            with self.assertRaisesRegex(ValueError, "symlink_not_allowed"):
                checkpoints.load_checkpoint(link, map_location="cpu")

    def test_load_checkpoint_rejects_oversized_safetensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "big.safetensors"
            path.write_bytes(b"")
            os.truncate(path, io_limits.MAX_SAFETENSORS_FILE_BYTES + 1)
            with self.assertRaisesRegex(ValueError, "file_too_large"):
                checkpoints.load_checkpoint(path, map_location="cpu")

    def test_script_support_checkpoint_resolvers_cover_live_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pt_path = Path(tmp) / "model.pt"
            sft_path = Path(tmp) / "model.safetensors"
            pt_path.write_bytes(b"pt")
            sft_path.write_bytes(b"sft")

            resolved = script_support.resolve_checkpoint(sft_path)
            preferred = script_support.resolve_preferred_checkpoint(pt_path)

        self.assertEqual(resolved, sft_path)
        self.assertEqual(preferred, sft_path)

    def test_save_training_checkpoint_updates_latest_marker(self) -> None:
        if torch is None:
            self.skipTest("requires torch")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.pt"

            checkpoints.save_training_checkpoint(path, {"epoch": 3})

            self.assertTrue(path.exists())
            self.assertEqual((Path(tmp) / "latest_checkpoint.txt").read_text(encoding="utf-8"), "last.pt")

    def test_train_module_does_not_call_torch_load_for_best_checkpoint_eval(self) -> None:
        train_main = (ROOT / "src" / "ai_image_detector" / "train_main.py").read_text(encoding="utf-8")
        train_post = (ROOT / "src" / "ai_image_detector" / "train_post.py").read_text(encoding="utf-8")
        combined = train_main + train_post
        self.assertIn("best = load_checkpoint(best_path, map_location=device)", combined)
        self.assertIn("save_training_checkpoint(", combined)
        self.assertIn("ckpt = load_training_checkpoint(resume_path, map_location=device)", combined)
        self.assertNotIn("torch.load(", combined)

    def test_train_module_checkpoint_io_stays_on_plain_model_when_compile_is_enabled(self) -> None:
        runtime_text = (ROOT / "src" / "ai_image_detector" / "runtime.py").read_text(encoding="utf-8")
        train_text = (ROOT / "src" / "ai_image_detector" / "train_main.py").read_text(encoding="utf-8")
        self.assertIn("def maybe_compile_model(", runtime_text)
        self.assertIn("train_model = maybe_compile_model(model, enabled=bool(args.compile))", train_text)
        self.assertIn('"state_dict": model.state_dict()', train_text)
        self.assertIn('model.load_state_dict(ckpt["state_dict"])', train_text)
        self.assertIn("logits = train_model(x, metadata_features=metadata_features)", train_text)

    def test_video_training_checkpoint_io_stays_on_plain_model_when_compile_is_enabled(self) -> None:
        runtime_text = (ROOT / "src" / "ai_image_detector" / "runtime.py").read_text(encoding="utf-8")
        video_text = (ROOT / "src" / "ai_image_detector" / "video_temporal.py").read_text(encoding="utf-8")
        self.assertIn("def maybe_compile_model(", runtime_text)
        self.assertIn("train_model = maybe_compile_model(model, enabled=bool(args.compile))", video_text)
        self.assertIn('"state_dict": model.state_dict()', video_text)
        self.assertIn('model.load_state_dict(ckpt["state_dict"])', video_text)
        self.assertIn("logit = train_model(x)", video_text)

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
        if torch is None:
            self.skipTest("requires torch")
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
