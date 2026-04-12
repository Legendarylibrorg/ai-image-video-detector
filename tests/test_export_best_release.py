from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from tests._support import ROOT


class ExportBestReleaseTests(unittest.TestCase):
    def test_export_best_release_writes_release_bundle_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ens = root / "artifacts_ens"
            video = root / "video_artifacts"
            release = ens / "release"
            (ens / "m1").mkdir(parents=True, exist_ok=True)
            (ens / "m5_metadata").mkdir(parents=True, exist_ok=True)
            (ens / "distill").mkdir(parents=True, exist_ok=True)
            video.mkdir(parents=True, exist_ok=True)

            (ens / "m1" / "best.safetensors").write_bytes(b"x")
            (ens / "m1" / "calibration.json").write_text(
                json.dumps({"threshold": 0.61, "temperature": 0.7, "objective": "balanced"}),
                encoding="utf-8",
            )
            (ens / "m1" / "best_metrics.json").write_text(
                json.dumps(
                    {
                        "auc": 0.98,
                        "balanced_accuracy": 0.95,
                        "precision_ai": 0.94,
                        "recall_ai": 0.95,
                        "precision_real": 0.96,
                        "recall_real": 0.95,
                        "predicts_single_class": False,
                        "promotion_eligible": True,
                        "promotion_reason": "ok",
                        "threshold_operable": True,
                        "composite_metrics": {"ece": 0.04, "brier": 0.05},
                    }
                ),
                encoding="utf-8",
            )
            (ens / "m1" / "config.json").write_text(
                json.dumps({"args": {"backbone": "effb0", "img_size": 256, "use_metadata_features": False}}),
                encoding="utf-8",
            )
            (ens / "m1" / "inference_spec.json").write_text(json.dumps({"schema": "ai-image-detector-runtime-v1"}), encoding="utf-8")
            (ens / "m1" / "best_model_summary.json").write_text(json.dumps({"epoch": 3}), encoding="utf-8")
            (ens / "m1" / "test_metrics.json").write_text(
                json.dumps({"auc": 0.97, "f1": 0.93, "precision_ai": 0.92, "recall_ai": 0.94, "ece": 0.04, "brier": 0.05}),
                encoding="utf-8",
            )
            (ens / "m5_metadata" / "best.safetensors").write_bytes(b"x")
            (ens / "m5_metadata" / "calibration.json").write_text(
                json.dumps({"threshold": 0.55, "temperature": 0.8, "objective": "balanced"}),
                encoding="utf-8",
            )
            (ens / "m5_metadata" / "best_metrics.json").write_text(
                json.dumps(
                    {
                        "auc": 0.99,
                        "balanced_accuracy": 0.96,
                        "precision_ai": 0.95,
                        "recall_ai": 0.96,
                        "precision_real": 0.97,
                        "recall_real": 0.95,
                        "predicts_single_class": False,
                        "promotion_eligible": True,
                        "promotion_reason": "ok",
                        "threshold_operable": True,
                        "composite_metrics": {"ece": 0.03, "brier": 0.04},
                    }
                ),
                encoding="utf-8",
            )
            (ens / "m5_metadata" / "config.json").write_text(
                json.dumps({"args": {"backbone": "tiny", "img_size": 256, "use_metadata_features": True}}),
                encoding="utf-8",
            )
            (ens / "m5_metadata" / "inference_spec.json").write_text(
                json.dumps({"schema": "ai-image-detector-runtime-v1", "auxiliary_features": {"enabled": True}}),
                encoding="utf-8",
            )
            (ens / "m5_metadata" / "best_model_summary.json").write_text(json.dumps({"epoch": 4}), encoding="utf-8")
            (ens / "m5_metadata" / "test_metrics.json").write_text(
                json.dumps({"auc": 0.98, "f1": 0.94, "precision_ai": 0.95, "recall_ai": 0.94, "ece": 0.03, "brier": 0.04}),
                encoding="utf-8",
            )
            (ens / "test_metrics.json").write_text(json.dumps({"auc": 0.98}), encoding="utf-8")
            (ens / "robust_eval.json").write_text(json.dumps({"clean": {"auc": 0.97}}), encoding="utf-8")
            (ens / "ensemble_config.json").write_text(json.dumps({"threshold": 0.6}), encoding="utf-8")
            (ens / "domain_config.json").write_text(json.dumps({"base_threshold": 0.6}), encoding="utf-8")
            (ens / "final_run_summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
            (ens / "run_manifest.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
            (ens / "final_thresholds.json").write_text(json.dumps({"ensemble": 0.6}), encoding="utf-8")
            (ens / "prod_manifest.json").write_text(json.dumps({"models": ["dummy"]}), encoding="utf-8")
            (ens / "distill" / "best.safetensors").write_bytes(b"x")
            (ens / "distill" / "best_model_summary.json").write_text(json.dumps({"metrics": {"val_acc": 0.9}}), encoding="utf-8")
            (video / "best_video.safetensors").write_bytes(b"x")
            (video / "best_video_metrics.json").write_text(json.dumps({"val_acc": 0.88}), encoding="utf-8")

            subprocess.run(
                [
                    "python3",
                    "scripts/export_best_release.py",
                    "--ens-out",
                    str(ens),
                    "--video-artifacts",
                    str(video),
                    "--out",
                    str(release),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            manifest = json.loads((release / "release_manifest.json").read_text(encoding="utf-8"))
            latest = (ens / "latest_release.txt").read_text(encoding="utf-8").strip()
            prod_exists = (release / "prod_manifest.json").exists()
            m1_exists = (release / "m1" / "best.safetensors").exists()
            calibration_exists = (release / "m1" / "calibration.json").exists()
            distill_exists = (release / "distill" / "best.safetensors").exists()
            video_exists = (release / "best_video.safetensors").exists()
            public_model_exists = (release / "public_model" / "best.safetensors").exists()
            public_spec_exists = (release / "public_model" / "inference_spec.json").exists()
            public_test_metrics_exists = (release / "public_model" / "test_metrics.json").exists()
            public_profile_exists = (release / "public_model" / "inference_profile.json").exists()
            public_manifest = json.loads((release / "public_model" / "model_manifest.json").read_text(encoding="utf-8"))
            public_profile = json.loads((release / "public_model" / "inference_profile.json").read_text(encoding="utf-8"))
            latest_public = (ens / "latest_public_model.txt").read_text(encoding="utf-8").strip()
            release_path = str(release.resolve())

        self.assertTrue(prod_exists)
        self.assertTrue(m1_exists)
        self.assertTrue(calibration_exists)
        self.assertTrue(distill_exists)
        self.assertTrue(video_exists)
        self.assertTrue(public_model_exists)
        self.assertTrue(public_spec_exists)
        self.assertTrue(public_test_metrics_exists)
        self.assertTrue(public_profile_exists)
        self.assertIn("release_manifest.json", manifest["copied_files"])
        self.assertEqual(manifest["public_model"]["member"], "m5_metadata")
        self.assertEqual(public_manifest["member"], "m5_metadata")
        self.assertTrue(public_manifest["use_metadata_features"])
        self.assertEqual(public_manifest["selection_reason"], "best_promotable_metadata_aware_model")
        self.assertEqual(public_manifest["test_metrics"]["auc"], 0.98)
        self.assertEqual(public_profile["schema"], "ai-image-detector-inference-profile-v1")
        self.assertEqual(public_profile["recommended_output"]["label"], "AI-generated|Real|Unknown")
        self.assertEqual(public_profile["example_output"]["label"], "AI-generated")
        self.assertTrue(latest_public.endswith("public_model/best.safetensors"))
        self.assertEqual(latest, release_path)

    def test_export_best_release_prefers_stronger_pixel_model_over_weaker_metadata_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ens = root / "artifacts_ens"
            release = ens / "release"
            (ens / "m1").mkdir(parents=True, exist_ok=True)
            (ens / "m5_metadata").mkdir(parents=True, exist_ok=True)

            (ens / "m1" / "best.safetensors").write_bytes(b"x")
            (ens / "m1" / "calibration.json").write_text(
                json.dumps({"threshold": 0.61, "temperature": 0.7, "objective": "balanced"}),
                encoding="utf-8",
            )
            (ens / "m1" / "best_metrics.json").write_text(
                json.dumps(
                    {
                        "auc": 0.98,
                        "balanced_accuracy": 0.95,
                        "precision_ai": 0.94,
                        "recall_ai": 0.95,
                        "precision_real": 0.96,
                        "recall_real": 0.95,
                        "predicts_single_class": False,
                        "promotion_eligible": True,
                        "promotion_reason": "ok",
                        "threshold_operable": True,
                    }
                ),
                encoding="utf-8",
            )
            (ens / "m1" / "config.json").write_text(
                json.dumps({"args": {"backbone": "effb0", "img_size": 256, "use_metadata_features": False}}),
                encoding="utf-8",
            )
            (ens / "m1" / "inference_spec.json").write_text(json.dumps({"schema": "ai-image-detector-runtime-v1"}), encoding="utf-8")
            (ens / "m1" / "best_model_summary.json").write_text(json.dumps({"epoch": 3}), encoding="utf-8")
            (ens / "m1" / "test_metrics.json").write_text(
                json.dumps({"auc": 0.99, "f1": 0.96, "precision_ai": 0.97, "recall_ai": 0.95, "precision_real": 0.98, "recall_real": 0.96}),
                encoding="utf-8",
            )

            (ens / "m5_metadata" / "best.safetensors").write_bytes(b"x")
            (ens / "m5_metadata" / "calibration.json").write_text(
                json.dumps({"threshold": 0.55, "temperature": 0.8, "objective": "balanced"}),
                encoding="utf-8",
            )
            (ens / "m5_metadata" / "best_metrics.json").write_text(
                json.dumps(
                    {
                        "auc": 0.97,
                        "balanced_accuracy": 0.92,
                        "precision_ai": 0.90,
                        "recall_ai": 0.91,
                        "precision_real": 0.92,
                        "recall_real": 0.90,
                        "predicts_single_class": False,
                        "promotion_eligible": True,
                        "promotion_reason": "ok",
                        "threshold_operable": True,
                    }
                ),
                encoding="utf-8",
            )
            (ens / "m5_metadata" / "config.json").write_text(
                json.dumps({"args": {"backbone": "tiny", "img_size": 256, "use_metadata_features": True}}),
                encoding="utf-8",
            )
            (ens / "m5_metadata" / "inference_spec.json").write_text(
                json.dumps({"schema": "ai-image-detector-runtime-v1", "auxiliary_features": {"enabled": True}}),
                encoding="utf-8",
            )
            (ens / "m5_metadata" / "best_model_summary.json").write_text(json.dumps({"epoch": 4}), encoding="utf-8")
            (ens / "m5_metadata" / "test_metrics.json").write_text(
                json.dumps({"auc": 0.95, "f1": 0.91, "precision_ai": 0.92, "recall_ai": 0.90, "precision_real": 0.93, "recall_real": 0.89}),
                encoding="utf-8",
            )

            (ens / "prod_manifest.json").write_text(json.dumps({"models": ["dummy"]}), encoding="utf-8")

            subprocess.run(
                [
                    "python3",
                    "scripts/export_best_release.py",
                    "--ens-out",
                    str(ens),
                    "--out",
                    str(release),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            public_manifest = json.loads((release / "public_model" / "model_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(public_manifest["member"], "m1")
        self.assertFalse(public_manifest["use_metadata_features"])
        self.assertEqual(public_manifest["selection_reason"], "best_quality_promotable_pixel_model")


if __name__ == "__main__":
    unittest.main()
