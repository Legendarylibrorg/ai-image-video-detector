from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from tests._support import ROOT


class PipelineReportingTests(unittest.TestCase):
    def test_dataset_report_writes_qa_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data = root / "data_best"
            prepared = root / "prepared"
            video = root / "video_data"
            reports = root / "reports"
            cache_file = root / "sources.txt"

            for base in (data, prepared):
                for split in ("train", "val", "test"):
                    for cls in ("ai", "real"):
                        bucket = base / split / cls
                        bucket.mkdir(parents=True, exist_ok=True)
                        (bucket / f"{split}_{cls}.jpg").write_bytes(b"x")
            for split in ("train", "val"):
                for cls in ("ai", "real"):
                    bucket = video / split / cls
                    bucket.mkdir(parents=True, exist_ok=True)
                    (bucket / f"{split}_{cls}.mp4").write_bytes(b"x")

            (data / "dataset_build_report.json").write_text(json.dumps({"full_targets_ok": True}), encoding="utf-8")
            (data / "dataset_run_summary.json").write_text(json.dumps({"hf_sources_used": 2}), encoding="utf-8")
            (prepared / "training_data_report.json").write_text(json.dumps({"complete_image_dataset": True}), encoding="utf-8")
            cache_file.write_text("repo/one\nrepo/two\n", encoding="utf-8")

            qa_out = reports / "dataset_qa_summary.json"
            provenance_out = reports / "dataset_provenance.json"
            proc = subprocess.run(
                [
                    "python3",
                    "scripts/write_pipeline_report.py",
                    "dataset",
                    "--data",
                    str(data),
                    "--prepared",
                    str(prepared),
                    "--video",
                    str(video),
                    "--cache-file",
                    str(cache_file),
                    "--out",
                    str(qa_out),
                    "--provenance-out",
                    str(provenance_out),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertEqual(proc.stdout, "")
            qa = json.loads(qa_out.read_text(encoding="utf-8"))
            provenance = json.loads(provenance_out.read_text(encoding="utf-8"))

        self.assertTrue(qa["qa_checks"]["collected_complete"])
        self.assertTrue(qa["qa_checks"]["prepared_complete"])
        self.assertTrue(qa["qa_checks"]["video_complete"])
        self.assertEqual(provenance["hf_discovery_sources"]["count"], 2)
        self.assertEqual(provenance["dataset_run_summary"]["hf_sources_used"], 2)

    def test_final_report_writes_manifest_thresholds_and_prod_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ens = root / "artifacts_ens"
            video_artifacts = root / "video_artifacts"
            data = root / "data_best"
            prepared = root / "prepared"
            video = root / "video_data"
            ens.mkdir(parents=True, exist_ok=True)
            video_artifacts.mkdir(parents=True, exist_ok=True)

            for name in ("m1", "m2", "m3", "m4"):
                model_dir = ens / name
                model_dir.mkdir(parents=True, exist_ok=True)
                (model_dir / "best.safetensors").write_bytes(b"x")
                (model_dir / "best_metrics.json").write_text(
                    json.dumps(
                        {
                            "auc": 0.9,
                            "f1": 0.8,
                            "balanced_accuracy": 0.89,
                            "precision_ai": 0.88,
                            "recall_ai": 0.89,
                            "precision_real": 0.90,
                            "recall_real": 0.88,
                            "predicts_single_class": False,
                            "promotion_eligible": True,
                            "promotion_reason": "ok",
                            "threshold_operable": True,
                            "composite_metrics": {"ece": 0.04, "brier": 0.05},
                        }
                    ),
                    encoding="utf-8",
                )
                (model_dir / "calibration.json").write_text(
                    json.dumps({"threshold": 0.42, "temperature": 0.7, "objective": "balanced"}),
                    encoding="utf-8",
                )
                (model_dir / "config.json").write_text(
                    json.dumps({"args": {"backbone": "effb0", "img_size": 256, "use_metadata_features": False}}),
                    encoding="utf-8",
                )
                (model_dir / "test_metrics.json").write_text(
                    json.dumps(
                        {
                            "auc": 0.96,
                            "f1": 0.92,
                            "precision_ai": 0.91,
                            "recall_ai": 0.92,
                            "precision_real": 0.93,
                            "recall_real": 0.91,
                            "ece": 0.04,
                            "brier": 0.05,
                        }
                    ),
                    encoding="utf-8",
                )
            metadata_dir = ens / "m5_metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            (metadata_dir / "best.safetensors").write_bytes(b"x")
            (metadata_dir / "best_metrics.json").write_text(
                json.dumps(
                    {
                        "auc": 0.99,
                        "f1": 0.82,
                        "balanced_accuracy": 0.91,
                        "precision_ai": 0.90,
                        "recall_ai": 0.91,
                        "precision_real": 0.92,
                        "recall_real": 0.90,
                        "predicts_single_class": False,
                        "promotion_eligible": True,
                        "promotion_reason": "ok",
                        "threshold_operable": True,
                        "composite_metrics": {"ece": 0.03, "brier": 0.04},
                    }
                ),
                encoding="utf-8",
            )
            (metadata_dir / "calibration.json").write_text(
                json.dumps({"threshold": 0.35, "temperature": 0.8, "objective": "balanced"}),
                encoding="utf-8",
            )
            (metadata_dir / "config.json").write_text(
                json.dumps({"args": {"backbone": "tiny", "img_size": 256, "use_metadata_features": True}}),
                encoding="utf-8",
            )
            (metadata_dir / "test_metrics.json").write_text(
                json.dumps(
                    {
                        "auc": 0.98,
                        "f1": 0.94,
                        "precision_ai": 0.95,
                        "recall_ai": 0.94,
                        "precision_real": 0.96,
                        "recall_real": 0.93,
                        "ece": 0.03,
                        "brier": 0.04,
                    }
                ),
                encoding="utf-8",
            )
            (ens / "test_metrics.json").write_text(json.dumps({"auc": 0.97, "f1": 0.91}), encoding="utf-8")
            (ens / "ensemble_config.json").write_text(json.dumps({"threshold": 0.55, "fit": {"objective": "balanced"}}), encoding="utf-8")
            (ens / "domain_config.json").write_text(json.dumps({"base_threshold": 0.55, "thresholds": {"screenshot": 0.6}}), encoding="utf-8")
            (ens / "robust_eval.json").write_text(json.dumps({"clean": {"auc": 0.96}, "jpeg_q35": {"auc": 0.88}}), encoding="utf-8")
            distill_dir = ens / "distill"
            distill_dir.mkdir(parents=True, exist_ok=True)
            (distill_dir / "best.safetensors").write_bytes(b"x")
            (distill_dir / "best_model_summary.json").write_text(json.dumps({"metrics": {"val_acc": 0.87}}), encoding="utf-8")
            (video_artifacts / "best_video.safetensors").write_bytes(b"x")
            (video_artifacts / "best_video_metrics.json").write_text(json.dumps({"val_acc": 0.88, "threshold": 0.5}), encoding="utf-8")
            dataset_qa = root / "dataset_qa_summary.json"
            dataset_qa.write_text(json.dumps({"qa_checks": {"collected_complete": True}}), encoding="utf-8")

            final_out = ens / "final_run_summary.json"
            manifest_out = ens / "run_manifest.json"
            thresholds_out = ens / "final_thresholds.json"
            prod_manifest = ens / "prod_manifest.json"
            release_bundle = ens / "release"

            subprocess.run(
                [
                    "python3",
                    "scripts/write_pipeline_report.py",
                    "final",
                    "--data",
                    str(data),
                    "--prepared",
                    str(prepared),
                    "--video",
                    str(video),
                    "--ens-out",
                    str(ens),
                    "--ensemble-config",
                    str(ens / "ensemble_config.json"),
                    "--domain-config",
                    str(ens / "domain_config.json"),
                    "--video-artifacts",
                    str(video_artifacts),
                    "--dataset-qa",
                    str(dataset_qa),
                    "--robust-eval",
                    str(ens / "robust_eval.json"),
                    "--prod-manifest",
                    str(prod_manifest),
                    "--summary-out",
                    str(final_out),
                    "--manifest-out",
                    str(manifest_out),
                    "--thresholds-out",
                    str(thresholds_out),
                    "--release-bundle",
                    str(release_bundle),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            summary = json.loads(final_out.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
            thresholds = json.loads(thresholds_out.read_text(encoding="utf-8"))
            prod = json.loads(prod_manifest.read_text(encoding="utf-8"))

        self.assertEqual(summary["image_test_metrics"]["auc"], 0.97)
        self.assertEqual(summary["robust_eval"]["jpeg_q35"]["auc"], 0.88)
        self.assertEqual(summary["video_metrics"]["val_acc"], 0.88)
        self.assertEqual(summary["distilled_model"]["metrics"]["val_acc"], 0.87)
        self.assertTrue(summary["release_bundle"].endswith("release"))
        self.assertEqual(thresholds["ensemble"], 0.55)
        self.assertEqual(thresholds["domain_thresholds"]["screenshot"], 0.6)
        self.assertEqual(len(prod["models"]), 5)
        self.assertTrue(prod["distilled_model"].endswith("best.safetensors"))
        self.assertTrue(prod["robust_eval"].endswith("robust_eval.json"))
        self.assertTrue(prod["release_bundle"].endswith("release"))
        self.assertEqual(prod["public_model"]["member"], "m5_metadata")
        self.assertTrue(prod["public_model"]["use_metadata_features"])
        self.assertEqual(prod["public_model"]["selection_reason"], "best_promotable_metadata_aware_model")
        self.assertTrue(prod["public_model"]["public_checkpoint"].endswith("release/public_model/best.safetensors"))
        self.assertEqual(summary["public_model"]["member"], "m5_metadata")
        self.assertIn("preferred_checkpoints", manifest)
        self.assertTrue(manifest["artifacts"]["release_bundle"].endswith("release"))


if __name__ == "__main__":
    unittest.main()
