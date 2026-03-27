from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ExportBestReleaseTests(unittest.TestCase):
    def test_export_best_release_writes_release_bundle_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ens = root / "artifacts_ens"
            video = root / "video_artifacts"
            release = ens / "release"
            (ens / "m1").mkdir(parents=True, exist_ok=True)
            (ens / "distill").mkdir(parents=True, exist_ok=True)
            video.mkdir(parents=True, exist_ok=True)

            (ens / "m1" / "best.safetensors").write_bytes(b"x")
            (ens / "m1" / "calibration.json").write_text(json.dumps({"threshold": 0.61}), encoding="utf-8")
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
            release_path = str(release.resolve())

        self.assertTrue(prod_exists)
        self.assertTrue(m1_exists)
        self.assertTrue(calibration_exists)
        self.assertTrue(distill_exists)
        self.assertTrue(video_exists)
        self.assertIn("release_manifest.json", manifest["copied_files"])
        self.assertEqual(latest, release_path)


if __name__ == "__main__":
    unittest.main()
