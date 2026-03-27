from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkGateTests(unittest.TestCase):
    def test_skip_video_allows_strong_image_only_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ens = root / "artifacts_ens"
            video = root / "video_artifacts"
            ens.mkdir(parents=True, exist_ok=True)
            video.mkdir(parents=True, exist_ok=True)

            (ens / "test_metrics.json").write_text(
                json.dumps({
                    "auc": 0.97,
                    "f1": 0.93,
                    "precision": 0.91,
                    "recall": 0.92,
                    "ece": 0.03,
                    "brier": 0.07,
                }),
                encoding="utf-8",
            )
            (ens / "prod_manifest.json").write_text("{}", encoding="utf-8")
            (ens / "ensemble_config.json").write_text("{}", encoding="utf-8")
            (ens / "domain_config.json").write_text("{}", encoding="utf-8")
            (ens / "robust_eval.json").write_text(
                json.dumps({
                    "clean": {"auc": 0.97, "f1": 0.93},
                    "jpeg_q60": {"auc": 0.94, "f1": 0.90},
                    "jpeg_q35": {"auc": 0.91, "f1": 0.86},
                    "blur": {"auc": 0.92, "f1": 0.87},
                }),
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    "python3",
                    "scripts/benchmark_gate.py",
                    "--ens-out",
                    str(ens),
                    "--video-out",
                    str(video),
                    "--skip-video",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            self.assertIn('"video_best_val_acc": "skipped"', proc.stdout)
            self.assertIn('"robust_worst_auc": 0.91', proc.stdout)

    def test_gate_rejects_weak_robust_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ens = root / "artifacts_ens"
            video = root / "video_artifacts"
            ens.mkdir(parents=True, exist_ok=True)
            video.mkdir(parents=True, exist_ok=True)

            (ens / "test_metrics.json").write_text(
                json.dumps({
                    "auc": 0.97,
                    "f1": 0.93,
                    "precision": 0.91,
                    "recall": 0.92,
                    "ece": 0.03,
                    "brier": 0.07,
                }),
                encoding="utf-8",
            )
            (ens / "prod_manifest.json").write_text("{}", encoding="utf-8")
            (ens / "ensemble_config.json").write_text("{}", encoding="utf-8")
            (ens / "domain_config.json").write_text("{}", encoding="utf-8")
            (ens / "robust_eval.json").write_text(
                json.dumps({
                    "clean": {"auc": 0.97, "f1": 0.93},
                    "jpeg_q60": {"auc": 0.93, "f1": 0.89},
                    "jpeg_q35": {"auc": 0.88, "f1": 0.82},
                }),
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    "python3",
                    "scripts/benchmark_gate.py",
                    "--ens-out",
                    str(ens),
                    "--video-out",
                    str(video),
                    "--skip-video",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
            self.assertIn("robust_worst_auc 0.8800 < 0.9000", proc.stdout)
            self.assertIn("robust_worst_f1 0.8200 < 0.8500", proc.stdout)
            self.assertIn("robust_auc_drop 0.0900 > 0.0800", proc.stdout)


if __name__ == "__main__":
    unittest.main()
