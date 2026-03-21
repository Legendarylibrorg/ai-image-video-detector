from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkGateTests(unittest.TestCase):
    def test_skip_video_allows_image_only_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ens = root / "artifacts_ens"
            video = root / "video_artifacts"
            ens.mkdir(parents=True, exist_ok=True)
            video.mkdir(parents=True, exist_ok=True)

            (ens / "test_metrics.json").write_text(json.dumps({"auc": 0.99, "f1": 0.95}), encoding="utf-8")
            (ens / "prod_manifest.json").write_text("{}", encoding="utf-8")
            (ens / "ensemble_config.json").write_text("{}", encoding="utf-8")
            (ens / "domain_config.json").write_text("{}", encoding="utf-8")

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


if __name__ == "__main__":
    unittest.main()
