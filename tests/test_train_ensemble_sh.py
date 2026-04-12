from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from tests._support import ROOT


class TrainEnsembleShTests(unittest.TestCase):
    def test_common_training_flags_are_forwarded_to_each_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bin_dir = tmp / "bin"
            data_dir = tmp / "data"
            out_dir = tmp / "out"
            log_path = tmp / "aid_train_calls.log"

            bin_dir.mkdir()
            data_dir.mkdir()

            stub = bin_dir / "aid-train"
            stub.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "{ printf '%q ' \"$@\"; printf '\\n'; } >> \"$AID_TRAIN_LOG\"\n"
                "out=\"\"\n"
                "while [[ \"$#\" -gt 0 ]]; do\n"
                "  if [[ \"$1\" == \"--out\" ]]; then\n"
                "    out=\"$2\"\n"
                "    break\n"
                "  fi\n"
                "  shift\n"
                "done\n"
                "mkdir -p \"$out\"\n"
                "touch \"$out/best.safetensors\"\n",
                encoding="utf-8",
            )
            stub.chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = f"{bin_dir}:{env['PATH']}"
            env["AID_TRAIN_LOG"] = str(log_path)
            env["TRAIN_NO_PRETRAINED_BACKBONE"] = "1"
            env["TRAIN_NO_COMPILE"] = "1"
            env["TRAIN_NUM_WORKERS"] = "3"
            env["TRAIN_PATIENCE"] = "5"
            env["TRAIN_MIN_DELTA"] = "0.0005"
            env["TRAIN_DEGENERATE_PATIENCE"] = "3"

            subprocess.run(
                ["bash", "scripts/train_ensemble.sh", str(data_dir), str(out_dir), "1"],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            lines = log_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 5)
            metadata_lines = [line for line in lines if "--use-metadata-features" in line]
            self.assertEqual(len(metadata_lines), 1)
            self.assertIn("--init-from", metadata_lines[0])
            for line in lines:
                self.assertIn("--no-pretrained-backbone", line)
                self.assertIn("--no-compile", line)
                self.assertIn("--num-workers 3", line)
                self.assertIn("--patience 5", line)
                self.assertIn("--min-delta 0.0005", line)
                self.assertIn("--degenerate-patience 3", line)


if __name__ == "__main__":
    unittest.main()
