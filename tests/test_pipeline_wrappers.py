from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PipelineWrapperTests(unittest.TestCase):
    def test_full_pipeline_dry_run_allows_missing_venv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_venv = Path(tmpdir) / "missing-venv"
            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "MALWARE_SCAN": "0",
                    "VENV_DIR": str(missing_venv),
                }
            )
            proc = subprocess.run(
                ["bash", "scripts/full_pipeline_4090.sh"],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn(f"[DRY_RUN] source {missing_venv / 'bin' / 'activate'}", proc.stdout)
        self.assertIn("Pipeline complete.", proc.stdout)

    def test_one_command_4090_dry_run_skips_real_apt_and_missing_venv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bin_dir = tmp / "bin"
            bin_dir.mkdir()
            for name in ("apt-get", "sudo", "freshclam"):
                stub = bin_dir / name
                stub.write_text("#!/usr/bin/env bash\nexit 99\n", encoding="utf-8")
                stub.chmod(0o755)

            missing_venv = tmp / "missing-venv"
            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "MALWARE_SCAN": "0",
                    "VENV_DIR": str(missing_venv),
                    "PATH": f"{bin_dir}:{env['PATH']}",
                }
            )
            proc = subprocess.run(
                ["bash", "scripts/one_command_4090.sh"],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("[DRY_RUN] sudo apt-get update", proc.stdout)
        self.assertIn(f"[DRY_RUN] source {missing_venv / 'bin' / 'activate'}", proc.stdout)
        self.assertIn("[DRY_RUN] bash scripts/full_pipeline_4090.sh", proc.stdout)


if __name__ == "__main__":
    unittest.main()
