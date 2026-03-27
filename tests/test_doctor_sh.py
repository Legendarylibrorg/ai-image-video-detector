from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DoctorShTests(unittest.TestCase):
    def test_doctor_uses_custom_venv_dir_for_dep_and_token_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            venv_dir = tmp / "custom-venv"
            bin_dir = venv_dir / "bin"
            bin_dir.mkdir(parents=True)
            fake_python = bin_dir / "python"
            fake_python.write_text("#!/usr/bin/env bash\ncat >/dev/null\nexit 0\n", encoding="utf-8")
            fake_python.chmod(0o755)
            for name in ("hf", "aid-train", "aid-video-train"):
                fake_cli = bin_dir / name
                fake_cli.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
                fake_cli.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "VENV_DIR": str(venv_dir),
                    "HF_TOKEN": "from_env",
                    "HUGGINGFACE_HUB_TOKEN": "",
                    "DOCTOR_REQUIRE_TOKEN": "0",
                }
            )
            proc = subprocess.run(
                ["bash", "scripts/doctor.sh"],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn(f"doctor_ok: venv_present path={venv_dir}", proc.stdout)
        self.assertIn("doctor_ok: core_python_deps=ok", proc.stdout)
        self.assertIn("doctor_ok: core_cli_deps=ok", proc.stdout)
        self.assertIn("doctor_ok: hf_token_validation=ok", proc.stdout)
        self.assertNotIn("doctor_warn: venv_missing", proc.stdout)

    def test_doctor_can_require_gpu_and_clamav(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "DOCTOR_REQUIRE_GPU": "1",
                "DOCTOR_REQUIRE_CLAMAV": "1",
                "DOCTOR_REQUIRE_TOKEN": "0",
            }
        )
        proc = subprocess.run(
            ["bash", "scripts/doctor.sh"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )

        self.assertEqual(proc.returncode, 2)
        self.assertIn("doctor_fail: nvidia_smi_missing gpu_required=1", proc.stdout)
        self.assertIn("doctor_fail: clamscan_missing clamav_required=1", proc.stdout)

    def test_doctor_can_require_docker(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "DOCTOR_REQUIRE_DOCKER": "1",
                "DOCTOR_REQUIRE_TOKEN": "0",
            }
        )
        proc = subprocess.run(
            ["bash", "scripts/doctor.sh"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )

        self.assertEqual(proc.returncode, 2)
        self.assertIn("doctor_ok: docker_path_ready path=", proc.stdout)
        self.assertIn("doctor_fail: docker_missing docker_required=1", proc.stdout)


if __name__ == "__main__":
    unittest.main()
