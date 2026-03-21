from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SetupLinuxTests(unittest.TestCase):
    def run_setup_linux_with_stage_marker(self, stage: str) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            stage_dir = Path(tmpdir)
            (stage_dir / f"{stage}.done").write_text("done\n", encoding="utf-8")
            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "HF_SETUP_REQUIRE_TOKEN": "0",
                    "SETUP_ENV_FILE": str(stage_dir / ".env"),
                    "SETUP_STAGE_DIR": str(stage_dir),
                    "HF_TOKEN": "",
                    "HUGGINGFACE_HUB_TOKEN": "",
                }
            )
            proc = subprocess.run(
                ["bash", "scripts/setup_linux.sh"],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            return proc.stdout

    def run_setup_linux(self, *, env_file_text: str = "", extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            stage_dir = Path(tmpdir)
            env_file = stage_dir / ".env"
            env_file.write_text(env_file_text, encoding="utf-8")
            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "SETUP_ENV_FILE": str(env_file),
                    "SETUP_STAGE_DIR": str(stage_dir),
                    "HF_TOKEN": "",
                    "HUGGINGFACE_HUB_TOKEN": "",
                }
            )
            if extra_env:
                env.update(extra_env)
            return subprocess.run(
                ["bash", "scripts/setup_linux.sh"],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

    def test_setup_full_reruns_python_deps_even_with_stage_marker(self) -> None:
        out = self.run_setup_linux_with_stage_marker("python_deps")
        self.assertIn("setup_stage=python_deps status=run", out)
        self.assertNotIn("setup_stage=python_deps status=skip_done", out)

    def test_setup_full_reruns_hf_token_validation_even_with_stage_marker(self) -> None:
        out = self.run_setup_linux_with_stage_marker("hf_token")
        self.assertIn("setup_stage=hf_token status=run", out)
        self.assertNotIn("setup_stage=hf_token status=skip_done", out)

    def test_setup_full_uses_env_file_token_when_exported_token_is_empty(self) -> None:
        proc = self.run_setup_linux(
            env_file_text="HF_TOKEN=from_file\n",
            extra_env={"HF_SETUP_REQUIRE_TOKEN": "1"},
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("[DRY_RUN] validate_hf_token", proc.stdout)
        self.assertNotIn("hf_token_status=missing_noninteractive", proc.stdout)


if __name__ == "__main__":
    unittest.main()
