from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class InstallShTests(unittest.TestCase):
    def test_install_script_supports_repo_bootstrap_without_zip(self) -> None:
        text = (ROOT / "install.sh").read_text(encoding="utf-8")
        self.assertIn("git clone --depth 1", text)
        self.assertIn("source .venv/bin/activate", text)
        self.assertIn("./local.sh setup", text)
        self.assertNotIn("./local.sh doctor", text)
        self.assertIn("install_status=ready", text)

    def test_install_script_dry_run_works_inside_repo(self) -> None:
        proc = subprocess.run(
            ["bash", "./install.sh"],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "1",
                "INSTALL_SYSTEM_DEPS": "0",
                "INSTALL_ASSUME_LINUX": "1",
            },
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("install_stage=repo status=using_current repo=.", proc.stdout)
        self.assertIn("[DRY_RUN] SETUP_INSTALL_SYSTEM_DEPS=0 ./local.sh setup", proc.stdout)
        self.assertNotIn("[DRY_RUN] cd . && SETUP_INSTALL_SYSTEM_DEPS=0 ./local.sh setup", proc.stdout)
        self.assertNotIn("  cd .", proc.stdout)
        self.assertIn("source .venv/bin/activate", proc.stdout)
        self.assertIn("./local.sh setup", proc.stdout)
        self.assertNotIn("./local.sh doctor", proc.stdout)
        self.assertIn("install_status=ready", proc.stdout)
        self.assertNotIn(str(ROOT), proc.stdout)

    def test_install_script_keeps_cd_for_cloned_repo_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = subprocess.run(
                ["bash", str(ROOT / "install.sh")],
                cwd=tmpdir,
                env={
                    **os.environ,
                    "DRY_RUN": "1",
                    "INSTALL_SYSTEM_DEPS": "0",
                    "INSTALL_ASSUME_LINUX": "1",
                },
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("install_stage=repo status=cloned repo=ai-image-video-detector", proc.stdout)
        self.assertIn("[DRY_RUN] cd ai-image-video-detector && SETUP_INSTALL_SYSTEM_DEPS=0 ./local.sh setup", proc.stdout)
        self.assertIn("  cd ai-image-video-detector", proc.stdout)

    def test_install_script_reuses_extracted_repo_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            extracted_dir = Path(tmpdir) / "ai-image-video-detector-main"
            extracted_dir.mkdir()
            (extracted_dir / "scripts").mkdir()
            (extracted_dir / "install.sh").write_text((ROOT / "install.sh").read_text(encoding="utf-8"), encoding="utf-8")
            (extracted_dir / "local.sh").write_text((ROOT / "local.sh").read_text(encoding="utf-8"), encoding="utf-8")
            (extracted_dir / "scripts" / "install_deps.sh").write_text(
                (ROOT / "scripts" / "install_deps.sh").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            proc = subprocess.run(
                ["bash", "./install.sh"],
                cwd=extracted_dir,
                env={
                    **os.environ,
                    "DRY_RUN": "1",
                    "INSTALL_SYSTEM_DEPS": "0",
                    "INSTALL_ASSUME_LINUX": "1",
                },
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("install_stage=repo status=using_current repo=.", proc.stdout)
        self.assertIn("install_status=ready", proc.stdout)

    def test_install_script_reuses_extracted_repo_directory_from_parent_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            extracted_dir = tmp / "ai-image-video-detector-main"
            extracted_dir.mkdir()
            (extracted_dir / "scripts").mkdir()
            (extracted_dir / "install.sh").write_text((ROOT / "install.sh").read_text(encoding="utf-8"), encoding="utf-8")
            (extracted_dir / "local.sh").write_text((ROOT / "local.sh").read_text(encoding="utf-8"), encoding="utf-8")
            (extracted_dir / "scripts" / "install_deps.sh").write_text(
                (ROOT / "scripts" / "install_deps.sh").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            proc = subprocess.run(
                ["bash", str(extracted_dir / "install.sh")],
                cwd=tmp,
                env={
                    **os.environ,
                    "DRY_RUN": "1",
                    "INSTALL_SYSTEM_DEPS": "0",
                    "INSTALL_ASSUME_LINUX": "1",
                    "INSTALL_DIR": str(extracted_dir),
                },
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("install_stage=repo status=using_extracted repo=ai-image-video-detector-main", proc.stdout)
        self.assertIn("[DRY_RUN] cd ai-image-video-detector-main && SETUP_INSTALL_SYSTEM_DEPS=0 ./local.sh setup", proc.stdout)
        self.assertIn("  cd ai-image-video-detector-main", proc.stdout)


if __name__ == "__main__":
    unittest.main()
