from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from tests._support import ROOT


class InstallShTests(unittest.TestCase):
    def test_install_shell_scripts_are_bash_valid(self) -> None:
        for rel_path in (
            "install.sh",
            "scripts/lib/apt_packages_validate.sh",
        ):
            with self.subTest(script=rel_path):
                subprocess.run(["bash", "-n", rel_path], cwd=ROOT, check=True)

    def test_install_validate_module_compiles(self) -> None:
        import sys

        subprocess.run(
            [sys.executable, "-m", "py_compile", "scripts/lib/install_validate.py"],
            cwd=ROOT,
            check=True,
        )

    def test_install_script_supports_repo_bootstrap_without_zip(self) -> None:
        text = (ROOT / "install.sh").read_text(encoding="utf-8")
        self.assertIn("git clone --depth 1", text)
        self.assertIn("INSTALL_REV", text)
        self.assertIn("install_security_notice", text)
        self.assertIn("INSTALL_ALLOW_CUSTOM_REPO", text)
        self.assertIn("validate_clone_parameters_or_exit", text)
        self.assertIn("install_validate.py", text)
        self.assertIn("source .venv/bin/activate", text)
        self.assertIn("optional", text)
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
        self.assertIn("[DRY_RUN] env SETUP_INSTALL_SYSTEM_DEPS=0 ./local.sh setup", proc.stdout)
        self.assertNotIn("[DRY_RUN] cd . &&", proc.stdout)
        self.assertNotIn("  cd .", proc.stdout)
        self.assertIn("source .venv/bin/activate", proc.stdout)
        self.assertIn("(optional)", proc.stdout)
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
        self.assertIn("[DRY_RUN] cd ai-image-video-detector &&", proc.stdout)
        self.assertIn("env SETUP_INSTALL_SYSTEM_DEPS=0 ./local.sh setup", proc.stdout)
        self.assertIn("  cd ai-image-video-detector", proc.stdout)

    def test_install_script_reuses_extracted_repo_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            extracted_dir = Path(tmpdir) / "ai-image-video-detector-main"
            extracted_dir.mkdir()
            (extracted_dir / "scripts").mkdir()
            lib_dir = extracted_dir / "scripts" / "lib"
            lib_dir.mkdir()
            (lib_dir / "apt_packages_validate.sh").write_text(
                (ROOT / "scripts" / "lib" / "apt_packages_validate.sh").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (lib_dir / "install_validate.py").write_text(
                (ROOT / "scripts" / "lib" / "install_validate.py").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
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
            lib_dir = extracted_dir / "scripts" / "lib"
            lib_dir.mkdir()
            (lib_dir / "apt_packages_validate.sh").write_text(
                (ROOT / "scripts" / "lib" / "apt_packages_validate.sh").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (lib_dir / "install_validate.py").write_text(
                (ROOT / "scripts" / "lib" / "install_validate.py").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
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
        self.assertIn("[DRY_RUN] cd ai-image-video-detector-main &&", proc.stdout)
        self.assertIn("env SETUP_INSTALL_SYSTEM_DEPS=0 ./local.sh setup", proc.stdout)
        self.assertIn("  cd ai-image-video-detector-main", proc.stdout)

    def test_install_script_rejects_unofficial_repo_without_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = subprocess.run(
                ["bash", str(ROOT / "install.sh")],
                cwd=tmpdir,
                env={
                    **os.environ,
                    "DRY_RUN": "1",
                    "INSTALL_SYSTEM_DEPS": "0",
                    "INSTALL_ASSUME_LINUX": "1",
                    "REPO_URL": "https://github.com/example/other.git",
                    "INSTALL_ALLOW_CUSTOM_REPO": "0",
                },
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("install_fail", proc.stderr)

    def test_install_script_accepts_custom_https_repo_when_flag_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = subprocess.run(
                ["bash", str(ROOT / "install.sh")],
                cwd=tmpdir,
                env={
                    **os.environ,
                    "DRY_RUN": "1",
                    "INSTALL_SYSTEM_DEPS": "0",
                    "INSTALL_ASSUME_LINUX": "1",
                    "INSTALL_ALLOW_CUSTOM_REPO": "1",
                    "INSTALL_ALLOW_NON_OFFICIAL_GITHUB_REPO": "1",
                    "REPO_URL": "https://github.com/example/fork.git",
                },
                capture_output=True,
                text=True,
            )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("git clone --depth 1", proc.stdout)

    def test_install_script_rejects_github_fork_without_non_official_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = subprocess.run(
                ["bash", str(ROOT / "install.sh")],
                cwd=tmpdir,
                env={
                    **os.environ,
                    "DRY_RUN": "1",
                    "INSTALL_SYSTEM_DEPS": "0",
                    "INSTALL_ASSUME_LINUX": "1",
                    "INSTALL_ALLOW_CUSTOM_REPO": "1",
                    "INSTALL_ALLOW_NON_OFFICIAL_GITHUB_REPO": "0",
                    "REPO_URL": "https://github.com/example/fork.git",
                },
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("install_fail", proc.stderr)

    def test_apt_package_validator_rejects_injection_token(self) -> None:
        proc = subprocess.run(
            [
                "bash",
                "-c",
                f"source '{ROOT}/scripts/lib/apt_packages_validate.sh' && validate_apt_package_tokens_or_exit 'curl;rm'",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("invalid_apt_package_token", proc.stderr)


if __name__ == "__main__":
    unittest.main()
